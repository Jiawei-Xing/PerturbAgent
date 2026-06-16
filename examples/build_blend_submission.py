"""Build a DIR-blended Track B submission: LLM judge + Geneformer on covered rows.

The Geneformer in-silico probe (geneformer_probe.py) found that its signed
direction signal (dir_dlogit) is strongest on exactly the DE rows where the LLM
DIR judge is weakest (covered rows = pert & target both highly expressed in the
macrophage context). This script banks that complementarity:

  for each test row that Geneformer COVERS, replace the direction estimate with
      P_up|DE_new = (1-w) * P_up|DE_LLM  +  w * percentile(dir_dlogit)
  and leave UNCOVERED rows exactly as the LLM produced them.

P_DE (= pred_up + pred_down) is preserved exactly, so the blend touches ONLY the
DIR component -- DE AUROC is unchanged by construction; only the up/down split of
covered rows moves. percentile(dir_dlogit) ranks covered rows into (0,1) (higher
= more up), making it commensurate with the LLM probability.

    uv run python examples/build_blend_submission.py            # build + zip
    uv run python examples/build_blend_submission.py --validate # offline check only
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rankpct(x: np.ndarray) -> np.ndarray:
    """Rank -> (0,1) percentile, ties averaged."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[order] = np.arange(1, len(x) + 1)
    s = x[order]; i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r / (len(x) + 1.0)


def parse_trace(tr: str):
    try:
        d = json.loads(tr)
        return float(d["P_DE"]), float(d["P_up_given_DE"])
    except Exception:
        return None, None


def blended_pup(p_up_llm: np.ndarray, dir_dlogit: np.ndarray,
                covered: np.ndarray, w: float) -> np.ndarray:
    """Return new P_up|DE: blended on covered rows, untouched elsewhere."""
    out = p_up_llm.astype(float).copy()
    cov = covered == 1
    if cov.sum() == 0:
        return out
    gf_pct = np.full(len(out), 0.5)
    gf_pct[cov] = _rankpct(dir_dlogit[cov])
    out[cov] = (1 - w) * p_up_llm[cov] + w * gf_pct[cov]
    return out


# --------------------------------------------------------------------------- #
# Offline validation on labeled train rows
# --------------------------------------------------------------------------- #
def _rankdata(a):
    a = np.asarray(a, float); o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    s = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j+1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def auroc(y, s):
    y = np.asarray(y).astype(int); P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = _rankdata(s); return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def validate(w: float):
    """Apply the exact blend on labeled train rows (LLM preds from benchmark_b_dir)."""
    raw = json.loads((ROOT / "outputs/benchmark_b_dir/preds.json").read_text())
    llm = {k.split("|", 1)[1]: v[3] for k, v in raw.items() if k.startswith("dir_base|")}
    samp = pd.read_csv(ROOT / "outputs/benchmark_b_dir/sample.csv")
    gf = pd.read_csv(ROOT / "outputs/geneformer_probe/features_full.csv")[
        ["id", "dir_dlogit", "covered"]]
    df = samp.merge(gf, on="id", how="left")
    df["llm"] = df["id"].map(llm)
    de = df[(df.label != "none") & df.llm.notna()].reset_index(drop=True)
    de["covered"] = de["covered"].fillna(0)
    up = (de.label == "up").astype(int).to_numpy()
    new = blended_pup(de.llm.to_numpy(), de.dir_dlogit.fillna(0).to_numpy(),
                      de.covered.to_numpy(), w)
    cov = de.covered.to_numpy() == 1
    print(f"  validation pool: {len(de)} DE rows, {int(cov.sum())} covered")
    print(f"  aggregate DIR : LLM {auroc(up, de.llm):.4f} -> blend {auroc(up, new):.4f}  "
          f"(delta {auroc(up, new)-auroc(up, de.llm):+.4f})")
    print(f"  covered-only  : LLM {auroc(up[cov], de.llm.to_numpy()[cov]):.4f} -> "
          f"blend {auroc(up[cov], new[cov]):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path,
                    default=ROOT / "outputs/track_b_adversarial_sharp/submission.csv")
    ap.add_argument("--gf-test", type=Path,
                    default=ROOT / "outputs/geneformer_probe/features_test.csv")
    ap.add_argument("--weight", type=float, default=0.5, help="weight on Geneformer")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs/track_b_adversarial_blend")
    ap.add_argument("--validate", action="store_true", help="offline check only, no build")
    args = ap.parse_args()

    for w in ([0.3, 0.5, 0.7] if args.validate else [args.weight]):
        print(f"\n[validate w={w}]")
        validate(w)
    if args.validate:
        return

    sub = pd.read_csv(args.base)
    pde, pup = zip(*[parse_trace(t) for t in sub["reasoning_trace"]])
    pde = np.array(pde, float); pup = np.array(pup, float)
    gf = pd.read_csv(args.gf_test).set_index("id")
    cov = sub["id"].map(gf["covered"]).fillna(0).to_numpy()
    dlg = sub["id"].map(gf["dir_dlogit"]).fillna(0).to_numpy()

    bad = np.isnan(pde) | np.isnan(pup)
    pde = np.where(bad, sub["prediction_up"] + sub["prediction_down"], pde)
    pup = np.where(bad, 0.5, pup)
    new_pup = blended_pup(pup, dlg, cov, args.weight)

    out = sub.copy()
    out["prediction_up"] = (pde * new_pup).round(6)
    out["prediction_down"] = (pde * (1 - new_pup)).round(6)
    n_changed = int(((cov == 1)).sum())
    print(f"\n[build] base={args.base.parent.name}  weight={args.weight}  "
          f"covered rows changed: {n_changed}/{len(out)}")

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "submission.csv"
    out.to_csv(out_csv, index=False)
    with zipfile.ZipFile(args.out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, "submission.csv")
    print(f"[build] -> {out_csv} (+ submission.zip)")


if __name__ == "__main__":
    main()
