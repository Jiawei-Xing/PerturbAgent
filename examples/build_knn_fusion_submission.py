"""Fuse the gene-similarity kNN aggregator into the LLM agent's predictions.

knn_transfer_test.py showed gene-similarity transfer (legal across the disjoint
split) gives DE ~0.55 and DIR ~0.62 -- the DIR beating the LLM judge. This
fuses the two:

  fused_P_DE   = (1-w_de) *pct(P_DE_LLM)  + w_de *pct(prior_DE_kNN)
  fused_P_up|DE= (1-w_dir)*pct(P_up_LLM)  + w_dir*pct(prior_up_kNN)
  pred_up = fused_P_DE * fused_P_up|DE ;  pred_down = fused_P_DE * (1-fused_P_up|DE)

pct() = rank-percentile within the population (scale-robust; AUROC cares only
about rank). Uncovered rows (gene has no embedding) fall back to the LLM alone.

For the TEST submission the kNN knowledge base is the full train (test genes are
disjoint from train by construction). For weight selection the kNN is evaluated
leave-one-out with disjoint masking on the benchmark_b_dir rows (which also carry
LLM predictions), so weights are tuned on a high-power, honestly-split pool --
NOT the n=59 in-sample trap that sank the Geneformer blend.

    uv run python examples/build_knn_fusion_submission.py --validate   # tune weights
    uv run python examples/build_knn_fusion_submission.py              # build + zip
"""
from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
GF = ROOT / "outputs/geneformer_probe/model/geneformer"
SAFET = ROOT / "outputs/geneformer_probe/model/Geneformer-V2-104M/model.safetensors"


# ----- metrics -----
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


def pct(a):
    a = np.asarray(a, float)
    out = np.full(len(a), 0.5)
    m = ~np.isnan(a)
    if m.sum() > 1:
        out[m] = _rankdata(a[m]) / (m.sum() + 1.0)
    return out


def auroc(y, s):
    y = np.asarray(y).astype(int); P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = _rankdata(np.asarray(s, float))
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def boot_delta(y, s_new, s_old, n=4000, seed=0):
    y = np.asarray(y).astype(int); rng = np.random.default_rng(seed); N = len(y); d = []
    s_new = np.asarray(s_new, float); s_old = np.asarray(s_old, float)
    for _ in range(n):
        idx = rng.integers(0, N, N); yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        d.append(auroc(yy, s_new[idx]) - auroc(yy, s_old[idx]))
    d = np.array(d)
    return float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


# ----- embeddings + kNN -----
def load_emb():
    tok = pickle.load(open(GF / "token_dictionary_gc104M.pkl", "rb"))
    nm = pickle.load(open(GF / "gene_name_id_dict_gc104M.pkl", "rb"))
    nm_ci = {str(k).upper(): v for k, v in nm.items()}
    with safe_open(str(SAFET), framework="numpy") as f:
        W = f.get_tensor("bert.embeddings.word_embeddings.weight")
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return tok, nm_ci, W


def gene_emb(genes, tok, nm_ci, W):
    D = W.shape[1]
    out = np.zeros((len(genes), D)); cov = np.zeros(len(genes), bool)
    for i, g in enumerate(genes):
        e = nm_ci.get(str(g).upper())
        if e is not None and e in tok:
            out[i] = W[tok[e]]; cov[i] = True
    return out, cov


def knn_priors(q_pert, q_gene, kb, tok, nm_ci, W, power=2.0, disjoint=True):
    """Gene-similarity kNN priors for query rows against knowledge base kb."""
    Egq, covq = gene_emb(q_gene, tok, nm_ci, W)
    Egk, _ = gene_emb(kb["gene"].to_numpy(), tok, nm_ci, W)
    isde = (kb["label"].to_numpy() != "none").astype(float)
    isup = (kb["label"].to_numpy() == "up").astype(float)
    kb_p = kb["pert"].to_numpy(); kb_g = kb["gene"].to_numpy()
    sim = np.clip(Egq @ Egk.T, 0, None) ** power            # (Nq, Nkb)
    if disjoint:
        same = (q_pert[:, None] == kb_p[None, :]) | (q_gene[:, None] == kb_g[None, :])
        sim[same] = 0.0
    wsum = sim.sum(1)
    prior_de = np.where(wsum > 0, (sim * isde).sum(1) / np.where(wsum > 0, wsum, 1), np.nan)
    wde = sim * isde
    wdesum = wde.sum(1)
    prior_up = np.where(wdesum > 0, (wde * isup).sum(1) / np.where(wdesum > 0, wdesum, 1), np.nan)
    prior_de[~covq] = np.nan; prior_up[~covq] = np.nan
    return prior_de, prior_up


def fuse(p_llm, prior, w):
    """Rank-percentile convex blend; fall back to LLM where prior is NaN."""
    out = pct(p_llm).copy()
    m = ~np.isnan(prior)
    out[m] = (1 - w) * pct(p_llm)[m] + w * pct(np.where(m, prior, np.nan))[m]
    return out


# ----- validation -----
def validate(tok, nm_ci, W, power):
    raw = json.loads((ROOT / "outputs/benchmark_b_dir/preds.json").read_text())
    pde = {k.split("|", 1)[1]: v[2] for k, v in raw.items() if k.startswith("dir_base|")}
    pup = {k.split("|", 1)[1]: v[3] for k, v in raw.items() if k.startswith("dir_base|")}
    samp = pd.read_csv(ROOT / "outputs/benchmark_b_dir/sample.csv")
    samp = samp[samp["id"].isin(pde)].reset_index(drop=True)
    kb = pd.read_csv(ROOT / "data/train.csv")
    prior_de, prior_up = knn_priors(samp["pert"].to_numpy(), samp["gene"].to_numpy(),
                                    kb, tok, nm_ci, W, power=power, disjoint=True)
    lab = samp["label"].to_numpy()
    is_de = (lab != "none").astype(int)
    pde_llm = np.array([pde[i] for i in samp["id"]]); pup_llm = np.array([pup[i] for i in samp["id"]])
    dem = lab != "none"; is_up = (lab[dem] == "up").astype(int)

    print(f"[validate] n={len(samp)} ({int(dem.sum())} DE), gene-covered {int((~np.isnan(prior_de)).sum())}")
    print(f"  LLM alone : DE {auroc(is_de, pde_llm):.3f}  DIR {auroc(is_up, pup_llm[dem]):.3f}")
    print(f"  kNN alone : DE {auroc(is_de, np.nan_to_num(prior_de, nan=np.nanmean(prior_de))):.3f}  "
          f"DIR {auroc(is_up, np.nan_to_num(prior_up[dem], nan=np.nanmean(prior_up))):.3f}")
    print("  fused (DE w / DIR w grid):")
    base_de = auroc(is_de, pde_llm); base_dir = auroc(is_up, pup_llm[dem]); base_mean = (base_de + base_dir) / 2
    best = None
    for w_de in [0.0, 0.25, 0.5]:
        for w_dir in [0.0, 0.25, 0.5, 0.75]:
            fde = fuse(pde_llm, prior_de, w_de)
            fup = fuse(pup_llm, prior_up, w_dir)
            de = auroc(is_de, fde); dr = auroc(is_up, fup[dem]); mean = (de + dr) / 2
            tag = f"    w_de={w_de} w_dir={w_dir}: DE {de:.3f} DIR {dr:.3f} mean {mean:.3f} ({mean-base_mean:+.3f})"
            print(tag)
            if best is None or mean > best[0]:
                best = (mean, w_de, w_dir, de, dr)
    print(f"  base mean {base_mean:.3f}; BEST mean {best[0]:.3f} at w_de={best[1]} w_dir={best[2]}")
    # significance of the DIR fusion at the best dir weight
    fup = fuse(pup_llm, prior_up, best[2])
    md, lo, hi, pg = boot_delta(is_up, fup[dem], pct(pup_llm[dem]))
    print(f"  DIR fusion delta vs LLM (w_dir={best[2]}): median {md:+.3f} CI [{lo:+.3f},{hi:+.3f}] P(>0)={pg:.2f}")
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "outputs/track_b_adversarial_sharp/submission.csv")
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--w-de", type=float, default=None, help="override (else use validation best)")
    ap.add_argument("--w-dir", type=float, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/track_b_adversarial_knn_fusion")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    tok, nm_ci, W = load_emb()
    w_de_best, w_dir_best = validate(tok, nm_ci, W, args.power)
    if args.validate:
        return
    w_de = args.w_de if args.w_de is not None else w_de_best
    w_dir = args.w_dir if args.w_dir is not None else w_dir_best

    sub = pd.read_csv(args.base)
    orig_cols = list(sub.columns)                       # exact submission schema
    test = pd.read_csv(ROOT / "data/test.csv")[["id", "pert", "gene"]]
    sub = sub.merge(test, on="id", how="left")          # base has only id, not pert/gene
    assert sub["pert"].notna().all(), "id not matched to test.csv"
    tr = [json.loads(t) for t in sub["reasoning_trace"]]
    pde = np.array([d.get("P_DE", np.nan) for d in tr], float)
    pup = np.array([d.get("P_up_given_DE", np.nan) for d in tr], float)
    bad = np.isnan(pde); pde = np.where(bad, sub["prediction_up"] + sub["prediction_down"], pde)
    pup = np.where(np.isnan(pup), 0.5, pup)

    kb = pd.read_csv(ROOT / "data/train.csv")
    prior_de, prior_up = knn_priors(sub["pert"].to_numpy(), sub["gene"].to_numpy(),
                                    kb, tok, nm_ci, W, power=args.power, disjoint=False)
    fde = fuse(pde, prior_de, w_de)
    fup = fuse(pup, prior_up, w_dir)

    out = sub.copy()
    out["prediction_up"] = (fde * fup).round(6)
    out["prediction_down"] = (fde * (1 - fup)).round(6)
    out = out[orig_cols]                                 # drop merged pert/gene
    assert out[["prediction_up", "prediction_down"]].notna().all().all()
    print(f"\n[build] base={args.base.parent.name} w_de={w_de} w_dir={w_dir} "
          f"power={args.power}; gene-covered test rows: {int((~np.isnan(prior_de)).sum())}/{len(out)}")
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "submission.csv"
    out.to_csv(out_csv, index=False)
    with zipfile.ZipFile(args.out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, "submission.csv")
    print(f"[build] -> {out_csv} (+ submission.zip)")


if __name__ == "__main__":
    main()
