"""Fuse STATE macrophage-context DE into the 0.624 submission's DE side.

STATE's only earned signal is a weak DE magnitude (|delta_raw|, full-train
0.530); DIR is chance. So this touches ONLY P_DE -- rank-blend STATE's DE into
the shipped fused P_DE on STATE-covered rows -- and leaves the DIR ratio exactly
as in the 0.624 pick. Offline this was +0.000 (redundant with the kNN); this
builds it anyway to read the real LB number.

  new_P_DE = (1-w)*pct(P_DE_0.624) + w*pct(state_DE)   on covered rows
  pred_up/down rescaled to the new P_DE, DIR ratio preserved.
"""
from __future__ import annotations
import argparse
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATE_CSV = Path("/grid/siepel/home/xing/state/spike/state_features_test.csv")


def pct(a):
    a = np.asarray(a, float); o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a))
    return r / max(len(a) - 1, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "outputs/track_b_scgpt_fusion_wdir07/submission.csv")
    ap.add_argument("--w-state", type=float, default=0.3, help="DE rank-blend weight for STATE (offline-neutral optimum)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/track_b_state_fusion")
    args = ap.parse_args()

    sub = pd.read_csv(args.base)
    st = pd.read_csv(STATE_CSV)
    st["state_de"] = st["delta_raw"].abs()
    m = sub.merge(st[["id", "state_de"]], on="id", how="left")
    covered = m["state_de"].notna().to_numpy()
    print(f"rows {len(m)}; STATE-covered {covered.sum()} ({covered.mean():.0%})")

    pu = m["prediction_up"].to_numpy(float); pdn = m["prediction_down"].to_numpy(float)
    p_de = pu + pdn
    ratio = np.divide(pu, p_de, out=np.full_like(pu, 0.5), where=p_de > 0)   # DIR, preserved

    # rank-blend STATE DE into P_DE on covered rows only
    new_pde = p_de.copy()
    w = args.w_state
    cov_pct_base = pct(p_de[covered])
    cov_pct_state = pct(m["state_de"].to_numpy()[covered])
    blended = (1 - w) * cov_pct_base + w * cov_pct_state
    # map blended rank back onto the original P_DE value distribution (preserve scale)
    order = np.argsort(np.argsort(blended))
    sorted_pde = np.sort(p_de[covered])
    new_pde[covered] = sorted_pde[order]

    out = m.drop(columns=["state_de"]).copy()
    out["prediction_up"] = (new_pde * ratio).round(6)
    out["prediction_down"] = (new_pde * (1 - ratio)).round(6)
    out = out[list(sub.columns)]
    assert out[["prediction_up", "prediction_down"]].notna().all().all()
    # report what moved
    d_pde = np.abs((out.prediction_up + out.prediction_down).to_numpy() - p_de)
    d_ratio = np.abs(np.divide(out.prediction_up, (out.prediction_up + out.prediction_down).replace(0, np.nan)) - ratio)
    print(f"w_state={w}: mean|ΔP_DE|={d_pde.mean():.4f}  mean|ΔDIR ratio|={np.nanmean(d_ratio):.6f} (should be ~0)")

    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "submission.csv"
    out.to_csv(out_csv, index=False)
    with zipfile.ZipFile(args.out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, "submission.csv")
    print(f"[build] -> {out_csv} (+ submission.zip)")


if __name__ == "__main__":
    main()
