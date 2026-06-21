"""Does STATE macrophage-context perturbation prediction ADD to the GF⊕scGPT
gene-similarity kNN? The kNN already carries fused DIR ~0.66 / DE ~0.55. STATE
models the (pert,gene) interaction the kNN's gene-only similarity cannot, so even
if its standalone numbers are similar the errors may decorrelate.

STATE features come from spike/state_features.csv (dumped in the state venv):
delta_raw (vs macrophage basal) and delta_mc (vs model non-targeting). DE uses
|delta_raw|, DIR uses signed delta_mc -- the two control modes that each maximize
one component. Fusion = rank-percentile blend (the production merge), evaluated on
the disjoint-LOO rows the kNN scores. Reuses uce_knn_test's loaders + predict_gene.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from uce_knn_test import load_geneformer, load_scgpt, predict_gene, auroc, boot_ci, boot_delta  # noqa

STATE_CSV = Path("/grid/siepel/home/xing/state/spike/state_features.csv")


def rankpct(x):
    x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    gf = load_geneformer(); sc = load_scgpt(center=True)
    is_de, is_up, cov, knn_pde, knn_pup = predict_gene(df, [gf, sc], power=2.0)
    df = df.assign(knn_pde=knn_pde, knn_pup=knn_pup, is_de=is_de, is_up=is_up, cov=cov)

    st = pd.read_csv(STATE_CSV)
    st["state_de"] = st["delta_raw"].abs()
    st["state_dir"] = st["delta_mc"]
    m = df.merge(st[["id", "state_de", "state_dir"]], on="id", how="inner")
    print(f"merged rows: {len(m)} (kNN∩STATE covered)")

    # ---- DE: (up+down) vs none ----
    de = m[~m.knn_pde.isna()].copy()
    y = de.is_de.to_numpy()
    knn = de.knn_pde.to_numpy(); sta = de.state_de.to_numpy()
    print("\n=== DE (effect vs none) ===")
    for nm, s in [("kNN DE", knn), ("STATE DE", sta)]:
        a = auroc(y, s); lo, hi = boot_ci(y, s); print(f"  {nm:14s} {a:.3f} CI[{lo:.3f},{hi:.3f}]  (n={len(y)})")
    for w in (0.3, 0.5, 0.7):
        fused = (1 - w) * rankpct(knn) + w * rankpct(sta)
        a = auroc(y, fused); print(f"  fuse w_state={w:.1f}  {a:.3f}")
    best_w = 0.3
    fused = (1 - best_w) * rankpct(knn) + best_w * rankpct(sta)
    md, l, h, pg = boot_delta(y, fused, rankpct(knn))
    print(f"  paired [fuse(w={best_w}) - kNN]  {md:+.3f} CI[{l:+.3f},{h:+.3f}] P(>0)={pg:.2f}")

    # ---- DIR: up vs down among true DE ----
    di = m[(m.is_de == 1) & ~m.knn_pup.isna()].copy()
    y = di.is_up.to_numpy()
    knn = di.knn_pup.to_numpy(); sta = di.state_dir.to_numpy()
    print("\n=== DIR (up vs down | DE) ===")
    for nm, s in [("kNN DIR", knn), ("STATE DIR", sta)]:
        a = auroc(y, s); lo, hi = boot_ci(y, s); print(f"  {nm:14s} {a:.3f} CI[{lo:.3f},{hi:.3f}]  (n={len(y)})")
    for w in (0.3, 0.5, 0.7):
        fused = (1 - w) * rankpct(knn) + w * rankpct(sta)
        a = auroc(y, fused); print(f"  fuse w_state={w:.1f}  {a:.3f}")
    for best_w in (0.3, 0.5):
        fused = (1 - best_w) * rankpct(knn) + best_w * rankpct(sta)
        md, l, h, pg = boot_delta(y, fused, rankpct(knn))
        print(f"  paired [fuse(w={best_w}) - kNN]  {md:+.3f} CI[{l:+.3f},{h:+.3f}] P(>0)={pg:.2f}")


if __name__ == "__main__":
    main()
