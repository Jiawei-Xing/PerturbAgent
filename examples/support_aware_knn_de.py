"""Support-aware kNN DE prototype + diagnosis.

The production GF⊕scGPT gene-sim kNN DE prior is gene-ONLY (pert-sim is chance),
so pde is a per-gene DE-propensity. Earlier finding: DE AUROC is 0.605 on
low-neighbor-support rows but inverts to 0.435 on high-support (dense-family)
rows. This builds the full gene-gene similarity matrix once, diagnoses the
inversion by support decile, then tests support-aware weightings against the
power=2 baseline with paired bootstrap at full power (n≈6600 DE rows).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from uce_knn_test import load_geneformer, load_scgpt, auroc, boot_ci, boot_delta  # noqa


def build_sim(df, spaces):
    """Ensemble gene-gene similarity matrix S (N×N), mean per-space cosine over
    spaces covering both rows, clipped at 0. Also disjoint mask."""
    perts = df["pert"].to_numpy(); genes = df["gene"].to_numpy()
    N = len(df)
    Ssum = np.zeros((N, N), np.float32); Scnt = np.zeros((N, N), np.float32)
    for E in spaces:
        D = len(next(iter(E.values()))); M = np.zeros((N, D), np.float32); c = np.zeros(N, bool)
        for i, g in enumerate(genes):
            v = E.get(str(g).upper())
            if v is not None:
                M[i] = v; c[i] = True
        cs = np.clip(M @ M.T, 0, None).astype(np.float32)
        cov = np.outer(c, c)
        Ssum[cov] += cs[cov]; Scnt[cov] += 1
    S = np.zeros((N, N), np.float32); nz = Scnt > 0; S[nz] = Ssum[nz] / Scnt[nz]
    covered = (Scnt.sum(1) > 0)
    same = (perts[:, None] == perts[None, :]) | (genes[:, None] == genes[None, :])
    return S, same, covered


def pde_from_weights(W, is_de):
    """Row-normalized weighted DE fraction; W already disjoint-masked."""
    den = W.sum(1); pde = np.full(len(W), np.nan)
    ok = den > 0
    num = (W * is_de[None, :]).sum(1)
    pde[ok] = num[ok] / den[ok]
    return pde, den


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    is_de = (df["label"].to_numpy() != "none").astype(float)
    gf = load_geneformer(); sc = load_scgpt(center=True)
    print("building sim matrix...")
    S, same, covered = build_sim(df, [gf, sc])
    N = len(df)

    def weighted(power, sharpen_hi=0.0, neutralize=0.0):
        """power: base exponent. sharpen_hi: add this*zsupport to per-row power.
        neutralize: pull pde toward 0.5 by support fraction (0..1)."""
        W = S ** power
        W[same] = 0.0
        pde, supp = pde_from_weights(W, is_de)
        if sharpen_hi != 0.0:
            zs = (supp - np.nanmean(supp)) / (np.nanstd(supp) + 1e-9)
            pde2 = np.full(N, np.nan)
            for i in range(N):
                if not covered[i] or supp[i] == 0:
                    continue
                p = max(0.5, power + sharpen_hi * zs[i])
                w = S[i] ** p; w[same[i]] = 0.0
                if w.sum() > 0:
                    pde2[i] = (w * is_de).sum() / w.sum()
            pde = pde2
        if neutralize > 0.0:
            sr = (supp - np.nanmin(supp)) / (np.nanmax(supp) - np.nanmin(supp) + 1e-9)
            a = 1.0 - neutralize * sr            # influence shrinks with support
            pde = a * pde + (1 - a) * 0.5
        return pde, supp

    base, supp = weighted(2.0)
    ok = ~np.isnan(base)
    a0 = auroc(is_de[ok], base[ok]); lo, hi = boot_ci(is_de[ok], base[ok])
    print(f"\nBASELINE power=2  DE AUROC {a0:.3f} CI[{lo:.3f},{hi:.3f}] (n={int(ok.sum())})")

    # ---- diagnosis: DE AUROC by support decile ----
    print("\n--- DE AUROC by neighbor-support decile (low->high) ---")
    sv = supp[ok]; bv = base[ok]; yv = is_de[ok]
    dec = np.quantile(sv, np.linspace(0, 1, 11))
    for k in range(10):
        m = (sv >= dec[k]) & (sv <= dec[k + 1] if k == 9 else sv < dec[k + 1])
        if m.sum() < 30 or len(set(yv[m])) < 2:
            continue
        print(f"  decile {k}: supp[{dec[k]:.1f},{dec[k+1]:.1f}]  AUROC {auroc(yv[m],bv[m]):.3f}  "
              f"mean_pde {bv[m].mean():.3f}  DE_rate {yv[m].mean():.3f}  n={int(m.sum())}")

    # what are high-support genes?
    g_supp = df.assign(supp=supp).groupby("gene")["supp"].first().sort_values(ascending=False)
    print("\n  top-12 highest-support genes:", list(g_supp.head(12).index))

    # ---- variants vs baseline (paired bootstrap) ----
    print("\n--- support-aware variants (paired Δ vs baseline power=2) ---")
    for nm, kw in [("sharpen_hi=+1.0", dict(sharpen_hi=1.0)),
                   ("sharpen_hi=+2.0", dict(sharpen_hi=2.0)),
                   ("sharpen_hi=-1.0", dict(sharpen_hi=-1.0)),
                   ("neutralize=0.5", dict(neutralize=0.5)),
                   ("neutralize=1.0", dict(neutralize=1.0)),
                   ("power=4 uniform", dict(power=4.0)),
                   ("power=1 uniform", dict(power=1.0))]:
        p = kw.pop("power", 2.0)
        pv, _ = weighted(p, **kw)
        m = ok & ~np.isnan(pv)
        a = auroc(is_de[m], pv[m])
        md, l, h, pg = boot_delta(is_de[m], pv[m], base[m])
        print(f"  {nm:18s} DE {a:.3f}  Δ {md:+.4f} CI[{l:+.4f},{h:+.4f}] P(>0)={pg:.2f} (n={int(m.sum())})")


if __name__ == "__main__":
    main()
