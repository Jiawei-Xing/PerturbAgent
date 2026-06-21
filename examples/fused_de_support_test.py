"""Does the support-aware kNN DE survive FUSION with the LLM (the shipped level)?
Baseline kNN DE = power2. Support-aware = power1 + down-weight high-support rows
toward neutral. Fuse each with the prompt-only LLM P_DE on the 2499-row wdir pool
(disjoint-LOO kNN against full train), rank-blend w_de=0.5, paired bootstrap."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from support_aware_knn_de import build_sim, pde_from_weights, load_geneformer, load_scgpt  # noqa
from uce_knn_test import auroc, boot_ci, boot_delta  # noqa


def rankpct(x):
    x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    is_de = (df["label"].to_numpy() != "none").astype(float)
    gf = load_geneformer(); sc = load_scgpt(center=True)
    print("building sim matrix...")
    S, same, covered = build_sim(df, [gf, sc])

    def knn_de(power, neutralize=0.0):
        W = S ** power; W[same] = 0.0
        pde, supp = pde_from_weights(W, is_de)
        if neutralize > 0:
            sr = (supp - np.nanmin(supp)) / (np.nanmax(supp) - np.nanmin(supp) + 1e-9)
            a = 1.0 - neutralize * sr
            pde = a * pde + (1 - a) * 0.5
        return pde

    base = knn_de(2.0)                      # production
    sa = knn_de(1.0, neutralize=1.0)        # support-aware
    df = df.assign(knn_base=base, knn_sa=sa)

    # LLM prompt-only P_DE on the wdir pool
    lab = pd.read_csv(ROOT / "outputs" / "wdir_tune_sample_labeled.csv")
    sub = pd.read_csv(ROOT / "outputs" / "wdir_tune_logprobs" / "submission.csv")
    sub["llm_de"] = sub["prediction_up"] + sub["prediction_down"]
    pool = lab.merge(sub[["id", "llm_de"]], on="id").merge(
        df[["id", "knn_base", "knn_sa"]], on="id")
    pool = pool[~pool.knn_base.isna() & ~pool.knn_sa.isna()].copy()
    y = (pool["label"].to_numpy() != "none").astype(int)
    print(f"pool: {len(pool)} covered rows, DE rate {y.mean():.3f}")

    llm = pool.llm_de.to_numpy()
    print(f"\nLLM DE alone      {auroc(y, llm):.3f}")
    print(f"kNN DE base(p2)   {auroc(y, pool.knn_base.to_numpy()):.3f}")
    print(f"kNN DE supp-aware {auroc(y, pool.knn_sa.to_numpy()):.3f}")

    for w in (0.3, 0.5, 0.7):
        fb = (1 - w) * rankpct(llm) + w * rankpct(pool.knn_base.to_numpy())
        fs = (1 - w) * rankpct(llm) + w * rankpct(pool.knn_sa.to_numpy())
        md, l, h, pg = boot_delta(y, fs, fb)
        print(f"  fuse w_knn={w}:  base {auroc(y, fb):.3f}  supp-aware {auroc(y, fs):.3f}  "
              f"Δ {md:+.4f} CI[{l:+.4f},{h:+.4f}] P(>0)={pg:.2f}")


if __name__ == "__main__":
    main()
