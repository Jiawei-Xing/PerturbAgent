"""Per-gene (support-adaptive) fusion weights vs the fixed w_de=0.5 / w_dir=0.7.

Motivation (from the support-stratified diagnosis): the kNN's reliability vs the
LLM varies by gene in OPPOSITE directions -- DE kNN strong on low-support genes
(0.605) and inverts on hubs (0.435); DIR kNN weak on low-support (0.545) and
strong on hubs (0.699). Support (embedding neighbourhood density) is label-free,
so per-gene weights w_de(support)↓ and w_dir(support)↑ are leak-safe.

Honest fused-level test on the wdir pool (LLM prompt-only preds + disjoint-LOO
kNN over full train). Reports DE, DIR, mean AUROC and paired bootstrap vs the
fixed-weight production blend.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from support_aware_knn_de import build_sim, load_geneformer, load_scgpt  # noqa
from uce_knn_test import auroc, boot_delta  # noqa


def rankpct(x):
    x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    is_de_all = (df["label"].to_numpy() != "none").astype(float)
    is_up_all = (df["label"].to_numpy() == "up").astype(float)
    gf = load_geneformer(); sc = load_scgpt(center=True)
    print("building sim...")
    S, same, covered = build_sim(df, [gf, sc])
    W = S ** 2.0; W[same] = 0.0
    den = W.sum(1)
    knn_de = np.where(den > 0, (W * is_de_all[None, :]).sum(1) / np.where(den > 0, den, 1), np.nan)
    wde = W * is_de_all[None, :]; dend = wde.sum(1)
    knn_dir = np.where(dend > 0, (wde * is_up_all[None, :]).sum(1) / np.where(dend > 0, dend, 1), np.nan)
    support = den
    df = df.assign(knn_de=knn_de, knn_dir=knn_dir, supp=support)

    lab = pd.read_csv(ROOT / "outputs" / "wdir_tune_sample_labeled.csv")
    sub = pd.read_csv(ROOT / "outputs" / "wdir_tune_logprobs" / "submission.csv")
    sub["llm_de"] = sub["prediction_up"] + sub["prediction_down"]
    sub["llm_dir"] = sub["prediction_up"] / (sub["prediction_up"] + sub["prediction_down"]).replace(0, np.nan)
    pool = lab.merge(sub[["id", "llm_de", "llm_dir"]], on="id").merge(
        df[["id", "knn_de", "knn_dir", "supp"]], on="id")
    pool = pool[~pool.knn_de.isna()].copy().reset_index(drop=True)
    y_de = (pool["label"].to_numpy() != "none").astype(int)
    dem = pool["label"].to_numpy() != "none"
    y_dir = (pool["label"].to_numpy()[dem] == "up").astype(int)
    spct = rankpct(pool["supp"].to_numpy())          # 0=sparsest, 1=densest
    print(f"pool {len(pool)} rows, {int(dem.sum())} DE")

    ld, ldir = pool.llm_de.to_numpy(), pool.llm_dir.fillna(0.5).to_numpy()
    kd, kdir = pool.knn_de.to_numpy(), pool.knn_dir.fillna(0.5).to_numpy()
    rl_de, rk_de = rankpct(ld), rankpct(kd)
    rl_dir, rk_dir = rankpct(ldir), rankpct(kdir)

    def score(wde_vec, wdir_vec):
        fde = (1 - wde_vec) * rl_de + wde_vec * rk_de
        fdir = (1 - wdir_vec) * rl_dir + wdir_vec * rk_dir
        de = auroc(y_de, fde); dr = auroc(y_dir, fdir[dem])
        return de, dr, fde, fdir

    # baseline fixed weights
    de0, dr0, fde0, fdir0 = score(np.full(len(pool), 0.5), np.full(len(pool), 0.7))
    print(f"\nFIXED  w_de=0.5 w_dir=0.7 :  DE {de0:.3f}  DIR {dr0:.3f}  mean {(de0+dr0)/2:.3f}")

    variants = {
        "DE-route (wde 0.7->0.1)": (0.7 - 0.6 * spct, np.full(len(pool), 0.7)),
        "DIR-route (wdir 0.4->0.95)": (np.full(len(pool), 0.5), 0.4 + 0.55 * spct),
        "both routes": (0.7 - 0.6 * spct, 0.4 + 0.55 * spct),
        "DE-route mild (0.6->0.3)": (0.6 - 0.3 * spct, np.full(len(pool), 0.7)),
    }
    for nm, (wdev, wdirv) in variants.items():
        de, dr, fde, fdir = score(wdev, wdirv)
        md_de, l_de, h_de, p_de = boot_delta(y_de, fde, fde0)
        md_dr, l_dr, h_dr, p_dr = boot_delta(y_dir, fdir[dem], fdir0[dem])
        print(f"\n{nm}")
        print(f"   DE  {de:.3f}  Δ {md_de:+.4f} CI[{l_de:+.4f},{h_de:+.4f}] P={p_de:.2f}")
        print(f"   DIR {dr:.3f}  Δ {md_dr:+.4f} CI[{l_dr:+.4f},{h_dr:+.4f}] P={p_dr:.2f}")
        print(f"   mean {(de+dr)/2:.3f}  (Δmean {((de+dr)/2)-((de0+dr0)/2):+.4f})")


if __name__ == "__main__":
    main()
