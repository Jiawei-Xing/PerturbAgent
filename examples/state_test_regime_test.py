"""Does STATE add value in the TEST-like regime? The disjoint test set is the
extreme low-train-support case. STATE is label-independent, so it may help where
the gene-sim kNN's transfer is weakest, even if redundant on average over train.

Stratify train-LOO rows by kNN neighbor support (sum of disjoint-masked weights):
- kNN-UNCOVERED rows (no GF/scGPT embedding) -> currently LLM-only on test; does
  STATE beat chance there? = pure additive coverage.
- LOW vs HIGH kNN-support quartiles among covered -> low-support ~ test conditions.
Report DE/DIR AUROC for kNN-only, STATE-only, fused, per stratum.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from uce_knn_test import load_geneformer, load_scgpt, auroc, boot_ci  # noqa

STATE_CSV = Path("/grid/siepel/home/xing/state/spike/state_features.csv")


def predict_with_support(df, spaces, power):
    """kNN gene-sim disjoint-LOO that also returns per-row support = sum of the
    disjoint-masked neighbor weights (low = test-like hard transfer)."""
    perts = df["pert"].to_numpy(); genes = df["gene"].to_numpy(); lab = df["label"].to_numpy()
    is_de = (lab != "none").astype(float); is_up = (lab == "up").astype(float)
    N = len(df)
    mats, covs = [], []
    for E in spaces:
        D = len(next(iter(E.values()))); M = np.zeros((N, D), np.float32); c = np.zeros(N, bool)
        for i, g in enumerate(genes):
            v = E.get(str(g).upper())
            if v is not None:
                M[i] = v; c[i] = True
        mats.append(M); covs.append(c)
    covs = np.array(covs); covered = covs.any(0)
    pde = np.full(N, np.nan); pup = np.full(N, np.nan); supp = np.zeros(N)
    for i in range(N):
        if not covered[i]:
            continue
        sim_sum = np.zeros(N); sim_cnt = np.zeros(N)
        for s, M in enumerate(mats):
            if not covs[s, i]:
                continue
            cs = np.clip(M @ M[i], 0, None); both = covs[s]
            sim_sum[both] += cs[both]; sim_cnt[both] += 1
        sim = np.zeros(N); nz = sim_cnt > 0; sim[nz] = sim_sum[nz] / sim_cnt[nz]
        w = sim ** power
        w[(perts == perts[i]) | (genes == genes[i])] = 0.0
        supp[i] = w.sum()
        if w.sum() > 0:
            pde[i] = (w * is_de).sum() / w.sum()
            wd = w * is_de
            if wd.sum() > 0:
                pup[i] = (wd * is_up).sum() / wd.sum()
    return is_de, is_up, covered, pde, pup, supp


def line(tag, y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    m = ~np.isnan(s)
    if m.sum() < 20 or len(set(y[m])) < 2:
        return f"  {tag:16s} n={m.sum():4d}  (too few)"
    a = auroc(y[m], s[m]); lo, hi = boot_ci(y[m], s[m])
    return f"  {tag:16s} {a:.3f} CI[{lo:.3f},{hi:.3f}]  n={int(m.sum())}"


def rankpct(x):
    x = np.asarray(x, float); o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def block(name, mask, is_de, is_up, knn_de, knn_up, st_de, st_up):
    print(f"\n### {name}  (n={int(mask.sum())})")
    y = is_de[mask]
    print("DE:")
    print(line("kNN", y, knn_de[mask]))
    print(line("STATE", y, st_de[mask]))
    # fuse only where both present
    both = mask & ~np.isnan(knn_de) & ~np.isnan(st_de)
    if both.sum() > 20:
        f = 0.7 * rankpct(knn_de[both]) + 0.3 * rankpct(st_de[both])
        print(line("fuse .7/.3", is_de[both], f))
    dem = mask & (is_de == 1)
    yd = is_up[dem]
    print(f"DIR (n_DE={int(dem.sum())}):")
    print(line("kNN", yd, knn_up[dem]))
    print(line("STATE", yd, st_up[dem]))


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    gf = load_geneformer(); sc = load_scgpt(center=True)
    is_de, is_up, cov, pde, pup, supp = predict_with_support(df, [gf, sc], 2.0)
    st = pd.read_csv(STATE_CSV).set_index("id")
    sid = df["id"].map(lambda x: x)
    st_de = np.array([abs(st.delta_raw[i]) if i in st.index else np.nan for i in df["id"]])
    st_up = np.array([st.delta_mc[i] if i in st.index else np.nan for i in df["id"]])

    knn_cov = ~np.isnan(pde)
    st_cov = ~np.isnan(st_de)
    print(f"rows {len(df)} | kNN-cov {knn_cov.mean():.0%} | STATE-cov {st_cov.mean():.0%} | "
          f"kNN-UNcov but STATE-cov {((~knn_cov)&st_cov).sum()}")

    # 1) kNN-uncovered rows -> STATE pure additive coverage (test: these get LLM-only)
    block("kNN-UNCOVERED (LLM-only on test)", (~knn_cov) & st_cov, is_de, is_up, pde, pup, st_de, st_up)

    # 2) covered rows split by kNN support quartiles (low support ~ test-like)
    cs = supp.copy(); cs[~knn_cov] = np.nan
    q = np.nanquantile(cs[knn_cov], [0.25, 0.75])
    low = knn_cov & (supp <= q[0]); high = knn_cov & (supp >= q[1])
    block("LOW kNN-support (test-like)", low, is_de, is_up, pde, pup, st_de, st_up)
    block("HIGH kNN-support", high, is_de, is_up, pde, pup, st_de, st_up)


if __name__ == "__main__":
    main()
