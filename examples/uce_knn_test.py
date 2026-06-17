"""UCE (ESM2 protein) embeddings + Geneformer⊕UCE ensemble for the kNN.

Phase-0 offline gate for the embedding swap (see README "What I learned"):
does a better / orthogonal gene embedding beat Geneformer token embeddings in
the disjoint-LOO gene-similarity kNN? Geneformer baseline (co-expression flavor)
is DE ~0.55 / DIR ~0.62-0.63. UCE represents each gene by the ESM2 embedding of
its protein -- a sequence/family axis ORTHOGONAL to co-expression, so even if its
standalone DIR is no better, the ensemble (errors decorrelate) can be.

Same honest evaluation as knn_transfer_test.py: leave-one-out with disjoint
masking (predict each train pair from only pairs sharing NEITHER its pert NOR its
gene). Gene-similarity only (pert-sim is chance, established). Ensemble = average
the per-space cosine similarities, then aggregate -- a rank-level blend.

Embeddings are provided as {UPPER_SYMBOL: unit_vector} dicts; spaces with no
vector for a symbol are dropped from that row's similarity (so the ensemble
gracefully falls back to whichever space covers the gene).

    uv run --extra serve python examples/uce_knn_test.py            # full train, CPU
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
GF = ROOT / "outputs" / "geneformer_probe" / "model" / "geneformer"
SAFET = ROOT / "outputs" / "geneformer_probe" / "model" / "Geneformer-V2-104M" / "model.safetensors"
UCE_PT = ROOT / "outputs" / "uce" / "MOUSE_PE.pt"   # written by extract step below


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
    r = _rankdata(np.asarray(s, float))
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def boot_ci(y, s, n=2000, seed=0):
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    rng = np.random.default_rng(seed); N = len(y); v = []
    for _ in range(n):
        idx = rng.integers(0, N, N); a = auroc(y[idx], s[idx])
        if not np.isnan(a):
            v.append(a)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def boot_delta(y, s_new, s_old, n=3000, seed=0):
    """Paired bootstrap of AUROC(new) - AUROC(old) on the same rows."""
    y = np.asarray(y).astype(int); s_new = np.asarray(s_new, float); s_old = np.asarray(s_old, float)
    rng = np.random.default_rng(seed); N = len(y); d = []
    for _ in range(n):
        idx = rng.integers(0, N, N); yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        d.append(auroc(yy, s_new[idx]) - auroc(yy, s_old[idx]))
    d = np.array(d)
    return float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


# ---------------- embedding providers: {UPPER_SYMBOL: unit_vector} ----------
def load_geneformer():
    tok = pickle.load(open(GF / "token_dictionary_gc104M.pkl", "rb"))
    nm = pickle.load(open(GF / "gene_name_id_dict_gc104M.pkl", "rb"))
    nm_ci = {str(k).upper(): v for k, v in nm.items()}
    with safe_open(str(SAFET), framework="numpy") as f:
        W = f.get_tensor("bert.embeddings.word_embeddings.weight")
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    E = {}
    for s, ensg in nm_ci.items():
        if ensg in tok:
            E[s] = W[tok[ensg]]
    return E


def load_scgpt(center=True):
    """scGPT whole-human gene token embeddings (encoder.embedding.weight, 60697x512),
    expression-trained like Geneformer but on more cells. Human vocab -> uppercase
    mouse-symbol match (~80% cov). Mean-center by default (cheap insurance)."""
    import json
    import torch
    vocab = json.load(open(ROOT / "outputs" / "scgpt" / "vocab.json"))
    sd = torch.load(ROOT / "outputs" / "scgpt" / "best_model.pt", map_location="cpu", weights_only=False)
    W = sd["encoder.embedding.weight"].float().numpy()
    if center:
        W = W - W.mean(0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    E = {}
    for sym, i in vocab.items():
        if sym.startswith("<"):
            continue
        E[str(sym).upper()] = W[i]
    return E


def load_uce(center=True):
    """ESM2 protein embeddings are strongly anisotropic (shared mean component);
    without mean-centering the residual can invert the direction signal, exactly
    like GenePT (DIR 0.169 raw). Default: mean-center over all genes, then unit-norm."""
    import torch
    d = torch.load(UCE_PT, map_location="cpu", weights_only=False)
    syms = list(d.keys())
    M = np.stack([np.asarray(d[s], np.float32).ravel() for s in syms])
    if center:
        M = M - M.mean(0)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return {str(s).upper(): M[i] for i, s in enumerate(syms)}


# ---------------- disjoint-LOO gene-similarity kNN over >=1 space ----------
def predict_gene(df, spaces, power):
    """spaces: list of {UPPER_SYMBOL: unit_vec}. Ensemble = mean per-space cosine."""
    perts = df["pert"].to_numpy(); genes = df["gene"].to_numpy()
    lab = df["label"].to_numpy()
    is_de = (lab != "none").astype(float); is_up = (lab == "up").astype(float)
    N = len(df)

    # per-space row matrices (zeros where missing) + per-space coverage mask
    mats, covs = [], []
    for E in spaces:
        D = len(next(iter(E.values())))
        M = np.zeros((N, D), np.float32); c = np.zeros(N, bool)
        for i, g in enumerate(genes):
            v = E.get(str(g).upper())
            if v is not None:
                M[i] = v; c[i] = True
        mats.append(M); covs.append(c)
    covs = np.array(covs)                       # (S, N)
    covered = covs.any(0)                        # row usable if ANY space covers it

    prior_de = np.full(N, np.nan); prior_up = np.full(N, np.nan)
    for i in range(N):
        if not covered[i]:
            continue
        # average cosine across spaces that cover BOTH row i and the neighbour
        sim_sum = np.zeros(N); sim_cnt = np.zeros(N)
        for s, M in enumerate(mats):
            if not covs[s, i]:
                continue
            cs = np.clip(M @ M[i], 0, None)
            both = covs[s]                       # neighbours covered in this space
            sim_sum[both] += cs[both]; sim_cnt[both] += 1
        sim = np.zeros(N); nz = sim_cnt > 0
        sim[nz] = sim_sum[nz] / sim_cnt[nz]
        w = sim ** power
        w[(perts == perts[i]) | (genes == genes[i])] = 0.0   # disjoint mask + self
        if w.sum() > 0:
            prior_de[i] = (w * is_de).sum() / w.sum()
            wd = w * is_de
            if wd.sum() > 0:
                prior_up[i] = (wd * is_up).sum() / wd.sum()
    return is_de, is_up, covered, prior_de, prior_up


def report(name, df, spaces, power):
    is_de, is_up, cov, pde, pup = predict_gene(df, spaces, power)
    ok = ~np.isnan(pde)
    a_de = auroc(is_de[ok], pde[ok]); lo, hi = boot_ci(is_de[ok], pde[ok])
    dem = (is_de == 1) & ~np.isnan(pup)
    a_dir = auroc(is_up[dem], pup[dem]); dlo, dhi = boot_ci(is_up[dem], pup[dem])
    print(f"\n[{name}]  covered {cov.mean():.0%}")
    print(f"   DE  AUROC {a_de:.3f}  95% CI [{lo:.3f},{hi:.3f}]  (n={int(ok.sum())})")
    print(f"   DIR AUROC {a_dir:.3f}  95% CI [{dlo:.3f},{dhi:.3f}]  (n={int(dem.sum())})")
    return is_de, is_up, pde, pup


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--emb", choices=["uce", "scgpt"], default="uce", help="which second embedding to test vs Geneformer")
    ap.add_argument("--no-center", action="store_true", help="skip mean-centering (anisotropy check)")
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    gf = load_geneformer()
    name = args.emb.upper()
    other = (load_scgpt if args.emb == "scgpt" else load_uce)(center=not args.no_center)
    # report symbol coverage over the actual gene universe
    genes = set(g.upper() for g in df["gene"])
    print(f"[knn] {len(df)} train pairs; power={args.power}; second emb = {name} (center={not args.no_center})")
    print(f"   gene coverage: Geneformer {len(genes & set(gf))/len(genes):.0%}, "
          f"{name} {len(genes & set(other))/len(genes):.0%}, "
          f"either {len(genes & (set(gf)|set(other)))/len(genes):.0%}  (of {len(genes)} genes)")

    gd_de, gd_up, gf_pde, gf_pup = report("Geneformer only (gene-sim)", df, [gf], args.power)
    report(f"{name} only (gene-sim)", df, [other], args.power)
    en_de, en_up, en_pde, en_pup = report(f"Geneformer ⊕ {name} ensemble", df, [gf, other], args.power)

    # paired DIR delta: ensemble vs Geneformer, on the rows both score
    dem = (gd_de == 1) & ~np.isnan(gf_pup) & ~np.isnan(en_pup)
    md, l, h, pg = boot_delta(gd_up[dem], en_pup[dem], gf_pup[dem])
    print(f"\n[paired DIR  ensemble - Geneformer]  median {md:+.3f} CI [{l:+.3f},{h:+.3f}] P(>0)={pg:.2f}  (n={int(dem.sum())})")
    okde = ~np.isnan(gf_pde) & ~np.isnan(en_pde)
    md, l, h, pg = boot_delta(gd_de[okde], en_pde[okde], gf_pde[okde])
    print(f"[paired DE   ensemble - Geneformer]  median {md:+.3f} CI [{l:+.3f},{h:+.3f}] P(>0)={pg:.2f}  (n={int(okde.sum())})")


if __name__ == "__main__":
    main()
