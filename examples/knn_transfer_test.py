"""Embedding-similarity (kNN) transfer test -- the disjoint-split-legal lever.

The split blocks gene-IDENTITY shortcuts but NOT functional-SIMILARITY transfer:
an unseen test gene can borrow the perturbation-response statistics of
functionally similar train genes. This is the intended signal and the
PerturbQA / MLGenX "LLM-guided kNN aggregator" approach. We previously tested
only weak versions (STRING pathway-neighbors -> analogue_p_de 0.488; GO/category
priors -> chance); this tries a DENSE gene+perturbation embedding kNN.

Embeddings: Geneformer V2-104M token (word) embeddings -- a free, ~20k-gene
functional space already on disk. (GenePT text embeddings are the gold standard;
this is the cheap first read -- if it shows signal, upgrade.)

Honest evaluation = leave-one-out with DISJOINT masking that mirrors the
competition split: to predict train pair (p, g), aggregate over ONLY the train
pairs that share NEITHER p NOR g (so neither the perturbation nor the gene was
ever "seen"). Weighted by sim_pert(p,p') * sim_gene(g,g'). Reports DE and DIR
AUROC for pert-only, gene-only, and joint similarity, with bootstrap CIs.

    uv run python examples/knn_transfer_test.py            # full train, CPU
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


def load_embeddings():
    tok = pickle.load(open(GF / "token_dictionary_gc104M.pkl", "rb"))   # ENSG -> id
    nm = pickle.load(open(GF / "gene_name_id_dict_gc104M.pkl", "rb"))   # symbol -> ENSG
    nm_ci = {str(k).upper(): v for k, v in nm.items()}
    with safe_open(str(SAFET), framework="numpy") as f:
        W = f.get_tensor("bert.embeddings.word_embeddings.weight")      # (V, 768)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return tok, nm_ci, W


def emb_of(sym, tok, nm_ci, W):
    e = nm_ci.get(str(sym).upper())
    if e is None or e not in tok:
        return None
    return W[tok[e]]


def predict(df, tok, nm_ci, W, mode, power):
    """Leave-one-out disjoint kNN prior for DE and direction."""
    perts = df["pert"].to_numpy(); genes = df["gene"].to_numpy()
    lab = df["label"].to_numpy()
    is_de = (lab != "none").astype(float)
    is_up = (lab == "up").astype(float)

    # unique pert/gene embeddings + index per row
    upert = {p: emb_of(p, tok, nm_ci, W) for p in set(perts)}
    ugene = {g: emb_of(g, tok, nm_ci, W) for g in set(genes)}
    covered = np.array([(upert[p] is not None) and (ugene[g] is not None)
                        for p, g in zip(perts, genes)])

    # build matrices over rows (use unit vectors; uncovered -> zeros so sim=0)
    Ep = np.array([upert[p] if upert[p] is not None else np.zeros(W.shape[1]) for p in perts])
    Eg = np.array([ugene[g] if ugene[g] is not None else np.zeros(W.shape[1]) for g in genes])

    N = len(df)
    prior_de = np.full(N, np.nan); prior_up = np.full(N, np.nan)
    de_mask_for_up = is_de.astype(bool)
    for i in range(N):
        if not covered[i]:
            continue
        sp = Ep @ Ep[i] if mode in ("pert", "both") else np.ones(N)
        sg = Eg @ Eg[i] if mode in ("gene", "both") else np.ones(N)
        sp = np.clip(sp, 0, None); sg = np.clip(sg, 0, None)
        w = (sp * sg) ** power
        # DISJOINT: drop pairs sharing this pert OR gene (and self)
        w[(perts == perts[i]) | (genes == genes[i])] = 0.0
        if w.sum() > 0:
            prior_de[i] = (w * is_de).sum() / w.sum()
            wd = w * is_de                      # restrict direction to DE neighbours
            if wd.sum() > 0:
                prior_up[i] = (wd * is_up).sum() / wd.sum()
    return is_de, is_up, covered, prior_de, prior_up


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--power", type=float, default=4.0, help="similarity sharpening exponent")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok, nm_ci, W = load_embeddings()
    df = pd.read_csv(args.train)
    if args.limit:
        df = df.head(args.limit)
    print(f"[knn] {len(df)} train pairs; Geneformer token embeddings dim={W.shape[1]}; power={args.power}")

    for mode in ["pert", "gene", "both"]:
        is_de, is_up, cov, pde, pup = predict(df, tok, nm_ci, W, mode, args.power)
        ok = ~np.isnan(pde)
        a_de = auroc(is_de[ok], pde[ok]); lo, hi = boot_ci(is_de[ok], pde[ok])
        dem = (is_de == 1) & ~np.isnan(pup)
        a_dir = auroc(is_up[dem], pup[dem]); dlo, dhi = boot_ci(is_up[dem], pup[dem])
        print(f"\n[{mode}-similarity]  covered {cov.mean():.0%}")
        print(f"   DE  AUROC {a_de:.3f}  95% CI [{lo:.3f},{hi:.3f}]  (n={int(ok.sum())})")
        print(f"   DIR AUROC {a_dir:.3f}  95% CI [{dlo:.3f},{dhi:.3f}]  (n={int(dem.sum())})")


if __name__ == "__main__":
    main()
