"""Gate: does FUNCTIONAL-class guidance beat raw embedding cosine in the kNN?

Tests whether the leader's likely "LLM-guided kNN aggregator" has headroom,
cheaply and without any LLM calls. An LLM would supply each gene's functional
class; GO-BP + Reactome term sets are a high-coverage proxy. If restricting /
reweighting the gene-sim kNN by shared function beats raw cosine on disjoint-LOO
DIR/DE -- especially on hub genes where cosine washes out -- then the LLM is the
scalable way to supply that classification and the agent-integration is worth
building. If not, functional grouping == embedding neighborhood (saturated).

Variants vs the production GF⊕scGPT cosine kNN (power=2):
  func     : IDF-weighted functional cosine only (no embedding)
  gate     : embedding cosine, but zero neighbors with no shared function
  emb*func : embedding cosine * functional cosine
Reported on all covered rows, the GO-covered subset, and the hub (high-support)
subset. Reuses support_aware_knn_de.build_sim.
"""
from __future__ import annotations
import sys, glob, collections
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from support_aware_knn_de import build_sim, load_geneformer, load_scgpt  # noqa
from uce_knn_test import auroc, boot_ci, boot_delta  # noqa

LIBS = ["GO_Biological_Process_2023", "Reactome_2022"]


def load_terms():
    g2t = collections.defaultdict(set)
    for lib in LIBS:
        for line in open(ROOT / "outputs" / "func_db" / f"{lib}.gmt"):
            p = line.rstrip("\n").split("\t"); term = lib[:2] + ":" + p[0]
            for x in p[2:]:
                if x:
                    g2t[x.split(",")[0].upper()].add(term)
    return g2t


def func_sim_matrix(genes):
    """IDF-weighted functional-cosine similarity (N×N) over GO-BP+Reactome terms."""
    g2t = load_terms()
    # terms appearing in >=2 of our genes
    df = collections.Counter()
    rowterms = [g2t.get(str(g).upper(), set()) for g in genes]
    uniq_per_gene = {}
    for g in set(str(x).upper() for x in genes):
        for t in g2t.get(g, set()):
            df[t] += 1
    keep = {t: i for i, t in enumerate((t for t, c in df.items() if c >= 2))}
    Nt = len(keep)
    has = np.array([len(rt & keep.keys()) > 0 for rt in rowterms])
    idf = np.zeros(Nt)
    Ng = len(set(str(x).upper() for x in genes))
    for t, i in keep.items():
        idf[i] = np.log(Ng / df[t])
    B = np.zeros((len(genes), Nt), np.float32)
    for r, rt in enumerate(rowterms):
        for t in rt:
            j = keep.get(t)
            if j is not None:
                B[r, j] = idf[j]
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    F = (B @ B.T).astype(np.float32)
    print(f"functional terms kept (>=2 genes): {Nt}; rows with >=1 term: {has.mean():.0%}")
    return F, has


def knn(weight, same, is_de, covmask):
    W = weight.copy(); W[same] = 0.0
    den = W.sum(1); pde = np.full(len(W), np.nan); pup = np.full(len(W), np.nan)
    ok = (den > 0) & covmask
    num = (W * is_de[None, :]).sum(1)
    pde[ok] = num[ok] / den[ok]
    return pde


def knn_dir(weight, same, is_de, is_up, covmask):
    W = weight.copy(); W[same] = 0.0
    wde = W * is_de[None, :]; den = wde.sum(1)
    pup = np.full(len(W), np.nan); ok = (den > 0) & covmask
    num = (wde * is_up[None, :]).sum(1)
    pup[ok] = num[ok] / den[ok]
    return pup


def rep(tag, y, s, base=None):
    m = ~np.isnan(s) & ~np.isnan(y.astype(float))
    if m.sum() < 30 or len(set(y[m])) < 2:
        return f"  {tag:20s} (n={int(m.sum())} too few)"
    a = auroc(y[m], s[m]); lo, hi = boot_ci(y[m], s[m])
    out = f"  {tag:20s} {a:.3f} CI[{lo:.3f},{hi:.3f}] n={int(m.sum())}"
    if base is not None:
        mm = m & ~np.isnan(base)
        md, l, h, pg = boot_delta(y[mm], s[mm], base[mm])
        out += f"   Δvsemb {md:+.4f} P(>0)={pg:.2f}"
    return out


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    genes = df["gene"].to_numpy()
    is_de = (df["label"].to_numpy() != "none").astype(float)
    is_up = (df["label"].to_numpy() == "up").astype(float)
    gf = load_geneformer(); sc = load_scgpt(center=True)
    print("building embedding sim...")
    S, same, covered = build_sim(df, [gf, sc])
    print("building functional sim...")
    F, hasfunc = func_sim_matrix(genes)

    emb = S ** 2.0
    funcw = F ** 2.0
    gate = emb * (F > 0)                       # cosine restricted to shared-function neighbors
    embxf = emb * F                            # cosine * functional cosine

    supp = (emb * ~same).sum(1)
    hub = covered & (supp >= np.nanquantile(supp[covered], 0.75))

    weights = {"emb (baseline)": emb, "func only": funcw, "gate": gate, "emb*func": embxf}
    de = {k: knn(w, same, is_de, covered) for k, w in weights.items()}
    dr = {k: knn_dir(w, same, is_de, is_up, covered) for k, w in weights.items()}
    base_de = de["emb (baseline)"]; base_dr = dr["emb (baseline)"]

    for region, mask in [("ALL covered", covered),
                         ("GO-covered rows", covered & hasfunc),
                         ("HUB (top-25% support)", hub)]:
        print(f"\n=== {region} (n={int(mask.sum())}) ===")
        print(" DE:")
        for k in weights:
            print(rep(k, is_de[mask].astype(int), de[k][mask], None if "base" in k else base_de[mask]))
        demask = mask & (is_de == 1)
        print(f" DIR (n_DE={int(demask.sum())}):")
        for k in weights:
            print(rep(k, is_up[demask].astype(int), dr[k][demask], None if "base" in k else base_dr[demask]))


if __name__ == "__main__":
    main()
