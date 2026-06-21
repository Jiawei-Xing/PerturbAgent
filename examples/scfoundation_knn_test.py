"""scFoundation gene embeddings — the scale-bet rung of the embedding ladder
(50M+ cells, xTrimoGene). Final ceiling check: does adding scFoundation to the
production GF⊕scGPT gene-similarity kNN improve DIR/DE at full power?

scFoundation ships only the full mae_autobin checkpoint; the static per-gene
vector is pos_emb.weight (gene2vec-initialized positional embedding), rows
0..19263 aligned to OS_scRNA_gene_index.19264.tsv (human symbols). Mean-center
by default (the anisotropy lesson from GenePT/UCE); also report raw.

Reuses uce_knn_test's loaders, disjoint-LOO predict_gene, and paired bootstrap.

    .venv/bin/python examples/scfoundation_knn_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from uce_knn_test import load_geneformer, load_scgpt, predict_gene, auroc, boot_ci, boot_delta  # noqa

SCF = ROOT / "outputs" / "scfoundation"
CKPT = "/grid/siepel/home/xing/.cache/huggingface/hub/models--perturblab--scfoundation-gene/snapshots/a0e099dc2fa3af642b245a0c15e2db88d2e90d0a/model.pt"


def load_scfoundation(center=True):
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    W = sd["pos_emb.weight"].float().numpy()
    syms = [l.split("\t")[0] for l in open(SCF / "gene_index.tsv").read().splitlines()[1:]]
    W = W[: len(syms)]                       # drop the 3 appended special tokens
    assert W.shape[0] == len(syms) == 19264, (W.shape, len(syms))
    if center:
        W = W - W.mean(0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return {s.upper(): W[i] for i, s in enumerate(syms)}


def report(name, df, spaces, power):
    is_de, is_up, cov, pde, pup = predict_gene(df, spaces, power)
    ok = ~np.isnan(pde); a_de = auroc(is_de[ok], pde[ok]); lo, hi = boot_ci(is_de[ok], pde[ok])
    dem = (is_de == 1) & ~np.isnan(pup); a_dir = auroc(is_up[dem], pup[dem]); dlo, dhi = boot_ci(is_up[dem], pup[dem])
    print(f"[{name:28s}] cov {cov.mean():.0%}  DE {a_de:.3f}[{lo:.3f},{hi:.3f}]  DIR {a_dir:.3f}[{dlo:.3f},{dhi:.3f}] (n_dir={int(dem.sum())})")
    return is_de, is_up, pde, pup


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    gf = load_geneformer(); sc = load_scgpt(center=True)
    scf_c = load_scfoundation(center=True); scf_r = load_scfoundation(center=False)
    genes = set(g.upper() for g in df["gene"])
    print(f"gene coverage: GF {len(genes&set(gf))/len(genes):.0%}  scGPT {len(genes&set(sc))/len(genes):.0%}  "
          f"scFound {len(genes&set(scf_c))/len(genes):.0%}  (of {len(genes)})\n")

    report("GF only", df, [gf], 2.0)
    report("scFound only (centered)", df, [scf_c], 2.0)
    report("scFound only (raw)", df, [scf_r], 2.0)
    cur = report("GF+scGPT (production)", df, [gf, sc], 2.0)
    cand = report("GF+scGPT+scFound", df, [gf, sc, scf_c], 2.0)

    cur_de, cur_up, cur_pde, cur_pup = cur
    _, _, cand_pde, cand_pup = cand
    dem = (cur_de == 1) & ~np.isnan(cur_pup) & ~np.isnan(cand_pup)
    md, l, h, pg = boot_delta(cur_up[dem], cand_pup[dem], cur_pup[dem])
    print(f"\npaired DIR [GF+scGPT+scFound − GF+scGPT]  {md:+.4f} CI[{l:+.4f},{h:+.4f}] P(>0)={pg:.2f} (n={int(dem.sum())})")
    okde = ~np.isnan(cur_pde) & ~np.isnan(cand_pde)
    md, l, h, pg = boot_delta(cur_de[okde], cand_pde[okde], cur_pde[okde])
    print(f"paired DE  [GF+scGPT+scFound − GF+scGPT]  {md:+.4f} CI[{l:+.4f},{h:+.4f}] P(>0)={pg:.2f} (n={int(okde.sum())})")


if __name__ == "__main__":
    main()
