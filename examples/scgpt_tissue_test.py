"""Lever A: cell-type-matched scGPT embeddings for the kNN.

The kNN signal is gene co-regulation; BMDMs are myeloid/blood lineage, so a
BLOOD-tuned scGPT embedding may transfer direction better than whole-human.
Tests scGPT-blood (immune-matched) vs scGPT-brain (non-immune control) vs the
whole-human scGPT already in the 0.619 submission, all against Geneformer, in
the same disjoint-LOO gene-similarity harness (reused from uce_knn_test.py).

If blood > whole-human > brain, cell-type matching is the lever and we ensemble
blood in / swap it for whole-human in the fusion. brain is the falsification
control (should NOT beat whole-human if the effect is real cell-type matching).

    uv run --extra serve python examples/scgpt_tissue_test.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("U", ROOT / "examples" / "uce_knn_test.py")
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)


def load_scgpt_dir(d, center=False):
    vocab = json.load(open(ROOT / d / "vocab.json"))
    sd = torch.load(ROOT / d / "best_model.pt", map_location="cpu", weights_only=False)
    W = sd["encoder.embedding.weight"].float().numpy()
    if center:
        W = W - W.mean(0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return {str(s).upper(): W[i] for s, i in vocab.items() if not str(s).startswith("<")}


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    POWER = 2.0
    gf = U.load_geneformer()
    whole = load_scgpt_dir("outputs/scgpt")
    blood = load_scgpt_dir("outputs/scgpt_blood")
    brain = load_scgpt_dir("outputs/scgpt_brain")
    print(f"[lever-A] {len(df)} train pairs; power={POWER}\n")

    configs = [
        ("Geneformer", [gf]),
        ("scGPT whole-human", [whole]),
        ("scGPT BLOOD (immune)", [blood]),
        ("scGPT brain (control)", [brain]),
        ("GF ⊕ scGPT-whole (=0.619 sub)", [gf, whole]),
        ("GF ⊕ scGPT-BLOOD", [gf, blood]),
        ("GF ⊕ whole ⊕ BLOOD", [gf, whole, blood]),
    ]
    res = {}
    for name, sp in configs:
        is_de, is_up, cov, pde, pup = U.predict_gene(df, sp, POWER)
        ok = ~np.isnan(pde); dem = (is_de == 1) & ~np.isnan(pup)
        a_de = U.auroc(is_de[ok], pde[ok])
        a_dir = U.auroc(is_up[dem], pup[dem]); dlo, dhi = U.boot_ci(is_up[dem], pup[dem])
        print(f"{name:32s} cov {cov.mean():.0%}  DE {a_de:.3f}  DIR {a_dir:.3f} [{dlo:.3f},{dhi:.3f}]  (nDIR={int(dem.sum())})")
        res[name] = (is_de, is_up, pde, pup)

    # paired DIR deltas vs the current 0.619 base (GF ⊕ scGPT-whole)
    base = res["GF ⊕ scGPT-whole (=0.619 sub)"]
    print("\npaired DIR delta vs GF⊕whole (the 0.619 embedding):")
    for name in ["GF ⊕ scGPT-BLOOD", "GF ⊕ whole ⊕ BLOOD", "scGPT BLOOD (immune)"]:
        is_de, is_up, pde, pup = res[name]
        bde, bup = base[0], base[3]
        dem = (is_de == 1) & ~np.isnan(pup) & ~np.isnan(bup)
        md, l, h, pg = U.boot_delta(is_up[dem], pup[dem], bup[dem])
        print(f"   {name:24s} {md:+.3f} CI [{l:+.3f},{h:+.3f}] P(>0)={pg:.2f} (n={int(dem.sum())})")


if __name__ == "__main__":
    main()
