"""One more rung on the embedding ladder: bigger Geneformer (316M) + GenePT axis.

The validated lever is embedding richness/scale (GF-104M -> scGPT-33M-cells won).
This tests Geneformer-V2-316M (3x params, dim 1152 vs 768, same vocab) as a
drop-in, plus GenePT (text/literature axis, orthogonal to co-expression) as an
ensemble member, all against the shipped GF104⊕scGPT (disjoint-LOO DIR 0.668).

Ship ONLY if a config beats GF104⊕scGPT beyond the paired bootstrap CI on the
n≈3k high-power pool -- same bar that called scGPT (+0.029, P=1.00).

    uv run --extra serve python examples/embed_ladder_test.py
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("U", ROOT / "examples" / "uce_knn_test.py")
U = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(U)
GFDIR = ROOT / "outputs" / "geneformer_probe" / "model" / "geneformer"


def _gf_dicts():
    tok = pickle.load(open(GFDIR / "token_dictionary_gc104M.pkl", "rb"))
    nm = pickle.load(open(GFDIR / "gene_name_id_dict_gc104M.pkl", "rb"))
    return tok, {str(k).upper(): v for k, v in nm.items()}


def load_geneformer_size(safet):
    tok, nm_ci = _gf_dicts()
    with safe_open(str(safet), framework="numpy") as f:
        W = f.get_tensor("bert.embeddings.word_embeddings.weight")
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return {s: W[tok[e]] for s, e in nm_ci.items() if e in tok}


def load_genept(which="3large", center=True):
    fn = ("GenePT_gene_protein_embedding_model_3_text.pickle." if which == "3large"
          else "GenePT_gene_embedding_ada_text.pickle")
    d = pickle.load(open(ROOT / "outputs/genept/GenePT_emebdding_v2" / fn, "rb"))
    syms = list(d.keys())
    M = np.stack([np.asarray(d[s], np.float32).ravel() for s in syms])
    if center:
        M = M - M.mean(0)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return {str(s).upper(): M[i] for i, s in enumerate(syms)}


def main():
    df = pd.read_csv(ROOT / "data" / "train.csv")
    POWER = 2.0
    gf104 = load_geneformer_size(ROOT / "outputs/geneformer_probe/model/Geneformer-V2-104M/model.safetensors")
    gf316 = load_geneformer_size(ROOT / "outputs/geneformer_probe/model/Geneformer-V2-316M/model.safetensors")
    import importlib.util as _i
    _b = _i.spec_from_file_location("Bd", ROOT / "examples/build_scgpt_fusion_submission.py")
    Bd = _i.module_from_spec(_b); _b.loader.exec_module(Bd)
    scgpt = Bd.load_scgpt()
    genept = load_genept("3large")
    print(f"[ladder] {len(df)} pairs; power={POWER}\n")

    configs = [
        ("GF-104M", [gf104]),
        ("GF-316M", [gf316]),
        ("scGPT", [scgpt]),
        ("GF104 ⊕ scGPT  (=0.624 sub)", [gf104, scgpt]),
        ("GF316 ⊕ scGPT", [gf316, scgpt]),
        ("GF104 ⊕ GF316 ⊕ scGPT", [gf104, gf316, scgpt]),
        ("GF104 ⊕ scGPT ⊕ GenePT", [gf104, scgpt, genept]),
        ("GF316 ⊕ scGPT ⊕ GenePT", [gf316, scgpt, genept]),
    ]
    res = {}
    for name, sp in configs:
        is_de, is_up, cov, pde, pup = U.predict_gene(df, sp, POWER)
        ok = ~np.isnan(pde); dem = (is_de == 1) & ~np.isnan(pup)
        a_de = U.auroc(is_de[ok], pde[ok])
        a_dir = U.auroc(is_up[dem], pup[dem]); dlo, dhi = U.boot_ci(is_up[dem], pup[dem])
        print(f"{name:30s} cov {cov.mean():.0%}  DE {a_de:.3f}  DIR {a_dir:.3f} [{dlo:.3f},{dhi:.3f}]  (n={int(dem.sum())})")
        res[name] = (is_de, is_up, pde, pup)

    base = res["GF104 ⊕ scGPT  (=0.624 sub)"]
    print("\npaired DIR delta vs GF104⊕scGPT (the 0.624 embedding):")
    for name in ["GF316 ⊕ scGPT", "GF104 ⊕ GF316 ⊕ scGPT", "GF104 ⊕ scGPT ⊕ GenePT", "GF316 ⊕ scGPT ⊕ GenePT"]:
        is_de, is_up, pde, pup = res[name]; bup = base[3]
        dem = (is_de == 1) & ~np.isnan(pup) & ~np.isnan(bup)
        md, l, h, pg = U.boot_delta(is_up[dem], pup[dem], bup[dem])
        de_ok = ~np.isnan(pde) & ~np.isnan(base[2])
        mdd, ld, hd, pgd = U.boot_delta(res[name][0][de_ok], pde[de_ok], base[2][de_ok])
        print(f"   {name:26s} DIR {md:+.3f}[{l:+.3f},{h:+.3f}] P={pg:.2f} | DE {mdd:+.3f}[{ld:+.3f},{hd:+.3f}] P={pgd:.2f}")


if __name__ == "__main__":
    main()
