#!/usr/bin/env python
"""
Category-prior feature test -- is there transferable DE/DIR signal in simple
gene/pert *categories* (which survive the disjoint split, unlike gene identity)?

We already showed pert-STRING-analogy (analogue_p_de 0.488) and GRN reachability
(0.510) are chance for DE.  But those are network features.  This tests the cheaper,
more robust hypothesis the agent already half-uses via `gene_classify`: that
category membership -- "the readout is a ribosomal protein", "the perturbation is
core-essential machinery" -- carries a transferable prior on whether/which-way a
pair moves, even though the specific gene never appears in train.

No LLM, no GPU.  Scored on the same blinded 250 sample + full train, with the
competition's two micro-AUROCs.  This is the diagnostic for "is DE really capped,
or did we leave a structured prior on the table" and "would a sharpened DIR judge
leaning on these curated direction sets actually have signal to sharpen."
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

from tools.gene_classify import (  # noqa: E402
    _CELLCYCLE_DOWN, _ISR_UP, _P53_UP, _RIBOBIO_DOWN, _is_uncharacterized,
)

UP_SETS = _ISR_UP | _P53_UP
DOWN_SETS = _CELLCYCLE_DOWN | _RIBOBIO_DOWN

# Core-essential machinery: knocking these down triggers broad stress -> high DE.
# Name/prefix based so it transfers across the disjoint pert axis.
_ESSENTIAL_PREFIX = re.compile(
    r"^(Rp[sl]\d|Mrp[sl]\d|Psm[abcd]|Pol[r]?\d|Eif|Rps|Rpl|Snrp|Sf3|Prp|"
    r"Nup|Cct|Tubb|Tuba|Aar|.ars$)", re.IGNORECASE)
_ESSENTIAL_EXACT = {
    "Pcna", "Npm1", "Ncl", "Fbl", "Top2a", "Rrm2", "Rrm1", "Cdk1", "Plk1",
    "Hspa5", "Hsp90ab1", "Hsp90aa1", "Vcp", "Rpa1", "Rpa2", "Sod1",
}


def _cap(g: str) -> str:
    g = str(g).strip()
    return g[:1].upper() + g[1:].lower() if g else g


def gene_dir(g: str) -> int:
    c = _cap(g)
    if c in UP_SETS:
        return 1
    if c in DOWN_SETS or re.match(r"^M?rp[sl]\d", c):
        return -1
    return 0


def gene_stress(g: str) -> int:
    """1 if readout is in any curated stress-direction set (UP or DOWN)."""
    return 1 if gene_dir(g) != 0 else 0


def gene_unchar(g: str) -> int:
    return 1 if _is_uncharacterized(str(g)) else 0


def pert_essential(p: str) -> int:
    c = _cap(p)
    return 1 if (_ESSENTIAL_PREFIX.match(c) or c in _ESSENTIAL_EXACT) else 0


# --- AUROC (tie-aware) ------------------------------------------------------
def auroc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    n = len(s)
    npos, nneg = int(y.sum()), n - int(y.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sr = s[order]
    rk = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sr[j + 1] == sr[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return (rk[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def boot_ci(y, s, n=2000, seed=0):
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        a = auroc(y[idx], s[idx])
        if a == a:
            out.append(a)
    return np.percentile(out, [2.5, 97.5]) if out else (float("nan"),) * 2


def evaluate(df, tag):
    g_dir = df["gene"].map(gene_dir).to_numpy()
    g_stress = df["gene"].map(gene_stress).to_numpy()
    g_unchar = df["gene"].map(gene_unchar).to_numpy()
    p_ess = df["pert"].map(pert_essential).to_numpy()
    lab = df["label"].to_numpy()
    is_de = (lab != "none").astype(int)

    # Combined DE prior: essential pert raises DE; stress-set gene raises DE;
    # uncharacterized gene lowers DE.
    de_prior = p_ess + g_stress - g_unchar

    print(f"\n{'='*66}\n{tag}  (n={len(df)})  DE-rate={is_de.mean():.3f}")
    print(f"  pert essential: {p_ess.mean():.0%} | gene in stress-set: "
          f"{g_stress.mean():.0%} | gene unchar: {g_unchar.mean():.0%}")
    print("  -- DE AUROC ((up|down) vs none) --")
    for name, feat in [("pert_essential", p_ess), ("gene_stress_set", g_stress),
                       ("gene_unchar(neg)", -g_unchar), ("combined_prior", de_prior)]:
        lo, hi = boot_ci(is_de, feat)
        print(f"     {name:<18} {auroc(is_de, feat):.3f}  95%CI[{lo:.3f},{hi:.3f}]")

    # DIR among DE rows, using curated gene direction
    de = lab != "none"
    is_up = (lab[de] == "up").astype(int)
    gd = g_dir[de]
    print("  -- DIR AUROC (up vs down, DE rows) --")
    lo, hi = boot_ci(is_up, gd)
    print(f"     gene_dir (all DE, n={de.sum()})      {auroc(is_up, gd):.3f}  "
          f"95%CI[{lo:.3f},{hi:.3f}]")
    nz = gd != 0
    if nz.sum() >= 10 and len(set(is_up[nz])) == 2:
        lo, hi = boot_ci(is_up[nz], gd[nz])
        print(f"     gene_dir (in-set only, n={nz.sum()})  "
              f"{auroc(is_up[nz], gd[nz]):.3f}  95%CI[{lo:.3f},{hi:.3f}]  "
              f"[up-rate in-set: {is_up[nz].mean():.2f} vs base {is_up.mean():.2f}]")


def main():
    for tag, p in [("BLINDED 250 sample", ROOT / "outputs/benchmark_b_250/sample.csv"),
                   ("FULL train", ROOT / "data/train.csv")]:
        evaluate(pd.read_csv(p), tag)


if __name__ == "__main__":
    main()
