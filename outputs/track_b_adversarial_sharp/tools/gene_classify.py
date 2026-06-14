"""
gene_classify -- Cheap, offline structural/biological priors for a gene symbol.

Two signals that matter a lot on this mouse-BMDM CRISPRi screen:

1. **Uncharacterized targets.** Many measurement genes are RIKEN cDNAs
   (``*Rik``), predicted ``Gm#####`` models, or accession-style symbols. The
   model has no specific knowledge of these, so absent a concrete mechanism
   they should default toward ``none`` / low confidence rather than a
   hallucinated direction.

2. **Stress-program membership.** The dominant biology here is essential-gene
   knockdown triggering broad stress responses. Genes split cleanly by
   expected direction under that stress:
     * UP   -- integrated stress response (ATF4 targets) and p53 targets.
     * DOWN -- ribosomal proteins / ribosome biogenesis, cell-cycle and
               proliferation genes (growth program shut-down).
   These curated sets are small but high-precision when they fire.

All checks are pure string/lookup operations -- no network, no train data.
"""

from __future__ import annotations

import re

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "gene_classify",
        "description": (
            "Classify a mouse gene symbol with offline priors: whether it is "
            "an uncharacterized/predicted gene (RIKEN cDNA, Gm-model, "
            "accession-style), and whether it belongs to a stress program with "
            "a known expected direction under essential-gene knockdown "
            "(integrated stress response / p53 -> up; ribosomal / cell-cycle "
            "-> down)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gene": {
                    "type": "string",
                    "description": "Mouse gene symbol (e.g. 'Ddit3').",
                },
            },
            "required": ["gene"],
        },
    },
}

# Integrated stress response / ATF4 targets -> tend to go UP under
# translational, amino-acid, and proteostasis stress.
_ISR_UP = {
    "Atf3", "Atf4", "Ddit3", "Trib3", "Chac1", "Asns", "Slc7a11", "Mthfd2",
    "Vegfa", "Gdf15", "Cebpb", "Nupr1", "Stc2", "Sesn2", "Ppp1r15a", "Wars",
    "Cth", "Phgdh", "Psat1", "Shmt2", "Aldh18a1", "Ddit4", "Eif4ebp1",
    "Slc7a5", "Slc3a2", "Mt1", "Mt2",
}

# p53 transcriptional targets -> tend to go UP under genotoxic / ribosomal /
# nucleolar stress (which essential-gene knockdown frequently triggers).
_P53_UP = {
    "Cdkn1a", "Mdm2", "Bax", "Bbc3", "Gadd45a", "Sesn2", "Zmat3", "Eda2r",
    "Phlda3", "Ccng1", "Trp53inp1", "Btg2", "Sesn1", "Ei24", "Aen", "Cdkn1b",
}

# Cell-cycle / proliferation genes -> go DOWN when growth arrests.
_CELLCYCLE_DOWN = {
    "Mki67", "Ccnb1", "Ccnb2", "Ccna2", "Ccne1", "Ccne2", "Cdk1", "Cdk4",
    "Top2a", "Pcna", "Mcm2", "Mcm3", "Mcm4", "Mcm5", "Mcm6", "Mcm7", "Bub1",
    "Bub1b", "Plk1", "Aurka", "Aurkb", "Foxm1", "Cdc20", "Cdc6", "E2f1",
    "Birc5", "Cenpa", "Cenpe", "Kif11", "Tyms", "Rrm2",
}

# Ribosome biogenesis / nucleolar factors (in addition to the Rps*/Rpl*
# structural proteins detected by prefix) -> DOWN with biosynthesis shutdown.
_RIBOBIO_DOWN = {
    "Npm1", "Ncl", "Fbl", "Nop56", "Nop58", "Bop1", "Wdr12", "Pes1", "Ddx21",
    "Nip7", "Mybbp1a", "Rrp9", "Utp20", "Gnl3", "Nolc1", "Las1l",
}

_UNCHAR_PATTERNS = [
    re.compile(r"Rik$", re.IGNORECASE),          # RIKEN cDNA, e.g. 2810002D19Rik
    re.compile(r"^Gm\d+$", re.IGNORECASE),        # predicted gene model
    re.compile(r"^(AW|BC|AI|AK|AA|AV)\d{5,}$", re.IGNORECASE),  # accession-style
    re.compile(r"^LOC\d+$", re.IGNORECASE),
    re.compile(r"^\d", ),                          # starts with a digit
]


def _is_uncharacterized(gene: str) -> bool:
    g = gene.strip()
    return any(p.search(g) for p in _UNCHAR_PATTERNS)


def _stress_membership(gene: str) -> list[str]:
    g = gene.strip()
    g_cap = g[:1].upper() + g[1:].lower() if g else g
    hits: list[str] = []
    if g_cap in _ISR_UP:
        hits.append("integrated-stress-response/ATF4 target (expected UP)")
    if g_cap in _P53_UP:
        hits.append("p53 target (expected UP)")
    if g_cap in _CELLCYCLE_DOWN:
        hits.append("cell-cycle/proliferation gene (expected DOWN)")
    if g_cap in _RIBOBIO_DOWN:
        hits.append("ribosome-biogenesis/nucleolar factor (expected DOWN)")
    if re.match(r"^Rp[sl]\d", g_cap):
        hits.append("ribosomal protein (expected DOWN)")
    if re.match(r"^Mrp[sl]\d", g_cap):
        hits.append("mitochondrial ribosomal protein (expected DOWN)")
    return hits


def gene_classify(gene: str) -> str:
    """Return uncharacterized flag + stress-program membership for a gene.

    Examples:
        >>> "RIKEN" in gene_classify("2810002D19Rik")
        True
        >>> "UP" in gene_classify("Ddit3")
        True
        >>> "DOWN" in gene_classify("Rps6")
        True
    """
    if not gene or not gene.strip():
        return "Error: provide a gene symbol."
    lines = [f"Classification of '{gene}':"]
    if _is_uncharacterized(gene):
        lines.append(
            "  - UNCHARACTERIZED (RIKEN cDNA / predicted / accession-style "
            "symbol): no specific functional knowledge; lean toward 'none' "
            "unless a concrete mechanism is established."
        )
    else:
        lines.append("  - Named/characterized gene symbol.")
    mem = _stress_membership(gene)
    if mem:
        for m in mem:
            lines.append(f"  - Stress program: {m}")
    else:
        lines.append("  - No membership in the curated stress-program sets.")
    return "\n".join(lines)
