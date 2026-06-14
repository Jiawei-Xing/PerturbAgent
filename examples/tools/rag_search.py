"""
rag_search -- Retrieval-augmented literature evidence (BM25 passage re-ranking).

An upgrade over ``pubmed_search``: instead of returning whole top-ranked
abstracts (lexical *document* match, which drags in off-topic papers when a
multi-gene query collides with a popular unrelated topic), this tool

  1. casts a wider net of candidate abstracts via several focused ``esearch``
     queries (direct co-mention, plus each gene with macrophage/regulation
     context),
  2. splits every abstract into sentence-level *passages*, and
  3. re-ranks those passages with BM25 against a perturbation+gene query,
     surfacing only the few sentences that actually mention the pair / its
     regulation.

BM25 rewards rare-term overlap via IDF, so a sentence naming *both* ``pert``
and ``gene`` outranks generic boilerplate even if it lives inside an otherwise
tangential paper -- exactly the denoising the keyword tool lacks.  The scoring
is a self-contained ~40-line implementation (no extra dependency, runs fully
offline once abstracts are fetched).

Network use is identical in spirit to ``pubmed_search`` (NCBI E-utilities over
HTTPS); failures degrade to an informative string.  As with the keyword tool,
an empty result for an uncharacterized RIKEN/Gm target is itself a signal that
supports leaning toward 'none'.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

# Reuse the vetted E-utilities helpers (request headers, esearch/efetch parsing).
from .pubmed_search import _esearch, _efetch

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "Retrieval-augmented literature search. Fetches candidate PubMed "
            "abstracts about a perturbation and a target gene, then returns the "
            "individual sentences most relevant to whether the perturbation "
            "regulates the gene (BM25 passage re-ranking). Prefer this over "
            "pubmed_search for regulation/sign evidence. Empty means no "
            "literature was found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pert": {
                    "type": "string",
                    "description": "Perturbation (knocked-down) gene symbol, e.g. 'Stat1'.",
                },
                "gene": {
                    "type": "string",
                    "description": "Measured target gene symbol, e.g. 'Irf1'.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 5, max 8).",
                },
            },
            "required": ["pert", "gene"],
        },
    },
}

# Generic terms expanded into the query so regulation/sign sentences rank up
# even when pert/gene co-mention is sparse.  IDF keeps these from dominating.
_CONTEXT_TERMS = [
    "regulat", "express", "induc", "repress", "activat", "inhibit",
    "target", "knockdown", "knockout", "downstream", "upregulat",
    "downregulat", "macrophage",
]

_WORD = re.compile(r"[a-z0-9]+")
# Split abstracts into sentences without dragging in nltk; good enough for
# biomedical prose (avoids splitting on common abbreviations' trailing dot by
# requiring a following capital/space-capital).
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) >= 2]


def _passages(articles: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Flatten (title, abstract) pairs into (source_title, sentence) passages."""
    out: list[tuple[str, str]] = []
    for title, abstract in articles:
        for sent in _SENT.split(abstract):
            sent = sent.strip()
            if len(sent) >= 30:  # drop fragments
                out.append((title, sent))
    return out


def _bm25_rank(
    passages: list[tuple[str, str]],
    query_terms: list[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[float, int]]:
    """Return (score, passage_index) sorted high-to-low."""
    docs = [_tokenize(s) for _, s in passages]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n
    df: Counter = Counter()
    for d in docs:
        for term in set(d):
            df[term] += 1
    idf = {
        t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
        for t in set(query_terms)
        if df.get(t, 0) > 0
    }
    scored: list[tuple[float, int]] = []
    for i, d in enumerate(docs):
        if not d:
            continue
        tf = Counter(d)
        dl = len(d)
        s = 0.0
        for t in query_terms:
            if t not in idf:
                continue
            f = tf.get(t, 0)
            if f == 0:
                continue
            s += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        if s > 0:
            scored.append((s, i))
    scored.sort(reverse=True)
    return scored


def _gather_candidates(pert: str, gene: str, *, per_query: int) -> list[tuple[str, str]]:
    """Union candidate abstracts from a few focused esearch queries."""
    queries = [
        f"{pert} AND {gene}",                 # direct co-mention (best, often empty)
        f"{gene} regulation expression",      # what regulates the target
        f"{pert} macrophage regulation",      # the perturbation's biology
    ]
    pmids: list[str] = []
    seen: set[str] = set()
    for q in queries:
        try:
            for pid in _esearch(q, per_query):
                if pid not in seen:
                    seen.add(pid)
                    pmids.append(pid)
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.34)  # NCBI politeness (3 req/s without an API key)
    if not pmids:
        return []
    try:
        return _efetch(pmids)
    except Exception:  # noqa: BLE001
        return []


def rag_search(pert: str, gene: str, top_k: int = 5) -> str:
    """Return the BM25-top passages relating `pert` to `gene` as a string."""
    if not pert or not gene:
        return "Error: provide both pert and gene."
    top_k = min(max(1, int(top_k)), 8)
    try:
        articles = _gather_candidates(pert, gene, per_query=8)
    except Exception as e:  # noqa: BLE001
        return f"Error retrieving literature for {pert}/{gene}: {e}"
    if not articles:
        return (
            f"No literature found relating '{pert}' and '{gene}'. "
            "For an uncharacterized target this favors 'none'."
        )

    passages = _passages(articles)
    if not passages:
        return f"No usable abstract text for '{pert}'/'{gene}'."

    query_terms = _tokenize(f"{pert} {gene}") + _CONTEXT_TERMS
    ranked = _bm25_rank(passages, query_terms)
    if not ranked:
        return (
            f"Found {len(articles)} abstract(s) for '{pert}'/'{gene}' but none "
            "mention the pair or its regulation."
        )

    pl, gl = pert.lower(), gene.lower()
    lines = [
        f"Top {min(top_k, len(ranked))} retrieved passages for {pert} -> {gene} "
        f"(BM25 over {len(passages)} sentences from {len(articles)} abstracts):"
    ]
    for rank, (_, idx) in enumerate(ranked[:top_k], 1):
        title, sent = passages[idx]
        both = pl in sent.lower() and gl in sent.lower()
        flag = " [co-mentions both]" if both else ""
        sent = sent[:300] + ("..." if len(sent) > 300 else "")
        lines.append(f"  [{rank}]{flag} {sent}")
        lines.append(f"      (from: {title[:90]})")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test (requires network): a well-studied pair should co-mention.
    print(rag_search("Stat1", "Irf1"))
