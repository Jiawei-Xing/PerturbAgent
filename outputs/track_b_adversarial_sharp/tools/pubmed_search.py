"""
pubmed_search -- Retrieve literature evidence from PubMed (NCBI E-utilities).

A free, retrieval-only tool: ``esearch`` ranks abstracts for a query and
``efetch`` returns their titles + abstracts.  Useful for direct co-mention
evidence ("does pert X regulate gene Y?") and for sign evidence ("is X a
negative/positive regulator of Y?").  Note that many measurement targets in
this screen are uncharacterized RIKEN cDNAs with no literature -- an empty
result is itself informative and supports leaning toward 'none'.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pubmed_search",
        "description": (
            "Search PubMed for a free-text query and return the top abstracts "
            "(titles + truncated abstract text). Use for literature evidence "
            "of regulation or interaction between genes. An empty result means "
            "no literature was found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "PubMed query, e.g. 'Stat1 Irf1 macrophage regulation'."
                    ),
                },
                "retmax": {
                    "type": "integer",
                    "description": "Max abstracts to return (default 4, max 8).",
                },
            },
            "required": ["query"],
        },
    },
}


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"Accept": "application/xml"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _esearch(query: str, retmax: int) -> list[str]:
    q = urllib.parse.quote(query)
    url = (
        f"{_EUTILS}/esearch.fcgi?db=pubmed&retmode=json"
        f"&retmax={retmax}&term={q}"
    )
    import json
    data = json.loads(_get(url).decode())
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch(pmids: list[str]) -> list[tuple[str, str]]:
    ids = ",".join(pmids)
    url = f"{_EUTILS}/efetch.fcgi?db=pubmed&retmode=xml&id={ids}"
    root = ET.fromstring(_get(url))
    out = []
    for art in root.findall(".//PubmedArticle"):
        title_el = art.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"
        abstract = " ".join(
            "".join(a.itertext()).strip()
            for a in art.findall(".//Abstract/AbstractText")
        ).strip()
        out.append((title, abstract))
    return out


def pubmed_search(query: str, retmax: int = 4) -> str:
    """Return top PubMed abstracts for `query` as a readable string."""
    if not query or not query.strip():
        return "Error: provide a query."
    retmax = min(max(1, int(retmax)), 8)
    try:
        pmids = _esearch(query, retmax)
        if not pmids:
            return f"No PubMed results for '{query}'."
        articles = _efetch(pmids)
    except Exception as e:  # noqa: BLE001
        return f"Error querying PubMed for '{query}': {e}"
    if not articles:
        return f"No PubMed results for '{query}'."

    lines = [f"PubMed results for '{query}' ({len(articles)} shown):"]
    for i, (title, abstract) in enumerate(articles, 1):
        lines.append(f"  [{i}] {title}")
        if abstract:
            snippet = abstract[:500] + ("..." if len(abstract) > 500 else "")
            lines.append(f"      {snippet}")
    return "\n".join(lines)
