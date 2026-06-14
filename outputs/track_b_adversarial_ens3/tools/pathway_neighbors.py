"""
pathway_neighbors -- Turn an unseen perturbation into seen analogues.

The competition split is disjoint on both axes, so a test perturbation never
appears in the training data and a direct lookup is useless.  This tool finds
the perturbation's functional neighborhood via STRING protein-protein
interactions, then intersects those partners with the perturbations that *do*
appear in train and reports how *their* knockdowns behaved.  That gives the
EFFECT/NULL debate an analogy-based prior: "this gene isn't in train, but its
complex-mates are, and knocking them down was differentially-expressing X% of
the time."
"""

from __future__ import annotations

import json
import urllib.request

from ._traindata import load_train

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pathway_neighbors",
        "description": (
            "Find STRING interaction partners of a perturbation gene and "
            "report which of them appear as perturbations in the training "
            "data, along with how their CRISPRi knockdowns behaved "
            "(up/down/none distribution). Use to reason by analogy about an "
            "unseen perturbation's downstream impact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pert": {
                    "type": "string",
                    "description": "Perturbed gene symbol (e.g. 'Aars').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max STRING partners to fetch (default 25).",
                },
            },
            "required": ["pert"],
        },
    },
}


def _string_partners(pert: str, limit: int) -> list[tuple[str, float]]:
    url = (
        "https://string-db.org/api/json/interaction_partners?"
        f"identifiers={pert}&species=10090&limit={limit}"
    )
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    out = []
    for e in data:
        name = e.get("preferredName_B") or e.get("stringId_B")
        if name:
            out.append((name, float(e.get("score", 0.0))))
    return out


def pathway_neighbors(pert: str, limit: int = 25) -> str:
    """Report STRING neighbors of `pert` that are perturbations in train,
    with their knockdown effect distributions."""
    limit = min(max(1, int(limit)), 50)
    try:
        partners = _string_partners(pert, limit)
    except Exception as e:  # noqa: BLE001
        return f"Error querying STRING DB for '{pert}': {e}"
    if not partners:
        return f"No STRING interaction partners found for '{pert}' (mouse)."

    df = load_train()
    pert_lower = {p.lower(): p for p in df["pert"].astype(str).unique()}
    lab = df["label"].astype(str).str.lower()

    lines = [
        f"STRING neighbors of '{pert}' (mouse) and their training-data "
        f"knockdown behavior:"
    ]
    analogue_rows = []  # (name, score, n, up, down, none)
    other_partners = []
    for name, score in partners:
        train_name = pert_lower.get(name.lower())
        if train_name is None:
            other_partners.append((name, score))
            continue
        hits = df[df["pert"].astype(str).str.lower() == name.lower()]
        h_lab = lab[hits.index]
        analogue_rows.append((
            train_name, score, len(hits),
            int((h_lab == "up").sum()),
            int((h_lab == "down").sum()),
            int((h_lab == "none").sum()),
        ))

    if analogue_rows:
        tot_up = tot_down = tot_none = 0
        lines.append(
            f"  {len(analogue_rows)} neighbor(s) are perturbations in train:"
        )
        for name, score, n, up, down, none in sorted(
            analogue_rows, key=lambda r: -r[1]
        ):
            tot_up += up
            tot_down += down
            tot_none += none
            lines.append(
                f"    - {name} (STRING {score:.2f}): {n} targets -> "
                f"{up} up, {down} down, {none} none"
            )
        tot = tot_up + tot_down + tot_none
        if tot:
            de = tot_up + tot_down
            lines.append(
                f"  Pooled over analogues: P(DE)={de / tot:.1%}, "
                f"and among DE, up:down = "
                f"{(tot_up / tot_down) if tot_down else float('inf'):.2f}:1."
            )
    else:
        lines.append(
            "  None of the STRING neighbors appear as perturbations in train "
            "(no direct analogue available)."
        )

    if other_partners:
        names = ", ".join(n for n, _ in other_partners[:15])
        lines.append(f"  Other interaction partners (not in train): {names}")
    return "\n".join(lines)
