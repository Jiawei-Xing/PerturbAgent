"""
train_base_rates -- Report global label priors from the training data.

The competition metric is AUROC (pure ranking), so a well-calibrated prior
anchors the absolute level of P(DE) and P(up|DE).  Most perturbations in this
screen are essential cellular machinery whose knockdown triggers broad stress
programs rather than clean one-to-one regulation, so the base rates carry real
signal: when no specific mechanism is found, defaulting toward these priors is
the rational fallback.
"""

from __future__ import annotations

from ._traindata import load_train

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "train_base_rates",
        "description": (
            "Return global label base rates (fraction up / down / none) and "
            "the up:down ratio among differentially expressed pairs, computed "
            "from the training data. Use as a prior when no specific mechanism "
            "is found."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


def train_base_rates() -> str:
    """Return label base rates over the training data as a readable string."""
    df = load_train()
    n = len(df)
    if n == 0:
        return "No training rows available."
    lab = df["label"].astype(str).str.lower()
    n_up = int((lab == "up").sum())
    n_down = int((lab == "down").sum())
    n_none = int((lab == "none").sum())
    n_de = n_up + n_down
    ud = (n_up / n_down) if n_down else float("inf")
    p_de = n_de / n
    p_up_given_de = (n_up / n_de) if n_de else float("nan")
    return (
        f"Training base rates over {n} (perturbation, gene) pairs:\n"
        f"  up:   {n_up} ({n_up / n:.1%})\n"
        f"  down: {n_down} ({n_down / n:.1%})\n"
        f"  none: {n_none} ({n_none / n:.1%})\n"
        f"  P(differentially expressed) = {p_de:.1%}\n"
        f"  Among DE pairs, up:down = {ud:.2f} : 1 "
        f"(P(up | DE) = {p_up_given_de:.1%})"
    )
