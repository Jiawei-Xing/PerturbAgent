"""
Shared training-data loader for Track B tools.

All tools that read ``train.csv`` go through ``load_train()`` so that the
offline benchmark can point them at a *blinded* copy via the
``MLGENX_TRAIN_CSV`` environment variable.  This matters because the real
competition split is disjoint along both the perturbation and gene axes --
a test perturbation or gene never appears in train -- so to estimate honest
performance on held-out train rows the lookup tables must exclude every row
that shares a perturbation or gene with the evaluation sample.  Pointing
``MLGENX_TRAIN_CSV`` at a pre-filtered CSV gives the tools the same
"the answer is not in the table" behavior they will see at test time.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "train.csv"
_CACHE: dict[str, pd.DataFrame] = {}


def train_csv_path() -> Path:
    """Resolve the active train CSV (env override or repo default)."""
    return Path(os.environ.get("MLGENX_TRAIN_CSV", str(_DEFAULT_CSV)))


def load_train() -> pd.DataFrame:
    """Load the active train CSV, cached per resolved path."""
    p = str(train_csv_path())
    if p not in _CACHE:
        _CACHE[p] = pd.read_csv(p)
    return _CACHE[p]
