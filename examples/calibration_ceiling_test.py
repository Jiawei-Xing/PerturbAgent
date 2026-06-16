"""Calibration / probability-mapping ceiling test (a documented no-op).

The competition metric is two AUROCs, and AUROC is invariant to any *monotonic*
transform of the scores. The metric also decouples cleanly:
    DE  is ranked by  pred_up + pred_down            == P_DE
    DIR is ranked by  pred_up / (pred_up+pred_down)  == P_up|DE
so recalibrating P_DE and P_up|DE separately and monotonically (Platt, isotonic,
temperature, logit) changes NEITHER AUROC. Calibration is mathematically inert
here.

The only non-monotonic lever is TIE-BREAKING: rows at identical scores (the
integer-percent <prob>NN</prob> judge output produces many) get 0.5 credit, so
reordering them with a better-than-chance signal is a real gain. This script
quantifies that headroom by computing the ORACLE ceiling -- the AUROC if a
perfect, label-knowing tie-breaker reordered every tie optimally. That is the
absolute maximum any probability mapping could ever extract; a realistic
test-time tie-breaker can only approach it to the extent it beats chance, and
every external feature tested on this disjoint split is chance (see
grn_feature_test.py / geneformer_probe.py).

Run on the cached production benchmark predictions (sharp DE judge, numeric,
seed 42), which store [pred_up, pred_down] per id on a labeled train sample.

    uv run python examples/calibration_ceiling_test.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    s = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def auroc(y, s) -> float:
    y = np.asarray(y).astype(int); P = int(y.sum()); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = _rankdata(np.asarray(s, float))
    return (r[y == 1].sum() - P * (P + 1) / 2.0) / (P * N)


def ceiling(y, s):
    """(oracle, adversarial) AUROC: ties broken in favour of / against labels."""
    y = np.asarray(y).astype(int); eps = 1e-9
    return auroc(y, np.asarray(s, float) + eps * y), auroc(y, np.asarray(s, float) - eps * y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path,
                    default=ROOT / "outputs" / "benchmark_b_sharp_parser_num" / "preds.json")
    ap.add_argument("--sample", type=Path,
                    default=ROOT / "outputs" / "benchmark_b_sharp_parser_num" / "sample.csv")
    ap.add_argument("--key", default="debate1_num", help="condition prefix in preds.json")
    args = ap.parse_args()

    raw = json.loads(args.preds.read_text())
    preds = {k.split("|", 1)[1]: v for k, v in raw.items() if k.startswith(args.key + "|")}
    samp = pd.read_csv(args.sample)
    samp = samp[samp["id"].isin(preds)].reset_index(drop=True)
    pu = np.array([preds[i][0] for i in samp["id"]])
    pdn = np.array([preds[i][1] for i in samp["id"]])
    lab = samp["label"].to_numpy()

    P_DE = pu + pdn
    de = (lab != "none").astype(int)
    mask = lab != "none"
    den = np.where((pu + pdn) == 0, 1.0, pu + pdn)
    P_up = pu / den
    up = (lab[mask] == "up").astype(int)
    Pup_de = P_up[mask]

    print(f"n={len(samp)}  DE rows={int(mask.sum())}  up={int(up.sum())} down={int((up==0).sum())}")
    print(f"current DE AUROC  = {auroc(de, P_DE):.4f}")
    print(f"current DIR AUROC = {auroc(up, Pup_de):.4f}")

    print("\n-- (1) monotonic recalibration is a no-op --")
    transforms = [("x^3", lambda x: x ** 3), ("sqrt", np.sqrt),
                  ("logit", lambda x: np.log((x + 1e-6) / (1 - x + 1e-6))),
                  ("rank", _rankdata)]
    for name, f in transforms:
        print(f"   '{name:5}'  DIR {auroc(up, f(Pup_de)):.4f}   DE {auroc(de, f(P_DE)):.4f}")

    print("\n-- (2) tie-break oracle ceiling (the only non-monotonic lever) --")
    b, w = ceiling(up, Pup_de)
    print(f"   DIR: current {auroc(up, Pup_de):.4f}  ->  oracle {b:.4f}  (floor {w:.4f})   headroom +{b-auroc(up,Pup_de):.4f}")
    b2, w2 = ceiling(de, P_DE)
    print(f"   DE : current {auroc(de, P_DE):.4f}  ->  oracle {b2:.4f}  (floor {w2:.4f})   headroom +{b2-auroc(de,P_DE):.4f}")
    cur_mean = (auroc(de, P_DE) + auroc(up, Pup_de)) / 2
    print(f"   mean: current {cur_mean:.4f}  ->  oracle {(b+b2)/2:.4f}  (needs a PERFECT, label-knowing tie-breaker)")
    print(f"   DIR unique P_up|DE values: {len(np.unique(Pup_de))} over {int(mask.sum())} DE rows "
          f"(integer-% quantization is the tie source)")


if __name__ == "__main__":
    main()
