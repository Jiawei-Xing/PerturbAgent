#!/usr/bin/env python
"""
GRN feature test -- does a directed regulatory network carry indirect-effect signal?

Motivation
----------
The Track B DE-detection bottleneck (~0.51) is dominated by *indirect / secondary*
effects: knocking down a perturbation moves a readout gene through a chain of
regulatory edges that no single paper documents, so literature RAG can't reach it
(see outputs/benchmark_b_rag, a clean negative).  A gene regulatory network is the
conceptually correct tool for that: instead of needing a documented (pert, gene)
relationship, you *propagate* the perturbation through the network and read off the
effect on the gene.

This script tests -- cheaply, with no LLM and no GPU -- whether that propagated
signal actually correlates with the truth on the same blinded 250-row benchmark
sample the agent was scored on (and on full train for higher power).  It mirrors the
`analogue_p_de` disproof methodology: compute a fixed network feature, AUROC it
against the labels, compare to chance (0.50), to `analogue_p_de` (0.488), and to the
LLM's own DE 0.549 / DIR 0.638.

Network
-------
OmniPath (literature-curated *signaling*, directed+signed) + CollecTRI (TF->target,
signed), mouse (organism 10090).  Signaling is included so that *non-TF*
perturbations (kinases, metabolic enzymes, ...) have outgoing edges -- a pure
TF->target network would leave most perturbations as sources with no out-edges.

Features (per (pert, gene) pair)
--------------------------------
From an UNSIGNED directed graph (any directed edge, weight 1) -- candidate DE signal:
  * reachable        : is there a directed path pert -> gene within `cutoff` hops?
  * recip_dist       : 1 / shortest_path_len  (0 if unreachable)
  * connectivity     : alpha-damped signed-agnostic influence |sum_paths alpha^len|
From a SIGNED directed graph (definite +/- edges only) -- candidate DIR signal:
  * kd_up_score      : -sign(signed_influence)  (knockdown inverts the propagated
                       sign; higher => predicted up-regulated)
Control:
  * pert_outdeg      : out-degree of pert (hubness) -- if DE "signal" is really just
                       "pert is a famous hub", this control will carry it instead.

Scoring uses the competition's two micro-AUROCs:
  DE  = (up|down) vs none, scored by the DE feature
  DIR = up vs down among DE-positive rows, scored by kd_up_score
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / "outputs" / "grn_feature_test" / "omnipath_mouse.tsv"


# --------------------------------------------------------------------------- #
# AUROC (tie-aware; identical to benchmark_track_b.auroc)
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    r = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(a), float)
    ranks[order] = r
    return ranks


def auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(y_score, float))
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
# Build graphs
# --------------------------------------------------------------------------- #
def load_graphs(net_path: Path):
    """Return (adj_unsigned, adj_signed, canon) as dict node -> list[(nbr, sign)].

    adj_unsigned uses every directed edge with sign +1 (magnitude only).
    adj_signed uses only edges with a definite stimulation/inhibition sign.
    canon maps lowercase symbol -> canonical symbol seen in the network.
    """
    adj_u: dict[str, list[tuple[str, int]]] = defaultdict(list)
    adj_s: dict[str, list[tuple[str, int]]] = defaultdict(list)
    canon: dict[str, str] = {}
    n_dir = n_signed = 0
    with open(net_path, newline="") as fh:
        rd = csv.DictReader(fh, delimiter="\t")
        for row in rd:
            if row["is_directed"] != "True":
                continue
            s = row["source_genesymbol"].strip()
            t = row["target_genesymbol"].strip()
            if not s or not t or s == t:
                continue
            canon.setdefault(s.lower(), s)
            canon.setdefault(t.lower(), t)
            adj_u[s].append((t, 1))
            n_dir += 1
            stim = row["is_stimulation"] == "True"
            inh = row["is_inhibition"] == "True"
            if stim ^ inh:  # exactly one => definite sign
                adj_s[s].append((t, 1 if stim else -1))
                n_signed += 1
    print(f"[graph] {n_dir} directed edges, {n_signed} with definite sign, "
          f"{len(canon)} nodes")
    return adj_u, adj_s, canon


def _resolve(sym: str, canon: dict[str, str]) -> str | None:
    return canon.get(str(sym).lower())


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def bfs_dist(adj, src, dst, cutoff):
    """Shortest directed-path length src->dst within cutoff (or None)."""
    if src == dst:
        return 0
    frontier = {src}
    seen = {src}
    for d in range(1, cutoff + 1):
        nxt = set()
        for n in frontier:
            for nbr, _ in adj.get(n, ()):  # noqa: B007
                if nbr == dst:
                    return d
                if nbr not in seen:
                    seen.add(nbr)
                    nxt.add(nbr)
        if not nxt:
            break
        frontier = nxt
    return None


def propagate(adj, src, alpha, depth):
    """alpha-damped influence from src: sum over paths of alpha^len * prod(signs).

    Returns dict node -> influence.  For adj_unsigned (all signs +1) the magnitude
    is a connectivity score; for adj_signed it carries the propagated sign.
    """
    infl: dict[str, float] = defaultdict(float)
    vec = {src: 1.0}
    for _ in range(depth):
        nxt: dict[str, float] = defaultdict(float)
        for n, val in vec.items():
            for nbr, sign in adj.get(n, ()):  # noqa: B007
                contrib = val * sign * alpha
                nxt[nbr] += contrib
        for k, v in nxt.items():
            infl[k] += v
        vec = nxt
        if not vec:
            break
    return infl


def compute_features(df, adj_u, adj_s, canon, cutoff, alpha, depth):
    rows = []
    # cache per-source propagation (perts repeat across rows)
    prop_u_cache: dict[str, dict] = {}
    prop_s_cache: dict[str, dict] = {}
    for rec in df.itertuples(index=False):
        pert_c = _resolve(rec.pert, canon)
        gene_c = _resolve(rec.gene, canon)
        f = {
            "id": rec.id, "label": rec.label,
            "pert_in_net": pert_c is not None,
            "gene_in_net": gene_c is not None,
            "reachable": 0, "recip_dist": 0.0, "connectivity": 0.0,
            "kd_up_score": 0.0, "pert_outdeg": 0,
        }
        if pert_c is not None:
            f["pert_outdeg"] = len(adj_u.get(pert_c, ()))
            if pert_c not in prop_u_cache:
                prop_u_cache[pert_c] = propagate(adj_u, pert_c, alpha, depth)
                prop_s_cache[pert_c] = propagate(adj_s, pert_c, alpha, depth)
        if pert_c is not None and gene_c is not None:
            d = bfs_dist(adj_u, pert_c, gene_c, cutoff)
            if d is not None:
                f["reachable"] = 1
                f["recip_dist"] = 1.0 / d if d > 0 else 1.0
            f["connectivity"] = abs(prop_u_cache[pert_c].get(gene_c, 0.0))
            infl_s = prop_s_cache[pert_c].get(gene_c, 0.0)
            f["kd_up_score"] = -infl_s  # knockdown inverts propagated sign
        rows.append(f)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def evaluate(feat: pd.DataFrame, tag: str):
    lab = feat["label"].to_numpy()
    is_de = (lab != "none").astype(int)
    n = len(feat)
    cov_pert = feat["pert_in_net"].mean()
    cov_gene = feat["gene_in_net"].mean()
    cov_both = (feat["pert_in_net"] & feat["gene_in_net"]).mean()
    reach = feat["reachable"].mean()

    print(f"\n{'='*64}\n{tag}  (n={n})")
    print(f"  coverage: pert {cov_pert:.0%}, gene {cov_gene:.0%}, "
          f"both {cov_both:.0%}; reachable(pert->gene) {reach:.0%}")
    print(f"  labels: {dict(pd.Series(lab).value_counts())}")

    print("  -- DE AUROC ((up|down) vs none) --")
    for col in ["reachable", "recip_dist", "connectivity", "pert_outdeg"]:
        print(f"     {col:<14} {auroc(is_de, feat[col]):.3f}")

    # DIR: up vs down among DE rows; restrict to rows with a definite signed path
    de = feat[lab != "none"].copy()
    de_lab = de["label"].to_numpy()
    is_up = (de_lab == "up").astype(int)
    print("  -- DIR AUROC (up vs down, DE rows) --")
    print(f"     kd_up_score (all DE, n={len(de)})        "
          f"{auroc(is_up, de['kd_up_score']):.3f}")
    sub = de[de["kd_up_score"] != 0.0]
    if len(sub) >= 10 and sub["label"].nunique() == 2:
        print(f"     kd_up_score (signed-path only, n={len(sub)})  "
              f"{auroc((sub['label'].to_numpy()=='up').astype(int), sub['kd_up_score']):.3f}")
    else:
        print(f"     kd_up_score (signed-path only): n={len(sub)} too few / one class")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path,
                    default=ROOT / "outputs" / "benchmark_b_250" / "sample.csv")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--net", type=Path, default=NET)
    ap.add_argument("--cutoff", type=int, default=4, help="max BFS path length")
    ap.add_argument("--alpha", type=float, default=0.5, help="propagation damping")
    ap.add_argument("--depth", type=int, default=4, help="propagation depth")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "grn_feature_test")
    args = ap.parse_args()

    adj_u, adj_s, canon = load_graphs(args.net)
    args.out.mkdir(parents=True, exist_ok=True)

    for tag, path in [("BLINDED 250-row sample", args.sample),
                      ("FULL train (high power)", args.train)]:
        df = pd.read_csv(path)
        feat = compute_features(df, adj_u, adj_s, canon, args.cutoff,
                                args.alpha, args.depth)
        feat.to_csv(args.out / f"features_{path.stem}.csv", index=False)
        evaluate(feat, tag)
    print(f"\n[done] features written under {args.out}")


if __name__ == "__main__":
    main()
