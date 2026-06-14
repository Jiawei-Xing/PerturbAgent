#!/usr/bin/env python
"""
Blinded offline benchmark for the Track B adversarial agent.

Scores candidate configurations on *labeled* train rows -- which the model
cannot have memorized (the split is gene- and pert-disjoint) -- using the real
competition metric, so we establish a floor and pick a config before spending a
full test run.

Honest blinding
---------------
To reproduce test-time conditions, the train-reading tools (pathway_neighbors,
train_base_rates) must not see the evaluation rows.  We build a stratified
sample, then write a *blinded* train.csv that drops every row sharing a
perturbation OR a gene with the sample, and point the tools at it via
MLGENX_TRAIN_CSV.  This mimics the disjoint split: for a sampled row, neither
its pert nor its gene exists in the lookup tables.

Conditions compared (each scored on DE / DIR / mean AUROC):
  * direct  -- no debate; judges score the dossier directly (isolates the
               value the adversarial briefs add).
  * debate1 -- advocates argue once, then judges.
  * debate2 -- one rebuttal round before judging.
  * judge calibration -- logprob vs numeric.

The expensive evidence dossier is gathered once per row and reused across all
conditions.  Everything is cached to disk so reruns resume.

Requires a running GPT-OSS server; for logprob judges start it via
serve_with_logprobs_fix.py so -inf logprobs serialize.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))


# ---------------------------------------------------------------------------
# Scoring (mirrors kaggle_metric.score; numpy-only, tie-aware AUROC)
# ---------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    r = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        r[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(n, float)
    ranks[order] = r
    return ranks


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score(sample: pd.DataFrame, preds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    ids = sample["id"].tolist()
    labels = np.array(sample["label"].tolist())
    pu = np.array([preds[i][0] for i in ids], float)
    pd_ = np.array([preds[i][1] for i in ids], float)
    de = auroc((labels != "none").astype(int), pu + pd_)
    mask = labels != "none"
    denom = pu[mask] + pd_[mask]
    denom = np.where(denom == 0, 1.0, denom)
    dir_ = auroc((labels[mask] == "up").astype(int), pu[mask] / denom)
    return {"DE": de, "DIR": dir_, "mean": (de + dir_) / 2.0}


# ---------------------------------------------------------------------------
# Sampling + blinding
# ---------------------------------------------------------------------------

def build_sample(train_csv: Path, n: int, out: Path) -> pd.DataFrame:
    if out.exists():
        print(f"[sample] reusing {out}")
        return pd.read_csv(out)
    df = pd.read_csv(train_csv)
    frac = min(1.0, n / len(df))
    parts = [g.sample(frac=frac, random_state=0) for _, g in df.groupby("label")]
    sample = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"[sample] wrote {len(sample)} rows -> {out}")
    print(sample["label"].value_counts().to_string())
    return sample


def write_blinded_train(train_csv: Path, sample: pd.DataFrame, out: Path) -> Path:
    """Drop every train row sharing a pert OR gene with the sample."""
    df = pd.read_csv(train_csv)
    bad_pert = set(sample["pert"].astype(str).str.lower())
    bad_gene = set(sample["gene"].astype(str).str.lower())
    keep = ~(
        df["pert"].astype(str).str.lower().isin(bad_pert)
        | df["gene"].astype(str).str.lower().isin(bad_gene)
    )
    blinded = df[keep].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    blinded.to_csv(out, index=False)
    print(f"[blind] {len(df)} -> {len(blinded)} rows after removing sampled "
          f"perts/genes -> {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="token-abc123")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "benchmark_b")
    ap.add_argument("--sample-size", type=int, default=80)
    ap.add_argument("--advocate-max-tokens", type=int, default=8000)
    ap.add_argument("--judge-max-tokens", type=int, default=8000)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.train_csv, args.sample_size, args.out_dir / "sample.csv")
    blinded = write_blinded_train(args.train_csv, sample, args.out_dir / "train_blinded.csv")

    # Point the tools at the blinded table BEFORE importing the agent (which
    # imports the tools). _traindata reads the env var per call, so this holds.
    os.environ["MLGENX_TRAIN_CSV"] = str(blinded)

    from track_b_adversarial import predict_row, gather_dossier  # noqa: E402

    rows = sample.to_dict("records")
    lock = threading.Lock()

    # ── Stage 1: gather dossiers once per row (under blinding) ─────────
    dossier_cache_path = args.out_dir / "dossiers.json"
    dossiers: dict = json.loads(dossier_cache_path.read_text()) if dossier_cache_path.exists() else {}
    todo = [r for r in rows if r["id"] not in dossiers]
    print(f"[dossier] {len(todo)} to gather, {len(rows) - len(todo)} cached")

    def gather(r):
        text, n = gather_dossier(r["pert"], r["gene"])
        return r["id"], [text, n]

    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(gather, r): r for r in todo}
            done = 0
            for f in as_completed(futs):
                rid, val = f.result()
                with lock:
                    dossiers[rid] = val
                    done += 1
                    if done % 10 == 0 or done == len(todo):
                        dossier_cache_path.write_text(json.dumps(dossiers))
                        print(f"  [dossier {done}/{len(todo)}]")
        dossier_cache_path.write_text(json.dumps(dossiers))

    # ── Stage 2: run conditions over cached dossiers ──────────────────
    # (name, rounds, judge_mode, advocate_effort, judge_effort)
    conditions = [
        ("direct_lp", 0, "logprob", "medium", "medium"),
        ("debate1_lp", 1, "logprob", "medium", "medium"),
        ("debate2_lp", 2, "logprob", "medium", "medium"),
        ("debate1_num", 1, "numeric", "medium", "medium"),
    ]

    preds_cache_path = args.out_dir / "preds.json"
    preds_cache: dict = json.loads(preds_cache_path.read_text()) if preds_cache_path.exists() else {}

    def run_one(cond, r):
        name, rounds, jmode, aeff, jeff = cond
        key = f"{name}|{r['id']}"
        if key in preds_cache:
            return key, preds_cache[key]
        res = predict_row(
            r["pert"], r["gene"],
            api_base=args.api_base, api_key=args.api_key, model=args.model,
            advocate_effort=aeff, judge_effort=jeff, judge_mode=jmode,
            rounds=rounds, advocate_max_tokens=args.advocate_max_tokens,
            judge_max_tokens=args.judge_max_tokens, timeout_s=args.timeout_s,
            dossier=tuple(dossiers[r["id"]]),
        )
        return key, [res["prediction_up"], res["prediction_down"]]

    work = [(c, r) for c in conditions for r in rows if f"{c[0]}|{r['id']}" not in preds_cache]
    print(f"[run] {len(work)} predictions to make "
          f"({len(conditions)} conditions x {len(rows)} rows)")
    if work:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, c, r): (c, r) for c, r in work}
            done = 0
            for f in as_completed(futs):
                key, val = f.result()
                with lock:
                    preds_cache[key] = val
                    done += 1
                    if done % 10 == 0 or done == len(work):
                        preds_cache_path.write_text(json.dumps(preds_cache))
                        print(f"  [pred {done}/{len(work)}]")
        preds_cache_path.write_text(json.dumps(preds_cache))

    # ── Stage 3: score ────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"{'condition':<16}{'DE':>10}{'DIR':>10}{'mean':>10}")
    print("-" * 56)
    results = {}
    for name, *_ in conditions:
        preds = {r["id"]: tuple(preds_cache[f"{name}|{r['id']}"]) for r in rows}
        sc = score(sample, preds)
        results[name] = sc
        print(f"{name:<16}{sc['DE']:>10.3f}{sc['DIR']:>10.3f}{sc['mean']:>10.3f}")
    print("=" * 56)
    best = max(results.items(), key=lambda kv: kv[1]["mean"])
    print(f"BEST: {best[0]}  mean={best[1]['mean']:.4f} "
          f"(DE={best[1]['DE']:.3f}, DIR={best[1]['DIR']:.3f})")
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] -> {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
