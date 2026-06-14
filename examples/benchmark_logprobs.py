#!/usr/bin/env python
"""Offline benchmark: hard-parse vs logprob extraction for Track A.

Track A is scored by AUROC (pure ranking), so continuous, well-calibrated
scores can beat hard A/B/C labels -- but only if the answer-token logprobs
are not degenerate.  With high reasoning effort the model "decides" during
its hidden chain-of-thought, so the final answer-token distribution
collapses to ~0/1 and carries no graded signal; with low effort it may stay
graded.  This script measures that empirically on *labeled* train rows
(which the model cannot have memorized: splits are gene- and pert-disjoint).

For each condition (prompt x reasoning effort) it makes one logprobs-enabled
call per (row, seed) and extracts BOTH:
  * parse : hard A/B/C label -> corner (up/down/none), averaged over seeds
  * logprob: softmax over A/B/C answer-token logprobs -> continuous P(up),P(down)

Then it scores every (condition, method) with the real competition metric
(DE AUROC, DIR AUROC, and their mean) so we can pick the best Track-A-legal
extraction before spending a full test run.

Everything here stays within Track A rules: the fixed model, a single prompt
per condition, the standard 3 seeds, no tools.  Requires a server started via
serve_with_logprobs_fix.py so -inf logprobs serialize.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from mlgenx import format_prompt, parse_answer  # noqa: E402
from mlgenx.prompts import CELL_DESC  # noqa: E402
from track_a_logprobs import (  # noqa: E402
    append_answer_tag,
    extract_answer_tag,
    prediction_from_logprobs,
)

SEEDS = [42, 43, 44]
UNIFORM = (1.0 / 3.0, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(template_text: Optional[str], pert: str, gene: str) -> str:
    """Return the user prompt (with the answer-tag instruction appended)."""
    if template_text is None:
        base = format_prompt(pert, gene)
    else:
        base = template_text.format(pert=pert, gene=gene, cell_desc=CELL_DESC)
    return append_answer_tag(base)


# ---------------------------------------------------------------------------
# API call (one logprobs-enabled completion)
# ---------------------------------------------------------------------------

def call(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    seed: int,
    effort: str,
    max_tokens: int,
    timeout_s: int,
) -> Tuple[str, List[dict], Dict[str, float]]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": seed,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": effort,
        "logprobs": True,
        "top_logprobs": 20,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        out = json.loads(resp.read().decode())

    usage = out.get("usage", {}) or {}
    stats = {
        "ptoks": float(usage.get("prompt_tokens", 0)),
        "ctoks": float(usage.get("completion_tokens", 0)),
    }
    choices = out.get("choices", [])
    if not choices:
        return "", [], stats
    choice = choices[0]
    content = str((choice.get("message", {}) or {}).get("content", "") or "").strip()
    lp = (choice.get("logprobs") or {}).get("content") or []
    return content, lp, stats


def letter_to_pair(letter: Optional[str]) -> Optional[Tuple[float, float]]:
    if letter == "A":
        return (1.0, 0.0)
    if letter == "B":
        return (0.0, 1.0)
    if letter == "C":
        return (0.0, 0.0)
    return None


# ---------------------------------------------------------------------------
# Per-call extraction -> a compact, cacheable record
# ---------------------------------------------------------------------------

def extract(content: str, logprobs_content: List[dict]) -> Dict[str, object]:
    """Return both parse and logprob predictions for one call."""
    # Hard parse: prefer the explicit <answer> tag, else regex over content.
    letter = extract_answer_tag(content)
    parse_pair = letter_to_pair(letter)
    if parse_pair is None:
        parse_pair = parse_answer(content)  # (up, down) floats, uniform if unknown

    lp_pair = prediction_from_logprobs(logprobs_content)

    return {
        "parse_up": parse_pair[0],
        "parse_down": parse_pair[1],
        "lp_up": None if lp_pair is None else lp_pair[0],
        "lp_down": None if lp_pair is None else lp_pair[1],
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()


def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_cache(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def build_sample(train_csv: Path, n: int, sample_path: Path) -> pd.DataFrame:
    if sample_path.exists():
        print(f"[sample] reusing existing {sample_path}")
        return pd.read_csv(sample_path)
    df = pd.read_csv(train_csv)
    # Stratify by label, proportional, deterministic.
    frac = min(1.0, n / len(df))
    parts = [g.sample(frac=frac, random_state=0) for _, g in df.groupby("label")]
    sample = pd.concat(parts).sample(frac=1.0, random_state=0).reset_index(drop=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(sample_path, index=False)
    print(f"[sample] wrote {len(sample)} rows -> {sample_path}")
    print(sample["label"].value_counts())
    return sample


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), correct under ties -- matches scipy 'average'."""
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
    """ROC AUC via the rank (Mann-Whitney U) identity, tie-aware.

    Equivalent to sklearn.metrics.roc_auc_score so the benchmark matches the
    official scorer without depending on sklearn (absent in the serve env).
    """
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def score_method(
    sample: pd.DataFrame,
    preds: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """preds: id -> (prediction_up, prediction_down). Returns DE/DIR/mean.

    Mirrors kaggle_metric.score exactly: DE AUROC = (up+down) vs none scored
    by prediction_up+prediction_down; DIR AUROC = up vs down among DE-positive
    rows scored by prediction_up/(prediction_up+prediction_down).
    """
    ids = sample["id"].tolist()
    labels = np.array(sample["label"].tolist())
    pu = np.array([preds[i][0] for i in ids], float)
    pd_ = np.array([preds[i][1] for i in ids], float)

    de_true = (labels != "none").astype(int)
    de = auroc(de_true, pu + pd_)

    mask = labels != "none"
    dir_true = (labels[mask] == "up").astype(int)
    denom = pu[mask] + pd_[mask]
    denom = np.where(denom == 0, 1.0, denom)
    dir_ = auroc(dir_true, pu[mask] / denom)

    return {"DE": de, "DIR": dir_, "mean": (de + dir_) / 2.0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="token-abc123")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--v2-template", type=Path, default=ROOT / "examples" / "prompt_template_v2.txt")
    ap.add_argument("--v3-template", type=Path, default=ROOT / "examples" / "prompt_template_v3.txt")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "benchmark")
    ap.add_argument("--sample-size", type=int, default=300)
    # high reasoning_effort can emit 10k+ tokens; budget generously so the
    # answer tag is reached (Track A caps the *prompt* at 4096, not output).
    ap.add_argument("--max-tokens", type=int, default=24000)
    ap.add_argument("--timeout-s", type=int, default=1200)
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()

    v2_text = args.v2_template.read_text()
    v3_text = args.v3_template.read_text()

    # Conditions: (name, template_text_or_None, reasoning_effort).
    # v2_medium is the current production winner (cached from the prior run);
    # the new conditions isolate the prompt (v2->v3) and effort (medium->high)
    # levers and their combination. The old per-call max_tokens=8192 cache for
    # v2_medium is reused as-is (medium never truncated).
    conditions = [
        ("v2_medium", v2_text, "medium"),
        ("v2_high", v2_text, "high"),
        ("v3_medium", v3_text, "medium"),
        ("v3_high", v3_text, "high"),
    ]

    sample = build_sample(args.train_csv, args.sample_size, args.out_dir / "sample.csv")
    rows = sample.to_dict("records")

    cache_path = args.out_dir / "cache.json"
    cache = load_cache(cache_path)

    def key(cond: str, rid: str, seed: int) -> str:
        return f"{cond}|{rid}|{seed}"

    # Build the full work list across conditions.
    work = []
    for cond_name, tmpl, effort in conditions:
        for r in rows:
            for s in SEEDS:
                k = key(cond_name, r["id"], s)
                if k in cache:
                    continue
                work.append((cond_name, tmpl, effort, r, s, k))

    print(f"[run] {len(work)} calls to make "
          f"({len(conditions)} conditions x {len(rows)} rows x {len(SEEDS)} seeds, "
          f"{sum(1 for c in conditions for r in rows for s in SEEDS) - len(work)} cached)")

    done = 0
    total = len(work)

    def worker(item):
        cond_name, tmpl, effort, r, s, k = item
        prompt = build_prompt(tmpl, r["pert"], r["gene"])
        content, lp, stats = call(
            args.api_base, args.api_key, args.model, prompt, s, effort,
            args.max_tokens, args.timeout_s,
        )
        rec = extract(content, lp)
        rec.update(stats)
        rec["empty"] = (content == "" and not lp)
        return k, rec

    if work:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(worker, it): it for it in work}
            for fut in as_completed(futs):
                k, rec = fut.result()
                with _cache_lock:
                    cache[k] = rec
                    done += 1
                    if done % 25 == 0 or done == total:
                        save_cache(cache_path, cache)
                        print(f"  [{done}/{total}] cached")
        save_cache(cache_path, cache)

    # ----- aggregate over seeds and score each (condition x method) ---------
    print("\n" + "=" * 72)
    print(f"{'condition':<18}{'method':<10}{'DE':>8}{'DIR':>8}{'mean':>8}"
          f"{'empty':>8}{'lp_ok':>8}")
    print("-" * 72)

    results = {}
    for cond_name, _tmpl, _effort in conditions:
        parse_preds: Dict[str, Tuple[float, float]] = {}
        lp_preds: Dict[str, Tuple[float, float]] = {}
        n_empty = 0
        n_lp_ok = 0
        n_calls = 0
        for r in rows:
            rid = r["id"]
            pu_p = pd_p = 0.0
            pu_l = pd_l = 0.0
            for s in SEEDS:
                rec = cache[key(cond_name, rid, s)]
                n_calls += 1
                if rec.get("empty"):
                    n_empty += 1
                pu_p += rec["parse_up"]
                pd_p += rec["parse_down"]
                if rec["lp_up"] is not None:
                    pu_l += rec["lp_up"]
                    pd_l += rec["lp_down"]
                    n_lp_ok += 1
                else:
                    pu_l += UNIFORM[0]
                    pd_l += UNIFORM[1]
            ns = len(SEEDS)
            parse_preds[rid] = (pu_p / ns, pd_p / ns)
            lp_preds[rid] = (pu_l / ns, pd_l / ns)

        for method, preds in (("parse", parse_preds), ("logprob", lp_preds)):
            sc = score_method(sample, preds)
            results[(cond_name, method)] = sc
            lp_frac = n_lp_ok / max(1, n_calls)
            print(f"{cond_name:<18}{method:<10}"
                  f"{sc['DE']:>8.3f}{sc['DIR']:>8.3f}{sc['mean']:>8.3f}"
                  f"{n_empty:>8}{lp_frac:>8.2f}")

    print("=" * 72)
    best = max(results.items(), key=lambda kv: kv[1]["mean"])
    print(f"BEST: {best[0][0]} / {best[0][1]}  mean={best[1]['mean']:.4f} "
          f"(DE={best[1]['DE']:.3f}, DIR={best[1]['DIR']:.3f})")

    summary = {
        f"{c}|{m}": v for (c, m), v in results.items()
    }
    (args.out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] results -> {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
