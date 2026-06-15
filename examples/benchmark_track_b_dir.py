#!/usr/bin/env python
"""
Paired benchmark: plain DIR judge vs SHARPENED DIR judge.

The DE judge got a real, isolated +0.065 from sharpening; the DIRECTION judge --
the metric's equally-weighted, and the LLM's *strongest*, component (DIR ~0.64 vs
DE ~0.55) -- had no sharpened prompt.  This benchmark adds one.

Design (paired, noise-canceling)
--------------------------------
Two arms over the SAME blinded sample and the SAME cached dossiers, differing in
exactly one thing -- the DIR judge prompt:
  * dir_base   -- JUDGE_DIR_SYSTEM_NUMERIC      (current production)
  * dir_sharp  -- JUDGE_DIR_SYSTEM_NUMERIC_SHARP (base-rate anchor + repressor/
                  activator inversion + stress-flag use + spread)
Sharp DE is ON for both (MLGENX_SHARP_JUDGE=1), so P(DE) -- and therefore the DE
AUROC -- is identical across arms; the only thing that can move is DIR.  Because
the arms share dossiers, briefs, seed, and DE judge, the DIR delta is paired and
temperature noise largely cancels (same diagnostic used to validate the DE
sharpening).

Run on a 500-row sample (default) so the DIR estimate -- previously a lucky 0.638
draw on 250, true level ~0.58 -- is tight enough to trust.

Requires a running GPT-OSS server (numeric judges; no logprob plumbing needed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from benchmark_track_b import (  # noqa: E402  (reuse blinding + scoring)
    auroc, build_sample, score, write_blinded_train,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="token-abc123")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "outputs" / "benchmark_b_dir")
    ap.add_argument("--sample-size", type=int, default=500)
    ap.add_argument("--advocate-max-tokens", type=int, default=8000)
    ap.add_argument("--judge-max-tokens", type=int, default=8000)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Sharp DE on for both arms (production parity); DIR sharpness is per-arm.
    os.environ.setdefault("MLGENX_SHARP_JUDGE", "1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.train_csv, args.sample_size, args.out_dir / "sample.csv")
    blinded = write_blinded_train(args.train_csv, sample, args.out_dir / "train_blinded.csv")
    os.environ["MLGENX_TRAIN_CSV"] = str(blinded)

    from track_b_adversarial import predict_row, gather_dossier  # noqa: E402

    rows = sample.to_dict("records")
    lock = threading.Lock()

    # ── Stage 1: dossiers once per row ────────────────────────────────
    dossier_path = args.out_dir / "dossiers.json"
    dossiers = json.loads(dossier_path.read_text()) if dossier_path.exists() else {}
    todo = [r for r in rows if r["id"] not in dossiers]
    print(f"[dossier] {len(todo)} to gather, {len(rows) - len(todo)} cached")
    if todo:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(lambda r: (r["id"], list(gather_dossier(r["pert"], r["gene"]))), r): r
                    for r in todo}
            done = 0
            for f in as_completed(futs):
                rid, val = f.result()
                with lock:
                    dossiers[rid] = val
                    done += 1
                    if done % 10 == 0 or done == len(todo):
                        dossier_path.write_text(json.dumps(dossiers))
                        print(f"  [dossier {done}/{len(todo)}]")
        dossier_path.write_text(json.dumps(dossiers))

    # ── Stage 2: two arms (dir_base, dir_sharp) over shared dossiers ───
    arms = [("dir_base", False), ("dir_sharp", True)]
    preds_path = args.out_dir / "preds.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}

    def run_one(arm_name, dir_sharp, r):
        key = f"{arm_name}|{r['id']}"
        if key in preds:
            return key, preds[key]
        res = predict_row(
            r["pert"], r["gene"],
            api_base=args.api_base, api_key=args.api_key, model=args.model,
            advocate_effort="medium", judge_effort="medium", judge_mode="numeric",
            rounds=1, advocate_max_tokens=args.advocate_max_tokens,
            judge_max_tokens=args.judge_max_tokens, timeout_s=args.timeout_s,
            dossier=tuple(dossiers[r["id"]]), seed=args.seed, dir_sharp=dir_sharp,
        )
        tr = json.loads(res["reasoning_trace"])
        return key, [res["prediction_up"], res["prediction_down"],
                     tr.get("P_DE"), tr.get("P_up_given_DE")]

    work = [(a, s, r) for a, s in arms for r in rows if f"{a}|{r['id']}" not in preds]
    print(f"[run] {len(work)} predictions ({len(arms)} arms x {len(rows)} rows)")
    if work:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, a, s, r): (a, r) for a, s, r in work}
            done = 0
            for f in as_completed(futs):
                key, val = f.result()
                with lock:
                    preds[key] = val
                    done += 1
                    if done % 10 == 0 or done == len(work):
                        preds_path.write_text(json.dumps(preds))
                        print(f"  [pred {done}/{len(work)}]")
        preds_path.write_text(json.dumps(preds))

    # ── Stage 3: score + paired DIR diagnostic ────────────────────────
    print("\n" + "=" * 56)
    print(f"{'arm':<14}{'DE':>10}{'DIR':>10}{'mean':>10}")
    print("-" * 56)
    results = {}
    for arm_name, _ in arms:
        p = {r["id"]: (preds[f"{arm_name}|{r['id']}"][0],
                       preds[f"{arm_name}|{r['id']}"][1]) for r in rows}
        sc = score(sample, p)
        results[arm_name] = sc
        print(f"{arm_name:<14}{sc['DE']:>10.3f}{sc['DIR']:>10.3f}{sc['mean']:>10.3f}")
    print("=" * 56)
    d_mean = results["dir_sharp"]["mean"] - results["dir_base"]["mean"]
    d_dir = results["dir_sharp"]["DIR"] - results["dir_base"]["DIR"]
    print(f"DIR sharpening delta:  DIR {d_dir:+.3f}   mean {d_mean:+.3f}")

    # Paired noise-canceling check on P(up|DE) over true-DE rows.
    lab = {r["id"]: r["label"] for r in rows}
    de_ids = [r["id"] for r in rows if lab[r["id"]] != "none"]
    base_pup = np.array([preds[f"dir_base|{i}"][3] for i in de_ids], float)
    sharp_pup = np.array([preds[f"dir_sharp|{i}"][3] for i in de_ids], float)
    is_up = np.array([lab[i] == "up" for i in de_ids])
    corr = float(np.corrcoef(base_pup, sharp_pup)[0, 1])
    print(f"\nPaired P(up|DE) on {len(de_ids)} true-DE rows (corr base~sharp {corr:.2f}):")
    print(f"  true-UP   mean P(up|DE):  base {base_pup[is_up].mean():.3f} -> "
          f"sharp {sharp_pup[is_up].mean():.3f}  ({sharp_pup[is_up].mean()-base_pup[is_up].mean():+.3f})")
    print(f"  true-DOWN mean P(up|DE):  base {base_pup[~is_up].mean():.3f} -> "
          f"sharp {sharp_pup[~is_up].mean():.3f}  ({sharp_pup[~is_up].mean()-base_pup[~is_up].mean():+.3f})")
    sep_base = base_pup[is_up].mean() - base_pup[~is_up].mean()
    sep_sharp = sharp_pup[is_up].mean() - sharp_pup[~is_up].mean()
    print(f"  separation (UP-DOWN):     base {sep_base:+.3f} -> sharp {sep_sharp:+.3f}  "
          f"(bigger = better direction)")
    print(f"  spread (std P(up|DE)):    base {base_pup.std():.3f} -> sharp {sharp_pup.std():.3f}")

    results["_delta"] = {"DIR": d_dir, "mean": d_mean,
                         "sep_base": sep_base, "sep_sharp": sep_sharp}
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
