#!/usr/bin/env python
"""
Paired benchmark: zero-shot DE judge vs FEW-SHOT DE judge.

The LLM is the dominant DE signal (DE ~0.55-0.58 > kNN 0.549) and DE is the
bottleneck, so a prompt that lifts LLM DE is the most leveraged prompt target.
Few-shot = K labeled (perturbation, target) reference examples from THIS screen,
injected into the DE judge. Drawn from the BLINDED train (every row sharing a
sampled pert OR gene removed) so they're disjoint-safe -- no gene-identity leak,
only task-format + base-rate + pattern priors.

Design (paired, noise-canceling) -- mirrors benchmark_track_b_dir.py:
Two arms over the SAME blinded sample and SAME cached dossiers, differing in
exactly one thing -- the DE judge's few-shot block:
  * de_base  -- fewshot_k=0  (current production, zero-shot)
  * de_fs    -- fewshot_k=K  (K disjoint labeled examples)
Sharp DE judge ON for both (production parity). DIR judge identical, so DIR
should be ~flat; the only thing that can move is DE. Because the arms share
dossiers and seed, the DE delta is paired and temperature noise largely cancels.
The P_DE separation diagnostic (true-DE vs true-none) is the real test: a genuine
gain SEPARATES the two, a noise draw shifts both together.

Requires a running GPT-OSS server (numeric judges).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from benchmark_track_b import (  # noqa: E402
    auroc, build_sample, score, write_blinded_train,
)


def boot_delta(y, s_new, s_old, n=4000, seed=0):
    y = np.asarray(y).astype(int); rng = np.random.default_rng(seed); N = len(y); d = []
    s_new = np.asarray(s_new, float); s_old = np.asarray(s_old, float)
    for _ in range(n):
        idx = rng.integers(0, N, N); yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        d.append(auroc(yy, s_new[idx]) - auroc(yy, s_old[idx]))
    d = np.array(d)
    return float(np.median(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="token-abc123")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--train-csv", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "benchmark_b_fewshot")
    ap.add_argument("--sample-size", type=int, default=250)
    ap.add_argument("--fewshot-k", type=int, default=12)
    ap.add_argument("--advocate-max-tokens", type=int, default=8000)
    ap.add_argument("--judge-max-tokens", type=int, default=8000)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("MLGENX_SHARP_JUDGE", "1")   # production parity, both arms

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

    # ── Stage 2: two arms (de_base, de_fs) over shared dossiers ───────
    arms = [("de_base", 0), (f"de_fs{args.fewshot_k}", args.fewshot_k)]
    preds_path = args.out_dir / "preds.json"
    preds = json.loads(preds_path.read_text()) if preds_path.exists() else {}

    def run_one(arm_name, fsk, r):
        key = f"{arm_name}|{r['id']}"
        if key in preds:
            return key, preds[key]
        res = predict_row(
            r["pert"], r["gene"],
            api_base=args.api_base, api_key=args.api_key, model=args.model,
            advocate_effort="medium", judge_effort="medium", judge_mode="numeric",
            rounds=1, advocate_max_tokens=args.advocate_max_tokens,
            judge_max_tokens=args.judge_max_tokens, timeout_s=args.timeout_s,
            dossier=tuple(dossiers[r["id"]]), seed=args.seed, fewshot_k=fsk,
        )
        tr = json.loads(res["reasoning_trace"])
        return key, [res["prediction_up"], res["prediction_down"],
                     tr.get("P_DE"), tr.get("P_up_given_DE")]

    work = [(a, k, r) for a, k in arms for r in rows if f"{a}|{r['id']}" not in preds]
    print(f"[run] {len(work)} predictions ({len(arms)} arms x {len(rows)} rows)")
    if work:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {ex.submit(run_one, a, k, r): (a, r) for a, k, r in work}
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

    # ── Stage 3: score + paired DE diagnostic ─────────────────────────
    base_name, fs_name = arms[0][0], arms[1][0]
    print("\n" + "=" * 56)
    print(f"{'arm':<14}{'DE':>10}{'DIR':>10}{'mean':>10}")
    print("-" * 56)
    results = {}
    for arm_name, _ in arms:
        p = {r["id"]: (preds[f"{arm_name}|{r['id']}"][0], preds[f"{arm_name}|{r['id']}"][1]) for r in rows}
        sc = score(sample, p)
        results[arm_name] = sc
        print(f"{arm_name:<14}{sc['DE']:>10.3f}{sc['DIR']:>10.3f}{sc['mean']:>10.3f}")
    print("=" * 56)
    print(f"few-shot delta:  DE {results[fs_name]['DE']-results[base_name]['DE']:+.3f}   "
          f"mean {results[fs_name]['mean']-results[base_name]['mean']:+.3f}")

    # paired P_DE: separation of true-DE vs true-none (noise-canceling)
    lab = {r["id"]: r["label"] for r in rows}
    ids = [r["id"] for r in rows]
    base_pde = np.array([preds[f"{base_name}|{i}"][2] for i in ids], float)
    fs_pde = np.array([preds[f"{fs_name}|{i}"][2] for i in ids], float)
    is_de = np.array([lab[i] != "none" for i in ids])
    corr = float(np.corrcoef(base_pde, fs_pde)[0, 1])
    print(f"\nPaired P(DE) on {len(ids)} rows (corr base~fs {corr:.2f}):")
    print(f"  true-DE   mean P(DE):  base {base_pde[is_de].mean():.3f} -> fs {fs_pde[is_de].mean():.3f}")
    print(f"  true-none mean P(DE):  base {base_pde[~is_de].mean():.3f} -> fs {fs_pde[~is_de].mean():.3f}")
    sep_b = base_pde[is_de].mean() - base_pde[~is_de].mean()
    sep_f = fs_pde[is_de].mean() - fs_pde[~is_de].mean()
    print(f"  separation (DE-none):  base {sep_b:+.3f} -> fs {sep_f:+.3f}  (bigger = better detection)")
    print(f"  spread (std P(DE)):    base {base_pde.std():.3f} -> fs {fs_pde.std():.3f}")
    md, lo, hi, pg = boot_delta(is_de.astype(int), fs_pde, base_pde)
    print(f"  paired DE AUROC delta (fs - base):  {md:+.4f} CI[{lo:+.4f},{hi:+.4f}] P(>0)={pg:.2f}")

    results["_delta"] = {"DE": results[fs_name]["DE"] - results[base_name]["DE"],
                         "sep_base": sep_b, "sep_fs": sep_f, "P_gt0": pg}
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {args.out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
