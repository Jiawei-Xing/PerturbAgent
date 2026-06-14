#!/usr/bin/env python
"""
Build a seed-ensemble Track B submission.

Averages prediction_up / prediction_down across N per-seed runs of the
adversarial agent, using a validated single-seed submission.csv as the metadata
base (so the required Kaggle columns stay exactly as they were validated -- only
the two scored probability columns are replaced by the seed average).

Why average the predictions, not the labels: the metric is rank-based (AUROC),
and at temperature 1.0 the per-row probabilities are noisy (p_up|DE run-to-run
correlation ~0.40). Averaging independent seeds reduces that variance, which
tightens the ranking and was shown offline to lift DE robustly and stabilize the
high-variance DIR component.

Stdlib only (csv + json) so it runs anywhere without pandas.

    python examples/build_ensemble_submission.py \
        --base-submission outputs/track_b_adversarial_sharp/submission.csv \
        --seed-cache outputs/track_b_adversarial_sharp/responses_cache.json \
        --seed-cache outputs/track_b_adversarial_sharp_seed43/responses_cache.json \
        --seed-cache outputs/track_b_adversarial_sharp_seed44/responses_cache.json \
        --out-dir outputs/track_b_adversarial_ens3
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_rows(cache_path: Path) -> dict:
    d = json.loads(cache_path.read_text())
    return d.get("rows", d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-submission", type=Path, required=True,
                    help="validated single-seed submission.csv (metadata base)")
    ap.add_argument("--seed-cache", type=Path, action="append", required=True,
                    help="responses_cache.json per seed; repeat for each seed")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tools-dir", type=Path, default=ROOT / "examples" / "tools")
    args = ap.parse_args()

    caches = [load_rows(p) for p in args.seed_cache]
    print(f"[ens] {len(caches)} seed caches: "
          + ", ".join(f"{p.parent.name}({len(c)})"
                      for p, c in zip(args.seed_cache, caches)))

    with args.base_submission.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        base_rows = list(reader)
    if fieldnames is None:
        sys.exit("base submission has no header")

    n_avg, n_partial, missing_ids = 0, 0, []
    for row in base_rows:
        rid = row["id"]
        ups, downs, toks = [], [], []
        for c in caches:
            r = c.get(rid)
            if r and "prediction_up" in r:
                ups.append(float(r["prediction_up"]))
                downs.append(float(r["prediction_down"]))
                toks.append(int(r.get("tokens_used", 0) or 0))
        if not ups:
            missing_ids.append(rid)            # keep base row untouched
            continue
        row["prediction_up"] = round(sum(ups) / len(ups), 6)
        row["prediction_down"] = round(sum(downs) / len(downs), 6)
        if "tokens_used" in row:
            row["tokens_used"] = int(sum(toks))   # honest: total across seeds
        n_avg += 1
        if len(ups) < len(caches):
            n_partial += 1

    print(f"[ens] averaged {n_avg} rows "
          f"({n_partial} used fewer than {len(caches)} seeds); "
          f"{len(missing_ids)} rows fell back to base (no seed had them)")
    if missing_ids:
        print("      first missing:", missing_ids[:10])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = args.out_dir / "submission.csv"
    with sub_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(base_rows)

    # validate: no nulls, ids unique
    nulls = sum(1 for r in base_rows for v in r.values() if v is None or v == "")
    ids = [r["id"] for r in base_rows]
    assert nulls == 0, f"{nulls} null cells -- would score 0.0"
    assert len(ids) == len(set(ids)), "duplicate ids"
    print(f"[ens] wrote {sub_path} ({len(base_rows)} rows, 0 nulls, ids unique)")

    # prompt.txt + tools/ + zip (mirror the driver's bundle)
    prompt_src = args.base_submission.parent / "prompt.txt"
    prompt_dst = args.out_dir / "prompt.txt"
    if prompt_src.exists():
        shutil.copy(prompt_src, prompt_dst)
    out_tools = args.out_dir / "tools"
    if out_tools.exists():
        shutil.rmtree(out_tools)
    shutil.copytree(args.tools_dir, out_tools)

    zip_path = args.out_dir / "submission_track_b.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        if prompt_dst.exists():
            zf.write(prompt_dst, "prompt.txt")
        for tf in out_tools.rglob("*.py"):
            zf.write(tf, f"tools/{tf.name}")
    print(f"[ens] wrote {zip_path}  <-- upload to Kaggle")


if __name__ == "__main__":
    main()
