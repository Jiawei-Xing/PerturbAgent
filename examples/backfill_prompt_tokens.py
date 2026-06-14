"""One-off: backfill `prompt_tokens` into an existing track_a_logprobs cache.

The Kaggle Track A validator requires a `prompt_tokens` metadata column
(see data/sample_submission_track_a.csv) that track_a_logprobs.py did not
originally persist. This recomputes the rendered-prompt token count per row
using the GPT-OSS chat template (reasoning_effort=medium, the run setting),
summed across the 3 seeds to match how `tokens_used` aggregates, and writes
it into responses_cache.json. Re-running track_a_logprobs.py afterwards
rebuilds submission.csv from cache (all rows already cached -> no API calls).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pandas as pd
from transformers import AutoTokenizer

from mlgenx.prompts import CELL_DESC

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs" / "track_a" / "v2_logprobs"
TEMPLATE = ROOT / "examples" / "prompt_template_v2.txt"
TEST_CSV = ROOT / "data" / "test.csv"
SEEDS = [42, 43, 44]
MODEL = "openai/gpt-oss-120b"
REASONING_EFFORT = "medium"


def append_answer_tag(prompt: str) -> str:
    # Must match track_a_logprobs.append_answer_tag exactly.
    return (
        f"{prompt.rstrip()}\n\n"
        "Return ONLY the final choice in this exact format:\n"
        "<answer>A</answer>, <answer>B</answer>, or <answer>C</answer>\n"
        "Do not include any other text."
    )


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    template = TEMPLATE.read_text()
    test_df = pd.read_csv(TEST_CSV)

    cache_path = OUTDIR / "responses_cache.json"
    cache = json.loads(cache_path.read_text())
    rows = cache["rows"]

    n_done = 0
    for _, row in test_df.iterrows():
        rid = row["id"]
        prompt_raw = template.format(pert=row["pert"], gene=row["gene"], cell_desc=CELL_DESC)
        prompt = append_answer_tag(prompt_raw)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            reasoning_effort=REASONING_EFFORT,
            return_dict=True,
        )
        ptok = len(enc["input_ids"])
        c = rows.get(rid, {})
        for s in SEEDS:
            c[f"prompt_tokens_seed{s}"] = float(ptok)
        c["prompt_tokens"] = float(ptok * len(SEEDS))
        rows[rid] = c
        n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(test_df)} (last {rid} prompt_tokens/seed={ptok})", flush=True)

    cache_path.write_text(json.dumps(cache, indent=2))
    print(f"Backfilled prompt_tokens for {n_done} rows -> {cache_path}")


if __name__ == "__main__":
    main()
