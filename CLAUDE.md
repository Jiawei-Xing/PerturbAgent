# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Scaffolding for the **BioReasoning Challenge** (MLGenX LLM Perturbation Competition,
hosted on Kaggle). The task: given a `(perturbation, gene)` pair, predict the
**ternary** effect of CRISPRi knockdown of the perturbation on the gene in mouse
BMDMs — `up`, `down`, or `none` (5% FDR, |shrunken log2FC| ≥ log2(1.5)).

Submissions provide two probabilities per row, `prediction_up` and `prediction_down`;
`P(none)` is implicit as `1 - prediction_up - prediction_down`. The scoring metric is
the average of two micro AUROCs (see `kaggle_metric.py`):
- **DE AUROC**: (up+down) vs none, scored by `prediction_up + prediction_down`
- **DIR AUROC**: up vs down among DE-positive rows, scored by `prediction_up / (prediction_up + prediction_down)`

This repo is the **participant starter kit**, not the model. The `mlgenx` package is a
thin helper library (prompts + parsing + submission building); the real work happens in
`examples/`, which are meant to be edited/extended by participants.

## Environment / commands

Dependencies are managed with **`uv`** (not conda, despite the cluster default).

```bash
uv sync                       # core: mlgenx helpers (pandas, numpy, openai, dspy)
uv sync --extra train         # Track C fine-tuning: torch, transformers>=5.3, trl, peft
uv sync --extra serve         # vLLM serving (brings transformers<5)
```

**`train` and `serve` extras are mutually exclusive** — declared as conflicting in
`pyproject.toml` because `trl` needs `transformers>=5.3` while vLLM needs
`transformers<5`. Switch between them by re-running `uv sync` with the right extra.
Run example scripts with `uv run --extra <train|serve> python examples/...`.

There is **no test suite, linter, or CI**. The doctests embedded in `mlgenx/` and
`kaggle_metric.py` are the only executable checks:

```bash
uv run python -m doctest mlgenx/prompts.py mlgenx/parsing.py kaggle_metric.py -v
```

## The three tracks

All three share the same data, prompt helpers, and parsing, but differ in model and
constraints. Each example script reads `data/test.csv`, generates per-row predictions,
writes `submission.csv` + metadata, and zips a Kaggle-ready bundle.

| Track | Model | Script(s) | Constraint |
|-------|-------|-----------|------------|
| A | GPT-OSS-120B (fixed) | `track_a_prompt_only.py`, `track_a_logprobs.py` | single prompt, 3 seeds (42/43/44), no tools |
| B | GPT-OSS-120B (fixed) | `track_b_agentic.py`, `track_b_multiagent.py` | tools allowed, ≤250 calls/question |
| C | open model <10B | `finetune.py` → `track_c_finetune.py` | any fine-tuning, no tools at inference |

**Track A** averages predictions over 3 seeds. `track_a_logprobs.py` is a variant that
derives continuous probabilities from the softmax of A/B/C answer-token logprobs (asks
the model to wrap its choice in `<answer>X</answer>`) instead of hard text parsing.

**Track B** uses **DSPy ReAct** for text-based tool calling (works without native
function-calling API support). `track_b_multiagent.py` adds a coordinator that delegates
to `biology_expert` and `data_analyst` sub-agents, capturing nested traces hierarchically.

**Track C** is a two-step workflow that crosses the train/serve extra boundary:
1. `uv sync --extra train` → run `finetune.py` (LoRA via trl SFTTrainer, merges adapter
   into `outputs/finetuned_model/`).
2. **Patch the tokenizer** (one-time, see README): `transformers>=5.3` saves
   `extra_special_tokens` as a list, which `transformers<5` (vLLM) can't read — convert it
   to a dict.
3. `uv sync --extra serve` → serve with vLLM, then run `track_c_finetune.py`.

## Serving GPT-OSS-120B (Tracks A & B)

```bash
uv run --extra serve vllm serve openai/gpt-oss-120b \
    --port 8000 --enforce-eager --no-enable-prefix-caching
```

- **`--enforce-eager` is required** — without it GPT-OSS hits a vLLM CUDA-graph bug where
  requests after the first 1–2 return `content: null` / `finish_reason: "length"`.
- GPT-OSS-120B is a **reasoning model**: use `max_completion_tokens` (NOT the legacy
  `max_tokens`) so the budget covers reasoning + visible answer together; otherwise the
  model spends the whole budget reasoning and returns empty `content`. Responses split
  `message.reasoning` from `message.content`; both are `null` when reasoning exhausts the
  budget (the scripts default such rows to the uniform `(1/3, 1/3)` prior).
- For `track_a_logprobs.py`, start the server via **`serve_with_logprobs_fix.py`** instead
  of bare `vllm serve`. It monkeypatches Starlette's `JSONResponse` to clamp `-inf`/`nan`
  logprobs to `-9999.0`, since `json.dumps(allow_nan=False)` otherwise crashes on the
  `-inf` logprobs reasoning models emit. All args forward to `vllm serve`; the `serve`
  subcommand is injected automatically.

## Data (`data/`)

Row IDs are `{pert}_{gene}` (e.g. `Aars_Actb`). `train.csv` has a `label` column
(`up`/`down`/`none`); `test.csv` does not. Splits are disjoint along **both** the
perturbation axis and the gene axis — **no gene appears in more than one split**, so
naive memorization of gene behavior from train does not transfer. `kaggle_data_description.md`
is the authoritative data doc.

## mlgenx package (`mlgenx/`)

Just two modules, re-exported from `__init__.py`:
- `prompts.py` — `format_prompt(pert, gene, examples=None)` (zero-shot or few-shot) and
  `format_prompts_from_csv(...)`. Templates adapted from PerturbQA; `CELL_DESC` and the
  internal `_PROMPT_ZERO` template are imported directly by the example scripts.
- `parsing.py` — `parse_answer(text) -> (prediction_up, prediction_down)` maps a model
  response to a class via ordered regex patterns (extracts the post-"Answer:" portion
  first, then falls back to full text); unparseable → `(1/3, 1/3)`. Hard labels map to
  corners: up→`(1,0)`, down→`(0,1)`, none→`(0,0)`. Also `build_submission(...)`.

When changing prompt wording, keep `parsing.py`'s patterns in sync — the regexes assume
the A)/B)/C) answer format.

## Submission format

Every track needs the exact required metadata columns (listed per-track in README under
"How to Submit"). **Submissions missing metadata columns, or containing any null cell,
score 0.0.** Only `id`, `prediction_up`, `prediction_down` are scored; the rest
(reasoning traces, token counts, model name) are required metadata. Scripts fill empty
responses with `"none"` traces and `0` token counts automatically. `id` must match every
row in `test.csv`.

## Outputs & caching

Example scripts write to `outputs/track_{a,b,c}/<name>/` and maintain a
`responses_cache.json` keyed by row id — re-running resumes instead of re-querying the LLM.
`outputs/finetuned_model/` and `uv.lock` are gitignored; the per-track output dirs with
their cached responses and zips are committed as reference examples.
