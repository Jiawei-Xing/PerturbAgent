#!/usr/bin/env python
"""
Track B -- Adversarial (debate) tool-use agent.

Design rationale
----------------
The competition metric is the mean of two *independent* AUROCs, so we mirror
that structure with two adversarial sub-debates instead of one 3-way vote:

  * DEBATE 1 (-> DE AUROC):  EFFECT advocate vs NULL advocate.   Judge -> P(DE)
  * DEBATE 2 (-> DIR AUROC): UP advocate    vs DOWN advocate.    Judge -> P(up | DE)

  prediction_up   = P(DE) * P(up | DE)
  prediction_down = P(DE) * (1 - P(up | DE))

This arithmetic falls straight out of ``kaggle_metric.py`` (DE scored by
prediction_up+prediction_down; DIR scored by up/(up+down)), so each debate
optimizes exactly one component.

Because the train/test split is disjoint on both axes, direct memorization is
impossible -- the leverage is *external evidence* plus *analogy*.  A moderator
deterministically gathers a shared evidence dossier once per row (keeping us
far under the 250-call budget), each advocate argues its assigned side over
that dossier, and the two judges score which case is better supported.  Judges
emit a continuous, calibrated probability -- either via answer-token logprobs
(the lever that won Track A) or a numeric confidence -- because AUROC rewards
ranking, not hard labels.

Submission columns (Track B):
    id, prediction_up, prediction_down, reasoning_trace, tokens_used,
    num_tool_calls, prompt_tokens, num_distinct_tools, model_name
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

from mlgenx import format_prompt  # noqa: E402
from mlgenx.prompts import CELL_DESC, _PROMPT_ZERO  # noqa: E402
from tools import (  # noqa: E402
    gene_info,
    gene_classify,
    train_base_rates,
    pathway_neighbors,
    pubmed_search,
    rag_search,
)

# When MLGENX_USE_RAG=1, the literature section uses BM25 passage re-ranking
# (rag_search) instead of the keyword pubmed_search.  Off by default so the
# baseline dossier path stays reproducible; toggled per-benchmark for A/B.
import os as _os  # noqa: E402
_USE_RAG = _os.environ.get("MLGENX_USE_RAG", "0") == "1"
# When MLGENX_SHARP_JUDGE=1, the numeric DE judge uses the sharpened prompt
# (base-rate anchor + mechanistic-flag use + anti "famous-regulator" bias +
# spread). Off by default; independent of MLGENX_USE_RAG so they A/B separately.
_SHARP_JUDGE = _os.environ.get("MLGENX_SHARP_JUDGE", "0") == "1"

# When MLGENX_SHARP_DIR=1, the numeric DIR judge uses the sharpened prompt
# (up:down base-rate anchor + repressor/activator KD logic + stress-program
# direction flags + analogue up:down ratio + spread). The DE judge got a real
# +0.065 from sharpening; DIR -- the LLM's strongest component and equally
# weighted in the metric -- had no sharpened prompt until this. Default off;
# overridable per-call via predict_row(dir_sharp=...).
_SHARP_DIR = _os.environ.get("MLGENX_SHARP_DIR", "0") == "1"

# When MLGENX_FEWSHOT_K>0 (or predict_row(fewshot_k=...)), the DE judge is given
# K labeled (perturbation, target) reference examples from THIS screen, drawn
# from MLGENX_TRAIN_CSV. In the benchmark/test that file is the BLINDED train
# (every row sharing a sampled/test pert OR gene removed), so the examples are
# disjoint-safe -- no gene-identity leak, only task-format + base-rate + pattern
# priors. Zero-shot by default. Untested DE lever (the LLM DE is the dominant DE
# signal and DE is the bottleneck).
_FEWSHOT_K = int(_os.environ.get("MLGENX_FEWSHOT_K", "0"))
_FEWSHOT_CACHE: dict = {}


def _build_fewshot_de_block(k: int, seed: int) -> str:
    """K stratified (half unaffected / half DE) labeled examples from the blinded
    train, formatted to calibrate the DE judge. Cached per (k, seed, file)."""
    if k <= 0:
        return ""
    path = _os.environ.get("MLGENX_TRAIN_CSV", str(ROOT / "data" / "train.csv"))
    key = (k, seed, path)
    if key in _FEWSHOT_CACHE:
        return _FEWSHOT_CACHE[key]
    import csv as _csv
    import random as _random
    by: dict[str, list] = {"up": [], "down": [], "none": []}
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            if r.get("label") in by:
                by[r["label"]].append(r)
    rng = _random.Random(seed)
    n_none = k // 2
    none_pick = rng.sample(by["none"], min(n_none, len(by["none"])))
    de_pool = by["up"] + by["down"]
    de_pick = rng.sample(de_pool, min(k - n_none, len(de_pool)))
    picks = none_pick + de_pick
    rng.shuffle(picks)
    lines = []
    for r in picks:
        out = "unaffected" if r["label"] == "none" else "differentially expressed"
        lines.append(f"- {r['pert']} knockdown -> {r['gene']} : {out}")
    block = (
        "Reference outcomes from THIS screen (different perturbation/target pairs, "
        "for calibrating the ~45% base rate and the kinds of pairs that are or are "
        "not affected -- reason by analogy, do not assume your target matches any "
        "listed gene):\n" + "\n".join(lines) + "\n\n"
    )
    _FEWSHOT_CACHE[key] = block
    return block


TEST_CSV = ROOT / "data" / "test.csv"

# Neutral fallbacks (training base rates) for rows where a judge returns no
# usable answer.  Constant across rows, so they contribute ~chance ranking
# rather than a wrong commitment.
FALLBACK_P_DE = 0.447
TERNARY_DE_ANSWER_MAP = {"A": 0.85, "B": 0.85, "C": 0.15}
TERNARY_DIR_ANSWER_MAP = {"A": 0.85, "B": 0.15}
FALLBACK_P_UP_GIVEN_DE = 0.685


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ADVOCATE_SYSTEM = {
    "EFFECT": (
        "You are the EFFECT advocate in a structured scientific debate about a "
        "CRISPRi knockdown experiment in mouse bone marrow-derived macrophages "
        "(BMDMs). Your assigned position: knocking down the perturbation gene "
        "DOES significantly change expression of the target gene (it is "
        "differentially expressed, up OR down). Make the strongest evidence-"
        "based case for an effect: direct/indirect pathway or protein-"
        "interaction links, the target being a stress-responsive gene, or "
        "analogous perturbations in the data causing differential expression. "
        "Be rigorous, cite the dossier, and concede genuinely weak points. "
        "Write 4-8 sentences."
    ),
    "NULL": (
        "You are the NULL advocate in a structured scientific debate about a "
        "CRISPRi knockdown experiment in mouse BMDMs. Your assigned position: "
        "knocking down the perturbation gene does NOT significantly change the "
        "target gene (no significant effect). Make the strongest evidence-based "
        "case for no effect: lack of pathway connection, the target being an "
        "uncharacterized/buffered/housekeeping gene, redundancy, or the high "
        "base rate of 'none'. Be rigorous, cite the dossier, and concede "
        "genuinely weak points. Write 4-8 sentences."
    ),
    "UP": (
        "You are the UP advocate in a structured scientific debate about a "
        "CRISPRi knockdown experiment in mouse BMDMs. Assume the target gene "
        "IS differentially expressed; your assigned position is that it is "
        "UP-regulated after knockdown. Make the strongest case: the "
        "perturbation acting as a repressor of the target (knockdown -> "
        "derepression -> up), or the target being an integrated-stress-response "
        "/ p53 / inflammatory gene induced by the stress of losing an essential "
        "gene. Cite the dossier. Write 3-6 sentences."
    ),
    "DOWN": (
        "You are the DOWN advocate in a structured scientific debate about a "
        "CRISPRi knockdown experiment in mouse BMDMs. Assume the target gene "
        "IS differentially expressed; your assigned position is that it is "
        "DOWN-regulated after knockdown. Make the strongest case: the "
        "perturbation acting as an activator/required factor for the target "
        "(knockdown -> loss -> down), or the target being a ribosomal / "
        "cell-cycle / growth-program gene suppressed under stress. Cite the "
        "dossier. Write 3-6 sentences."
    ),
}

REBUTTAL_SUFFIX = (
    "\n\nYour opponent argued the following. Rebut its weakest points and "
    "reinforce your own position, without repeating yourself:\n\n{opponent}"
)

JUDGE_DE_SYSTEM_LOGPROB = (
    "You are an impartial expert judge weighing whether a CRISPRi knockdown "
    "significantly changes a target gene's expression in mouse BMDMs. You are "
    "given an evidence dossier and two briefs: one arguing there IS an effect "
    "(differentially expressed), one arguing there is NO effect. Weigh the "
    "evidence and the strength of each argument. Then output ONLY your verdict "
    "in this exact format:\n"
    "<answer>A</answer> if differentially expressed (effect), or "
    "<answer>B</answer> if no significant effect.\n"
    "Your confidence is read from the answer-token probability, so commit "
    "honestly in proportion to the evidence."
)

JUDGE_DIR_SYSTEM_LOGPROB = (
    "You are an impartial expert judge deciding the DIRECTION of a CRISPRi "
    "knockdown's effect on a target gene in mouse BMDMs, given that an effect "
    "exists. You are given an evidence dossier and two briefs: one arguing "
    "UP-regulation, one arguing DOWN-regulation. Weigh the evidence and the "
    "strength of each argument. Then output ONLY your verdict in this exact "
    "format:\n"
    "<answer>A</answer> if up-regulated, or <answer>B</answer> if "
    "down-regulated.\n"
    "Your confidence is read from the answer-token probability, so commit "
    "honestly in proportion to the evidence."
)

JUDGE_DE_SYSTEM_NUMERIC = (
    "You are an impartial expert judge weighing whether a CRISPRi knockdown "
    "significantly changes a target gene's expression in mouse BMDMs. You are "
    "given an evidence dossier and two briefs (EFFECT vs NULL). Weigh them and "
    "output ONLY your calibrated probability that the target IS differentially "
    "expressed, as an integer 0-100 in this exact format:\n"
    "<prob>NN</prob>"
)

# Sharpened DE judge (MLGENX_SHARP_JUDGE=1). Targets the two failure modes the
# 250-row benchmark exposed: (1) "famous-regulator" false positives -- the judge
# over-calls DE whenever the perturbation is a well-known broad regulator, even
# for an arbitrary target; (2) false negatives where the dossier ALREADY carries
# a mechanistic flag (ISR/p53/ribosomal from gene_classify, or concordant
# pathway-neighbor labels) that the judge ignored. Also forces probability spread
# to fix the flat ~0.40 P(DE) collapse (AUROC 0.51).
JUDGE_DE_SYSTEM_NUMERIC_SHARP = (
    "You are an impartial expert judge weighing whether a CRISPRi knockdown of "
    "the perturbation significantly changes the TARGET gene's expression in "
    "mouse BMDMs (5% FDR, |log2FC| >= log2(1.5)). You are given an evidence "
    "dossier and two briefs (EFFECT vs NULL).\n\n"
    "Base rate: in this screen ~45% of (perturbation, target) pairs are "
    "differentially expressed and ~55% are unaffected. Start from 45 and move "
    "ONLY for target-specific evidence.\n\n"
    "RAISE your probability when the dossier shows a SPECIFIC link to THIS "
    "target: (a) the perturbation's pathway neighbors, when knocked down, "
    "frequently changed their targets (high DE fraction among analogues); "
    "(b) literature directly co-mentions the perturbation regulating this "
    "target; (c) the target carries a mechanistic stress-program flag "
    "(ISR/ATF4, p53, ribosomal, cell-cycle) AND the perturbation plausibly "
    "triggers that program (e.g. translation/proteostasis/ribosome perturbations "
    "induce the ISR; DNA-damage perturbations trigger p53). Use these flags when "
    "present -- do not ignore them.\n\n"
    "LOWER your probability when: the target is an uncharacterized RIKEN (...Rik), "
    "Gm##### or predicted gene with no known regulation (these are usually "
    "unaffected); OR no pathway-neighbor, literature, or mechanistic link "
    "connects the perturbation to THIS target. Critically: a perturbation being "
    "a famous or pleiotropic regulator is NOT by itself evidence that it hits "
    "this specific gene -- most individual targets of even a broad regulator are "
    "unaffected. Do NOT inflate the probability just because the perturbation is "
    "well-known.\n\n"
    "Calibrate and SPREAD: use the full 5-95 range. Reserve >70 for specific "
    "direct or analogue evidence; use <25 for uncharacterized targets or no link. "
    "Do not cluster every answer near 40-55.\n\n"
    "Output ONLY your calibrated probability that the target IS differentially "
    "expressed, as an integer 0-100 in this exact format:\n"
    "<prob>NN</prob>"
)

JUDGE_DIR_SYSTEM_NUMERIC = (
    "You are an impartial expert judge deciding the DIRECTION of a CRISPRi "
    "knockdown's effect on a target gene in mouse BMDMs, given an effect "
    "exists. You are given an evidence dossier and two briefs (UP vs DOWN). "
    "Weigh them and output ONLY your calibrated probability that the target is "
    "UP-regulated (vs down), as an integer 0-100 in this exact format:\n"
    "<prob>NN</prob>"
)

# Sharpened DIR judge (MLGENX_SHARP_DIR=1 / dir_sharp=True). The DE sharpening
# anchored on the DE base rate and forced mechanistic-flag use; this does the
# same for direction. The directional biology of this screen is comparatively
# tractable (it's why DIR ~0.64 already beats DE ~0.55), so the gains are in
# (a) anchoring on the up-skewed base rate, (b) the repressor/activator KD rule,
# (c) actually using the stress-program direction flags the dossier surfaces,
# and (d) spreading instead of clustering near 50.
JUDGE_DIR_SYSTEM_NUMERIC_SHARP = (
    "You are an impartial expert judge deciding the DIRECTION of a CRISPRi "
    "knockdown's effect on the TARGET gene in mouse BMDMs, GIVEN that the target "
    "is differentially expressed. You are given an evidence dossier and two "
    "briefs (UP vs DOWN).\n\n"
    "Base rate: among differentially expressed pairs in this screen, ~68% go UP "
    "and ~32% go DOWN (knockdowns more often de-repress than reduce a target). "
    "Start from 65 and move for target-specific directional evidence.\n\n"
    "Core rule -- knockdown inverts the regulatory sign: if the perturbation "
    "(or its pathway) normally REPRESSES the target, knocking it down RELIEVES "
    "repression -> target UP; if it normally ACTIVATES/maintains the target, "
    "knockdown -> target DOWN. State which relationship the evidence supports, "
    "then apply this inversion.\n\n"
    "Use the dossier's directional flags -- do NOT ignore them:\n"
    "  - Target flagged ISR/ATF4 or p53 target (expected UP): these stress "
    "programs are INDUCED when an essential perturbation triggers translational/"
    "proteostasis or genotoxic/nucleolar stress -> push UP.\n"
    "  - Target flagged ribosomal protein, ribosome-biogenesis, or cell-cycle/"
    "proliferation (expected DOWN): the growth program is shut DOWN under that "
    "same stress -> push DOWN.\n"
    "  - Pathway-neighbor analogues report an up:down ratio among their "
    "knockdowns -- lean toward the majority direction when it is consistent.\n"
    "  - Direct literature stating the perturbation up- or down-regulates the "
    "target dominates the above.\n\n"
    "Calibrate and SPREAD: use the full 5-95 range. Reserve >80 (or <20) for a "
    "clear repressor/activator relationship, a concordant stress flag, or direct "
    "literature; stay near 55-70 when only the up-skewed base rate applies. Do "
    "NOT cluster every answer near 50.\n\n"
    "Output ONLY your calibrated probability that the target is UP-regulated "
    "(vs down), as an integer 0-100 in this exact format:\n"
    "<prob>NN</prob>"
)

# Static prompt content recorded in prompt.txt for the submission.
PROMPT_TXT = (
    "# Track B -- Adversarial debate agent\n\n"
    "Per (perturbation, gene) row, a moderator gathers a shared evidence "
    "dossier from the tools in tools/ (gene_info, gene_classify, "
    "train_base_rates, pathway_neighbors, pubmed_search). Two adversarial "
    "debates are then run over the dossier:\n"
    "  DEBATE 1 (DE):  EFFECT vs NULL  -> judge -> P(DE)\n"
    "  DEBATE 2 (DIR): UP vs DOWN      -> judge -> P(up|DE)\n"
    "  prediction_up = P(DE)*P(up|DE); prediction_down = P(DE)*(1-P(up|DE)).\n\n"
    "## Advocate system prompts\n\n"
    + "\n\n".join(f"### {k}\n{v}" for k, v in ADVOCATE_SYSTEM.items())
    + "\n\n## Judge system prompts (logprob mode)\n\n### DE\n"
    + JUDGE_DE_SYSTEM_LOGPROB
    + "\n\n### DIR\n" + JUDGE_DIR_SYSTEM_LOGPROB
    + "\n\n## User question template (zero-shot)\n\n"
    + _PROMPT_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def chat(
    api_base: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    effort: str,
    max_tokens: int,
    timeout_s: int,
    seed: int = 42,
    want_logprobs: bool = False,
) -> tuple[str, list, int]:
    """One chat completion. Returns (content, logprobs_content, total_tokens)."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": seed,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": effort,
    }
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 20
    data = json.dumps(payload).encode()

    # Retry transient server/connection failures. vLLM under concurrent load
    # occasionally drops a connection (RemoteDisconnected) or returns a 5xx;
    # without this, one blip kills an entire (resumable, but wasteful) run.
    out = None
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                out = json.loads(resp.read().decode())
            break
        except Exception as e:  # noqa: BLE001 -- transient HTTP/connection errors
            last_err = e
            if attempt < 3:
                time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s backoff
    if out is None:
        raise RuntimeError(f"chat() failed after retries: {last_err}")
    usage = out.get("usage", {}) or {}
    tokens = int(usage.get("total_tokens", 0) or 0)
    choices = out.get("choices", [])
    if not choices:
        return "", [], tokens
    choice = choices[0]
    content = str((choice.get("message", {}) or {}).get("content", "") or "").strip()
    lp = (choice.get("logprobs") or {}).get("content") or []
    return content, lp, tokens


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------

def prob_a_from_logprobs(logprobs_content: list) -> Optional[float]:
    """P(A) from a binary <answer>A|B</answer> via softmax over A/B logprobs.

    Mirrors the validated Track A extractor but for a two-way choice. Returns
    None if the answer token / its A,B logprobs cannot be located.
    """
    if not logprobs_content:
        return None
    tokens = [t.get("token", "") for t in logprobs_content]
    reconstructed = "".join(tokens)
    m = re.search(r"<answer>\s*([ABab])\s*</answer>", reconstructed)
    if not m:
        return None
    char_start = m.start(1)
    pos = 0
    idx = None
    for i, tok in enumerate(tokens):
        end = pos + len(tok)
        if pos <= char_start < end:
            idx = i
            break
        pos = end
    if idx is None:
        return None
    top = logprobs_content[idx].get("top_logprobs") or []
    lp_a = lp_b = None
    for e in top:
        tok = e.get("token", "").strip().upper()
        lp = e.get("logprob")
        if lp is None:
            continue
        if (tok == "A" or tok.endswith(">A")) and lp_a is None:
            lp_a = float(lp)
        elif (tok == "B" or tok.endswith(">B")) and lp_b is None:
            lp_b = float(lp)
    if lp_a is None and lp_b is None:
        return None
    floor = min(x for x in (lp_a, lp_b) if x is not None) - 20.0
    if lp_a is None:
        lp_a = floor
    if lp_b is None:
        lp_b = floor
    # softmax over the two
    m_ = max(lp_a, lp_b)
    ea, eb = math.exp(lp_a - m_), math.exp(lp_b - m_)
    return ea / (ea + eb)


def prob_from_numeric(
    content: str,
    answer_map: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """Parse a numeric probability, with fallbacks for common judge slips."""
    m = re.search(r"<prob\s*>?\s*([0-9]+(?:\.[0-9]+)?)\s*</prob>", content)
    if not m:
        m = re.search(r"<prob\s*>?\s*([0-9]+(?:\.[0-9]+)?)\b", content)
    if not m:
        m = re.search(r"\b([0-9]{1,3})\s*%", content)
    if not m:
        m = re.fullmatch(r"\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*", content)
    if m:
        v = float(m.group(1))
        return max(0.0, min(1.0, v / 100.0))
    if answer_map:
        ans = re.search(
            r"(?:<answer>\s*)?\b([ABCabc])\b(?:\s*</answer>|\)|\.)?",
            content,
        )
        if ans:
            return answer_map.get(ans.group(1).upper())
    return None


def dossier_features_from_text(pert: str, gene: str, text: str) -> dict[str, Any]:
    """Extract compact, model- and post-processing-friendly dossier signals."""
    sections: dict[str, str] = {}
    for m in re.finditer(r"== (?P<title>.*?) ==\n(?P<body>.*?)(?=\n\n== |\Z)", text, re.S):
        sections[m.group("title")] = m.group("body")

    def section(title_part: str) -> str:
        for title, body in sections.items():
            if title_part in title:
                return body
        return ""

    target_class = section(f"Target gene '{gene}' offline classification")
    pert_class = section(f"Perturbation '{pert}' offline classification")
    analogue = section(f"Pathway-neighbor analogues of '{pert}'")
    literature = section("Literature")

    p_de = None
    m = re.search(r"P\(DE\)=([0-9.]+)%", analogue)
    if m:
        p_de = float(m.group(1)) / 100.0

    up_down_ratio = None
    down_zero = False
    m = re.search(r"up:down = ([0-9.]+|inf)\s*:1", analogue)
    if m:
        if m.group(1) == "inf":
            down_zero = True
        else:
            up_down_ratio = float(m.group(1))

    no_literature = (
        "No PubMed results" in literature
        or "No literature found" in literature
        or "none mention the pair or its regulation" in literature
    )
    literature_error = "Error querying" in literature or "Error retrieving" in literature
    tool_error = "(tool error:" in text or "Error querying" in text or "Error retrieving" in text

    return {
        "target_uncharacterized": "UNCHARACTERIZED" in target_class,
        "target_stress_up": "expected UP" in target_class,
        "target_stress_down": "expected DOWN" in target_class,
        "pert_uncharacterized": "UNCHARACTERIZED" in pert_class,
        "pert_stress_up": "expected UP" in pert_class,
        "pert_stress_down": "expected DOWN" in pert_class,
        "analogue_found": "neighbor(s) are perturbations in train" in analogue,
        "analogue_p_de": p_de,
        "analogue_up_down_ratio": up_down_ratio,
        "analogue_down_zero": down_zero,
        "no_analogue": "None of the STRING neighbors appear as perturbations in train" in analogue,
        "no_literature": no_literature,
        "literature_error": literature_error,
        "tool_error": tool_error,
        "literature_mode": "rag" if _USE_RAG else "pubmed",
    }


# ---------------------------------------------------------------------------
# Evidence dossier (deterministic tool calls)
# ---------------------------------------------------------------------------

DOSSIER_TOOLS = [
    "gene_info", "gene_classify", "train_base_rates",
    "pathway_neighbors", "rag_search" if _USE_RAG else "pubmed_search",
]


def gather_dossier(pert: str, gene: str) -> tuple[str, int]:
    """Run the tools once and assemble a shared evidence dossier.

    Returns (dossier_text, num_tool_calls).
    """
    sections: list[tuple[str, str]] = []
    calls = 0

    def add(title: str, fn, *a):
        nonlocal calls
        calls += 1
        try:
            sections.append((title, fn(*a)))
        except Exception as e:  # noqa: BLE001
            sections.append((title, f"(tool error: {e})"))

    add(f"Perturbation '{pert}' annotation (mygene.info)", gene_info, pert)
    add(f"Target gene '{gene}' annotation (mygene.info)", gene_info, gene)
    add(f"Target gene '{gene}' offline classification", gene_classify, gene)
    add(f"Perturbation '{pert}' offline classification", gene_classify, pert)
    add("Training base rates", train_base_rates)
    add(f"Pathway-neighbor analogues of '{pert}'", pathway_neighbors, pert)
    if _USE_RAG:
        add("Literature (RAG passage re-ranking)", rag_search, pert, gene)
    else:
        add(
            "Literature (PubMed)", pubmed_search,
            f"{pert} {gene} mouse macrophage regulation expression",
        )

    text = "\n\n".join(f"== {t} ==\n{body}" for t, body in sections)
    return text, calls


# ---------------------------------------------------------------------------
# One row through the adversarial pipeline
# ---------------------------------------------------------------------------

def predict_row(
    pert: str,
    gene: str,
    *,
    api_base: str,
    api_key: str,
    model: str,
    advocate_effort: str,
    judge_effort: str,
    judge_mode: str,
    rounds: int,
    advocate_max_tokens: int,
    judge_max_tokens: int,
    timeout_s: int,
    dossier: Optional[tuple[Any, ...]] = None,
    seed: int = 42,
    dir_sharp: Optional[bool] = None,
    fewshot_k: Optional[int] = None,
) -> dict:
    """Run the full debate for one (pert, gene). Returns a result dict.

    ``rounds=0`` skips the advocates entirely (judges score the dossier
    directly) -- the no-debate baseline. ``rounds>=1`` runs the advocates,
    with ``rounds-1`` rebuttal rounds. Pass ``dossier`` (text, n_calls[, features])
    to reuse already-gathered evidence instead of re-querying the tools.
    """
    question = format_prompt(pert, gene)
    if dossier is None:
        dossier_text, n_tool_calls = gather_dossier(pert, gene)
        dossier_features = dossier_features_from_text(pert, gene, dossier_text)
    else:
        dossier_text = str(dossier[0])
        n_tool_calls = int(dossier[1])
        if len(dossier) > 2 and isinstance(dossier[2], dict):
            dossier_features = dossier[2]
        else:
            dossier_features = dossier_features_from_text(pert, gene, dossier_text)
    tokens = 0
    trace: dict[str, Any] = {
        "tool_calls": n_tool_calls,
        "dossier_features": dossier_features,
    }

    base_user = f"{question}\n\n=== EVIDENCE DOSSIER ===\n{dossier_text}"

    def advocate(role: str, opponent: Optional[str]) -> str:
        nonlocal tokens
        sys_p = ADVOCATE_SYSTEM[role]
        user = base_user
        if opponent:
            user = base_user + REBUTTAL_SUFFIX.format(opponent=opponent)
        content, _, tk = chat(
            api_base, api_key, model, sys_p, user, advocate_effort,
            advocate_max_tokens, timeout_s, seed=seed,
        )
        tokens += tk
        return content

    no_brief = "(no advocate brief; judge directly on the evidence dossier)"
    if rounds < 1:
        # No-debate baseline: judges score the dossier directly.
        effect = null = up = down = no_brief
    else:
        # Debate 1 (DE) and Debate 2 (DIR), round 1.
        effect = advocate("EFFECT", None)
        null = advocate("NULL", None)
        up = advocate("UP", None)
        down = advocate("DOWN", None)

        # Optional rebuttal round(s).
        for _ in range(max(0, rounds - 1)):
            effect2 = advocate("EFFECT", null)
            null2 = advocate("NULL", effect)
            up2 = advocate("UP", down)
            down2 = advocate("DOWN", up)
            effect, null, up, down = effect2, null2, up2, down2

    trace["briefs"] = {"EFFECT": effect, "NULL": null, "UP": up, "DOWN": down}

    # --- Judge 1: P(DE) ---
    if judge_mode == "logprob":
        de_sys = JUDGE_DE_SYSTEM_LOGPROB
    elif _SHARP_JUDGE:
        de_sys = JUDGE_DE_SYSTEM_NUMERIC_SHARP
    else:
        de_sys = JUDGE_DE_SYSTEM_NUMERIC
    fs_k = _FEWSHOT_K if fewshot_k is None else fewshot_k
    de_user = (
        _build_fewshot_de_block(fs_k, seed)
        + f"{base_user}\n\n=== EFFECT brief ===\n{effect}\n\n"
        f"=== NULL brief ===\n{null}"
    )
    de_content, de_lp, tk = chat(
        api_base, api_key, model, de_sys, de_user, judge_effort,
        judge_max_tokens, timeout_s, seed=seed,
        want_logprobs=(judge_mode == "logprob"),
    )
    tokens += tk
    if judge_mode == "logprob":
        p_de = prob_a_from_logprobs(de_lp)
    else:
        p_de = prob_from_numeric(de_content, TERNARY_DE_ANSWER_MAP)
    if p_de is None:
        p_de = FALLBACK_P_DE
        trace["de_fallback"] = True

    # --- Judge 2: P(up | DE) ---
    use_sharp_dir = _SHARP_DIR if dir_sharp is None else dir_sharp
    if judge_mode == "logprob":
        dir_sys = JUDGE_DIR_SYSTEM_LOGPROB
    elif use_sharp_dir:
        dir_sys = JUDGE_DIR_SYSTEM_NUMERIC_SHARP
    else:
        dir_sys = JUDGE_DIR_SYSTEM_NUMERIC
    dir_user = (
        f"{base_user}\n\n=== UP brief ===\n{up}\n\n"
        f"=== DOWN brief ===\n{down}"
    )
    dir_content, dir_lp, tk = chat(
        api_base, api_key, model, dir_sys, dir_user, judge_effort,
        judge_max_tokens, timeout_s, seed=seed,
        want_logprobs=(judge_mode == "logprob"),
    )
    tokens += tk
    if judge_mode == "logprob":
        p_up_de = prob_a_from_logprobs(dir_lp)
    else:
        p_up_de = prob_from_numeric(dir_content, TERNARY_DIR_ANSWER_MAP)
    if p_up_de is None:
        p_up_de = FALLBACK_P_UP_GIVEN_DE
        trace["dir_fallback"] = True

    trace["de_verdict"] = de_content
    trace["dir_verdict"] = dir_content
    trace["P_DE"] = round(p_de, 4)
    trace["P_up_given_DE"] = round(p_up_de, 4)

    pred_up = round(p_de * p_up_de, 6)
    pred_down = round(p_de * (1.0 - p_up_de), 6)

    return {
        "prediction_up": pred_up,
        "prediction_down": pred_down,
        "reasoning_trace": json.dumps(trace, default=str),
        "tokens_used": tokens,
        "num_tool_calls": n_tool_calls,
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

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
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Track B: adversarial debate agent")
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="token-abc123")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--judge-mode", choices=["logprob", "numeric"], default="logprob")
    ap.add_argument("--advocate-effort", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--judge-effort", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--rounds", type=int, default=1,
                    help="0=no debate (judge on dossier), 1=single briefs, 2=one rebuttal round")
    ap.add_argument("--advocate-max-tokens", type=int, default=8000)
    ap.add_argument("--judge-max-tokens", type=int, default=8000)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--test-csv", type=Path, default=TEST_CSV)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "track_b_adversarial")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--clear-cache", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows (0=all)")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed passed to predict_row (for seed ensembles)")
    args = ap.parse_args()

    model_name = args.model_name or args.model
    num_distinct_tools = len(DOSSIER_TOOLS)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = len(enc.encode(PROMPT_TXT))
    except Exception:
        prompt_tokens = len(PROMPT_TXT) // 4
    if prompt_tokens > 16384:
        ap.error(f"prompt.txt is ~{prompt_tokens} tokens, exceeds 16,384 limit.")
    print(f"prompt_tokens~{prompt_tokens}, tools={num_distinct_tools}, "
          f"judge_mode={args.judge_mode}, rounds={args.rounds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "responses_cache.json"
    dossier_cache_path = args.output_dir / "dossiers.json"
    if args.clear_cache:
        for p in (cache_path, dossier_cache_path):
            if p.exists():
                p.unlink()
    cache = load_cache(cache_path)
    cache.setdefault("rows", {})
    dossier_cache = load_cache(dossier_cache_path)
    test_df = pd.read_csv(args.test_csv)
    if args.limit:
        test_df = test_df.head(args.limit)
    total = len(test_df)
    lock = threading.Lock()
    new_count = 0

    def process(idx: int, row: pd.Series) -> None:
        nonlocal new_count
        rid = row["id"]
        with lock:
            if rid in cache["rows"] and "prediction_up" in cache["rows"][rid]:
                print(f"[{idx+1}/{total}] {rid} cache_hit")
                return
        try:
            with lock:
                dossier_val = dossier_cache.get(rid)
            if dossier_val is None:
                dossier_text, n_tool_calls = gather_dossier(row["pert"], row["gene"])
                dossier_features = dossier_features_from_text(
                    row["pert"], row["gene"], dossier_text
                )
                dossier_val = [dossier_text, n_tool_calls, dossier_features]
                with lock:
                    dossier_cache[rid] = dossier_val
                    save_cache(dossier_cache_path, dossier_cache)
            res = predict_row(
                row["pert"], row["gene"],
                api_base=args.api_base, api_key=args.api_key, model=args.model,
                advocate_effort=args.advocate_effort, judge_effort=args.judge_effort,
                judge_mode=args.judge_mode, rounds=args.rounds,
                advocate_max_tokens=args.advocate_max_tokens,
                judge_max_tokens=args.judge_max_tokens, timeout_s=args.timeout_s,
                dossier=tuple(dossier_val), seed=args.seed,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {rid}: {e}")
            res = {
                "prediction_up": round(FALLBACK_P_DE * FALLBACK_P_UP_GIVEN_DE, 6),
                "prediction_down": round(FALLBACK_P_DE * (1 - FALLBACK_P_UP_GIVEN_DE), 6),
                "reasoning_trace": json.dumps({"error": str(e)}),
                "tokens_used": 0, "num_tool_calls": 0,
            }
        res["model_name"] = model_name
        with lock:
            cache["rows"][rid] = res
            new_count += 1
            print(f"[{idx+1}/{total}] {rid} up={res['prediction_up']:.3f} "
                  f"down={res['prediction_down']:.3f} tok={res['tokens_used']}")
            if new_count % args.save_every == 0:
                save_cache(cache_path, cache)
                save_cache(dossier_cache_path, dossier_cache)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = [pool.submit(process, i, r) for i, (_, r) in enumerate(test_df.iterrows())]
        for f in as_completed(futs):
            f.result()
    save_cache(cache_path, cache)
    save_cache(dossier_cache_path, dossier_cache)
    print(f"Done. {new_count} new rows.")

    # ── Build submission ──────────────────────────────────────────────
    rows_out = []
    for _, row in test_df.iterrows():
        rid = row["id"]
        c = cache["rows"].get(rid, {})
        rows_out.append({
            "id": rid,
            "prediction_up": c.get("prediction_up", round(FALLBACK_P_DE * FALLBACK_P_UP_GIVEN_DE, 6)),
            "prediction_down": c.get("prediction_down", round(FALLBACK_P_DE * (1 - FALLBACK_P_UP_GIVEN_DE), 6)),
            "reasoning_trace": c.get("reasoning_trace", "none"),
            "tokens_used": int(c.get("tokens_used", 0)),
            "num_tool_calls": int(c.get("num_tool_calls", 0)),
            "prompt_tokens": prompt_tokens,
            "num_distinct_tools": num_distinct_tools,
            "model_name": c.get("model_name", model_name),
        })
    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(PROMPT_TXT)

    out_tools = args.output_dir / "tools"
    if out_tools.exists():
        shutil.rmtree(out_tools)
    shutil.copytree(ROOT / "examples" / "tools", out_tools)

    zip_path = args.output_dir / "submission_track_b.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")
        for tf in out_tools.rglob("*.py"):
            zf.write(tf, f"tools/{tf.name}")
    print(f"Wrote {sub_path}\nWrote {zip_path}  <-- upload to Kaggle")


if __name__ == "__main__":
    main()
