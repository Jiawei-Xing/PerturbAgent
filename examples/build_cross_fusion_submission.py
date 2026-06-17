"""Cross-base + kNN fusion submission.

The offline 499-row benchmark (run_logprobs_bench) showed the two LLM bases are
complementary: the debate agent has the better DE signal (0.591) while the
plain prompt-only logprobs base has the better DIR signal (0.583). Taking DE
from the agent and DIR from logprobs, then fusing the gene-similarity kNN on
top of each, reached offline mean 0.616 -- vs 0.605 offline / 0.606 LB for the
agent-only base + kNN.

  fused_P_DE   = (1-w_de) *pct(P_DE_agent)   + w_de *pct(prior_DE_kNN)
  fused_P_up|DE= (1-w_dir)*pct(DIR_logprobs) + w_dir*pct(prior_up_kNN)
  pred_up = fused_P_DE * fused_P_up|DE ;  pred_down = fused_P_DE * (1-fused_P_up|DE)

pct() = rank-percentile (AUROC cares only about rank). Uncovered rows fall back
to the LLM scores alone. Output keeps the agent submission's exact schema +
metadata columns (required; a missing one scores 0.0).

The agent base carries P_DE / P_up_given_DE in its reasoning_trace JSON; the
logprobs base supplies prediction_up/down from which DIR = up/(up+down).

    uv run --extra serve python examples/build_cross_fusion_submission.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
GF = ROOT / "outputs/geneformer_probe/model/geneformer"
SAFET = ROOT / "outputs/geneformer_probe/model/Geneformer-V2-104M/model.safetensors"


def _rankdata(a):
    a = np.asarray(a, float); o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    s = a[o]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j+1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def pct(a):
    a = np.asarray(a, float)
    out = np.full(len(a), 0.5)
    m = ~np.isnan(a)
    if m.sum() > 1:
        out[m] = _rankdata(a[m]) / (m.sum() + 1.0)
    return out


def load_emb():
    tok = pickle.load(open(GF / "token_dictionary_gc104M.pkl", "rb"))
    nm = pickle.load(open(GF / "gene_name_id_dict_gc104M.pkl", "rb"))
    nm_ci = {str(k).upper(): v for k, v in nm.items()}
    with safe_open(str(SAFET), framework="numpy") as f:
        W = f.get_tensor("bert.embeddings.word_embeddings.weight")
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    return tok, nm_ci, W


def gene_emb(genes, tok, nm_ci, W):
    D = W.shape[1]
    out = np.zeros((len(genes), D)); cov = np.zeros(len(genes), bool)
    for i, g in enumerate(genes):
        e = nm_ci.get(str(g).upper())
        if e is not None and e in tok:
            out[i] = W[tok[e]]; cov[i] = True
    return out, cov


def knn_priors(q_pert, q_gene, kb, tok, nm_ci, W, power=2.0, disjoint=False):
    Egq, covq = gene_emb(q_gene, tok, nm_ci, W)
    Egk, _ = gene_emb(kb["gene"].to_numpy(), tok, nm_ci, W)
    isde = (kb["label"].to_numpy() != "none").astype(float)
    isup = (kb["label"].to_numpy() == "up").astype(float)
    kb_p = kb["pert"].to_numpy(); kb_g = kb["gene"].to_numpy()
    sim = np.clip(Egq @ Egk.T, 0, None) ** power
    if disjoint:
        same = (q_pert[:, None] == kb_p[None, :]) | (q_gene[:, None] == kb_g[None, :])
        sim[same] = 0.0
    wsum = sim.sum(1)
    prior_de = np.where(wsum > 0, (sim * isde).sum(1) / np.where(wsum > 0, wsum, 1), np.nan)
    wde = sim * isde; wdesum = wde.sum(1)
    prior_up = np.where(wdesum > 0, (wde * isup).sum(1) / np.where(wdesum > 0, wdesum, 1), np.nan)
    prior_de[~covq] = np.nan; prior_up[~covq] = np.nan
    return prior_de, prior_up


def fuse(p_llm, prior, w):
    out = pct(p_llm).copy()
    m = ~np.isnan(prior)
    out[m] = (1 - w) * pct(p_llm)[m] + w * pct(np.where(m, prior, np.nan))[m]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", type=Path,
                    default=ROOT / "outputs/track_b_adversarial_sharp/submission.csv")
    ap.add_argument("--logprobs", type=Path,
                    default=ROOT / "outputs/track_a_logprobs_fulltest/submission.csv")
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--w-de", type=float, default=0.5)
    ap.add_argument("--w-dir", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/track_b_cross_fusion")
    args = ap.parse_args()

    tok, nm_ci, W = load_emb()

    agent = pd.read_csv(args.agent)
    orig_cols = list(agent.columns)                       # exact submission schema
    test = pd.read_csv(ROOT / "data/test.csv")[["id", "pert", "gene"]]
    agent = agent.merge(test, on="id", how="left")
    assert agent["pert"].notna().all(), "agent id not matched to test.csv"

    # DE base from the agent (P_DE in trace; fallback to up+down)
    tr = [json.loads(t) for t in agent["reasoning_trace"]]
    pde = np.array([d.get("P_DE", np.nan) for d in tr], float)
    pde = np.where(np.isnan(pde), agent["prediction_up"] + agent["prediction_down"], pde)

    # DIR base from the logprobs run (up / (up+down))
    lp = pd.read_csv(args.logprobs).set_index("id")
    miss = [i for i in agent["id"] if i not in lp.index]
    assert not miss, f"{len(miss)} agent ids missing from logprobs (e.g. {miss[:3]})"
    lu = lp.loc[agent["id"], "prediction_up"].to_numpy()
    ld = lp.loc[agent["id"], "prediction_down"].to_numpy()
    pup = lu / (lu + ld + 1e-9)

    kb = pd.read_csv(ROOT / "data/train.csv")
    prior_de, prior_up = knn_priors(agent["pert"].to_numpy(), agent["gene"].to_numpy(),
                                    kb, tok, nm_ci, W, power=args.power, disjoint=False)

    fde = fuse(pde, prior_de, args.w_de)
    fup = fuse(pup, prior_up, args.w_dir)

    out = agent.copy()
    out["prediction_up"] = (fde * fup).round(6)
    out["prediction_down"] = (fde * (1 - fup)).round(6)
    out = out[orig_cols]                                  # drop merged pert/gene
    assert out[["prediction_up", "prediction_down"]].notna().all().all()
    cov = int((~np.isnan(prior_de)).sum())
    print(f"[build] cross base (agent DE + logprobs DIR) + kNN  w_de={args.w_de} "
          f"w_dir={args.w_dir} power={args.power}; gene-covered {cov}/{len(out)}")
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "submission.csv"
    out.to_csv(out_csv, index=False)
    with zipfile.ZipFile(args.out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_csv, "submission.csv")
    print(f"[build] -> {out_csv} (+ submission.zip)")


if __name__ == "__main__":
    main()
