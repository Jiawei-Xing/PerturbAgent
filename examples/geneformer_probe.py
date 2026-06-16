"""Geneformer in-silico perturbation feature test (the foundation-model lever).

Tests whether a perturbation foundation model -- Geneformer V2-104M, applied with
a *macrophage context cell* -- produces a (pert, gene) directional signal that
beats chance on the disjoint mouse-BMDM CRISPRi task. This is the one external
lever the RAG / STATE / GRN / category negatives left untested, and it is the
learned, context-conditioned version of exactly what the GRN test lacked
("reachability, not propagated magnitude"). Coverage here is ~83% of train pairs
(both pert and target present in the macrophage cell) vs the GRN test's 16%, so
coverage is not the bottleneck -- discrimination is the open question.

No LLM and no train labels are read (the model is zero-shot). Mirrors
grn_feature_test.py: compute per-pair features, AUROC each against DE and DIR,
bootstrap CIs, on the blinded-250 sample and on full train (max power).

Mechanism, per (pert, gene), on the HPA "macrophages" reference transcriptome
encoded as a Geneformer rank-value sequence [<cls>, g0, g1, ..., <eos>]:

  baseline pass  : the macrophage cell, pert present (computed ONCE, shared).
  knockdown pass : the same cell with the pert token DELETED (CRISPRi LOF).

  de_target_shift : cosine distance between the TARGET gene's contextual
                    embedding baseline vs knockdown -- magnitude of the effect
                    on that specific gene (the DE signal).
  de_cell_shift   : cosine distance of the mean-pooled cell embedding -- global
                    perturbation magnitude (Geneformer's canonical signal; alt DE).
  dir_dlogit      : signed Delta of the masked-LM logit of the target gene token
                    (knockdown - baseline). Deleting a repressor raises the
                    target's predicted rank -> UP; deleting an activator lowers
                    it -> DOWN. (the DIR signal)

Mouse symbols are mapped to human orthologs by uppercasing into Geneformer's
human gene_name_id dictionary (case-folded ortholog proxy; ~97% of perts, ~80%
of targets resolve and sit in the macrophage cell).
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "geneformer_probe" / "model"
GF = MODEL_DIR / "geneformer"
V2 = MODEL_DIR / "Geneformer-V2-104M"
REF = ROOT / "outputs" / "geneformer_probe" / "ref" / "macrophage_ref.tsv"

MAX_LEN = 4096
PAD, MASK, CLS, EOS = 0, 1, 2, 3


# --------------------------------------------------------------------------- #
# Metrics (copied from grn_feature_test.py for a standalone script)
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    a_sorted = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true).astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(np.asarray(y_score, float))
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def boot_ci(y_true, y_score, n=2000, seed=0):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, float)
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        a = auroc(y_true[idx], y_score[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# Assets + macrophage context
# --------------------------------------------------------------------------- #
def load_assets():
    tok = pickle.load(open(GF / "token_dictionary_gc104M.pkl", "rb"))   # ENSG -> id
    med = pickle.load(open(GF / "gene_median_dictionary_gc104M.pkl", "rb"))  # ENSG -> median
    nm = pickle.load(open(GF / "gene_name_id_dict_gc104M.pkl", "rb"))   # symbol -> ENSG
    nm_ci = {str(k).upper(): v for k, v in nm.items()}
    return tok, med, nm_ci


def build_context(tok, med):
    """Rank-value encode the HPA macrophage transcriptome -> token sequence."""
    ref = pd.read_csv(REF, sep="\t", names=["ens", "sym", "ncpm"], header=0).dropna()
    ref = ref[(ref.ncpm > 0) & ref.ens.isin(med) & ref.ens.isin(tok)].copy()
    ref["val"] = ref.ncpm.to_numpy() / np.array([med[e] for e in ref.ens])
    ref = ref.sort_values("val", ascending=False)
    ens_order = ref.ens.tolist()[: MAX_LEN - 2]            # room for <cls>,<eos>
    ids = [CLS] + [tok[e] for e in ens_order] + [EOS]
    pos_of_ens = {e: i + 1 for i, e in enumerate(ens_order)}  # +1 for <cls>
    return ids, pos_of_ens


def sym_to_ens(sym, nm_ci):
    return nm_ci.get(str(sym).upper())


# --------------------------------------------------------------------------- #
# In-silico perturbation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_features(model, df, ctx_ids, pos_of_ens, tok, nm_ci, device, batch=8):
    base_ids = torch.tensor(ctx_ids, device=device)              # (L,)
    L = base_ids.shape[0]
    ens2tok = tok  # ENSG -> token id

    # resolve perts / targets to (ens, base_pos, token_id)
    def resolve(sym):
        e = sym_to_ens(sym, nm_ci)
        if e is None or e not in pos_of_ens:
            return None
        return (e, pos_of_ens[e], ens2tok[e])

    rows = df.to_dict("records")
    resolved = []
    for r in rows:
        p = resolve(r["pert"]); g = resolve(r["gene"])
        resolved.append((p, g))

    # ---- 1. baseline UNMASKED pass: cell embedding + every gene's embedding ----
    out = model.bert(base_ids.unsqueeze(0))
    h_base = out.last_hidden_state[0]                              # (L, H)
    cell_base = h_base[1:-1].mean(0)                              # exclude cls/eos

    # ---- 2. baseline MASKED logit per unique target (cell unchanged by pert) ----
    uniq_t = {g[0]: g for (_, g) in resolved if g is not None}
    logit_base = {}
    t_list = list(uniq_t.values())
    for i in range(0, len(t_list), batch):
        chunk = t_list[i : i + batch]
        inp = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        for j, (_, pos, _) in enumerate(chunk):
            inp[j, pos] = MASK
        logits = model(inp).logits                                # (b, L, V)
        for j, (e, pos, tid) in enumerate(chunk):
            logit_base[e] = float(logits[j, pos, tid])

    # ---- 3. per-pert knockdown passes (unmasked shift + masked target logit) ----
    by_pert = {}
    for k, (p, g) in enumerate(resolved):
        if p is None or g is None:
            continue
        by_pert.setdefault(p[0], {"pos": p[1], "tids": p[2], "targets": []})
        by_pert[p[0]]["targets"].append((k, g))                  # row idx, target

    feats = {k: {} for k in range(len(rows))}
    for ens_p, info in by_pert.items():
        ip = info["pos"]
        keep = torch.cat([base_ids[:ip], base_ids[ip + 1 :]])    # delete pert token
        # unmasked kd pass
        out_kd = model.bert(keep.unsqueeze(0))
        h_kd = out_kd.last_hidden_state[0]
        cell_kd = h_kd[1:-1].mean(0)
        cell_shift = 1.0 - torch.nn.functional.cosine_similarity(
            cell_base, cell_kd, dim=0).item()
        # masked kd logit, batched over this pert's targets
        tgts = info["targets"]
        for i in range(0, len(tgts), batch):
            chunk = tgts[i : i + batch]
            inp = keep.unsqueeze(0).repeat(len(chunk), 1).clone()
            posj = []
            for j, (ridx, g) in enumerate(chunk):
                e_g, pos_g, tid_g = g
                kpos = pos_g - 1 if pos_g > ip else pos_g        # shift after deletion
                inp[j, kpos] = MASK
                posj.append((ridx, e_g, pos_g, kpos, tid_g))
            logits = model(inp).logits
            for j, (ridx, e_g, pos_g, kpos, tid_g) in enumerate(posj):
                lk = float(logits[j, kpos, tid_g])
                tgt_shift = 1.0 - torch.nn.functional.cosine_similarity(
                    h_base[pos_g], h_kd[kpos], dim=0).item()
                feats[ridx] = {
                    "de_target_shift": tgt_shift,
                    "de_cell_shift": cell_shift,
                    "dir_dlogit": lk - logit_base.get(e_g, lk),
                    "covered": 1,
                }

    # fill uncovered rows with neutral zeros
    for k in range(len(rows)):
        if not feats[k]:
            feats[k] = {"de_target_shift": 0.0, "de_cell_shift": 0.0,
                        "dir_dlogit": 0.0, "covered": 0}
    fd = pd.DataFrame([feats[k] for k in range(len(rows))])
    return pd.concat([df.reset_index(drop=True), fd], axis=1)


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #
def evaluate(feat: pd.DataFrame, tag: str):
    lab = feat["label"].to_numpy()
    is_de = (lab != "none").astype(int)
    cov = feat["covered"].mean()
    print(f"\n{'='*64}\n{tag}  (n={len(feat)})")
    print(f"  coverage (pert&target in macrophage cell): {cov:.0%}")
    print(f"  labels: {dict(pd.Series(lab).value_counts())}")

    cov_mask = feat["covered"].to_numpy() == 1
    print("  -- DE AUROC ((up|down) vs none) --")
    res = {"coverage": float(cov), "DE": {}, "DIR": {}}
    for col in ["de_target_shift", "de_cell_shift"]:
        a = auroc(is_de, feat[col]); lo, hi = boot_ci(is_de, feat[col])
        print(f"     {col:<16} all rows  {a:.3f}  95% CI [{lo:.3f},{hi:.3f}]")
        res["DE"][col] = [a, lo, hi]
        # covered-only (uncovered rows have feature 0, which confounds ranking)
        ia, isc = is_de[cov_mask], feat[col].to_numpy()[cov_mask]
        if cov_mask.sum() >= 10 and 0 < ia.sum() < len(ia):
            a2 = auroc(ia, isc); lo2, hi2 = boot_ci(ia, isc)
            print(f"     {col:<16} covered   {a2:.3f}  95% CI [{lo2:.3f},{hi2:.3f}]  (n={int(cov_mask.sum())})")
            res["DE"][col + "_covered"] = [a2, lo2, hi2, int(cov_mask.sum())]

    de = feat[lab != "none"].copy()
    is_up = (de["label"].to_numpy() == "up").astype(int)
    print("  -- DIR AUROC (up vs down, DE rows) --")
    a = auroc(is_up, de["dir_dlogit"]); lo, hi = boot_ci(is_up, de["dir_dlogit"])
    print(f"     dir_dlogit (all DE, n={len(de)})       {a:.3f}  95% CI [{lo:.3f},{hi:.3f}]")
    res["DIR"]["dir_dlogit_allDE"] = [a, lo, hi, int(len(de))]
    sub = de[de["covered"] == 1]
    if len(sub) >= 10 and sub["label"].nunique() == 2:
        isu = (sub["label"].to_numpy() == "up").astype(int)
        a = auroc(isu, sub["dir_dlogit"]); lo, hi = boot_ci(isu, sub["dir_dlogit"])
        print(f"     dir_dlogit (covered DE, n={len(sub)})    {a:.3f}  95% CI [{lo:.3f},{hi:.3f}]")
        res["DIR"]["dir_dlogit_covered"] = [a, lo, hi, int(len(sub))]
    return res


def main():
    global MAX_LEN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=Path,
                    default=ROOT / "outputs" / "benchmark_b_250" / "sample.csv")
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "train.csv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows")
    ap.add_argument("--predict", type=Path, default=None,
                    help="label-less CSV (e.g. data/test.csv): write features, skip eval")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "outputs" / "geneformer_probe")
    args = ap.parse_args()
    MAX_LEN = args.max_len

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # embedding shifts are ~1e-4; bf16 mantissa too coarse
    print(f"[load] Geneformer V2-104M on {device} ({dtype})")
    tok, med, nm_ci = load_assets()
    model = BertForMaskedLM.from_pretrained(V2, torch_dtype=dtype).to(device).eval()
    ctx_ids, pos_of_ens = build_context(tok, med)
    print(f"[ctx] macrophage context: {len(ctx_ids)} tokens "
          f"(<cls> + {len(ctx_ids)-2} genes + <eos>)")

    args.out.mkdir(parents=True, exist_ok=True)
    runs = [("BLINDED 250-row sample", args.sample), ("FULL train", args.train)]
    if args.predict is not None:
        runs.append(("PREDICT", args.predict))
    all_res = {}
    for tag, path in runs:
        if not path.exists():
            print(f"[skip] {tag}: {path} not found")
            continue
        df = pd.read_csv(path)
        if args.limit:
            df = df.head(args.limit)
        print(f"\n[run] {tag}: {len(df)} rows ...")
        feat = compute_features(model, df, ctx_ids, pos_of_ens, tok, nm_ci,
                                device, batch=args.batch)
        name = "test" if tag == "PREDICT" else tag.split()[0].lower()
        feat.to_csv(args.out / f"features_{name}.csv", index=False)
        if "label" in df.columns:                       # eval only if labeled
            all_res[tag] = evaluate(feat, tag)
        else:
            print(f"  [features-only] {int(feat['covered'].sum())}/{len(feat)} covered "
                  f"-> features_{name}.csv")

    (args.out / "results.json").write_text(json.dumps(all_res, indent=2))
    print(f"\n[done] -> {args.out/'results.json'}")


if __name__ == "__main__":
    main()
