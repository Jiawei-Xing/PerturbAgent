"""GenePert-style trained classifier vs the kNN aggregator.

Instead of non-parametric gene-similarity kNN, train a model on gene+pert
embeddings -> label, the GenePert approach. Honest evaluation = 5-fold
DOUBLY-disjoint CV: assign perts and genes each to 5 random groups; for fold k,
eval = rows with pert-group k AND gene-group k, train = rows with pert-group != k
AND gene-group != k (so eval perts and genes are both unseen, mirroring the
competition split). Pool eval predictions, report DE and DIR AUROC.

Features: Geneformer V2-104M token embeddings (mean-centered, unit-norm) for the
perturbation and target gene, optionally their element-wise product (interaction).
Models: L2-regularized logistic regression and a 1-hidden-layer MLP, in torch.

Compare against the kNN baseline (knn_transfer_test.py): DE ~0.55, DIR ~0.63.

    uv run --extra serve python examples/trained_classifier_test.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def auroc(y, s):
    y = np.asarray(y).astype(int); P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    r = _rankdata(np.asarray(s, float))
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def boot(y, s, n=2000, seed=0):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y)); a = auroc(y[idx], s[idx])
        if not np.isnan(a):
            v.append(a)
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def load_emb():
    tok = pickle.load(open(GF / "token_dictionary_gc104M.pkl", "rb"))
    nm = pickle.load(open(GF / "gene_name_id_dict_gc104M.pkl", "rb"))
    nm_ci = {str(k).upper(): v for k, v in nm.items()}
    with safe_open(str(SAFET), framework="numpy") as f:
        W = f.get_tensor("bert.embeddings.word_embeddings.weight")
    W = W - W.mean(0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    E = {s: W[tok[e]] for s, e in nm_ci.items() if e in tok}
    return E


def vec(sym, E, D):
    return E.get(str(sym).upper(), None)


def build_features(df, E, use_pert, use_prod):
    D = len(next(iter(E.values())))
    X, cov = [], []
    for r in df.itertuples():
        g = vec(r.gene, E, D); p = vec(r.pert, E, D)
        if g is None or (use_pert and p is None):
            X.append(np.zeros(D * (1 + use_pert + use_prod))); cov.append(False); continue
        parts = [g]
        if use_pert:
            parts.append(p)
        if use_prod:
            parts.append(g * (p if p is not None else 0))
        X.append(np.concatenate(parts)); cov.append(True)
    return np.array(X, np.float32), np.array(cov)


class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden, 1))
        else:
            self.net = nn.Linear(d, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_predict(Xtr, ytr, Xte, hidden, wd, epochs=300):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = MLP(Xtr.shape[1], hidden).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Xt), yt); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(Xte, device=dev))).cpu().numpy()


def cv_eval(df, E, use_pert, use_prod, hidden, wd, seed=0):
    rng = np.random.default_rng(seed)
    uperts = {p: rng.integers(0, 5) for p in df["pert"].unique()}
    ugenes = {g: rng.integers(0, 5) for g in df["gene"].unique()}
    pg = df["pert"].map(uperts).to_numpy(); gg = df["gene"].map(ugenes).to_numpy()
    X, cov = build_features(df, E, use_pert, use_prod)
    lab = df["label"].to_numpy()
    is_de = (lab != "none").astype(np.float32); is_up = (lab == "up").astype(np.float32)
    pde = np.full(len(df), np.nan); pup = np.full(len(df), np.nan)
    for k in range(5):
        te = (pg == k) & (gg == k) & cov
        tr = (pg != k) & (gg != k) & cov
        if te.sum() < 5 or tr.sum() < 50:
            continue
        pde[te] = fit_predict(X[tr], is_de[tr], X[te], hidden, wd)
        trd = tr & (is_de == 1)
        ted = te & (is_de == 1)
        if ted.sum() >= 3 and trd.sum() >= 30:
            pup[ted] = fit_predict(X[trd], is_up[trd], X[ted], hidden, wd)
    okde = ~np.isnan(pde); okup = ~np.isnan(pup)
    de = auroc(is_de[okde], pde[okde]); dl = boot(is_de[okde], pde[okde])
    dr = auroc(is_up[okup], pup[okup]); drl = boot(is_up[okup], pup[okup])
    return de, dl, dr, drl, int(okde.sum()), int(okup.sum())


def main():
    torch.manual_seed(0)
    E = load_emb()
    df = pd.read_csv(ROOT / "data/train.csv")
    print(f"[clf] {len(df)} train pairs; kNN baseline: DE ~0.55, DIR ~0.63\n")
    configs = [
        ("logistic gene-only", dict(use_pert=0, use_prod=0, hidden=0, wd=1e-2)),
        ("logistic pert+gene", dict(use_pert=1, use_prod=0, hidden=0, wd=1e-2)),
        ("logistic pert+gene+prod", dict(use_pert=1, use_prod=1, hidden=0, wd=1e-2)),
        ("MLP-128 pert+gene+prod", dict(use_pert=1, use_prod=1, hidden=128, wd=1e-3)),
        ("MLP-128 gene-only", dict(use_pert=0, use_prod=0, hidden=128, wd=1e-3)),
    ]
    for name, cfg in configs:
        de, dl, dr, drl, nde, nup = cv_eval(df, E, **cfg)
        print(f"{name:26s} DE {de:.3f}[{dl[0]:.3f},{dl[1]:.3f}]  "
              f"DIR {dr:.3f}[{drl[0]:.3f},{drl[1]:.3f}]  (n={nde}/{nup})")


if __name__ == "__main__":
    main()
