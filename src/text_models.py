"""Phase 5b: text models on description_day0 embeddings (+ combined with tabular). Dev only.

Same three evaluation periods and the same selection rule as Phase 5a (mean recall@110 over WF 2023,
WF 2024-H1, val). Neural nets: the LAST 6 MONTHS OF EACH TRAINING PERIOD are an early-stopping set (never val).
Default (refit=True): train on the earlier part only to find the best epoch count, then REFIT on the full training
period (stopping months included) for that many epochs, same seed -> the nets see the same data as boosting.
refit=False keeps the stop-set-only model (trained on the earlier part, best epoch on the stopping months).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from metrics import recall_at_k
from tabular_models import make_hgb, make_lr, views

ROOT = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT / "data" / "embeddings"
torch.set_num_threads(8)


def load_embeddings(name: str, dev: pd.DataFrame) -> np.ndarray:
    E = np.load(EMB_DIR / f"{name}.npy").astype(np.float32)
    ids = pd.read_parquet(EMB_DIR / f"{name}_ids.parquet")["cve_id"]
    assert (ids.to_numpy() == dev["cve_id"].to_numpy()).all(), "embedding rows must follow dev order"
    return E


def es_split(dev: pd.DataFrame, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a training period into (train part, early-stopping part = its last 6 calendar months)."""
    last = dev.loc[fit_idx, "published"].max().to_period("M")
    es_months = pd.period_range(last - 5, last, freq="M")
    in_es = dev["published"].dt.to_period("M").isin(es_months).to_numpy()
    return fit_idx & ~in_es, fit_idx & in_es


# ------------------------------------------------------------------ text logistic regression
def make_text_lr(C: float):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


# ------------------------------------------------------------------ MLP (PyTorch, CPU)
class MLP(torch.nn.Module):
    def __init__(self, d_in: int, hidden: tuple[int, ...], dropout: float = 0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            d = h
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xes=None, yes=None, hidden=(64,), lr=1e-3, seed=0, max_epochs=60, patience=6, batch=512,
              wd=1e-4, fixed_epochs=None):
    """Class-weighted BCE (pos_weight = negatives/positives of the training data). With an ES set: early stopping,
    returns the best-epoch model. With fixed_epochs (refit): trains exactly that many epochs, no ES set."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    if fixed_epochs is None:
        Xes_t, yes_t = torch.tensor(Xes, dtype=torch.float32), torch.tensor(yes, dtype=torch.float32)
    pos_w = torch.tensor((len(ytr) - ytr.sum()) / max(ytr.sum(), 1), dtype=torch.float32)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    model = MLP(Xtr.shape[1], hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    hist, best, best_state, bad = [], np.inf, None, 0
    for epoch in range(fixed_epochs or max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        if fixed_epochs is not None:
            hist.append((epoch + 1, tot / len(Xtr_t), np.nan))
            continue
        with torch.no_grad():
            es_loss = loss_fn(model(Xes_t), yes_t).item()
        hist.append((epoch + 1, tot / len(Xtr_t), es_loss))
        if es_loss < best - 1e-4:
            best, best_state, bad = es_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    hist = pd.DataFrame(hist, columns=["epoch", "train_loss", "es_loss"])
    if fixed_epochs is None:
        model.load_state_dict(best_state)
        hist.attrs["best_epoch"] = int(hist.loc[hist["es_loss"].idxmin(), "epoch"])
    model.eval()
    return model, hist


def mlp_predict(model, X) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


# ------------------------------------------------------------------ generic evaluation over the three periods
def eval_periods(dev: pd.DataFrame, score_fn, k_main=110, k_gen=305) -> dict:
    """score_fn(fit_idx, ev_idx) -> scores for ev rows. Returns recall per period + selection score."""
    out, scores = {}, {}
    for name, (fit_idx, ev_idx) in views(dev).items():
        ev = dev[ev_idx].copy()
        ev["p"] = score_fn(fit_idx, ev_idx)
        out[f"{name}@{k_main}"] = recall_at_k(ev, "p", k_main)
        if name == "val":
            out[f"val@{k_gen}"] = recall_at_k(ev, "p", k_gen)
        scores[name] = ev["p"].to_numpy()
    p = [out[f"wf_2023@{k_main}"], out[f"wf_2024h1@{k_main}"], out[f"val@{k_main}"]]
    out["worst"] = min(p)
    out["selection"] = float(np.mean(p))
    out["_scores"] = scores
    return out


def text_lr_scorer(E, y, C):
    def f(fit_idx, ev_idx):
        return make_text_lr(C).fit(E[fit_idx], y[fit_idx]).predict_proba(E[ev_idx])[:, 1]
    return f


def _fit_with_refit(F_tr, y_tr, F_es, y_es, F_full, y_full, hidden, lr, seed, refit, curves, epochs_log):
    model, hist = train_mlp(F_tr, y_tr, F_es, y_es, hidden, lr, seed)
    best = hist.attrs["best_epoch"]
    if curves is not None:
        curves.append(hist.assign(period=len(curves)))
    if epochs_log is not None:
        epochs_log.append(best)
    if refit:                     # same seed, full training period, the epoch count found above
        model, _ = train_mlp(F_full, y_full, hidden=hidden, lr=lr, seed=seed, fixed_epochs=best)
    return model


def mlp_scorer(dev, F, y, hidden, lr, seed, curves=None, refit=True, epochs_log=None):
    def f(fit_idx, ev_idx):
        tr, es = es_split(dev, fit_idx)
        model = _fit_with_refit(F[tr], y[tr], F[es], y[es], F[fit_idx], y[fit_idx], hidden, lr, seed, refit, curves, epochs_log)
        return mlp_predict(model, F[ev_idx])
    return f


def tab_text_scorer(dev, X_tab, E, y, hidden, lr, seed, curves=None, refit=True, epochs_log=None):
    """(a) one network on [tabular (encoded/scaled inside the period) | embedding]. Encoders are fit on the data
    the network trains on: the earlier part for the epoch search, the full period for the refit."""
    from sklearn.base import clone
    dense = lambda A: A.toarray() if hasattr(A, "toarray") else np.asarray(A)

    def feats(pre, idx):
        return np.hstack([dense(pre.transform(X_tab[idx])), E[idx]])

    def f(fit_idx, ev_idx):
        tr, es = es_split(dev, fit_idx)
        pre = make_lr().named_steps["pre"]                     # same one-hot/scaler as the tabular LR
        pre.fit(X_tab[tr])
        model, hist = train_mlp(feats(pre, tr), y[tr], feats(pre, es), y[es], hidden, lr, seed)
        best = hist.attrs["best_epoch"]
        if curves is not None:
            curves.append(hist.assign(period=len(curves)))
        if epochs_log is not None:
            epochs_log.append(best)
        if refit:
            pre = clone(make_lr().named_steps["pre"]).fit(X_tab[fit_idx])
            model, _ = train_mlp(feats(pre, fit_idx), y[fit_idx], hidden=hidden, lr=lr, seed=seed, fixed_epochs=best)
        return mlp_predict(model, feats(pre, ev_idx))
    return f


def oof_text_score(dev, E, y, fit_idx, C, n_splits=5, seed=0) -> np.ndarray:
    """Out-of-fold text-LR scores for the rows of a training period (grouped by cv_group: twins stay together)."""
    idx = np.flatnonzero(fit_idx)
    oof = np.zeros(len(idx))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for a, b in cv.split(idx, y[idx], groups=dev["cv_group"].to_numpy()[idx]):
        oof[b] = make_text_lr(C).fit(E[idx[a]], y[idx[a]]).predict_proba(E[idx[b]])[:, 1]
    return oof


def hgb_text_scorer(dev, X_tab, E, y, C, hgb_params):
    """(b) gradient boosting + text score as an extra feature: OOF on the training rows, full-period text model
    for the evaluation rows -> the boosting model never sees a text score fitted on its own label."""
    import tabular_models as tm

    def f(fit_idx, ev_idx):
        Xf, Xe = X_tab[fit_idx].copy(), X_tab[ev_idx].copy()
        Xf["text_score"] = oof_text_score(dev, E, y, fit_idx, C)
        Xe["text_score"] = make_text_lr(C).fit(E[fit_idx], y[fit_idx]).predict_proba(E[ev_idx])[:, 1]
        if "text_score" not in tm.NUM:
            tm.NUM.append("text_score")
        try:
            m = make_hgb(**hgb_params).fit(Xf, y[fit_idx])
            return m.predict_proba(Xe)[:, 1]
        finally:
            tm.NUM.remove("text_score")
    return f


def month_pct_rank(dev_part: pd.DataFrame, s: np.ndarray) -> np.ndarray:
    return pd.Series(s, index=dev_part.index).groupby(dev_part["published"].dt.to_period("M")).rank(pct=True).to_numpy()
