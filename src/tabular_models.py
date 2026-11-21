"""Phase 5a: tabular models (no text). Features = day0_tabular + KEV history. All learned preprocessing
(encoders, rare-category grouping, target encoding, scaling) lives inside the Pipeline, so it is fit on the
training part of each fold only.

Evaluation views (all on dev; test is never touched):
  val     : fit on train (2022-01 -> 2024-06), score val (2024-07 -> 2024-12)       -> recall@K per month
  wf_2023 : fit on 2022, score 2023                                               -> recall@K per month
  wf_2024h1: fit on 2022-2023, score 2024-01 -> 2024-06                           -> recall@K per month
  gcv     : 5-fold StratifiedGroupKFold by cv_group (vendor|month) inside train  -> recall at the same
            share-of-month budget (folds mix months, so a fixed K per month does not apply)
Selection score = mean recall@K_MAIN over the three periods: WF 2023, WF 2024-H1, val (equal weights).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

from features import CATEGORICAL, CVSS_PARTS, NUMERIC, KEV_HIST, day0_tabular, kev_history_features
from metrics import recall_at_budget, recall_at_k, pr_auc

HIGH_CARD = ["vendor_key", "cna", "cwe"]
CVSS_CAT = [f"cvss_{p}" for p in CVSS_PARTS]
NUM = NUMERIC + ["vendor_kev_log_all", "vendor_kev_log_365", "vendor_kev_days_since", "vendor_kev_any",
                 "product_kev_log_all"]


def build_X(df: pd.DataFrame, kev: pd.DataFrame) -> pd.DataFrame:
    X = day0_tabular(df)
    H = kev_history_features(df, kev)
    X["vendor_kev_log_all"] = np.log1p(H["vendor_kev_n_all"])
    X["vendor_kev_log_365"] = np.log1p(H["vendor_kev_n_365"])
    X["vendor_kev_days_since"] = H["vendor_kev_days_since"].fillna(3650)   # never -> 10 years
    X["vendor_kev_any"] = (H["vendor_kev_n_all"] > 0).astype(int)
    X["product_kev_log_all"] = np.log1p(H["product_kev_n_all"])
    return X


# ------------------------------------------------------------------ models
def make_lr(C: float = 0.1, min_freq: int = 10) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", C=C, max_iter=5000))])


def make_hgb(encoding: str = "target", learning_rate: float = 0.05, max_leaf_nodes: int = 15,
             min_samples_leaf: int = 50, max_iter: int = 300, l2: float = 1.0) -> Pipeline:
    if encoding == "target":     # smoothed target encoding, cross-fitted (cv=5) inside fit -> no in-sample leak
        # seeded splitter: with cv=5 scikit-learn shuffles WITHOUT a seed -> results changed between runs
        hc = ("hc", TargetEncoder(target_type="binary", smooth="auto",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0)), HIGH_CARD)
        n_hc = len(HIGH_CARD)
    else:                        # one-hot of frequent categories only (rare ones grouped as 'infrequent')
        hc = ("hc", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False), HIGH_CARD)
        n_hc = None
    pre = ColumnTransformer([
        hc,
        ("cvss", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CVSS_CAT),
        ("num", "passthrough", NUM)], verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        max_iter=max_iter, l2_regularization=l2, class_weight="balanced", early_stopping=False,
        categorical_features=[f"{c}" for c in CVSS_CAT], random_state=0)
    return Pipeline([("pre", pre), ("clf", clf)])


# ------------------------------------------------------------------ evaluation views
def views(dev: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pub = dev["published"]
    tr = dev["split"].eq("train").to_numpy()
    return {
        "val": (tr, dev["split"].eq("val").to_numpy()),
        "wf_2023": ((pub.dt.year == 2022).to_numpy(), (pub.dt.year == 2023).to_numpy()),
        "wf_2024h1": ((pub.dt.year <= 2023).to_numpy(), (tr & (pub.dt.year == 2024)).to_numpy()),
    }


def evaluate(make_model, dev: pd.DataFrame, X: pd.DataFrame, k_main: int, k_gen: int, frac: float,
             gcv: bool = True, keep_scores: bool = False) -> dict:
    out, scores = {}, {}
    y = dev["label"].to_numpy()
    for name, (fit_idx, ev_idx) in views(dev).items():
        m = make_model().fit(X[fit_idx], y[fit_idx])
        ev = dev[ev_idx].copy()
        ev["p"] = m.predict_proba(X[ev_idx])[:, 1]
        out[f"{name}@{k_main}"] = recall_at_k(ev, "p", k_main)
        if name == "val":
            out[f"val@{k_gen}"] = recall_at_k(ev, "p", k_gen)
            out["val_pr_auc"] = pr_auc(ev, "p")
            scores["val"] = ev
    if gcv:
        tr = dev["split"].eq("train").to_numpy()
        dtr, Xtr = dev[tr].reset_index(drop=True), X[tr].reset_index(drop=True)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        oof = np.zeros(len(dtr))
        for a, b in cv.split(Xtr, dtr["label"], groups=dtr["cv_group"]):
            oof[b] = make_model().fit(Xtr.iloc[a], dtr["label"].iloc[a]).predict_proba(Xtr.iloc[b])[:, 1]
        out[f"gcv@{frac:.1%}"] = recall_at_budget(dtr.assign(p=oof), "p", frac)
    out["wf_mean"] = np.mean([out[f"wf_2023@{k_main}"], out[f"wf_2024h1@{k_main}"]])
    out["wf_worst"] = min(out[f"wf_2023@{k_main}"], out[f"wf_2024h1@{k_main}"])
    # SELECTION RULE (all Phase 5 models): mean recall@K_MAIN over the three periods (WF 2023, WF 2024-H1, val)
    out["selection"] = np.mean([out[f"wf_2023@{k_main}"], out[f"wf_2024h1@{k_main}"], out[f"val@{k_main}"]])
    if keep_scores:
        out["_scores"] = scores
    return out


def grid_search(kind: str, grid: dict, dev, X, k_main, k_gen, frac, log=print) -> pd.DataFrame:
    rows = []
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        params = dict(zip(keys, vals))
        maker = (lambda p=params: make_lr(**p)) if kind == "lr" else (lambda p=params: make_hgb(**p))
        t = time.time()
        res = evaluate(maker, dev, X, k_main, k_gen, frac)
        rows.append({**params, **res, "sec": round(time.time() - t)})
        log(f"  {kind} {params} -> val@{k_main} {res[f'val@{k_main}']:.3f} | wf_mean {res['wf_mean']:.3f} "
            f"| selection {res['selection']:.3f} ({time.time() - t:.0f}s)")
    return pd.DataFrame(rows).sort_values("selection", ascending=False).reset_index(drop=True)
