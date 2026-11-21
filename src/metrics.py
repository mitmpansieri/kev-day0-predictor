"""Evaluation metrics shared by all phases.

recall@K (the main metric): each calendar month the team can emergency-patch K CVEs. Take the K highest-scored
CVEs published that month; recall@K = exploited CVEs caught / all exploited CVEs, pooled over months.
Ties at the cut-off are scored by their EXPECTED value under a random tie-break (exact, deterministic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _month(df: pd.DataFrame) -> pd.Series:
    return df["published"].dt.to_period("M")


def caught_at_k(score: np.ndarray, y: np.ndarray, k: float) -> tuple[float, float]:
    """Expected (positives caught, patches used) when taking the top-k of one month, random tie-break."""
    n = len(score)
    if n == 0 or k <= 0:
        return 0.0, 0.0
    if k >= n:
        return float(y.sum()), float(n)
    order = np.argsort(-score, kind="stable")
    s, yy = score[order], y[order]
    thr = s[int(k) - 1] if float(k).is_integer() else s[int(np.floor(k))]
    above = s > thr
    tie = s == thr
    slots = k - above.sum()
    frac = slots / tie.sum()
    return float(yy[above].sum() + frac * yy[tie].sum()), float(k)


def recall_at_k(df: pd.DataFrame, score: str | pd.Series, k: float) -> float:
    sc = df[score] if isinstance(score, str) else score
    caught = 0.0
    for _, idx in df.groupby(_month(df)).indices.items():
        c, _ = caught_at_k(sc.to_numpy()[idx], df["label"].to_numpy()[idx], k)
        caught += c
    return caught / df["label"].sum()


def recall_curve(df: pd.DataFrame, score: str | pd.Series, ks) -> np.ndarray:
    return np.array([recall_at_k(df, score, k) for k in ks])


def random_recall_at_k(df: pd.DataFrame, k: float) -> float:
    """Expected recall@K of a random ranking: per month min(k, n)/n of that month's positives."""
    caught = sum(min(k, len(idx)) / len(idx) * df["label"].to_numpy()[idx].sum()
                 for idx in df.groupby(_month(df)).indices.values())
    return caught / df["label"].sum()


def rule_at_own_capacity(df: pd.DataFrame, flag: pd.Series) -> dict:
    months = _month(df).nunique()
    tp = int((flag & (df["label"] == 1)).sum())
    return {"patches_per_month": flag.sum() / months, "caught": tp,
            "recall": tp / df["label"].sum(), "precision": tp / max(flag.sum(), 1)}


def pr_auc(df: pd.DataFrame, score: str | pd.Series) -> float:
    sc = df[score] if isinstance(score, str) else score
    return average_precision_score(df["label"], sc)


def bootstrap_recall_at_k(df: pd.DataFrame, scores: dict[str, pd.Series], ks: list[float],
                          n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap: each replicate resamples CVEs within each month (Poisson(1) weights, same weights for
    every method) and re-draws a random tie-break. Returns one row per (replicate, method, k) with recall."""
    rng = np.random.default_rng(seed)
    y_all = df["label"].to_numpy()
    groups = list(df.groupby(_month(df)).indices.values())
    S = {m: s.to_numpy(dtype=float) for m, s in scores.items()}
    out = []
    for b in range(n_boot):
        w_all = rng.poisson(1.0, len(df)).astype(float)
        jitter = rng.random(len(df)) * 1e-6               # random tie-break, << any real score gap
        tot_pos = (w_all * y_all).sum()
        for m, s_all in S.items():
            caught = dict.fromkeys(ks, 0.0)
            for idx in groups:
                s, y, w = s_all[idx] + jitter[idx], y_all[idx], w_all[idx]
                order = np.argsort(-s)
                cw, wy = np.cumsum(w[order]), (w * y)[order]
                for k in ks:                          # take items until the weighted capacity k is used
                    full = cw <= k
                    c = wy[full].sum()
                    j = full.sum()
                    if j < len(order) and w[order][j] > 0:  # fractional boundary item
                        c += wy[j] * (k - (cw[j - 1] if j > 0 else 0.0)) / w[order][j]
                    caught[k] += c
            for k in ks:
                out.append((b, m, k, caught[k] / tot_pos if tot_pos else np.nan))
    return pd.DataFrame(out, columns=["rep", "method", "k", "recall"])


def ci_table(boot: pd.DataFrame, point: dict[tuple[str, float], float], alpha: float = 0.05) -> pd.DataFrame:
    g = boot.groupby(["method", "k"])["recall"]
    t = pd.DataFrame({"lo": g.quantile(alpha / 2), "hi": g.quantile(1 - alpha / 2)})
    t["point"] = [point[i] for i in t.index]
    return t[["point", "lo", "hi"]]


def recall_at_budget(df: pd.DataFrame, score: str | pd.Series, frac: float) -> float:
    """Like recall_at_k, but each month's capacity = frac x that month's CVE count (for sets of different size)."""
    sc = df[score] if isinstance(score, str) else score
    caught = 0.0
    for idx in df.groupby(_month(df)).indices.values():
        c, _ = caught_at_k(sc.to_numpy()[idx], df["label"].to_numpy()[idx], frac * len(idx))
        caught += c
    return caught / df["label"].sum()
