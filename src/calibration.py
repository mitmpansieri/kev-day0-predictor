"""Step 7 / Decision 11d: calibrate the late-fusion score (a within-month rank) into a probability. DEV ONLY.

1. Rebuild the out-of-time (OOT) scores of the headline model on the three selection periods, exactly as in 05b:
   gradient boosting (05a settings) + text MLP (minilm, (64,), lr 1e-3, 5 seeds, ES on last 6 months + refit),
   fused = mean of the two within-month percentile ranks. Sanity check: selection 46.7, worst period 42.6.
2. Candidates: Platt (logistic on the fused score) and isotonic. Choice = lower MEAN LEAVE-ONE-PERIOD-OUT Brier
   (fit on two periods, score the third): in-sample Brier always favours the more flexible isotonic.
3. Refit the winner on the three periods pooled -> outputs/models/calibrator_fusion.joblib (+ .json summary).
Calibration is monotone: it never changes the ranking or recall@K; it only turns ranks into probabilities.

Run: .venv/bin/python src/calibration.py [fusion|hgb]   (Decision 12: hgb = headline -> calibrator_hgb.joblib)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import load_kev_raw                       # noqa: E402
from metrics import recall_at_k                         # noqa: E402
from tabular_models import build_X, make_hgb, views     # noqa: E402
import text_models as txm                               # noqa: E402

HGB_BEST = dict(encoding="onehot", learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100)
EMB_NAME, NN_H, NN_LR, SEEDS = "minilm", (64,), 1e-3, range(5)
PERIODS = ["wf_2023", "wf_2024h1", "val"]
OOT_PATH = ROOT / "data" / "oot_fused_scores.parquet"
MODELS, TAB, FIG = ROOT / "outputs" / "models", ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"


def build_oot_scores() -> pd.DataFrame:
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    y = dev["label"].to_numpy()
    kev = load_kev_raw(sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1])
    X_tab = build_X(dev, kev)
    E = txm.load_embeddings(EMB_NAME, dev)
    parts = []
    for name, (fit_idx, ev_idx) in views(dev).items():
        t = time.time()
        part = dev.loc[ev_idx, ["cve_id", "published", "label"]].copy()
        part["period"] = name
        part["s_hgb"] = make_hgb(**HGB_BEST).fit(X_tab[fit_idx], y[fit_idx]).predict_proba(X_tab[ev_idx])[:, 1]
        epochs = []
        part["s_text"] = np.mean([txm.mlp_scorer(dev, E, y, NN_H, NN_LR, s, epochs_log=epochs)(fit_idx, ev_idx)
                                  for s in SEEDS], axis=0)
        part["fused"] = (txm.month_pct_rank(part, part["s_hgb"].to_numpy())
                         + txm.month_pct_rank(part, part["s_text"].to_numpy())) / 2
        print(f"  {name:10s} n={len(part):,} pos={int(part['label'].sum())} epochs={epochs} ({time.time() - t:.0f}s)",
              flush=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


MIN_POS = 10   # user decision 2026-11-20: every isotonic step must contain >= 10 exploited CVEs


class MergedIsotonic:
    """Isotonic regression whose steps are merged (top down) until each holds >= min_pos positives; a remainder
    at the bottom joins the lowest step. Merging adjacent steps keeps it monotone. Predicts a STEP function:
    p(s) = rate of the step with the largest lower score bound <= s (scores below all bounds -> lowest step)."""

    def __init__(self, min_pos: int = MIN_POS):
        self.min_pos = min_pos

    def fit(self, s: np.ndarray, y: np.ndarray):
        o = np.argsort(s, kind="stable")
        s, y = np.asarray(s)[o], np.asarray(y)[o]
        f = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s, y).predict(s)
        starts = np.r_[0, np.flatnonzero(np.diff(f) != 0) + 1]          # original steps (runs of equal fit)
        ends = np.r_[starts[1:], len(s)]
        merged, acc = [], None                                           # walk from the top step down
        for a, b in zip(starts[::-1], ends[::-1]):
            acc = (a, acc[1] if acc else b)
            if y[acc[0]:acc[1]].sum() >= self.min_pos:
                merged.append(acc); acc = None
        if acc:                                                          # bottom remainder -> lowest step
            merged[-1] = (acc[0], merged[-1][1]) if merged else acc
        merged = merged[::-1]
        self.lower_ = np.array([s[a] for a, _ in merged])
        self.rate_ = np.array([y[a:b].mean() for a, b in merged])
        self.n_ = np.array([b - a for a, b in merged]); self.pos_ = np.array([int(y[a:b].sum()) for a, b in merged])
        return self

    def predict(self, s: np.ndarray) -> np.ndarray:
        i = np.searchsorted(self.lower_, np.asarray(s), side="right") - 1
        return self.rate_[np.clip(i, 0, len(self.rate_) - 1)]

    def steps(self) -> pd.DataFrame:
        return pd.DataFrame({"fused_from": self.lower_, "n": self.n_, "positives": self.pos_, "prob": self.rate_})


SUFFIX = {"fusion": "", "hgb": "_hgb"}


def to_x_hgb(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(s / (1 - s))


def fit_cal(kind: str, s: np.ndarray, y: np.ndarray):
    if kind == "platt":
        return LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
    if kind == "isotonic_merged":
        return MergedIsotonic().fit(s, y)
    return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s, y)


def predict_cal(model, s: np.ndarray) -> np.ndarray:
    p = model.predict_proba(s.reshape(-1, 1))[:, 1] if hasattr(model, "predict_proba") else model.predict(s)
    return np.clip(p, 1e-6, 1 - 1e-6)


def main(model: str = "fusion"):
    t0 = time.time()
    if OOT_PATH.exists():
        oot = pd.read_parquet(OOT_PATH)
        print(f"loaded cached OOT scores {OOT_PATH.name}")
    else:
        print("rebuilding OOT scores (dev only) ...", flush=True)
        oot = build_oot_scores()
        oot.to_parquet(OOT_PATH, index=False)

    # sanity check: must reproduce the Phase 5 numbers of the chosen model
    col, exp_sel, exp_worst = {"fusion": ("fused", 46.7, 42.6), "hgb": ("s_hgb", 47.4, 39.7)}[model]
    rec = {p: recall_at_k(oot[oot["period"] == p], col, 110) for p in PERIODS}
    sel, worst = 100 * np.mean(list(rec.values())), 100 * min(rec.values())
    print(f"[{model}] recall@110 by period:", {p: round(100 * r, 1) for p, r in rec.items()},
          f"selection {sel:.1f} (expected {exp_sel}), worst {worst:.1f} (expected {exp_worst})")
    assert abs(sel - exp_sel) < 0.06 and abs(worst - exp_worst) < 0.06, "OOT scores do not reproduce Phase 5"
    # calibrator input x: fused rank (fusion) or log-odds of the boosting probability (hgb; Platt is fit on
    # log-odds, isotonic is unaffected by a monotone transform)
    oot["x"] = oot["fused"] if model == "fusion" else to_x_hgb(oot["s_hgb"].to_numpy())

    # leave-one-period-out comparison
    rows = []
    for kind in ["platt", "isotonic", "isotonic_merged"]:
        for held in PERIODS:
            a, b = oot[oot["period"] != held], oot[oot["period"] == held]
            m = fit_cal(kind, a["x"].to_numpy(), a["label"].to_numpy())
            p = predict_cal(m, b["x"].to_numpy())
            rows.append({"method": kind, "held_out": held, "n": len(b), "positives": int(b["label"].sum()),
                         "brier": brier_score_loss(b["label"], p), "log_loss": log_loss(b["label"], p),
                         "mean_pred": p.mean(), "observed_rate": b["label"].mean()})
        m = fit_cal(kind, oot["x"].to_numpy(), oot["label"].to_numpy())
        p = predict_cal(m, oot["x"].to_numpy())
        rows.append({"method": kind, "held_out": "in-sample (pooled)", "n": len(oot), "positives": int(oot["label"].sum()),
                     "brier": brier_score_loss(oot["label"], p), "log_loss": log_loss(oot["label"], p),
                     "mean_pred": p.mean(), "observed_rate": oot["label"].mean()})
    base = oot["label"].mean()
    rows.append({"method": "constant base rate (reference)", "held_out": "pooled", "n": len(oot),
                 "positives": int(oot["label"].sum()),
                 "brier": brier_score_loss(oot["label"], np.full(len(oot), base)),
                 "log_loss": log_loss(oot["label"], np.full(len(oot), base)),
                 "mean_pred": base, "observed_rate": base})
    res = pd.DataFrame(rows)
    lopo = res[res["held_out"].isin(PERIODS)].groupby("method")[["brier", "log_loss"]].mean()
    # user rule (2026-11-20): merged isotonic, unless its leave-one-period-out Brier is worse than Platt's
    chosen = "isotonic_merged" if lopo.loc["isotonic_merged", "brier"] <= lopo.loc["platt", "brier"] else "platt"
    TAB.mkdir(parents=True, exist_ok=True)
    res.to_csv(TAB / f"05d_calibration{SUFFIX[model]}.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\nmean leave-one-period-out:\n", lopo.to_string(float_format=lambda v: f"{v:.6f}"))
    print(f"CHOSEN: {chosen}")

    # final calibrator on the three periods pooled
    final = fit_cal(chosen, oot["x"].to_numpy(), oot["label"].to_numpy())
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS / f"calibrator_{model}.joblib")
    q = [0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
    grid = np.array(q) if model == "fusion" else np.quantile(oot["x"], q)   # hgb: at pooled score quantiles
    summary = {"chosen": chosen, "model": model,
               "input": "fused score" if model == "fusion" else "logit(clip(s_hgb, 1e-7, 1-1e-7))",
               "fit_on": "pooled OOT scores: WF 2023, WF 2024-H1, val",
               "n": len(oot), "positives": int(oot["label"].sum()),
               "lopo_brier": lopo["brier"].round(7).to_dict(), "lopo_log_loss": lopo["log_loss"].round(6).to_dict(),
               "base_rate": round(float(base), 6),
               "prob_at_score_quantile": {str(g): round(float(v), 5) for g, v in zip(q, predict_cal(final, grid))}}
    if chosen == "isotonic_merged":
        summary["steps"] = final.steps().round(6).to_dict(orient="records")
        print(final.steps().to_string(index=False))
    (MODELS / f"calibrator_{model}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["prob_at_score_quantile"], indent=1))

    # reliability figure: bins by fused score (most of the action is in the top few %)
    edges = [0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0001]
    oot["bin"] = pd.cut(oot["x"].rank(pct=True), edges, right=False)      # bins by pooled score percentile
    oot["p_cal"] = predict_cal(final, oot["x"].to_numpy())
    g = oot.groupby("bin", observed=True).agg(pred=("p_cal", "mean"), obs=("label", "mean"), n=("label", "size"),
                                              pos=("label", "sum"))
    print(g.to_string())
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    lo = 1e-4
    ax.plot([lo, 1], [lo, 1], color="#999", lw=0.8, ls="--", label="perfect")
    ax.scatter(g["pred"].clip(lo), g["obs"].clip(lo), s=30 + g["pos"] * 2, color="#2a78d6", zorder=3,
               label=f"{chosen} (bins of fused score; size ~ positives)")
    ax.set(xscale="log", yscale="log", xlabel="predicted probability (mean in bin)",
           ylabel="observed exploitation rate in bin",
           title="Calibration of the late-fusion score\n(pooled OOT: WF 2023, WF 2024-H1, val)")
    ax.legend(fontsize=8, loc="upper left")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"05d_calibration{SUFFIX[model]}.png", dpi=150, bbox_inches="tight")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fusion")   # usage: calibration.py [fusion|hgb]
