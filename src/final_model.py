"""The frozen headline model (TEST PROTOCOL 1): late fusion of gradient boosting + 5-seed text MLP.

One recipe, used for every fit so the dry run and the test run share the same code:
  fit_headline(df_fit, X_fit, E_fit) ->
    gradient boosting make_hgb(onehot, lr 0.03, 15 leaves, min_leaf 100) on all fit rows
    text MLP (minilm, (64,), lr 1e-3), per seed 0-4: early-stopping set = last 6 calendar months of the fit rows,
    best epoch, then refit on ALL fit rows for that many epochs (same seed)   [= Phase 5 recipe, text_models.py]
  score_headline(models, df_eval, X_eval, E_eval) -> s_hgb, s_text (mean of 5 seeds), fused (within-month ranks)

CLI (Phase 6 prep step A, DEV ONLY):
  .venv/bin/python src/final_model.py fit-dev  -> fits on all of dev (train+val), saves outputs/models/dev_*,
                                                  writes epoch counts + sha256 to outputs/models/dev_models.json
  .venv/bin/python src/final_model.py fit-all  -> Step H: deployment/demo gradient boosting on ALL labelled data
                                                  (dev + test) -> outputs/models/deploy_hgb.joblib + deploy_models.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import text_models as txm                                   # noqa: E402
from features import load_kev_raw                           # noqa: E402
from tabular_models import build_X, make_hgb                # noqa: E402

HGB_BEST = dict(encoding="onehot", learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100)
EMB_NAME, NN_H, NN_LR, SEEDS = "minilm", (64,), 1e-3, (0, 1, 2, 3, 4)
KEV_FILE = ROOT / "data" / "raw" / "kev_20261120.csv"       # frozen snapshot (TEST PROTOCOL 1)
MODELS = ROOT / "outputs" / "models"


def fit_headline(df_fit: pd.DataFrame, X_fit: pd.DataFrame, E_fit: np.ndarray, log=print) -> dict:
    df_fit, X_fit = df_fit.reset_index(drop=True), X_fit.reset_index(drop=True)
    y = df_fit["label"].to_numpy()
    t = time.time()
    hgb = make_hgb(**HGB_BEST).fit(X_fit, y)
    log(f"  gradient boosting fit on {len(y):,} CVEs / {int(y.sum())} positives ({time.time() - t:.0f}s)")
    tr, es = txm.es_split(df_fit, np.ones(len(df_fit), bool))
    es_months = df_fit.loc[es, "published"].dt.to_period("M")
    log(f"  MLP early-stopping set: {es_months.min()} -> {es_months.max()} ({es.sum():,} CVEs, {int(y[es].sum())} pos)")
    mlps, epochs = [], []
    for s in SEEDS:
        _, hist = txm.train_mlp(E_fit[tr], y[tr], E_fit[es], y[es], NN_H, NN_LR, s)
        best = hist.attrs["best_epoch"]
        m, _ = txm.train_mlp(E_fit, y, hidden=NN_H, lr=NN_LR, seed=s, fixed_epochs=best)
        mlps.append(m); epochs.append(best)
    log(f"  MLP best epochs (seeds {list(SEEDS)}): {epochs}")
    return {"hgb": hgb, "mlps": mlps, "epochs": epochs, "es_months": [str(es_months.min()), str(es_months.max())]}


def score_headline(models: dict, df_eval: pd.DataFrame, X_eval: pd.DataFrame, E_eval: np.ndarray) -> pd.DataFrame:
    out = df_eval[["cve_id", "published", "label"]].copy()
    out["s_hgb"] = models["hgb"].predict_proba(X_eval)[:, 1]
    seed_scores = np.column_stack([txm.mlp_predict(m, E_eval) for m in models["mlps"]])
    for i, s in enumerate(SEEDS):
        out[f"s_text_seed{s}"] = seed_scores[:, i]
    out["s_text"] = seed_scores.mean(axis=1)
    out["fused"] = (txm.month_pct_rank(out, out["s_hgb"].to_numpy())
                    + txm.month_pct_rank(out, out["s_text"].to_numpy())) / 2
    return out


def fuse_seed(out: pd.DataFrame, seed: int) -> np.ndarray:
    """Fusion with ONE text seed (for the seed-spread sensitivity)."""
    return (txm.month_pct_rank(out, out["s_hgb"].to_numpy())
            + txm.month_pct_rank(out, out[f"s_text_seed{seed}"].to_numpy())) / 2


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_models(models: dict, prefix: str, meta: dict) -> dict:
    MODELS.mkdir(parents=True, exist_ok=True)
    files = {"hgb": MODELS / f"{prefix}_hgb.joblib"}
    joblib.dump(models["hgb"], files["hgb"])
    for s, m in zip(SEEDS, models["mlps"]):
        files[f"mlp_s{s}"] = MODELS / f"{prefix}_text_mlp_s{s}.pt"
        torch.save(m.state_dict(), files[f"mlp_s{s}"])
    meta = {**meta, "epochs": models["epochs"], "es_months": models["es_months"],
            "settings": {"hgb": HGB_BEST, "text": {"embedding": EMB_NAME, "hidden": list(NN_H), "lr": NN_LR,
                                                   "seeds": list(SEEDS)}},
            "sha256": {k: _sha(p) for k, p in files.items()}}
    (MODELS / f"{prefix}_models.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_models(prefix: str) -> dict:
    meta = json.loads((MODELS / f"{prefix}_models.json").read_text())
    hgb_p = MODELS / f"{prefix}_hgb.joblib"
    assert _sha(hgb_p) == meta["sha256"]["hgb"], "gradient boosting file changed since it was frozen"
    mlps = []
    for s in SEEDS:
        p = MODELS / f"{prefix}_text_mlp_s{s}.pt"
        assert _sha(p) == meta["sha256"][f"mlp_s{s}"], f"MLP seed {s} file changed since it was frozen"
        m = txm.MLP(384, NN_H); m.load_state_dict(torch.load(p)); m.eval(); mlps.append(m)
    return {"hgb": joblib.load(hgb_p), "mlps": mlps, "epochs": meta["epochs"], "es_months": meta["es_months"],
            "meta": meta}


def fit_dev():
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    kev = load_kev_raw(KEV_FILE)
    X = build_X(dev, kev)
    E = txm.load_embeddings(EMB_NAME, dev)
    print(f"fit on ALL of dev: {dev['published'].min():%Y-%m-%d} -> {dev['published'].max():%Y-%m-%d}, "
          f"{len(dev):,} CVEs, {int(dev['label'].sum())} positives")
    models = fit_headline(dev, X, E)
    meta = save_models(models, "dev", {"fit_on": "dev (train+val)", "n": len(dev), "positives": int(dev["label"].sum()),
                                       "kev_file": KEV_FILE.name, "created": pd.Timestamp.now().isoformat(timespec="seconds")})
    # reload + rescore a slice to prove the saved files reproduce the in-memory models
    back = load_models("dev")
    a = score_headline(models, dev.tail(3000), X.tail(3000), E[-3000:])
    b = score_headline(back, dev.tail(3000), X.tail(3000), E[-3000:])
    assert np.allclose(a["s_hgb"], b["s_hgb"]) and np.allclose(a["s_text"], b["s_text"])
    print("saved + reload check OK:", json.dumps({k: meta[k] for k in ["epochs", "es_months", "n", "positives"]}))


def fit_all():
    """Step H (after Phase 6): DEPLOYMENT / DEMO model = gradient boosting (Decision 12 settings, no re-tuning) refit
    on ALL labelled data (dev + test, published 2022-01-01 -> 2026-08-20 label cutoff). NEVER used for a reported
    number: every test result comes from the frozen dev model (dev_hgb.joblib)."""
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    test = pd.read_parquet(ROOT / "data" / "test_locked.parquet")
    full = pd.concat([dev, test], ignore_index=True)
    assert full["cve_id"].is_unique
    kev = load_kev_raw(KEV_FILE)
    X = build_X(full, kev)
    y = full["label"].to_numpy()
    t = time.time()
    hgb = make_hgb(**HGB_BEST).fit(X, y)
    print(f"deployment model: fit on {len(full):,} CVEs / {int(y.sum())} positives "
          f"({full['published'].min():%Y-%m-%d} -> {full['published'].max():%Y-%m-%d}) in {time.time() - t:.0f}s")
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "deploy_hgb.joblib"
    joblib.dump(hgb, path)
    meta = {"role": "DEPLOYMENT / DEMO model (Step H). NOT used for any reported number; Phase 6 results = dev_hgb.joblib",
            "fit_on": "dev + test (all labelled data)", "n": len(full), "positives": int(y.sum()),
            "published_from": f"{full['published'].min():%Y-%m-%d}", "published_to": f"{full['published'].max():%Y-%m-%d}",
            "kev_file": KEV_FILE.name, "settings": {"hgb": HGB_BEST, "max_iter": 300, "random_state": 0},
            "created": pd.Timestamp.now().isoformat(timespec="seconds"), "sha256": {"hgb": _sha(path)},
            "frozen_phase6_model_sha256": json.loads((MODELS / "dev_models.json").read_text())["sha256"]["hgb"]}
    (MODELS / "deploy_models.json").write_text(json.dumps(meta, indent=2))
    back = joblib.load(path)
    assert _sha(path) == meta["sha256"]["hgb"] and np.allclose(back.predict_proba(X.tail(2000))[:, 1], hgb.predict_proba(X.tail(2000))[:, 1])
    assert _sha(MODELS / "dev_hgb.joblib") == meta["frozen_phase6_model_sha256"], "frozen Phase 6 model must be untouched"
    print("saved", path.name, "sha256", meta["sha256"]["hgb"][:16], "| frozen dev_hgb.joblib unchanged")


def calibrate_deploy():
    """Phase 8a step 0: indicative probability for the DEMO model (deploy_hgb.joblib).
    The demo's PRIMARY output is the within-month percentile (rank vs a reference month). The probability is
    indicative only: Platt on logit(score) fit on TIME-ORDERED out-of-fold scores of the deploy recipe
    (same settings): for each half-year 2023H1 .. 2026H2(Jul 1 - Aug 20), fit on all labelled CVEs published
    BEFORE it, score it. Reference month = deploy-model scores of the last complete labelled month (2026-07)."""
    from sklearn.metrics import brier_score_loss, log_loss
    from calibration import fit_cal, predict_cal, to_x_hgb
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    test = pd.read_parquet(ROOT / "data" / "test_locked.parquet")
    full = pd.concat([dev, test], ignore_index=True)
    kev = load_kev_raw(KEV_FILE)
    X, y = build_X(full, kev), full["label"].to_numpy()
    half = full["published"].dt.year.astype(str) + "H" + np.where(full["published"].dt.month <= 6, "1", "2")
    oof = np.full(len(full), np.nan)
    rows = []
    for h in sorted(half[full["published"] >= "2023-01-01"].unique()):
        ev = (half == h).to_numpy()
        start = full.loc[ev, "published"].min().normalize()
        fit = (full["published"] < start).to_numpy()
        m = make_hgb(**HGB_BEST).fit(X[fit], y[fit])
        oof[ev] = m.predict_proba(X[ev])[:, 1]
        rows.append({"half-year": h, "fit CVEs": int(fit.sum()), "fit positives": int(y[fit].sum()),
                     "scored CVEs": int(ev.sum()), "scored positives": int(y[ev].sum())})
        print(f"  {h}: fit on {fit.sum():,} CVEs before {start:%Y-%m-%d}, scored {ev.sum():,}", flush=True)
    ok = ~np.isnan(oof)
    s, yy, hh = oof[ok], y[ok], half[ok].to_numpy()
    cal = fit_cal("platt", to_x_hgb(s), yy)
    # honest check: leave-one-half-year-out Platt -> Brier on the held-out half-year
    lo = np.zeros(len(s))
    for h in np.unique(hh):
        c = fit_cal("platt", to_x_hgb(s[hh != h]), yy[hh != h])
        lo[hh == h] = predict_cal(c, to_x_hgb(s[hh == h]))
    p_all = predict_cal(cal, to_x_hgb(s))
    base = np.full(len(yy), yy.mean())
    summ = {"oof CVEs": int(ok.sum()), "oof positives": int(yy.sum()), "base rate": float(yy.mean()),
            "Brier leave-one-half-year-out": float(brier_score_loss(yy, lo)), "Brier constant": float(brier_score_loss(yy, base)),
            "log loss LOHO": float(log_loss(yy, np.clip(lo, 1e-6, 1 - 1e-6))), "log loss constant": float(log_loss(yy, base)),
            "mean predicted (in-sample)": float(p_all.mean())}
    q = np.quantile(s, [0.5, 0.9, 0.99, 0.999])
    summ["p at OOF score quantiles 50/90/99/99.9%"] = [float(v) for v in predict_cal(cal, to_x_hgb(q))]
    joblib.dump(cal, MODELS / "calibrator_deploy.joblib")
    (MODELS / "calibrator_deploy.json").write_text(json.dumps({"method": "Platt on logit(score), time-ordered OOF half-years",
        "coef": float(cal.coef_[0][0]), "intercept": float(cal.intercept_[0]), "folds": rows, **summ,
        "role": "INDICATIVE probability for the demo; primary demo output = within-month percentile"}, indent=2))
    # reference month for the demo percentile
    deploy = joblib.load(MODELS / "deploy_hgb.joblib")
    ref = (full["published"].dt.to_period("M") == pd.Period("2026-07", "M")).to_numpy()
    pd.DataFrame({"cve_id": full.loc[ref, "cve_id"].to_numpy(), "score": deploy.predict_proba(X[ref])[:, 1]}
                 ).to_csv(MODELS / "demo_reference_2026-07.csv", index=False)
    print(json.dumps(summ, indent=1))
    print(f"reference month 2026-07: {ref.sum():,} CVEs -> outputs/models/demo_reference_2026-07.csv")


if __name__ == "__main__":
    if sys.argv[1:] == ["fit-dev"]:
        fit_dev()
    elif sys.argv[1:] == ["fit-all"]:
        fit_all()
    elif sys.argv[1:] == ["calibrate-deploy"]:
        calibrate_deploy()
    else:
        print(__doc__)
