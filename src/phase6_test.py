"""Phase 6: the ONE evaluation on test, exactly as fixed in CLAUDE.md (TEST PROTOCOL). Same code for the dry run.

  (Decision 12: headline = gradient boosting, late fusion = alternative; both come from final_model.fit_headline)
  --dry-run                 fit the recipe on TRAIN only (MLP stopping set 2024-01..06) and evaluate on VAL
                            (= 05b's val view; must reproduce fusion val@110 42.6, gradient boosting 47.1).
                            Outputs: outputs/tables/06dry_*.csv, outputs/figures/06dry_*.png, data/phase6_scores_dry.parquet
  --test --confirm-phase6   load the FROZEN dev models (outputs/models/dev_*, sha256-checked), score test_locked.parquet.
                            Needs data/embeddings/test/ (gpu_run.py embed-test) and data/epss_test.parquet.
                            Refuses to run if test outputs exist (one run only); --rerun-after-crash overrides, logged.

Everything is computed for the evaluation set as a whole and per period (quarters, years, EPSS v3/v4 split):
recall@110/@305 per month with paired bootstrap CIs (2,000, seed 0), paired differences vs the headline,
PR-AUC, recall-vs-K curve, rules at their own capacity vs the model at the same K, EPSS coverage/lag,
calibration (descriptive), seed spread, uncertain-phrase sensitivity (evaluation rows edited, models unchanged).
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
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_model as fm                                                     # noqa: E402
import text_models as txm                                                    # noqa: E402
from calibration import MergedIsotonic                                       # noqa: E402,F401 (unpickling)
from description_leak import PHRASES, phrase_flags                           # noqa: E402
from features import load_kev_raw                                            # noqa: E402
from metrics import (bootstrap_recall_at_k, ci_table, pr_auc, random_recall_at_k, recall_at_k,  # noqa: E402
                     recall_curve, rule_at_own_capacity)
from tabular_models import build_X                                           # noqa: E402

K_MAIN, K_GEN, KS = 110, 305, [110, 305]
N_BOOT = 2000
# Decision 12 (2026-11-21, pre-test): headline = gradient boosting; late fusion = reported alternative
HEAD, ALT = "gradient boosting (headline)", "late fusion (alternative)"
EPSS_V4 = pd.Timestamp("2025-03-17")
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
BLUE, ORANGE, AQUA, VIOLET, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.grid": True,
                     "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


# ------------------------------------------------------------------ data
def load_eval(mode: str):
    kev = load_kev_raw(fm.KEV_FILE)
    if mode == "dry":
        dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
        X = build_X(dev, kev)
        E = txm.load_embeddings(fm.EMB_NAME, dev)
        tr, va = dev["split"].eq("train").to_numpy(), dev["split"].eq("val").to_numpy()
        print("DRY RUN: fit on train (train-only recipe), evaluate on val", flush=True)
        models = fm.fit_headline(dev[tr], X[tr], E[tr])
        ev = dev[va].reset_index(drop=True)
        Xe, Ee = X[va].reset_index(drop=True), E[va]
        epss_path = ROOT / "data" / "epss_val.parquet"
        var_path = None                                   # val has no desc_day0_uncertain rows
    else:
        models = fm.load_models("dev")
        print(f"TEST: frozen dev models loaded, epochs {models['epochs']} (sha256 verified)", flush=True)
        ev = pd.read_parquet(ROOT / "data" / "test_locked.parquet")
        Xe = build_X(ev, kev)
        tdir = ROOT / "data" / "embeddings" / "test"
        ids = pd.read_parquet(tdir / "minilm_ids.parquet")["cve_id"]
        assert (ids.to_numpy() == ev["cve_id"].to_numpy()).all(), "test embedding rows must follow test order"
        Ee = np.load(tdir / "minilm.npy").astype(np.float32)
        epss_path = ROOT / "data" / "epss_test.parquet"
        var_path = tdir / "phrase_removed"           # all test rows with a phrase; uncertain ones picked below
    return ev, Xe, Ee, models, epss_path, var_path, kev


# ------------------------------------------------------------------ helpers
def boot_ci(df, scores, ks=KS, seed=0):
    point = {(m, k): recall_at_k(df, s, k) for m, s in scores.items() for k in ks}
    boot = bootstrap_recall_at_k(df, scores, ks, n_boot=N_BOOT, seed=seed)
    return ci_table(boot, point), boot


def fmt_ci(t, m, k):
    r = t.loc[(m, k)]
    return f"{100 * r['point']:.1f} [{100 * r['lo']:.0f}–{100 * r['hi']:.0f}]"


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    mode = "dry" if "--dry-run" in sys.argv else "test" if ("--test" in sys.argv and "--confirm-phase6" in sys.argv) else None
    if mode is None:
        sys.exit(__doc__)
    P = "06dry_" if mode == "dry" else "06_"
    if mode == "test" and (TAB / "06_recall.csv").exists() and "--rerun-after-crash" not in sys.argv:
        sys.exit("test outputs already exist: the test set is evaluated ONCE (use --rerun-after-crash only after a crash)")
    t0 = time.time()
    ev, Xe, Ee, models, epss_path, var_path, kev = load_eval(mode)
    sc = fm.score_headline(models, ev, Xe, Ee)
    ev = ev.assign(fused=sc["fused"].to_numpy(), s_hgb=sc["s_hgb"].to_numpy(), s_text=sc["s_text"].to_numpy())
    ev["phrase_day0"] = phrase_flags(ev["description_day0"], PHRASES).any(axis=1)
    if epss_path.exists():
        ep = pd.read_parquet(epss_path)
        ev = ev.merge(ep[["cve_id", "epss", "epss_percentile", "epss_date", "epss_lag_days"]], on="cve_id", how="left")
    else:
        print(f"WARNING: {epss_path.name} missing -> EPSS left out", flush=True)
        ev["epss"] = np.nan; ev["epss_lag_days"] = np.nan
    has_epss = ev["epss"].notna().any()
    n_months = ev["published"].dt.to_period("M").nunique()
    print(f"evaluation set: {len(ev):,} CVEs, {int(ev['label'].sum())} positives, {n_months} months "
          f"({ev['published'].min():%Y-%m-%d} -> {ev['published'].max():%Y-%m-%d})", flush=True)

    S = {HEAD: ev["s_hgb"], ALT: ev["fused"],
         "phrase, then CVSS (CNA)": 100 * ev["phrase_day0"] + ev["cvss_cna_score"].fillna(-1),
         "CVSS (CNA) ranking": ev["cvss_cna_score"].fillna(-1),
         "generous CVSS (fallback) ranking": ev["cvss_score"].fillna(-1)}
    if has_epss:
        S["EPSS (first score, <=7 d after pub)"] = ev["epss"].fillna(-1)        # lookahead favours EPSS
        S["EPSS (score by day 1)"] = ev["epss"].where(ev["epss_lag_days"] <= 1).fillna(-1)     # morning after pub
        S["EPSS (day-0 score only)"] = ev["epss"].where(ev["epss_lag_days"] == 0).fillna(-1)   # like-for-like

    # ---------- 1. recall@K with CIs, paired differences, PR-AUC
    ci, boot = boot_ci(ev, S)
    rows = []
    for m in S:
        rows.append({"method": m, f"recall@{K_MAIN} [95% CI]": fmt_ci(ci, m, K_MAIN),
                     f"recall@{K_GEN} [95% CI]": fmt_ci(ci, m, K_GEN),
                     f"recall@{K_MAIN}": 100 * ci.loc[(m, K_MAIN), "point"], f"recall@{K_GEN}": 100 * ci.loc[(m, K_GEN), "point"],
                     "PR-AUC": pr_auc(ev, S[m])})
    rows.append({"method": "random", f"recall@{K_MAIN}": 100 * random_recall_at_k(ev, K_MAIN),
                 f"recall@{K_GEN}": 100 * random_recall_at_k(ev, K_GEN), "PR-AUC": ev["label"].mean()})
    rec = pd.DataFrame(rows); rec.to_csv(TAB / f"{P}recall.csv", index=False)
    piv = boot.pivot_table(index=["rep", "k"], columns="method", values="recall")
    drows = []
    for m in S:
        if m == HEAD:
            continue
        for k in KS:
            d = 100 * (piv[HEAD] - piv[m]).xs(k, level="k")
            drows.append({"headline minus": m, "K": k, "diff (pts)": 100 * (ci.loc[(HEAD, k), "point"] - ci.loc[(m, k), "point"]),
                          "CI lo": d.quantile(.025), "CI hi": d.quantile(.975), "P(headline better)": (d > 0).mean()})
    diffs = pd.DataFrame(drows); diffs.to_csv(TAB / f"{P}diffs.csv", index=False)

    # ---------- 2. rules at their own capacity vs the model at the same K
    rules = {"CVSS>=9 (CNA)": ev["cvss_cna_score"] >= 9, "CVSS>=7 (CNA)": ev["cvss_cna_score"] >= 7,
             "phrase OR CVSS>=9 (CNA)": ev["phrase_day0"] | (ev["cvss_cna_score"] >= 9),
             "generous CVSS>=9 (fallback)": ev["cvss_score"] >= 9}
    orows = []
    for name, flag in rules.items():
        r = rule_at_own_capacity(ev, flag)
        k = r["patches_per_month"]
        orows.append({"rule": name, "patches/month": round(k), "rule recall %": 100 * r["recall"],
                      "rule precision %": 100 * r["precision"],
                      "headline recall % at same K": 100 * recall_at_k(ev, S[HEAD], k),
                      "late fusion recall % at same K": 100 * recall_at_k(ev, S[ALT], k)})
    own = pd.DataFrame(orows); own.to_csv(TAB / f"{P}own_capacity.csv", index=False)

    # ---------- 3. recall-vs-K curve
    ks = np.unique(np.r_[np.arange(10, 100, 10), np.arange(100, 1001, 25), K_MAIN, K_GEN])
    curve = pd.DataFrame({m: recall_curve(ev, s, ks) for m, s in S.items()}, index=ks)
    curve["random"] = [random_recall_at_k(ev, k) for k in ks]
    curve.index.name = "K per month"; curve.to_csv(TAB / f"{P}recall_vs_k.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    col = {HEAD: BLUE, ALT: AQUA, "phrase, then CVSS (CNA)": ORANGE, "CVSS (CNA) ranking": INK2,
           "generous CVSS (fallback) ranking": "#999999", "random": "#cccccc"}
    for m in curve:
        ax.plot(curve.index, 100 * curve[m], lw=2.2 if m == HEAD else 1.3, color=col.get(m, VIOLET),
                ls="--" if m in ("generous CVSS (fallback) ranking", "random") else "-", label=m)
    for k in KS:
        ax.axvline(k, color=GRID, lw=1)
    ax.set(xlabel="emergency patches per month (K)", ylabel="% of exploited CVEs caught",
           title=f"Recall vs capacity ({'val, dry run' if mode == 'dry' else 'test'})", ylim=(0, 100))
    ax.legend(fontsize=7.5, loc="lower right"); save(fig, f"{P}recall_vs_k")

    # ---------- 4. by period: quarter, year (2025 = WF 2022-24 -> 2025), EPSS v3/v4
    def period_table(groups: pd.Series, methods, label):
        prow = []
        for g, part in ev.groupby(groups):
            if part["label"].sum() == 0:
                continue
            c, _ = boot_ci(part, {m: S[m].loc[part.index] for m in methods}, ks=[K_MAIN])
            prow.append({label: str(g), "CVEs": len(part), "positives": int(part["label"].sum()),
                         "months": part["published"].dt.to_period("M").nunique(),
                         **{f"{m} @{K_MAIN}": fmt_ci(c, m, K_MAIN) for m in methods}})
        return pd.DataFrame(prow)
    pm = [HEAD, ALT, "phrase, then CVSS (CNA)"] + (["EPSS (first score, <=7 d after pub)", "EPSS (score by day 1)"] if has_epss else [])
    byq = period_table(ev["published"].dt.to_period("Q"), pm, "quarter"); byq.to_csv(TAB / f"{P}by_quarter.csv", index=False)
    byy = period_table(ev["published"].dt.year, pm, "year"); byy.to_csv(TAB / f"{P}by_year.csv", index=False)
    epss_split = period_table(np.where(ev["published"] < EPSS_V4, "before 2025-03-17 (EPSS v3)", "from 2025-03-17 (EPSS v4)"),
                              pm, "EPSS model period") if has_epss else pd.DataFrame()
    epss_split.to_csv(TAB / f"{P}epss_v3_v4.csv", index=False)

    # ---------- 5. EPSS coverage / lag
    if has_epss:
        lag = ev["epss_lag_days"].map(lambda v: "none within 7 d" if pd.isna(v) else f"{int(v)} d")
        cov = pd.crosstab(lag, ev["label"]).rename(columns={0: "negatives", 1: "positives"})
        cov["share of CVEs %"] = 100 * cov.sum(axis=1) / len(ev)
        cov.index.name = "first EPSS score at"
        cov.to_csv(TAB / f"{P}epss_coverage.csv")

    # ---------- 5b. "not yet in KEV" (TEST PROTOCOL 3): EPSS lists CISA KEV among its inputs (assumption), so an
    # EPSS score dated on/after a CVE's KEV addition may reflect the label. Per EPSS variant: keep only exploited
    # CVEs whose KEV dateAdded is STRICTLY AFTER the date of the EPSS score that variant uses (no score used ->
    # kept: EPSS cannot benefit). Excluded exploited CVEs are DROPPED from the evaluation (no credit, no slot).
    nyk = pd.DataFrame()
    if has_epss:
        kev_date = kev.groupby("cve_id")["date"].min()
        ev["kev_date"] = ev["cve_id"].map(kev_date)
        edate = pd.to_datetime(ev["epss_date"])
        variants = {"EPSS (first score, <=7 d after pub)": ev["epss_lag_days"].notna(),
                    "EPSS (score by day 1)": ev["epss_lag_days"] <= 1,
                    "EPSS (day-0 score only)": ev["epss_lag_days"] == 0}
        nrows = []
        for vname, used in variants.items():
            pos = ev["label"] == 1
            assert ev.loc[pos, "kev_date"].notna().all(), "every positive must have a KEV date"
            leak = pos & used & ~(ev["kev_date"] > edate)
            sub = ev[~leak]
            c, _ = boot_ci(sub, {m: S[m].loc[sub.index] for m in S})
            row = {"EPSS variant defining the cut": vname, "positives kept": int(sub["label"].sum()),
                   "positives dropped (KEV on/before EPSS date)": int(leak.sum())}
            for m in S:
                for k in KS:
                    row[f"{m} @{k}"] = fmt_ci(c, m, k)
            nrows.append(row)
        nyk = pd.DataFrame(nrows)
        nyk.to_csv(TAB / f"{P}not_yet_in_kev.csv", index=False)

    # ---------- 6. calibration (descriptive; calibrator fixed, never refit here)
    from calibration import predict_cal, to_x_hgb
    cal = joblib.load(ROOT / "outputs" / "models" / "calibrator_hgb.joblib")          # Decision 12: Platt on logit
    p = np.clip(predict_cal(cal, to_x_hgb(ev["s_hgb"].to_numpy())), 1e-6, 1 - 1e-6)
    ev["p_calibrated"] = p
    base = np.full(len(ev), ev["label"].mean())
    calt = pd.DataFrame([{"Brier (calibrated)": brier_score_loss(ev["label"], p), "Brier (constant base rate)": brier_score_loss(ev["label"], base),
                          "log loss (calibrated)": log_loss(ev["label"], p), "log loss (constant)": log_loss(ev["label"], base),
                          "mean predicted": p.mean(), "observed rate": ev["label"].mean(),
                          "note": "IN-SAMPLE for the dry run (val is part of the calibrator's fit data)" if mode == "dry" else "out-of-sample"}])
    calt.to_csv(TAB / f"{P}calibration.csv", index=False)
    bins = pd.cut(ev["s_hgb"].rank(pct=True), [0, 0.5, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0001], right=False)
    rel = ev.groupby(bins, observed=True).agg(n=("label", "size"), positives=("label", "sum"),
                                              predicted=("p_calibrated", "mean"), observed=("label", "mean"))
    rel.to_csv(TAB / f"{P}calibration_bins.csv")

    # ---------- 7. sensitivities: single seeds; uncertain phrases removed (evaluation rows edited)
    srows = []
    for s in fm.SEEDS:
        f = fm.fuse_seed(sc, s)
        srows.append({"text seed": str(s), f"recall@{K_MAIN}": 100 * recall_at_k(ev, pd.Series(f, index=ev.index), K_MAIN),
                      f"recall@{K_GEN}": 100 * recall_at_k(ev, pd.Series(f, index=ev.index), K_GEN)})
    seeds = pd.DataFrame(srows)
    seeds.loc[len(seeds)] = {"text seed": "mean", f"recall@{K_MAIN}": seeds[f"recall@{K_MAIN}"].mean(), f"recall@{K_GEN}": seeds[f"recall@{K_GEN}"].mean()}
    seeds.loc[len(seeds)] = {"text seed": "worst", f"recall@{K_MAIN}": seeds[f"recall@{K_MAIN}"].iloc[:5].min(), f"recall@{K_GEN}": seeds[f"recall@{K_GEN}"].iloc[:5].min()}
    seeds.loc[len(seeds)] = {"text seed": "5-seed ensemble (late fusion)", f"recall@{K_MAIN}": 100 * ci.loc[(ALT, K_MAIN), "point"],
                             f"recall@{K_GEN}": 100 * ci.loc[(ALT, K_GEN), "point"]}
    seeds.to_csv(TAB / f"{P}seeds.csv", index=False)

    unc = ev["desc_day0_uncertain"].fillna(False).to_numpy()
    sens = [{"variant": "as evaluated", "edited CVEs": 0, "edited positives": 0,
             f"gradient boosting (headline) @{K_MAIN}": 100 * ci.loc[(HEAD, K_MAIN), "point"], f"late fusion @{K_MAIN}": 100 * ci.loc[(ALT, K_MAIN), "point"]}]
    if unc.any():
        from description_leak import _phrase_names, drop_sentences_with
        inp = pd.read_parquet(Path(str(var_path) + "_input.parquet"))
        Ev = np.load(Path(str(var_path) + "_minilm.npy")).astype(np.float32)
        keep = inp["cve_id"].isin(set(ev.loc[unc, "cve_id"])).to_numpy()
        inp, Ev = inp[keep].reset_index(drop=True), Ev[keep]
        idx = pd.Series(np.arange(len(ev)), index=ev["cve_id"]).loc[inp["cve_id"]].to_numpy()
        assert set(idx) == set(np.flatnonzero(unc)), "every uncertain test CVE needs a phrase-removed embedding"
        ev2 = ev.copy(); E2 = Ee.copy()
        ev2.loc[idx, "description_day0"] = inp["text"].to_numpy(); E2[idx] = Ev
        sc2 = fm.score_headline(models, ev2, build_X(ev2, kev), E2)
        sens.append({"variant": "uncertain phrases removed (evaluation rows)", "edited CVEs": int(unc.sum()),
                     "edited positives": int(ev.loc[unc, "label"].sum()),
                     f"gradient boosting (headline) @{K_MAIN}": 100 * recall_at_k(ev2, sc2["s_hgb"], K_MAIN),
                     f"late fusion @{K_MAIN}": 100 * recall_at_k(ev2, sc2["fused"], K_MAIN)})
    pd.DataFrame(sens).to_csv(TAB / f"{P}sensitivity.csv", index=False)

    # ---------- save scores + print summary
    keep = ["cve_id", "published", "label", "fused", "s_hgb", "s_text", "p_calibrated", "epss", "epss_lag_days"]
    ev[keep].to_parquet(ROOT / "data" / f"phase6_scores_{mode}.parquet", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    fl = lambda v: f"{v:.3f}" if isinstance(v, float) else v
    print("\n== recall@K ==\n" + rec.drop(columns=[f"recall@{K_MAIN}", f"recall@{K_GEN}"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n== headline minus comparator ==\n" + diffs.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n== rules at own capacity vs model at same K ==\n" + own.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n== by quarter ==\n" + byq.to_string(index=False))
    print("\n== by year ==\n" + byy.to_string(index=False))
    if len(nyk):
        print("\n== not yet in KEV (exploited CVEs counted only if KEV dateAdded > date of the EPSS score used) ==")
        print(nyk.set_index("EPSS variant defining the cut").T.to_string())
    if has_epss:
        print("\n== EPSS v3/v4 ==\n" + epss_split.to_string(index=False))
        print("\n== EPSS coverage (lag days) ==\n" + cov.to_string(index=False))
    print("\n== calibration ==\n" + calt.T.to_string())
    print(rel.to_string())
    print("\n== text seeds ==\n" + seeds.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print("\n== sensitivity ==\n" + pd.DataFrame(sens).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
