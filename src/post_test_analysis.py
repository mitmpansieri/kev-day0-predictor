"""POST-TEST EXPLORATORY ANALYSIS (after Phase 6; changes NO reported Phase 6 number).

Everything here was decided AFTER the test results were seen, so it is exploratory: it explains the results,
it does not replace them. Outputs are prefixed 06x_ to keep them apart from the pre-registered 06_ tables.

  1. drift   : 2025 vs 2026 for the headline, the fusion, the best rule, CVSS and EPSS side by side; K as a share of
               monthly CVEs; recall at a CONSTANT share of each month (the dev share, 110 / dev CVEs per month);
               where exploited CVEs land in the monthly ranking; vendor / KEV-history mix of exploited CVEs.
  2. rolling : at the start of each test quarter, refit gradient boosting (frozen 05a settings, no tuning) on all
               CVEs whose 90-day label window had CLOSED by then (published < quarter start - 90 days; dev + test),
               score that quarter; compare with the frozen dev model (paired bootstrap).
Run: .venv/bin/python src/post_test_analysis.py      (~5-10 min CPU; log outputs/logs/step9_post_test.log)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_model as fm                                                     # noqa: E402
from description_leak import PHRASES, phrase_flags                           # noqa: E402
from features import load_kev_raw, vendor_or_cna                             # noqa: E402
from metrics import (bootstrap_recall_at_k, ci_table, recall_at_budget, recall_at_k,  # noqa: E402
                     random_recall_at_k)
from tabular_models import build_X, make_hgb                                 # noqa: E402

K, N_BOOT, LABEL_DELAY = 110, 2000, pd.Timedelta(days=90)
TAB = ROOT / "outputs" / "tables"
HEAD, ALT, RULE = "gradient boosting (headline)", "late fusion (alternative)", "phrase, then CVSS (CNA)"
EPSS7, EPSS1 = "EPSS (first score, <=7 d after pub)", "EPSS (score by day 1)"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s)
    def flush(self):
        self.o.flush(); self.f.flush()


def ci_fmt(df, scores, k=K):
    point = {(m, k): recall_at_k(df, s, k) for m, s in scores.items()}
    boot = bootstrap_recall_at_k(df, scores, [k], n_boot=N_BOOT, seed=0)
    return ci_table(boot, point), boot


def f(c, m, k=K):
    r = c.loc[(m, k)]
    return f"{100 * r['point']:.1f} [{100 * r['lo']:.0f}–{100 * r['hi']:.0f}]"


def month_pct(df, s):
    """Within-month percentile rank (1 = top of the month)."""
    return s.groupby(df["published"].dt.to_period("M")).rank(pct=True)


def main():
    t0 = time.time()
    kev = load_kev_raw(fm.KEV_FILE)
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    test = pd.read_parquet(ROOT / "data" / "test_locked.parquet")
    sc = pd.read_parquet(ROOT / "data" / "phase6_scores_test.parquet")
    assert (sc["cve_id"].to_numpy() == test["cve_id"].to_numpy()).all()
    ev = test.assign(**{c: sc[c].to_numpy() for c in ["s_hgb", "fused", "epss", "epss_lag_days"]})
    ev["phrase_day0"] = phrase_flags(ev["description_day0"], PHRASES).any(axis=1)
    ev["year"] = ev["published"].dt.year
    S = {HEAD: ev["s_hgb"], ALT: ev["fused"], RULE: 100 * ev["phrase_day0"] + ev["cvss_cna_score"].fillna(-1),
         "CVSS (CNA) ranking": ev["cvss_cna_score"].fillna(-1), EPSS7: ev["epss"].fillna(-1),
         EPSS1: ev["epss"].where(ev["epss_lag_days"] <= 1).fillna(-1)}

    # ------------------------------------------------------------ 1a. recall by year, all methods side by side
    dev_months = dev["published"].dt.to_period("M").nunique()
    share_dev = K / (len(dev) / dev_months)                        # K as a share of an average dev month
    rows = []
    for y, part in ev.groupby("year"):
        c, _ = ci_fmt(part, {m: s.loc[part.index] for m, s in S.items()})
        r = {"year": y, "positives": int(part["label"].sum())}
        r.update({f"{m} @110": f(c, m) for m in S})
        r["random @110"] = f"{100 * random_recall_at_k(part, K):.1f}"
        rows.append(r)
    by_year = pd.DataFrame(rows); by_year.to_csv(TAB / "06x_drift_by_year.csv", index=False)

    # 1b. capacity vs volume
    vol = []
    for name, part in [("dev 2022-24", dev)] + [(str(y), p) for y, p in ev.groupby("year")]:
        mo = part["published"].dt.to_period("M").nunique()
        vol.append({"period": name, "months": mo, "CVEs / month": len(part) / mo,
                    "exploited / month": part["label"].sum() / mo,
                    "K=110 as % of monthly CVEs": 100 * K / (len(part) / mo),
                    "CVSS>=9 (CNA) / month": (part["cvss_cna_score"] >= 9).sum() / mo,
                    "CVEs from vendors with earlier KEV entries / month": np.nan})
    volume = pd.DataFrame(vol)

    # 1c. recall at a CONSTANT SHARE of each month (= share_dev), to separate the volume effect
    rows = []
    for y, part in ev.groupby("year"):
        r = {"year": y, "budget": f"{100 * share_dev:.2f}% of each month's CVEs (= 110 in an average dev month)",
             "patches / month (avg)": share_dev * len(part) / part["published"].dt.to_period("M").nunique()}
        r.update({f"{m}": 100 * recall_at_budget(part, s.loc[part.index], share_dev) for m, s in S.items()})
        rows.append(r)
    const_share = pd.DataFrame(rows); const_share.to_csv(TAB / "06x_drift_constant_share.csv", index=False)

    # 1d. where do exploited CVEs land in the monthly ranking?
    rows = []
    for m, s in S.items():
        pct = month_pct(ev, s)
        for y in sorted(ev["year"].unique()):
            p = pct[(ev["year"] == y) & (ev["label"] == 1)]
            rows.append({"method": m, "year": y, "median percentile of exploited CVEs": 100 * p.median(),
                         "share of exploited in top 1% of month": 100 * (p > 0.99).mean(),
                         "share in top 5%": 100 * (p > 0.95).mean()})
    ranks = pd.DataFrame(rows); ranks.to_csv(TAB / "06x_drift_rank_of_exploited.csv", index=False)

    # 1e. vendor / KEV-history mix of exploited CVEs, 2025 vs 2026 (and the competition for slots)
    X_te = build_X(ev, kev)
    ev["vendor_key"] = vendor_or_cna(ev)
    ev["vendor_kev_any"] = X_te["vendor_kev_any"].to_numpy()
    dev_v = vendor_or_cna(dev)
    seen, seen_pos = set(dev_v), set(dev_v[dev["label"] == 1])
    ev["vendor_in_dev"] = ev["vendor_key"].isin(seen)
    ev["vendor_exploited_in_dev"] = ev["vendor_key"].isin(seen_pos)
    top110 = pd.Series(False, index=ev.index)
    for _, idx in ev.groupby(ev["published"].dt.to_period("M")).indices.items():
        top110.iloc[idx[np.argsort(-ev["s_hgb"].to_numpy()[idx], kind="stable")[:K]]] = True
    ev["in_top110"] = top110
    mix = []
    for y, part in ev.groupby("year"):
        pos = part[part["label"] == 1]
        mo = part["published"].dt.to_period("M").nunique()
        mix.append({"year": y, "exploited": len(pos),
                    "% vendor seen in dev": 100 * pos["vendor_in_dev"].mean(),
                    "% vendor exploited in dev": 100 * pos["vendor_exploited_in_dev"].mean(),
                    "% vendor had KEV entry before publication": 100 * pos["vendor_kev_any"].mean(),
                    "% with CNA CVSS": 100 * pos["cvss_cna_score"].notna().mean(),
                    "% CNA CVSS >= 9": 100 * (pos["cvss_cna_score"] >= 9).mean(),
                    "% exploitation phrase day 0": 100 * pos["phrase_day0"].mean(),
                    "% vendor missing (CNA used)": 100 * pos["vendor_key"].str.startswith("cna:").mean(),
                    "headline catch rate, vendor had KEV entry %": 100 * pos.loc[pos["vendor_kev_any"] == 1, "in_top110"].mean(),
                    "headline catch rate, vendor without KEV entry %": 100 * pos.loc[pos["vendor_kev_any"] == 0, "in_top110"].mean(),
                    "ALL CVEs from vendors with KEV entry / month": (part["vendor_kev_any"] == 1).sum() / mo})
    mix = pd.DataFrame(mix); mix.to_csv(TAB / "06x_drift_vendor_mix.csv", index=False)
    volume.loc[1:, "CVEs from vendors with earlier KEV entries / month"] = mix["ALL CVEs from vendors with KEV entry / month"].to_numpy()
    X_dev_any = build_X(dev, kev)["vendor_kev_any"]
    volume.loc[0, "CVEs from vendors with earlier KEV entries / month"] = X_dev_any.sum() / dev_months
    volume.to_csv(TAB / "06x_drift_volume.csv", index=False)

    vrows = []
    for y, part in ev[ev["label"] == 1].groupby("year"):
        for v, g in part.groupby("vendor_key"):
            vrows.append({"year": y, "vendor": v, "exploited": len(g), "caught by headline @110": int(g["in_top110"].sum()),
                          "vendor seen in dev": bool(g["vendor_in_dev"].iloc[0]),
                          "vendor exploited in dev": bool(g["vendor_exploited_in_dev"].iloc[0])})
    vend = pd.DataFrame(vrows).sort_values(["year", "exploited"], ascending=[True, False])
    vend.to_csv(TAB / "06x_drift_vendors.csv", index=False)

    # ------------------------------------------------------------ 2. rolling retrain simulation
    full = pd.concat([dev, test], ignore_index=True)
    X_full = build_X(full, kev)
    y_full = full["label"].to_numpy()
    n_dev = len(dev)
    frozen = joblib.load(ROOT / "outputs" / "models" / "dev_hgb.joblib")
    chk = frozen.predict_proba(X_full.iloc[n_dev:])[:, 1]
    assert np.allclose(chk, ev["s_hgb"].to_numpy()), "feature rebuild must reproduce the frozen test scores"
    print("check: frozen model on rebuilt test features reproduces the Phase 6 scores exactly", flush=True)

    rolling = np.full(len(test), np.nan)
    q_test = test["published"].dt.to_period("Q")
    info = []
    for q in sorted(q_test.unique()):
        start = q.start_time
        fit = (full["published"] < start - LABEL_DELAY).to_numpy()
        # sanity: every label used is knowable at `start` (KEV addition, if any within 90 d, happened before start)
        known = full.loc[fit & (y_full == 1), "kev_date_added"]
        assert (pd.to_datetime(known) < start).all()
        t = time.time()
        m = make_hgb(**fm.HGB_BEST).fit(X_full[fit], y_full[fit])
        sel = (q_test == q).to_numpy()
        rolling[sel] = m.predict_proba(X_full.iloc[n_dev:][sel])[:, 1]
        info.append({"quarter": str(q), "train cut (published before)": f"{start - LABEL_DELAY:%Y-%m-%d}",
                     "train CVEs": int(fit.sum()), "train positives": int(y_full[fit].sum()),
                     "of which test-period positives": int(y_full[fit][n_dev:].sum()) if fit[n_dev:].any() else 0})
        print(f"  {q}: fit on {fit.sum():,} CVEs / {int(y_full[fit].sum())} pos (published < "
              f"{start - LABEL_DELAY:%Y-%m-%d}) in {time.time() - t:.0f}s", flush=True)
    info = pd.DataFrame(info)
    info["of which test-period positives"] = [int(((full["published"] < pd.Period(q).start_time - LABEL_DELAY).to_numpy()
                                                  [n_dev:] & (y_full[n_dev:] == 1)).sum()) for q in info["quarter"]]
    ev["s_rolling"] = rolling
    R = {"frozen (dev model)": ev["s_hgb"], "rolling quarterly retrain": ev["s_rolling"]}

    def roll_row(name, part):
        c, boot = ci_fmt(part, {m: s.loc[part.index] for m, s in R.items()})
        piv = boot.pivot_table(index="rep", columns="method", values="recall")
        d = 100 * (piv["rolling quarterly retrain"] - piv["frozen (dev model)"])
        return {"period": name, "CVEs": len(part), "positives": int(part["label"].sum()),
                "frozen @110": f(c, "frozen (dev model)"), "rolling @110": f(c, "rolling quarterly retrain"),
                "frozen": 100 * c.loc[("frozen (dev model)", K), "point"], "rolling": 100 * c.loc[("rolling quarterly retrain", K), "point"],
                "frozen lo": 100 * c.loc[("frozen (dev model)", K), "lo"], "frozen hi": 100 * c.loc[("frozen (dev model)", K), "hi"],
                "rolling lo": 100 * c.loc[("rolling quarterly retrain", K), "lo"], "rolling hi": 100 * c.loc[("rolling quarterly retrain", K), "hi"],
                "rolling - frozen (pts)": 100 * (c.loc[("rolling quarterly retrain", K), "point"] - c.loc[("frozen (dev model)", K), "point"]),
                "diff CI": f"[{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}]", "P(rolling better)": (d > 0).mean()}

    rrows = [roll_row(str(q), ev[q_test == q]) for q in sorted(q_test.unique())]
    rrows += [roll_row(str(y), p) for y, p in ev.groupby("year")] + [roll_row("all test", ev)]
    roll = pd.DataFrame(rrows).merge(info, left_on="period", right_on="quarter", how="left").drop(columns="quarter")
    roll.to_csv(TAB / "06x_rolling_retrain.csv", index=False)
    ev[["cve_id", "published", "label", "s_hgb", "s_rolling"]].to_parquet(ROOT / "data" / "post_test_rolling_scores.parquet", index=False)

    # ------------------------------------------------------------ print
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    fl = lambda v: f"{v:.1f}"
    print("\n== 1a. recall@110 by year ==\n" + by_year.set_index("year").T.to_string())
    print("\n== 1b. volume vs capacity ==\n" + volume.to_string(index=False, float_format=fl))
    print("\n== 1c. recall at a constant share of each month ==\n" + const_share.set_index("year").T.to_string(float_format=fl))
    print("\n== 1d. rank of exploited CVEs (within-month percentile) ==\n" + ranks.to_string(index=False, float_format=fl))
    print("\n== 1e. vendor / KEV mix of exploited CVEs ==\n" + mix.set_index("year").T.to_string(float_format=fl))
    for y in sorted(vend["year"].unique()):
        print(f"\n top vendors {y}:\n" + vend[vend["year"] == y].head(12).to_string(index=False))
    print("\n== 2. rolling retrain ==\n" + roll[["period", "positives", "frozen @110", "rolling @110", "rolling - frozen (pts)",
                                                 "diff CI", "P(rolling better)", "train CVEs", "train positives",
                                                 "of which test-period positives"]].to_string(index=False, float_format=fl))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout = Tee(ROOT / "outputs" / "logs" / "step9_post_test.log")
    main()
