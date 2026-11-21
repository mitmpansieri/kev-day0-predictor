"""Builds notebooks/06_test_evaluation.ipynb (Phase 6). Takeaways in src/nb06_takeaways.json.

The notebook does NOT re-run the test evaluation (that ran ONCE: src/phase6_test.py --test --confirm-phase6).
It reads the cached Phase 6 outputs (outputs/tables/06_*.csv, data/phase6_scores_test.parquet) and the
post-test exploratory outputs (src/post_test_analysis.py -> outputs/tables/06x_*.csv,
data/post_test_rolling_scores.parquet), and draws the figures.
"""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TKF = ROOT / "src" / "nb06_takeaways.json"
TK = json.loads(TKF.read_text()) if TKF.exists() else {}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
def takeaway(key):
    return md("**Takeaway**: " + TK.get(key, "_(filled after the first run)_"))

EXPLORATORY = ("> ⚠️ **POST-TEST EXPLORATORY ANALYSIS.** Designed after the test results were seen. It explains the "
               "results; it changes **no** reported Phase 6 number and no model, K, metric or comparator.")

cells = [
md("""# Phase 6 — The one evaluation on the test set (2025-01 → 2026-08)

**Question:** on the day a CVE is published, can we pick the ones attackers will exploit within 90 days better than
today's rules, at the same emergency-patching capacity?

**How this was run (TEST PROTOCOL in CLAUDE.md, fixed before the test set was opened):**
* Headline = **gradient boosting** (Decision 12, chosen on dev evidence only); alternative = **late fusion**
  (gradient boosting + text MLP). Both refit once on all of dev (2022-01 → 2024-12), frozen, sha256-checked.
* Main metric: **recall@110 per calendar month** (share of exploited CVEs caught when patching 110 CVEs a month),
  also K = 305; paired within-month bootstrap (2,000 resamples) for 95% CIs and differences.
* The test set was scored **once** (`src/phase6_test.py --test --confirm-phase6`). This notebook only reads those
  cached results; nothing here re-fits or re-scores the frozen models.
* Sections 1–6 are **pre-registered**. Sections 7–8 are **post-test exploratory** and are labelled as such."""),
code("""import os, sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt

FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
BLUE, ORANGE, AQUA, VIOLET, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9.5})
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
def T(name): return pd.read_csv(TAB / f"{name}.csv")
SHORT = {"gradient boosting (headline)": "gradient\\nboosting", "late fusion (alternative)": "late\\nfusion",
         "phrase, then CVSS (CNA)": "phrase, then\\nCVSS (CNA)", "CVSS (CNA) ranking": "CVSS (CNA)\\nranking",
         "EPSS (first score, <=7 d after pub)": "EPSS\\n(<= 7 d)"}

HEAD, ALT = "gradient boosting (headline)", "late fusion (alternative)"
EPSS7, EPSS1, EPSS0 = "EPSS (first score, <=7 d after pub)", "EPSS (score by day 1)", "EPSS (day-0 score only)"
COL = {HEAD: BLUE, ALT: AQUA, "phrase, then CVSS (CNA)": ORANGE, "CVSS (CNA) ranking": INK2,
       "generous CVSS (fallback) ranking": "#999999", EPSS7: VIOLET, EPSS1: "#a196e8", EPSS0: "#d2cdf4", "random": "#cccccc"}
sc = pd.read_parquet(ROOT / "data" / "phase6_scores_test.parquet")
months = sc["published"].dt.to_period("M")
print(f"test: {len(sc):,} CVEs, {int(sc['label'].sum())} exploited within 90 days ({sc['label'].mean():.2%}), "
      f"{months.nunique()} months ({sc['published'].min():%Y-%m-%d} -> {sc['published'].max():%Y-%m-%d})")
print(f"CVEs per month: {len(sc) / months.nunique():,.0f} on average (dev: 2,607) -> K = 110 is "
      f"{100 * 110 / (len(sc) / months.nunique()):.1f}% of a test month (dev: 4.2%)")"""),

md("""## 1. Main result: recall at fixed capacity
Each method ranks every month's CVEs; we patch the top K and count the exploited CVEs caught. EPSS is shown three
ways: the first score within 7 days of publication (**lookahead: favours EPSS**, our models use day-0 information
only), the score available by day 1, and day-0 scores only (like-for-like)."""),
code("""rec = T("06_recall").set_index("method")
show = rec[["recall@110 [95% CI]", "recall@305 [95% CI]", "PR-AUC"]].copy()
show.loc["random", ["recall@110 [95% CI]", "recall@305 [95% CI]"]] = [f"{rec.loc['random', 'recall@110']:.1f}", f"{rec.loc['random', 'recall@305']:.1f}"]
display(show.round({"PR-AUC": 3}))
d = T("06_diffs")
d["difference [95% CI]"] = [f"{r['diff (pts)']:+.1f} [{r['CI lo']:+.1f}, {r['CI hi']:+.1f}]" for _, r in d.iterrows()]
d["P(headline better)"] = (100 * d["P(headline better)"]).round(0).astype(int).astype(str) + "%"
display(d.pivot(index="headline minus", columns="K", values=["difference [95% CI]", "P(headline better)"]))

order = [HEAD, ALT, EPSS7, EPSS1, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", "generous CVSS (fallback) ranking", EPSS0]
ci = rec.loc[order, "recall@110 [95% CI]"].str.extract(r"([\\d.]+) \\[(\\d+)–(\\d+)\\]").astype(float)
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(order))[::-1]
ax.barh(y, ci[0], color=[COL[m] for m in order], height=0.62)
ax.errorbar(ci[0], y, xerr=[ci[0] - ci[1], ci[2] - ci[0]], fmt="none", ecolor=INK, lw=1, capsize=3)
for yi, v in zip(y, ci[0]): ax.annotate(f"{v:.1f}%", (v, yi), xytext=(22, -3), textcoords="offset points", fontsize=8.5)
ax.set_yticks(y, order); ax.grid(axis="y", visible=False); ax.set_xlim(0, 60)
ax.set_xlabel("% of exploited CVEs caught at 110 patches / month (95% CI)")
ax.set_title("Test set (2025-01 → 2026-08): recall@110", loc="left")
save(fig, "06_01_recall_at_110")"""),
takeaway("main"),

md("""## 2. Same capacity as today's rules
Each rule uses its **own** test-period capacity (how many CVEs it flags per month); the models get exactly that
many patches."""),
code("""own = T("06_own_capacity").set_index("rule")
display(own.round(1))"""),
takeaway("own"),

md("""## 3. Recall vs capacity on test
The full curve, so no single K drives the conclusion (K = 110 and 305 marked; fixed in Phase 4 from train)."""),
code("""curve = T("06_recall_vs_k").set_index("K per month")
fig, ax = plt.subplots(figsize=(9, 4.6))
for m in [HEAD, ALT, EPSS7, EPSS0, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", "generous CVSS (fallback) ranking", "random"]:
    ax.plot(curve.index, 100 * curve[m], color=COL[m], lw=2.4 if m == HEAD else 1.4,
            ls="--" if m in ("generous CVSS (fallback) ranking", "random", EPSS0) else "-", label=m)
for k in (110, 305):
    ax.axvline(k, color=INK, lw=0.8, ls=":"); ax.annotate(f"K={k}", (k, 2), fontsize=8, xytext=(3, 0), textcoords="offset points")
ax.set(xlim=(0, 1000), ylim=(0, 100), xlabel="emergency patches per month (K)", ylabel="% of exploited CVEs caught")
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_title("Test: recall vs capacity", loc="left")
save(fig, "06_02_recall_vs_k")
display((100 * curve.loc[[50, 110, 200, 305, 500, 1000], [HEAD, ALT, EPSS7, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking"]]).round(1))"""),
takeaway("curve"),

md("""## 4. The EPSS comparison and its caveats
EPSS (FIRST) is the industry benchmark, but the comparison is not like-for-like:
1. **Lookahead.** Most CVEs get their first EPSS score the day *after* NVD publication; the ≤ 7-day line uses it anyway.
2. **Different target.** EPSS predicts exploitation activity in the next 30 days, not KEV listing within 90 days.
3. **Model change inside test.** EPSS v4 replaced v3 on 2025-03-17.
4. **EPSS may use the label.** *Assumption (FIRST documentation not reachable from the sandbox): EPSS v3/v4 list
   CISA KEV among their data sources, so an EPSS score dated after a CVE's KEV addition may reflect the label.*
   → Section 5."""),
code("""cov = T("06_epss_coverage").set_index("first EPSS score at")
cov["share of positives %"] = 100 * cov["positives"] / cov["positives"].sum()
display(cov.round(1))
display(T("06_epss_v3_v4").set_index("EPSS model period"))"""),
takeaway("epss"),

md("""## 5. "Not yet in KEV": removing the label leak from the EPSS comparison (pre-registered)
For each EPSS line, an exploited CVE counts **only if CISA added it to KEV strictly after the date of the EPSS
score that line uses**. Exploited CVEs already in KEV by then are dropped from the evaluation for **every** method
(no credit, no capacity slot), so all methods are scored on the same CVEs."""),
code("""nyk = T("06_not_yet_in_kev").set_index("EPSS variant defining the cut")
methods = [HEAD, ALT, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", "generous CVSS (fallback) ranking", EPSS7, EPSS1, EPSS0]
tbl = pd.DataFrame({cut: {m: nyk.loc[cut, f"{m} @110"] for m in methods} for cut in [EPSS7, EPSS0]})
tbl.columns = [f"@110, cut by {c} ({nyk.loc[c, 'positives kept']} kept, {nyk.loc[c, 'positives dropped (KEV on/before EPSS date)']} dropped)" for c in tbl.columns]
tbl["all exploited CVEs (Section 1)"] = rec.loc[methods, "recall@110 [95% CI]"]
display(tbl)

pt = lambda s: float(s.split()[0])
plot_m = [HEAD, ALT, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", EPSS7]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
x = np.arange(len(plot_m)); w = 0.38
a = [pt(rec.loc[m, "recall@110 [95% CI]"]) for m in plot_m]; b = [pt(nyk.loc[EPSS7, f"{m} @110"]) for m in plot_m]
ax.bar(x - w/2, a, w, color=[COL[m] for m in plot_m], alpha=0.45, label="all 294 exploited CVEs")
ax.bar(x + w/2, b, w, color=[COL[m] for m in plot_m], label="185 not yet in KEV when EPSS scored them")
for xi, (u, v) in enumerate(zip(a, b)):
    ax.annotate(f"{u:.1f}", (xi - w/2, u), ha="center", xytext=(0, 3), textcoords="offset points", fontsize=8)
    ax.annotate(f"{v:.1f}", (xi + w/2, v), ha="center", xytext=(0, 3), textcoords="offset points", fontsize=8, weight="bold")
ax.set_xticks(x, [SHORT[m] for m in plot_m], fontsize=8); ax.grid(axis="x", visible=False)
ax.set_ylabel("recall@110 (%)"); ax.set_ylim(0, 45); ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Recall@110 before / after removing CVEs already in KEV", loc="left")
save(fig, "06_03_not_yet_in_kev")"""),
takeaway("nyk"),

md("""## 6. Robustness (pre-registered): walk-forward 2022-24 → 2025, quarters, seeds, calibration
The headline was trained on all of dev (2022–2024), so **test year 2025 is the walk-forward fold 2022-24 → 2025**;
2026 is a further 8 months out."""),
code("""byy = T("06_by_year").set_index("year")
display(byy)
byq = T("06_by_quarter").set_index("quarter")
display(byq)
seeds = T("06_seeds").set_index("text seed"); display(seeds.round(1))"""),
takeaway("robust"),

md("""### Calibration of the headline probability (descriptive)
The calibrator (Platt on logit of the gradient-boosting score, Decision 12) was fit on dev's pooled out-of-time
scores and is **not refit on test**. Bins = percentiles of the score."""),
code("""cal = T("06_calibration"); display(cal.T)
bins = T("06_calibration_bins").rename(columns={"s_hgb": "score percentile bin"})
display(bins.assign(predicted=lambda d: (100 * d["predicted"]).round(2), observed=lambda d: (100 * d["observed"]).round(2)))
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([0, 8], [0, 8], color=GRID, lw=1.5, ls="--", label="perfect calibration")
ax.plot(100 * bins["predicted"], 100 * bins["observed"], "o-", color=BLUE, lw=2, label="headline (Platt), test")
r = bins.iloc[-1]
ax.annotate(f"top 0.5% of scores (n={r['n']:,})", (100 * r["predicted"], 100 * r["observed"]), fontsize=8, xytext=(6, -12), textcoords="offset points", color=INK2)
ax.set(xlabel="mean predicted probability (%)", ylabel="observed exploited share (%)", xlim=(0, 8), ylim=(0, 8))
ax.legend(frameon=False, loc="lower right"); ax.set_title("Reliability on test", loc="left")
save(fig, "06_04_calibration")"""),
takeaway("calibration"),

md("## 7. Why does recall fall in 2026? (post-test exploratory)\n" + EXPLORATORY + """

Three questions: does **everyone** drop (or only our model)? Is it **volume** (a fixed K = 110 covering a shrinking
share of each month)? Did the **mix** of exploited CVEs change (new vendors, vendors without KEV history)?
Computed by `src/post_test_analysis.py` (log `outputs/logs/step9_post_test.log`)."""),
code("""dy = T("06x_drift_by_year").set_index("year"); display(dy.T)
vol = T("06x_drift_volume").set_index("period"); display(vol.round(1))
cs = T("06x_drift_constant_share").set_index("year"); display(cs.T)

ms = [HEAD, ALT, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", EPSS7]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, (title, get) in zip(axes, [("Fixed K = 110 / month (as reported)", lambda m, y: pt(dy.loc[y, f"{m} @110"])),
                                   ("Constant share: 4.2% of each month (= 110 in a dev month)", lambda m, y: cs.loc[y, m])]):
    x = np.arange(len(ms)); w = 0.38
    for j, (yr, alpha) in enumerate([(2025, 0.5), (2026, 1.0)]):
        v = [get(m, yr) for m in ms]
        ax.bar(x + (j - 0.5) * w, v, w, color=[COL[m] for m in ms], alpha=alpha, label=str(yr))
        for xi, vi in zip(x, v): ax.annotate(f"{vi:.0f}", (xi + (j - 0.5) * w, vi), ha="center", xytext=(0, 2), textcoords="offset points", fontsize=7.5)
    ax.set_xticks(x, [SHORT[m] for m in ms], fontsize=8); ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=10)
axes[0].set_ylabel("recall (%)"); axes[0].legend(frameon=False, title="light = 2025, dark = 2026", fontsize=8)
save(fig, "06_05_drift_by_year")"""),
takeaway("drift_year"),
code("""ranks = T("06x_drift_rank_of_exploited"); display(ranks.pivot(index="method", columns="year").round(1))
mix = T("06x_drift_vendor_mix").set_index("year"); display(mix.T.round(1))
vend = T("06x_drift_vendors")
for yr in (2025, 2026):
    print(f"top vendors of exploited CVEs, {yr}:"); display(vend[vend["year"] == yr].head(10).set_index("vendor"))"""),
takeaway("drift_mix"),

md("## 8. Would retraining every quarter help? (post-test exploratory)\n" + EXPLORATORY + """

At the start of each test quarter we refit gradient boosting **with the frozen settings** (no tuning) on every CVE
whose 90-day label window had **closed** by that day (published more than 90 days before the quarter start; dev +
earlier test rows), then score that quarter. This is what a team retraining quarterly would have had. Note the
frozen model saw all of dev, including Oct–Dec 2024 CVEs whose labels were not yet knowable on 2025-01-01; the
rolling model respects the delay, so in 2025Q1 it has *less* data, and in 2025Q2 it trains on exactly dev (same
model as frozen: a built-in check)."""),
code("""roll = T("06x_rolling_retrain")
q = roll[roll["period"].str.contains("Q")].reset_index(drop=True)
display(roll[["period", "positives", "frozen @110", "rolling @110", "rolling - frozen (pts)", "diff CI", "P(rolling better)",
              "train cut (published before)", "train CVEs", "train positives", "of which test-period positives"]].set_index("period"))
fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(len(q))
for name, c, off in [("frozen", BLUE, -0.08), ("rolling", ORANGE, 0.08)]:
    ax.errorbar(x + off, q[name], yerr=[q[name] - q[f"{name} lo"], q[f"{name} hi"] - q[name]], color=c, fmt="o-", lw=2, capsize=3,
                label="frozen dev model (as reported)" if name == "frozen" else "rolling quarterly retrain (exploratory)")
for xi, n in zip(x, q["positives"]): ax.annotate(f"{n} exploited", (xi, 1), ha="center", fontsize=7.5, color=INK2)
ax.set_xticks(x, q["period"]); ax.set_ylim(0, 70); ax.set_ylabel("recall@110 (%) with 95% CI")
ax.legend(frameon=False, loc="upper right"); ax.set_title("Gradient boosting by test quarter: frozen vs retrained each quarter", loc="left")
save(fig, "06_06_rolling_retrain")"""),
takeaway("rolling"),

md("""## 9. Limitations
* **CNA CVSS is not fully day-0** (pre-test check 7a): CNAs later changed their own vector for 5.5% of exploited vs
  1.9% of other CVEs; 2 exploited CVEs crossed 9.0 only later. Using the first known vector cost ~0.8 points on dev
  (a lower bound). Description edits are label-correlated too (Decision 10e), though re-scoring with the earliest
  text did not inflate results on dev.
* **Uncertain day-0 phrases** (check 7b): on dev, removing them cost the fusion 2.0 points and helped gradient boosting
  (+0.4). Test has **no** uncertain rows (57 phrases verified at publication, 6 removed), so the test check was N/A.
* **EPSS comparison**: lookahead favours EPSS; different target (30-day activity); v3 → v4 switch inside test; EPSS is
  *assumed* to use KEV (documentation not reachable) → Section 5.
* **Label = KEV listing**, which is CISA's selection of exploited CVEs (US-government-relevant products, evidence
  required); exploitation never reported to CISA counts as "not exploited".
* **Drift**: recall fell to 23.9% in 2026 (Section 7); K = 110 was fixed from 2022–24 volumes and covers 1.6% of a 2026 month.
* **Small numbers**: 294 exploited CVEs over 20 months; quarterly CIs are ±15–20 points.
* **Calibration** degrades out of time: average over-prediction as the base rate fell, under-prediction in the top 0.5%."""),

md("## 10. Side-by-side: every method, validation (dry run, train-only fit) and test"),
code("""dry = T("06dry_recall").set_index("method")
methods_all = [HEAD, ALT, EPSS7, EPSS1, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", "generous CVSS (fallback) ranking", EPSS0, "random"]
kind = {HEAD: "model (headline)", ALT: "model (alternative)", EPSS7: "benchmark (lookahead)", EPSS1: "benchmark",
        EPSS0: "benchmark (like-for-like)", "phrase, then CVSS (CNA)": "day-0 rule (best)", "CVSS (CNA) ranking": "day-0 rule (today)",
        "generous CVSS (fallback) ranking": "rule with hindsight", "random": "reference"}
rows = []
for m in methods_all:
    r = {"method": m, "type": kind[m]}
    if m == "random":
        r.update({"val recall@110": f"{dry.loc[m, 'recall@110']:.1f}", "test recall@110 [95% CI]": f"{rec.loc[m, 'recall@110']:.1f}",
                  "test recall@305 [95% CI]": f"{rec.loc[m, 'recall@305']:.1f}"})
    else:
        r.update({"val recall@110": dry.loc[m, "recall@110 [95% CI]"], "test recall@110 [95% CI]": rec.loc[m, "recall@110 [95% CI]"],
                  "test recall@305 [95% CI]": rec.loc[m, "recall@305 [95% CI]"],
                  "test 2025 @110": dy.loc[2025, f"{m} @110"] if f"{m} @110" in dy else "",
                  "test 2026 @110": dy.loc[2026, f"{m} @110"] if f"{m} @110" in dy else "",
                  "not yet in KEV @110": nyk.loc[EPSS7, f"{m} @110"]})
    r["test PR-AUC"] = round(rec.loc[m, "PR-AUC"], 3)
    rows.append(r)
rows.append({"method": "gradient boosting, rolling quarterly retrain", "type": "exploratory (post-test)",
             "test recall@110 [95% CI]": roll.set_index("period").loc["all test", "rolling @110"],
             "test 2025 @110": roll.set_index("period").loc["2025", "rolling @110"], "test 2026 @110": roll.set_index("period").loc["2026", "rolling @110"]})
final = pd.DataFrame(rows).set_index("method").astype(object).fillna("")
display(final); final.to_csv(TAB / "06_final_comparison.csv")"""),
takeaway("final"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "notebooks" / "06_test_evaluation.ipynb")
print("written")
