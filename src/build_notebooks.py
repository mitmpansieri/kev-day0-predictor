"""Builds the 5 notebooks of this repo (notebooks/1_data ... 5_ciso_tool) from the repo's own small files.

  .venv/bin/python src/build_notebooks.py                 # writes notebooks/*.ipynb
  .venv/bin/python src/run_notebook.py notebooks/1_data.ipynb   # executes one (or open in Jupyter and Run All)

Notebooks 1-4 read only outputs/tables/*.csv, outputs/models/*.json, data/raw/kev_*.csv and data/shap_test.parquet.
Notebook 5 runs ciso_tool.py (deployment model; NVD reads are cached in data/nvd_tool_cache/).
"""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
def tk(text): return md("**Takeaway**: " + text)
POST = "> ⚠️ **Post-test exploratory analysis**: designed after the test results were seen; it explains them and changes no reported number."

SETUP = """import os, sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl")); os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, AQUA, VIOLET, RED, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#d64545", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9.5})
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 90)
def T(name, **kw): return pd.read_csv(TAB / f"{name}.csv", **kw)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
pt = lambda s: float(str(s).split()[0])
HEAD, ALT, EPSS7 = "gradient boosting (headline)", "late fusion (alternative)", 'EPSS (first score, <=7 d after pub)'"""

NAV = ("Notebooks: [1 data](1_data.ipynb) · [2 train & validate](2_train_validate.ipynb) · [3 test](3_test.ipynb) · "
       "[4 results for the CISO](4_results_ciso.ipynb) · [5 CISO tool](5_ciso_tool.ipynb) · detailed appendices in `appendix/`.")

# =====================================================================================================  1 DATA
nb1 = [
md(f"""# 1 · Data: collection, the evolution of cybersecurity, and the leakage hunt
**Goal of the project:** on the day a CVE is published, rank it by the chance attackers exploit it within 90 days
(= CISA adds it to the **KEV** catalogue within 90 days), so a vulnerability team spends its limited **K emergency
patches a month** where it matters. **Stakeholder:** Head of Vulnerability Management → CISO.
**Current policy (assumed):** emergency-patch every CVE with CVSS ≥ 9 — common severity-based practice, *not verified for
a specific organisation*; notebooks 2–3 therefore compare against several rules and every capacity K.

{NAV}"""),
code(SETUP),
md("""## 1.1 Data collection
| source | what | how | snapshot |
|---|---|---|---|
| **NVD API 2.0** | every CVE published since 2022-01-01 (CNA, description, CWE, vendor/product, CVSS with the source of each score) | `src/build_cve_dataset.py`, 90-day windows, paging, retries, cache | 2026-11-20 |
| **NVD change history** | first recorded version of descriptions / CVSS (audit sample) | `src/nvd_history.py` | 2026-11-20 |
| **CISA KEV** | label (dateAdded), vendor KEV history (entries **before** publication only), ransomware flag (evaluation only) | one CSV | `data/raw/kev_20261120.csv` |
| **FIRST EPSS** | benchmark only | `src/epss.py` | 2026-11-21 |

Everything is re-downloadable with `scripts/download_all.sh`. **Label:** KEV `dateAdded` − NVD publication ≤ 90 days
(KEV before publication counts). **Right-censoring:** CVEs published < 90 days before the KEV snapshot are dropped →
labels complete up to **2026-08-20**."""),
code("""py = T("03_positives_by_year")
kev = pd.read_csv(sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1])
print(f"KEV snapshot: {len(kev):,} entries, dateAdded {kev['dateAdded'].min()} -> {kev['dateAdded'].max()}, CVE years {kev['cveID'].str[4:8].min()}-{kev['cveID'].str[4:8].max()}")
print(f"dataset: {py['cves'].sum():,} CVEs (2022-01-01 -> 2026-08-20), {py['positives'].sum()} exploited within 90 days ({py['positives'].sum() / py['cves'].sum():.2%})")
display(py.set_index("year"))
fig, ax = plt.subplots(figsize=(8, 3.3)); ax2 = ax.twinx()
ax.bar(py["year"].astype(str), py["cves"], color=GRID); ax2.plot(py["year"].astype(str), py["positives"], "o-", color=RED, lw=2)
for x, (p, r) in enumerate(zip(py["positives"], py["rate_%"])): ax2.annotate(f"{p} ({r:.2f}%)", (x, p), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
ax.set_ylabel("CVEs published (grey)"); ax2.set_ylabel("exploited within 90 d (red)", color=RED); ax2.grid(False)
ax.set_title("CVEs per year vs exploited within 90 days (2026 = Jan 1 - Aug 20)", loc="left"); save(fig, "1_01_problem")"""),
tk("**196,403 CVEs (2022 → Aug 2026), only 613 (0.31%, ~1 in 320) exploited within 90 days.** Volume doubled from 25,073 (2022) to 48,150 (2025) and 2026 is on track for ~85,000, while exploited CVEs stay in the low hundreds a year (85–159). The emergency list must stay small while the haystack grows."),

md("""## 1.2 The evolution of cybersecurity (descriptive only — no modelling choice uses this section)
NVD data cover 2022-01 → 2026-08; the KEV catalogue reaches back to CVEs from 2002 for the long view."""),
code("""mo = T("00_evo_monthly"); mix = T("00_evo_cvss_mix").set_index("year"); kbm = T("00_kev_by_month"); kcy = T("00_kev_by_cve_year")
ven = T("00_evo_vendors"); dtk = T("00_evo_days_to_kev").set_index("year")
fig, axes = plt.subplots(2, 2, figsize=(13, 7.6))
a = axes[0, 0]; a.plot(pd.PeriodIndex(mo["month"], freq="M").to_timestamp(), mo["cves"], color=BLUE, lw=1.8); a.set_title("CVEs published per month (NVD)", loc="left")
a = axes[0, 1]; bottom = np.zeros(len(mix))
for s, c in zip(["CRITICAL", "HIGH", "MEDIUM", "LOW", "no CNA score"], [RED, ORANGE, "#e7c24a", AQUA, GRID]):
    a.bar(mix.index.astype(str), mix[s], bottom=bottom, color=c, label=s); bottom += mix[s].to_numpy()
a.set_title("CNA CVSS severity mix per year (% of CVEs)", loc="left"); a.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
a = axes[1, 0]; a.bar(pd.PeriodIndex(kbm["dateAdded"], freq="M").to_timestamp(), kbm["kev_additions"], width=25, color=VIOLET); a.set_yscale("log")
a.set_title("KEV additions per month (log scale; launch batch Nov 2021)", loc="left")
a = axes[1, 1]; a.bar(kcy["cve_year"], kcy["kev_entries"], color=INK2); a2 = a.twinx(); a2.grid(False)
a2.plot(kcy["cve_year"], kcy["median_years_cve_to_kev"], "o-", color=ORANGE, ms=3); a.set_title("KEV entries by CVE year (bars) and median years to KEV (orange)", loc="left")
fig.tight_layout(); save(fig, "1_02_evolution")
display(mix[["CVEs", "CNA CVSS >= 9 %", "no CNA score", "cvss_source=other %"]].round(1)); display(dtk)
print("share of exploited CVEs (%) by vendor ('cna:' = vendor missing):"); display(ven.pivot(index="vendor", columns="year", values="share of exploited %").round(1).sort_values(2025, ascending=False))"""),
tk("**Volume is accelerating** (~2,100 CVEs a month in 2022 → ~6,800 in 2026). **CNAs now score most CVEs themselves** (CVEs without a CNA CVSS: 57% → 19%), and **'critical' is inflating** (CNA CVSS ≥ 9: 3.7% → 8.9% of CVEs, i.e. ~80 → ~600 patches a month under a CVSS ≥ 9 policy). **Exploitation is listed faster**: among KEV-listed CVEs, 52% (2022) vs 71% (2026) within 30 days, about a quarter on or before publication day. **KEV is now mostly about new CVEs**: CVEs from 2017 waited a median 5 years for KEV; CVEs since 2021 are listed the same year. Microsoft is consistently 17–24% of exploited CVEs; Cisco rose to 9.6% in 2026."),

md("""## 1.3 The leakage hunt
NVD returns **today's** version of each record, and some fields are added *because* a CVE was exploited. Each leak
was found in the data and fixed before modelling (decisions dated in `docs/decision_log.md`)."""),
code("""adp = T("00_leak_adp_by_quarter"); b = T("03b_train_baselines_cna_vs_fallback"); cwe = T("03c_train_cwe_source").set_index("cwe_fallback_source")
ed = T("03e_history_description_changes").set_index("label"); sens = T("05b_sensitivity_original_text"); kh = T("00_leak_kev_history"); rs = T("04_random_vs_date_split")
leaks = pd.DataFrame([
 {"leak": "CVSS from CISA-ADP / NVD (added after publication)", "evidence": f"2022Q1: {adp.iloc[0]['% of CVEs with ADP CVSS']:.1f}% of CVEs vs {adp.iloc[0]['% of exploited with ADP CVSS']:.1f}% of exploited CVEs have an ADP score",
  "fix": "CVSS features from the CNA's own score only", "before -> after": f"CVSS>=9 rule on train: {b.iloc[1]['recall_%']}% at {b.iloc[1]['patches/month']}/mo (with late scores) -> {b.iloc[0]['recall_%']}% at {b.iloc[0]['patches/month']}/mo (CNA only)"},
 {"leak": "CWE assigned later by NVD analysts", "evidence": f"{cwe.loc['nvd', 'share_of_pos_%']}% of exploited train CVEs got their CWE from NVD", "fix": "CNA CWE only", "before -> after": "feature defined before any model was fit"},
 {"leak": "descriptions edited after exploitation", "evidence": f"edited: {ed.loc['positives (all)', 'desc_changed_%']}% of exploited vs {ed.loc['negatives (sample)', 'desc_changed_%']}% of other CVEs",
  "fix": "description_day0: exploitation phrases not in the first recorded version removed", "before -> after": f"earliest vs today's text: rank correlation {sens.iloc[0]['Spearman (sample CVEs)']:.3f}"},
 {"leak": "vendor KEV history counting future / own entries", "evidence": "today's KEV includes entries added after publication", "fix": "only entries added BEFORE publication, never the CVE's own",
  "before -> after": f"val recall@110: guarded {kh.iloc[0]['val recall@110 %']} / own-entry {kh.iloc[1]['val recall@110 %']} / naive {kh.iloc[2]['val recall@110 %']} (guarantee, no inflation here)"},
 {"leak": "random train/test split", "evidence": f"{rs.iloc[1].iloc[-1]:.0f}% of random-split val positives had an exploited same-vendor+month twin in train", "fix": "split by date + walk-forward",
  "before -> after": f"same model: random split {rs.iloc[1]['recall@3.4% of monthly CVEs']}% vs date split {rs.iloc[0]['recall@3.4% of monthly CVEs']}%"},
 {"leak": "CNA revised its own CVSS later (residual)", "evidence": "5.5% of exploited vs 1.9% of other CVEs", "fix": "not fixable from day-0 data; sensitivity run", "before -> after": "selection score 47.4 -> 46.6 with first known vector"}]).set_index("leak")
display(leaks)
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.plot(adp["published"], adp["% of exploited with ADP CVSS"], "o-", color=RED, lw=2, label="% of exploited CVEs with a CISA-ADP CVSS")
ax.plot(adp["published"], adp["% of CVEs with ADP CVSS"], "o-", color=INK2, lw=2, label="% of all CVEs with a CISA-ADP CVSS")
ax.tick_params(axis="x", rotation=60, labelsize=7.5); ax.legend(frameon=False); ax.set_title("CISA-ADP scores were added to exploited CVEs after the fact", loc="left"); save(fig, "1_03_leak_adp")"""),
tk("**Five shortcuts in today's NVD/KEV data would have leaked the answer.** CISA-ADP CVSS scores were back-filled mostly for exploited CVEs (2022Q1: 46% of exploited vs 3.4% of all) and would make the CVSS ≥ 9 rule look like 35.1% recall instead of the honest 21.9%; NVD-assigned CWEs cover 55% of exploited CVEs; descriptions were edited after exploitation (30.4% vs 12.3%); vendor KEV history must stop at the publication date (guarded by design — the leaky variants did not inflate validation here, 45.6% / 35.0% vs 47.1%); and a random split inflates recall from 38.2% to 59.1%. **Residual:** CNAs later revise their own CVSS (5.5% vs 1.9%), costing 0.8 points when tested."),
]

# =====================================================================================================  2 TRAIN / VALIDATE
nb2 = [
md(f"""# 2 · Train and validate: task, metric, baselines, models, selection (all before the test set)
{NAV}"""),
code(SETUP),
md("""## 2.1 Task, splits, metric
* **Supervised** binary classification on publication-day information only; the output is used as a **monthly ranking**.
* **Split by date:** train 2022-01 → 2024-06, validation 2024-07 → 2024-12, **test 2025-01 → 2026-08-20 (opened once,
  notebook 3)**; walk-forward folds 2022 → 2023 and 2022–23 → 2024-H1 inside dev.
* **recall@K per calendar month**: each month patch the top K; recall = exploited caught / all exploited.
  K = **110** = monthly capacity of the CVSS ≥ 9 (CNA) rule in train; K = **305** = the hindsight rule's capacity.
  Fixed before modelling; the full recall-vs-K curve is always shown.
* **Uncertainty:** paired within-month Poisson bootstrap (2,000) → 95% CIs. **Selection rule:** mean recall@110 over
  WF 2023, WF 2024-H1 and validation (never validation alone)."""),
code("""split = pd.DataFrame({"period": ["train", "validation", "test"], "dates": ["2022-01 -> 2024-06", "2024-07 -> 2024-12", "2025-01 -> 2026-08-20"],
                      "CVEs": [74268, 19574, 102561], "exploited": [251, 68, 294], "months": [30, 6, 20]})
split["rate %"] = (100 * split["exploited"] / split["CVEs"]).round(3); split["CVEs / month"] = (split["CVEs"] / split["months"]).round(0)
split["K=110 as % of a month"] = (100 * 110 / split["CVEs / month"]).round(1); display(split.set_index("period"))"""),
tk("A date split is harder and honest: test months are twice as large as train months (5,128 vs 2,476 CVEs), so the fixed K = 110 covers 4.4% of a train month but only 2.1% of a test month."),
md("## 2.2 Baselines: business rules on validation (day-0 information)"),
code("""bv = T("04_val_comparison").set_index("method"); display(bv)
order = ["CVSS (CNA) ranking", "phrase, then CVSS (CNA)", "generous CVSS (fallback) ranking", "random"]
v = [pt(bv.loc[m, "recall@110 % [95% CI]"]) for m in order]
fig, ax = plt.subplots(figsize=(8, 2.8)); ax.barh(order, v, color=[INK2, ORANGE, "#999999", GRID]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
for i, x in enumerate(v): ax.annotate(f"{x:.1f}%", (x, i), xytext=(4, -3), textcoords="offset points", fontsize=8.5)
ax.set_xlabel("validation recall@110 (%)"); ax.set_title("The bar to beat: day-0 rules", loc="left"); save(fig, "2_01_baselines")"""),
tk("**The best rule a careful team could run today — 'exploitation phrase first, then CNA CVSS' — catches 28.1% [17–38] of exploited validation CVEs with 110 patches a month**; plain CVSS ranking 23.7%, random 3.5%. Hindsight CVSS (ADP-filled) does *not* help at K = 110 (15.1%)."),
md("""## 2.3 Tabular ML vs deep learning on text
* **Supervised, hand-built features:** logistic regression and gradient boosting (scikit-learn HistGradientBoosting) on CNA
  CVSS, CNA, vendor, CWE, description length, day-0 exploitation phrase, vendor/product KEV history.
* **Pretrained (self-supervised) representations:** MiniLM / BGE sentence embeddings of the day-0 description (encoders
  pretrained on unlabeled text) + a supervised head (logistic regression or a small MLP, early stopping, 5 seeds).
* **Fine-tuned transformer:** MiniLM fine-tuned end-to-end on our labels (Apple GPU).
* **Combinations:** tabular + text MLP; boosting + out-of-fold text score; **late fusion** (mean of within-month ranks)."""),
code("""a5 = T("05a_comparison").set_index("method"); b5 = T("05b_comparison").set_index("method"); c5 = T("05c_comparison").set_index("method")
V110, V305 = "val recall@110 [CI] (ensemble for nets)", "val recall@305 [CI] (ensemble for nets)"
rows = [("phrase, then CVSS (CNA)", "best day-0 rule", b5.loc["phrase, then CVSS (CNA)"]), ("gradient boosting", "supervised, tabular", b5.loc["gradient boosting (05a)"]),
        ("text LR (MiniLM embeddings)", "pretrained + supervised head", b5.loc["text LR"]), ("text MLP (MiniLM, 5 seeds)", "pretrained + supervised head", b5.loc["text MLP (5 seeds)"]),
        ("tabular + text MLP", "tabular + text", b5.loc["(a) tabular+text MLP (5 seeds)"]), ("boosting + OOF text score", "tabular + text", b5.loc["(b) gradient boosting + OOF text score"]),
        ("late fusion: boosting + text MLP", "tabular + text (rank fusion)", b5.loc["(c) late fusion: gradient boosting + text MLP (5 seeds)"])]
out = [{"method": n, "learning": k, "selection (3-period mean)": r["selection (3-period mean)"], "worst period": r["worst period"], "val recall@110 [CI]": r[V110], "val recall@305 [CI]": r[V305]} for n, k, r in rows]
lr = a5.loc["logistic regression"]; lv = pt(lr["val recall@110 [CI]"])
out.append({"method": "logistic regression", "learning": "supervised, tabular", "selection (3-period mean)": round(np.mean([lr["WF 2023"], lr["WF 2024-H1"], lv]), 1),
            "worst period": min(lr["WF 2023"], lr["WF 2024-H1"], lv), "val recall@110 [CI]": lr["val recall@110 [CI]"], "val recall@305 [CI]": lr["val recall@305 [CI]"]})
ft = c5.loc["fine-tuned MiniLM (lr1e-5_full, 2 seeds)"]
out.append({"method": "fine-tuned MiniLM (2 seeds)", "learning": "pretrained + supervised fine-tuning", "selection (3-period mean)": ft["selection (3-period mean)"],
            "worst period": ft["worst period"], "val recall@110 [CI]": ft["val recall@110 [CI]"], "val recall@305 [CI]": ft["val recall@305 [CI]"]})
ladder = pd.DataFrame(out).set_index("method").sort_values("selection (3-period mean)", ascending=False); display(ladder)
col = {"best day-0 rule": ORANGE, "supervised, tabular": BLUE, "pretrained + supervised head": VIOLET, "pretrained + supervised fine-tuning": RED, "tabular + text": AQUA, "tabular + text (rank fusion)": AQUA}
lad = ladder.iloc[::-1]; fig, ax = plt.subplots(figsize=(9, 3.9))
ax.barh(lad.index, lad["selection (3-period mean)"], color=[col[k] for k in lad["learning"]])
for i, (s, w) in enumerate(zip(lad["selection (3-period mean)"], lad["worst period"])): ax.annotate(f"{s:.1f} (worst {w:.1f})", (s, i), xytext=(4, -3), textcoords="offset points", fontsize=8)
ax.set_xlim(0, 62); ax.grid(axis="y", visible=False); ax.set_xlabel("selection score: mean recall@110 over WF 2023, WF 2024-H1, validation (%)")
ax.set_title("Model ladder (dev only). blue tabular, violet/red pretrained text, green combined", loc="left", fontsize=10); save(fig, "2_02_model_ladder")"""),
tk("**Gradient boosting on hand-built day-0 features is the best single model (selection 47.4%, validation 47.1% [36–60])**, 13 points above the best rule (34.1%). Pretrained sentence embeddings + a supervised head reach 39.2–39.5% — above the rule, below the tabular model — and **fine-tuning MiniLM on 251 positives did worse (34.6%)** than keeping the self-supervised encoder frozen (overfitting with few labels). Late fusion has the best worst period (42.6%) at a near-identical mean (46.7%)."),
md("""## 2.4 Model selection — fixed before the test set was opened
* **Decision 11:** highest selection score; within 2 points, the better worst period → late fusion (46.7, worst 42.6) over
  gradient boosting (47.4, worst 39.7).
* **Pre-test checks:** the fusion leans on uncertain day-0 phrases (−2.0 points when removed; boosting +0.4); boosting
  was better on validation (47.1 vs 42.6) and on the validation "not yet in KEV" subset (52.5 vs 35.0).
* **Decision 12 (dev evidence only, before any test label was seen): headline = gradient boosting**, late fusion =
  reported alternative. The test protocol was written down first (`docs/decision_log.md`), and a dry run on validation
  reproduced every Phase 5 number."""),
code("""sd = T("05d_sensitivity_dev"); sd["model"] = sd["model"].str.replace(" (headline)", "", regex=False)
display(sd.set_index(["variant", "model"])[["wf_2023@110", "wf_2024h1@110", "val@110", "selection", "worst"]])
p = sd[sd["variant"] == "base (as in 05b)"].set_index("model")
fig, ax = plt.subplots(figsize=(7.5, 3.3)); x = np.arange(3); w = 0.38
for j, (m, c) in enumerate([("gradient boosting", BLUE), ("late fusion", AQUA)]):
    v = p.loc[m, ["wf_2023@110", "wf_2024h1@110", "val@110"]].astype(float).to_numpy(); ax.bar(x + (j - 0.5) * w, v, w, color=c, label=m)
    for xi, vi in zip(x, v): ax.annotate(f"{vi:.0f}", (xi + (j - 0.5) * w, vi), ha="center", xytext=(0, 2), textcoords="offset points", fontsize=8)
ax.set_xticks(x, ["WF 2022 -> 2023", "WF 2022-23 -> 2024-H1", "train -> validation"]); ax.grid(axis="x", visible=False); ax.legend(frameon=False)
ax.set_ylabel("recall@110 (%)"); ax.set_title("The two finalists on the three dev periods", loc="left"); save(fig, "2_03_selection")"""),
tk("Gradient boosting and late fusion tie on dev (47.4 vs 46.7). Boosting was chosen as headline on **dev evidence only** — better on validation, robust to the uncertain-phrase check, simpler to deploy and explain — before the test set was opened; both were scored on test."),
]

# =====================================================================================================  3 TEST
nb3 = [
md(f"""# 3 · The single test evaluation (Phase 6) and the EPSS comparison
Test = 2025-01-01 → 2026-08-20 · 102,561 CVEs · 294 exploited. Frozen dev models (sha256-checked), protocol fixed in
advance, run **once** (`src/phase6_test.py --test --confirm-phase6`). This notebook reads the cached results.

{NAV}"""),
code(SETUP),
md("## 3.1 Recall at fixed capacity"),
code("""rec = T("06_recall").set_index("method"); dif = T("06_diffs"); own = T("06_own_capacity").set_index("rule"); curve = T("06_recall_vs_k").set_index("K per month")
display(rec[["recall@110 [95% CI]", "recall@305 [95% CI]", "PR-AUC"]].round(3))
d = dif[dif["K"] == 110].set_index("headline minus")
display(d.assign(**{"difference [95% CI]": [f"{a:+.1f} [{b:+.1f}, {c:+.1f}]" for a, b, c in zip(d["diff (pts)"], d["CI lo"], d["CI hi"])]})[["difference [95% CI]", "P(headline better)"]])
print("rules at their own capacity vs the model at the same K:"); display(own.round(1))
COL = {HEAD: BLUE, ALT: AQUA, "phrase, then CVSS (CNA)": ORANGE, "CVSS (CNA) ranking": INK2, EPSS7: VIOLET, "EPSS (day-0 score only)": "#c9c3f0", "random": "#cccccc"}
fig, ax = plt.subplots(figsize=(9, 4.4))
for m in COL: ax.plot(curve.index, 100 * curve[m], color=COL[m], lw=2.4 if m == HEAD else 1.4, ls="--" if m in ("random", "EPSS (day-0 score only)") else "-", label=m)
for k in (110, 305): ax.axvline(k, color=INK, lw=0.8, ls=":")
ax.set(xlim=(0, 1000), ylim=(0, 100), xlabel="emergency patches per month (K)", ylabel="% of exploited CVEs caught"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("Test: recall vs capacity", loc="left"); save(fig, "3_01_recall_vs_k")"""),
tk("**Gradient boosting caught 33.7% [28–39] of exploited CVEs with 110 patches a month**, vs 25.2% for the best rule (+8.5 [+1.7, +15.3], P 99%) and 18.6% for CVSS ranking (+15.1). It leads **every day-0 rule at every capacity from 10 to 1,000** (smallest lead +1.2 points at K = 30), and **at the CVSS ≥ 9 rule's own 385 patches a month catches 51.7% vs 41.2%** — so the conclusion does not hinge on the assumed CVSS ≥ 9 policy or on K. Test recall is below validation (47.1%): see drift in notebook 4."),
md("""## 3.2 EPSS — the industry benchmark, with caveats
1. **Lookahead:** most CVEs get their first EPSS score the day *after* publication (shown ≤ 7 days, by day 1 and day-0 only).
2. **Different target:** exploitation activity in the next 30 days, not KEV within 90 days. 3. **EPSS v4** replaced v3 on 2025-03-17.
4. **EPSS may use the label:** *assumed* that EPSS lists CISA KEV among its sources (documentation not reachable from the
   project sandbox) → the pre-registered **"not yet in KEV"** analysis keeps only exploited CVEs added to KEV *after* the EPSS score date."""),
code("""cov = T("06_epss_coverage").set_index("first EPSS score at"); display(cov.round(1))
display(T("06_epss_v3_v4").set_index("EPSS model period"))
nyk = T("06_not_yet_in_kev").set_index("EPSS variant defining the cut")
methods = [HEAD, ALT, "phrase, then CVSS (CNA)", "CVSS (CNA) ranking", "generous CVSS (fallback) ranking", EPSS7, "EPSS (day-0 score only)"]
tbl = pd.DataFrame({"all 294 exploited": rec.loc[methods, "recall@110 [95% CI]"], "185 not yet in KEV at the EPSS date": [nyk.loc[EPSS7, f"{m} @110"] for m in methods]}); display(tbl)
mm = [HEAD, "phrase, then CVSS (CNA)", EPSS7]; x = np.arange(3); w = 0.38
a = [pt(rec.loc[m, "recall@110 [95% CI]"]) for m in mm]; b = [pt(nyk.loc[EPSS7, f"{m} @110"]) for m in mm]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(x - w/2, a, w, color=[COL[m] for m in mm], alpha=0.45, label="all exploited"); ax.bar(x + w/2, b, w, color=[COL[m] for m in mm], label="not yet in KEV when EPSS scored")
for xi, (u, v) in enumerate(zip(a, b)):
    ax.annotate(f"{u:.1f}", (xi - w/2, u), ha="center", xytext=(0, 2), textcoords="offset points", fontsize=8); ax.annotate(f"{v:.1f}", (xi + w/2, v), ha="center", xytext=(0, 2), textcoords="offset points", fontsize=8, weight="bold")
ax.set_xticks(x, ["gradient\\nboosting", "phrase, then\\nCVSS (CNA)", "EPSS\\n(<= 7 d)"]); ax.grid(axis="x", visible=False); ax.set_ylim(0, 45); ax.legend(frameon=False, fontsize=8)
ax.set_title("recall@110 after removing CVEs already in KEV", loc="left"); save(fig, "3_02_not_yet_in_kev")"""),
tk("Against EPSS the model is ahead but not clearly (+4.1 [−2.7, +10.5], P 86%); EPSS ranks its very top better at ≤ 50 patches a month, and only 30% of CVEs have an EPSS score on publication day (day-0 EPSS: 5.1%). **But 109 of the 294 exploited CVEs (37%) were already in KEV when EPSS produced its score; on the other 185, EPSS falls to 14.6% [9–20] while the model keeps 30.4% [24–37]** (best rule 29.4%). The gap is similar under EPSS v3 (43.2 vs 40.9) and v4 (32.8 vs 30.8)."),
md("## 3.3 Robustness (pre-registered): walk-forward 2022-24 → 2025, quarters, calibration"),
code("""display(T("06_by_year").set_index("year")); display(T("06_by_quarter").set_index("quarter"))
cal = T("06_calibration"); display(cal.T)
bins = T("06_calibration_bins"); fig, ax = plt.subplots(figsize=(5, 4.2))
ax.plot([0, 8], [0, 8], color=GRID, ls="--", lw=1.5, label="perfect"); ax.plot(100 * bins["predicted"], 100 * bins["observed"], "o-", color=BLUE, lw=2, label="headline (Platt), test")
ax.set(xlim=(0, 8), ylim=(0, 8), xlabel="mean predicted (%)", ylabel="observed exploited share (%)"); ax.legend(frameon=False, loc="lower right"); ax.set_title("Reliability on test", loc="left"); save(fig, "3_03_calibration")"""),
tk("**Test year 2025 (= walk-forward 2022-24 → 2025): 41.9% [34–50]**, in line with the dev periods; **2026 (Jan–Aug): 23.9% [17–32]**, falling to 7.7% in Jul–Aug while EPSS stays at 29.6% in both years. Probabilities are only modestly better than a constant (Brier 0.002816 vs 0.002858), over-predict on average (0.42% vs 0.29%) and under-predict in the top 0.5% (4.4% vs 7.6%) — ranking is what was validated."),
]

# =====================================================================================================  4 RESULTS / CISO
nb4 = [
md(f"""# 4 · Results for the CISO: explanations, drift, business impact, recommendations
All numbers use the **frozen Phase 6 model** on the test period. {NAV}"""),
code(SETUP),
md("""## 4.1 Why the model ranks what it ranks (SHAP)
Permutation SHAP on the 22 raw inputs (the standard tree explainer mis-handles the model's categorical CVSS splits;
these values add up exactly to the model output), 2,000 random test CVEs, log-odds units."""),
code("""import shap
sh = pd.read_parquet(ROOT / "data" / "shap_test.parquet")
FEATS = [c[6:] for c in sh.columns if c.startswith("shap__")]; SV = sh[[f"shap__{f}" for f in FEATS]].to_numpy(); rand = sh["in_random"].to_numpy()
imp = T("07_shap_importance"); display(imp.head(10).set_index("feature"))
num = {f: pd.to_numeric(sh[f"x__{f}"], errors="coerce") for f in FEATS}
data = np.column_stack([num[f].to_numpy(float) if num[f].notna().any() else np.full(len(sh), np.nan) for f in FEATS])
names = dict(zip(imp["feature"], imp["feature"]))
NAME = {"vendor_key": "vendor (or CNA)", "cna": "CNA (who published)", "cwe": "CWE (CNA)", "n_cwes": "number of CWEs", "cvss_cna_score": "CVSS score (CNA)", "has_cna_cvss": "has CNA CVSS",
        "phrase_day0": "exploitation phrase (day 0)", "desc_len": "description length", "vendor_kev_log_all": "vendor KEV entries (all time)", "vendor_kev_log_365": "vendor KEV entries (365 d)",
        "vendor_kev_days_since": "days since vendor's last KEV entry", "vendor_kev_any": "vendor has any KEV entry", "product_kev_log_all": "product KEV entries", **{f"cvss_{p}": f"CVSS {p}" for p in ["AV", "AC", "PR", "UI", "S", "C", "I", "A", "AT"]}}
expl = shap.Explanation(values=SV[rand], base_values=sh.loc[rand, "base_value"].to_numpy(), data=data[rand], feature_names=[NAME[f] for f in FEATS])
plt.figure(); shap.plots.beeswarm(expl, max_display=10, show=False, plot_size=(9, 4.8)); plt.title("SHAP beeswarm (2,000 random test CVEs)", loc="left", fontweight="bold"); save(plt.gcf(), "4_01_beeswarm")
ex = T("07_examples"); display(ex[["example", "cve_id", "vendor", "product", "cvss_cna_score", "cwe", "label", "days_to_kev", "s_hgb", "month_pct", "in_top110"]])"""),
tk("**The model reads who is affected and whether attackers already target that vendor, then severity**: publisher (CNA, mean |SHAP| 0.79), the vendor's earlier KEV entries (0.65; the top driver for exploited CVEs, 1.14) and CNA CVSS (0.41). Examples: **caught** — Fortinet auth bypass CVE-2026-24858 (CVSS 9.8, Fortinet's 24 earlier KEV entries; #1 of its month, in KEV the same day); **missed** — Microsoft AD FS privilege escalation CVE-2026-56155 (local, 7.8; top 6.4% of a ~9,000-CVE month, outside the 110); **false alarm** — Cisco FMC auth bypass to root CVE-2026-20079 (CVSS 10), which *was* added to KEV, after 189 days."),
md("## 4.2 Drift: why recall fell in 2026\n" + POST),
code("""dy = T("06x_drift_by_year").set_index("year"); display(dy.T)
display(T("06x_drift_volume").set_index("period").round(1)); display(T("07_shap_kev_by_year").set_index("year").T.round(2))
roll = T("06x_rolling_retrain"); q = roll[roll["period"].str.contains("Q")].reset_index(drop=True)
display(roll.set_index("period").loc[["2025", "2026", "all test"], ["positives", "frozen @110", "rolling @110", "rolling - frozen (pts)", "diff CI"]])
fig, ax = plt.subplots(figsize=(9, 3.8)); xq = np.arange(len(q))
for nm, c, off in [("frozen", BLUE, -0.08), ("rolling", ORANGE, 0.08)]:
    ax.errorbar(xq + off, q[nm], yerr=[q[nm] - q[f"{nm} lo"], q[f"{nm} hi"] - q[nm]], fmt="o-", color=c, capsize=3, lw=2, label="frozen model (reported)" if nm == "frozen" else "retrained each quarter (post-test)")
ax.set_xticks(xq, q["period"]); ax.set_ylim(0, 70); ax.set_ylabel("recall@110 (%)"); ax.legend(frameon=False); ax.set_title("Recall by test quarter", loc="left"); save(fig, "4_02_drift")"""),
tk("**Not everyone dropped**: from 2025 to 2026 the model fell 41.9 → 23.9 and the best rule 29.5 → 20.0, while EPSS stayed at 29.6. Two causes: **volume** (6,800 CVEs a month in 2026, so K = 110 is 1.6% of a month) and **crowding** — CVEs from vendors with KEV history rose from 1,246 to 2,376 a month, and the share of *non-exploited* CVEs pushed up strongly by KEV history went 18.8% → 26.6%. **Retraining every quarter (90-day label delay respected) recovered part of it: 2026 recall 31.3% vs 23.9%** (+7.4 [+0.0, +15.2]); over the whole test period it is a wash (+1.0)."),
md("""## 4.3 Business impact
Headline = the **frozen** model as evaluated in Phase 6. The retrained-model figures are **post-test exploratory**.
**Edit the assumptions and re-run.**"""),
code("""# ================= ASSUMPTIONS (edit me) =================
HOURS_PER_EMERGENCY_PATCH = 6.0     # out-of-cycle patch: triage, test, change approval, deploy, verify
COST_PER_HOUR_USD = 100.0           # fully loaded analyst/engineer hour
SHARE_OF_FEED = 1.0                 # share of all NVD CVEs that affect your estate (1.0 = whole feed, as studied)
# ========================================================="""),
code("""sr = T("07_same_recall"); sc_ = T("07_same_capacity").set_index("comparison"); pol = T("07_policy"); rw = T("07_ransomware")
sr["saved / month (frozen)"] = (sr["rule patches / month"] - sr["GB (frozen) patches / month to match"]) * SHARE_OF_FEED
sr["saved / month (retrained, post-test)"] = (sr["rule patches / month"] - sr["GB rolling-retrain patches / month to match"]) * SHARE_OF_FEED
sr["USD / year (frozen)"] = sr["saved / month (frozen)"] * 12 * HOURS_PER_EMERGENCY_PATCH * COST_PER_HOUR_USD
sr["USD / year (retrained, post-test)"] = sr["saved / month (retrained, post-test)"] * 12 * HOURS_PER_EMERGENCY_PATCH * COST_PER_HOUR_USD
display(sr[["period", "rule", "rule patches / month", "rule recall %", "GB (frozen) patches / month to match", "saved / month (frozen)", "USD / year (frozen)",
            "GB rolling-retrain patches / month to match", "saved / month (retrained, post-test)", "USD / year (retrained, post-test)"]].round(1))
main = sr[(sr["period"] == "all test") & (sr["rule"] == "CVSS>=9 (CNA)")].iloc[0]; y26 = sr[(sr["period"] == "2026 (Jan-Aug)") & (sr["rule"] == "CVSS>=9 (CNA)")].iloc[0]
hours = [2, 4, 6, 8, 10]
sens = pd.DataFrame({"hours per patch": hours,
    "frozen: hours saved / yr": [main["saved / month (frozen)"] * 12 * h for h in hours],
    "frozen: USD / yr": [main["saved / month (frozen)"] * 12 * h * COST_PER_HOUR_USD for h in hours],
    "frozen, 2026 rate: USD / yr": [y26["saved / month (frozen)"] * 12 * h * COST_PER_HOUR_USD for h in hours],
    "retrained (post-test): USD / yr": [main["saved / month (retrained, post-test)"] * 12 * h * COST_PER_HOUR_USD for h in hours],
    "retrained, 2026 rate (post-test): USD / yr": [y26["saved / month (retrained, post-test)"] * 12 * h * COST_PER_HOUR_USD for h in hours]}).set_index("hours per patch")
print("Same recall as the CVSS>=9 (CNA) rule with fewer emergency patches - sensitivity to hours per patch:"); display(sens.round(0))
display(sc_.round(1)); print("ransomware-linked exploited CVEs (n = 43):"); display(rw[rw["subset"] == "ransomware-linked exploited"].set_index("method")[["recall@110 [95% CI]", "recall@385 [95% CI]"]])
print("operating policy (post-test exploratory):"); display(pol.round(1))
fig, ax = plt.subplots(figsize=(8, 3.6))
for c_, col_, ls in [("frozen: USD / yr", BLUE, "-"), ("frozen, 2026 rate: USD / yr", ORANGE, "-"), ("retrained (post-test): USD / yr", BLUE, ":"), ("retrained, 2026 rate (post-test): USD / yr", ORANGE, ":")]:
    ax.plot(sens.index, sens[c_] / 1e6, ls, color=col_, marker="o", lw=2, label=c_.replace(": USD / yr", ""))
ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("hours per emergency patch"); ax.set_ylabel(f"USD million / year at ${COST_PER_HOUR_USD:.0f}/h"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Savings vs the CVSS >= 9 rule at equal recall", loc="left"); save(fig, "4_03_business")"""),
tk("**Headline (frozen model, test 2025-01 → 2026-08): to match the CVSS ≥ 9 rule's coverage (41.2% of exploited CVEs) the model needs 198 emergency patches a month instead of 385 — ~187 fewer, ~13,500 analyst hours and ~$1.35M a year** at 6 h per patch and $100/h ($0.45M at 2 h to $2.24M at 10 h). At equal capacity it catches **+15 to +27 more exploited CVEs a year** at 110 patches a month, and on ransomware-linked CVEs 42.2% vs 24.6–27.9% for the rules and EPSS (n = 43, wide CIs). **But the saving is concentrated in 2025**: at the 2026 rate the frozen model saves only 37 patches a month ($0.09–0.44M). *Post-test exploratory:* with quarterly retraining the model needs **148 patches a month (~$1.7M a year at 6 h; $0.57–2.84M over 2–10 h)** and keeps the advantage in 2026 (302 vs 606 patches)."),
md("""## 4.4 Recommendations for the CISO
1. **Replace "patch every CVSS ≥ 9" with a model-ranked emergency list** (gradient boosting on day-0 data): same workload,
   more exploited CVEs caught (33.7% vs 25.2% of the best rule at 110 a month), or the same coverage with about half the patches.
2. **Retrain every quarter** (respecting the 90-day label delay) and **review capacity against CVE volume each quarter**
   (e.g. ~4.2% of each month's CVEs): without both, 2026 showed the advantage can vanish (post-test exploratory evidence).
3. **Use EPSS as a second opinion, not the primary ranking**: it is a next-day signal, ranks its very top slightly better at
   ≤ 50 patches a month, and much of its apparent strength comes from CVEs already in KEV (14.6% vs 30.4% on the rest).
4. **Track live**: each month, recall on newly KEV-listed CVEs and the share of PATCH NOW CVEs from KEV-history vendors
   (the 2026 crowding signal). Every ranked CVE comes with its top reasons (notebook 5)."""),
md("""## 4.5 Limitations
* **The baseline policy is assumed.** "Patch CVSS ≥ 9" is common severity-based practice, not a verified policy of a specific
  organisation. The conclusion does not rest on it: the model leads every day-0 rule at every K from 10 to 1,000 patches a
  month and each rule at its own capacity (CVSS ≥ 9, CVSS ≥ 7, phrase-or-CVSS ≥ 9, hindsight CVSS ≥ 9); only EPSS (with a
  one-day lookahead) is ahead, and only at ≤ 50 patches a month (notebook 3).
* **Label = CISA KEV within 90 days**: CISA's selection; exploits never reported to CISA count as "not exploited".
* **Day-0 is approximate**: CNAs later changed their own CVSS for 5.5% of exploited vs 1.9% of other CVEs (−0.8 points);
  description edits are label-correlated (history audited for a sample).
* **EPSS comparison**: lookahead, different target, v3→v4 switch, EPSS's use of KEV is an assumption.
* **Drift and small numbers**: 294 exploited test CVEs; quarterly CIs ±15–20 points; ransomware subset 43 CVEs.
* **Money**: whole NVD feed; a real organisation sees only its own estate (`SHARE_OF_FEED`); hours per patch is an assumption (2–10 h shown).
* **Probabilities** are indicative (over-prediction on average after the base rate fell)."""),
]

# =====================================================================================================  5 TOOL
nb5 = [
md(f"""# 5 · The CISO tool: "I just saw a new NVD entry — do we patch it now?"
`ciso_tool.py` scores CVEs with the **deployment model** (gradient boosting, same frozen settings, refit on all labelled
data 2022-01 → 2026-08-20; never used for any reported number) from **day-0 information only**:
* NVD record parsed with the **same code as training** (CNA CVSS, CNA CWE, CNA vendor/product, description);
* exploitation wording in the description counts only if the record was fetched within 7 days of publication (otherwise it
  cannot be proven day-0 and is removed);
* vendor/product **KEV history** counts only KEV entries added **before** the CVE's publication, from the newest local
  KEV snapshot (the tool never downloads KEV).

**Output per CVE:** within-month **percentile** vs the July 2026 reference month (primary), **indicative probability**
(Platt on time-ordered out-of-fold scores) and how many times the 0.31% base rate it is, **decision** PATCH NOW / MONITOR at
the **4.2%-of-monthly-CVEs budget**, and the **top 3 SHAP reasons** in plain words. Single CVEs are shown as a readable card,
a day's feed as a ranked table. CNA identifiers (often UUIDs) are shown as organisation names from `data/nvd_sources.csv`
(NVD source API, keyless; display only). NVD is read without an API key (the tool never reads `.env`); responses are cached in
`data/nvd_tool_cache/`, so this notebook re-runs offline. The same logic is available as a local web app with a 'Past CVEs' tab (frozen model's day-0 verdicts on the test period): `streamlit run app.py`.

{NAV}"""),
code("""import os, sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl")); os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np
import ciso_tool as ct
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70); pd.set_option("display.max_columns", 20)
tool = ct.Tool(); print(tool.describe())
COLS = ["rank", "cve_id", "vendor", "product", "cna", "cna_cvss", "percentile", "probability_%", "decision", "reason_1", "reason_2", "reason_3"]"""),
md("""## 5.1 Daily feed: everything NVD published on one day, ranked
Demo day **2026-09-08** (Microsoft Patch Tuesday), after the 2026-08-20 label cutoff, so none of these CVEs were in the
deployment model's training data. Top 20 shown with reasons."""),
code("""DAY = "2026-09-08"
feed = tool.score(ct.fetch_day(DAY), explain_top=20)
n_now = (feed["decision"] == "PATCH NOW").sum()
print(f"{DAY}: {len(feed):,} CVEs published; PATCH NOW: {n_now} ({100 * n_now / len(feed):.1f}%) - the 4.2% budget is a MONTHLY share, so busy days such as Patch Tuesday take more")
display(feed.head(20)[COLS].round(1))
notes = feed[feed["notes"] != ""]; print(f"CVEs with a day-0 note (wording removed / KEV snapshot older than publication): {len(notes)}")"""),
md("""### Anecdotal: which of these are in CISA KEV now?
> ⚠️ **Anecdotal, not a metric.** One day, a few CVEs, a KEV snapshot taken ~10 weeks later (2026-11-20, so the 90-day
> window is not complete). Shown only to illustrate the tool; the evaluated performance is in notebook 3."""),
code("""kev = pd.read_csv(sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1], usecols=["cveID", "dateAdded"])
f = feed.merge(kev.rename(columns={"cveID": "cve_id"}), on="cve_id", how="left")
in_kev = f[f["dateAdded"].notna()]
print(f"{len(in_kev)} of the {len(f):,} CVEs published on {DAY} are in the 2026-11-20 KEV snapshot:")
display(in_kev[["rank", "cve_id", "vendor", "product", "cna_cvss", "percentile", "decision", "dateAdded"]].round(1))
print(f"top 20 of the day that are in KEV: {f.head(20)['dateAdded'].notna().sum()} (anecdotal)")"""),
md("## 5.2 One CVE by ID (fetched from NVD)"),
code("""one_id = feed.iloc[0]["cve_id"]
one = tool.score(ct.fetch_cve(one_id))
print(ct.card(one.iloc[0], tool))"""),
md("""## 5.3 One CVE typed in by hand
For a CVE seen in a vendor advisory before NVD has it: vendor, CNA, CVSS vector (v3.x scored automatically), CWE and the
description. Example: a hypothetical unauthenticated remote code execution in a small vendor's product, vs the same bug in
a vendor with a long KEV history."""),
code("""desc = "A stack-based buffer overflow in the web management interface allows an unauthenticated remote attacker to execute arbitrary code via a crafted HTTP request."
vec = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
manual = pd.concat([ct.manual_cve("Acme Widgets", "security@acme.example", desc, vec, cwe="CWE-121", product="WidgetOS", published="2026-11-20", cve_id="MANUAL-ACME"),
                    ct.manual_cve("Ivanti", "3c1d8aa1-5a33-4ea4-8992-aadd6440af75", desc, vec, cwe="CWE-121", product="Connect Secure", published="2026-11-20", cve_id="MANUAL-IVANTI")])
for _, row in tool.score(manual).iterrows():
    print(ct.card(row, tool))"""),
tk("The tool turns the evaluated model into a daily decision: each NVD entry gets a percentile against a full reference month, an indicative probability, a PATCH NOW / MONITOR decision at a volume-scaled budget, and the three facts that drove the score. On the demo day (1,559 CVEs, Patch Tuesday) 182 were PATCH NOW — more than 4.2% because the budget is a *monthly* share and Patch Tuesday is the busiest day. In the hand-typed example the same unauthenticated RCE description (CVSS 9.8) is PATCH NOW for both vendors, but the indicative probability is **10.0% for Ivanti vs 3.1% for an unknown vendor**: **who** is affected (KEV history, publisher) matters as much as severity, as in the SHAP analysis — which is why the model must be retrained as attacker attention moves (notebook 4)."),
]

for name, cells in [("1_data", nb1), ("2_train_validate", nb2), ("3_test", nb3), ("4_results_ciso", nb4), ("5_ciso_tool", nb5)]:
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")
print("written 5 notebooks")
