"""Builds notebooks/04_baselines.ipynb (Phase 4). Run: .venv/bin/python src/build_nb04_baselines.py
Takeaway texts are filled from TAKEAWAYS below (written after inspecting the first run's numbers)."""
import json, sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TK = json.loads((ROOT / "src" / "nb04_takeaways.json").read_text()) if (ROOT / "src" / "nb04_takeaways.json").exists() else {}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
def takeaway(key):
    return md("**Takeaway**: " + TK.get(key, "_(filled after the first run)_"))

cells = [
md("""# Phase 4 — Baselines: what can a security team do today without a model?

Every model later has to beat these rules **at the same patching capacity**. All numbers are on the
**validation** set (CVEs published 2024-07-01 → 2024-12-31); train is used only to fix K and, in section 6,
to fit one simple model for the random-split comparison. The test set is not touched."""),
code("""import os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from metrics import (recall_at_k, recall_curve, random_recall_at_k, rule_at_own_capacity, pr_auc,
                     bootstrap_recall_at_k, ci_table, recall_at_budget)
from description_leak import PHRASES, phrase_flags

FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
BLUE, ORANGE, AQUA, YELLOW, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9.5})
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")

dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
train = dev[dev["split"] == "train"].reset_index(drop=True)
val = dev[dev["split"] == "val"].reset_index(drop=True)
for d in (train, val):
    d["phrase_day0"] = phrase_flags(d["description_day0"], PHRASES).any(axis=1)
months_val = val["published"].dt.to_period("M").nunique()
print(f"val: {len(val):,} CVEs over {months_val} months ({len(val) / months_val:,.0f}/month), "
      f"{int(val['label'].sum())} exploited ({val['label'].mean():.2%})")"""),

md("""## 1. The metric: recall@K at a fixed monthly capacity
The team can emergency-patch **K CVEs per month**. Each month we take the K highest-ranked CVEs published that
month; **recall@K** = share of all exploited CVEs that were in someone's top-K. Ties (many CVEs share a CVSS of
9.8) are scored by their expected value under a random tie-break.

**K is fixed from train, before looking at val:** main K = the monthly volume of the current rule
(CVSS ≥ 9 on the CNA's own score) in train; second K = the monthly volume of the *generous* rule
(fallback score incl. later CISA-ADP/NVD scores, Decision 9)."""),
code("""months_tr = train["published"].dt.to_period("M").nunique()
K_MAIN = int(round((train["cvss_cna_score"] >= 9).sum() / months_tr, -1))     # rounded to 10
K_GEN = int(round((train["cvss_score"] >= 9).sum() / months_tr))
KS = [K_MAIN, K_GEN]
print(f"K_MAIN = {K_MAIN}/month (CVSS>=9 CNA in train)  |  K_GEN = {K_GEN}/month (generous fallback CVSS>=9 in train)")
print(f"in val the same rules use {(val['cvss_cna_score'] >= 9).sum() / months_val:.0f} and "
      f"{(val['cvss_score'] >= 9).sum() / months_val:.0f} patches/month (more CVEs, more criticals)")
print(f"K_MAIN = {K_MAIN / (len(val) / months_val):.1%} of val's monthly CVEs; a random pick would catch "
      f"{random_recall_at_k(val, K_MAIN):.1%} of exploited CVEs")"""),
takeaway("k"),

md("""## 2. The rules at their own capacity (what each rule costs if followed literally)
* **CVSS ≥ 9 (CNA)** — the current rule, day-0 version (Decision 9).
* **CVSS ≥ 7 (CNA)** — the obvious "be safer" alternative.
* **Phrase OR CVSS ≥ 9 (CNA)** — also patch when the publication-day description says the vendor knows of
  exploitation (Decision 10f). A careful team could do this today.
* **Generous CVSS ≥ 9 (fallback)** — uses CISA-ADP/NVD scores that were often added later → an upper bound
  with hindsight, shown for comparison only."""),
code("""RULES = {
    "CVSS>=9 (CNA)": val["cvss_cna_score"] >= 9,
    "CVSS>=7 (CNA)": val["cvss_cna_score"] >= 7,
    "phrase OR CVSS>=9 (CNA)": val["phrase_day0"] | (val["cvss_cna_score"] >= 9),
    "generous CVSS>=9 (fallback)": val["cvss_score"] >= 9,
}
rng = np.random.default_rng(1)
boots = [rng.poisson(1.0, len(val)) for _ in range(2000)]
rows = []
for name, flag in RULES.items():
    r = rule_at_own_capacity(val, flag)
    rb = [((w * flag * val["label"]).sum() / (w * val["label"]).sum()) for w in boots]
    rows.append({"rule": name, "patches/month": round(r["patches_per_month"]), "caught": r["caught"],
                 "recall_%": round(100 * r["recall"], 1), "recall_95CI": f"{100*np.quantile(rb,.025):.0f}–{100*np.quantile(rb,.975):.0f}",
                 "precision_%": round(100 * r["precision"], 2)})
own = pd.DataFrame(rows).set_index("rule"); display(own)
own.to_csv(TAB / "04_val_rules_own_capacity.csv")"""),
takeaway("own"),

md("""## 3. Same capacity: recall@K for every rule
At a fixed K a threshold rule becomes a **ranking**: CVSS ≥ 9 and CVSS ≥ 7 both turn into "sort by CNA score"
(CVEs without a CNA score go last). "Phrase OR CVSS" ranks phrase CVEs first, then by CNA score.
Val has only ~68 exploited CVEs, so every number gets a **paired bootstrap 95% CI** (2,000 resamples within
month; same resample for every method, so differences are compared fairly)."""),
code("""SCORES = {
    "CVSS (CNA) ranking": val["cvss_cna_score"].fillna(-1),
    "phrase, then CVSS (CNA)": 100 * val["phrase_day0"] + val["cvss_cna_score"].fillna(-1),
    "generous CVSS (fallback) ranking": val["cvss_score"].fillna(-1),
}
point = {(m, k): recall_at_k(val, s, k) for m, s in SCORES.items() for k in KS}
boot = bootstrap_recall_at_k(val, SCORES, KS, n_boot=2000, seed=0)
ci = ci_table(boot, point)
for k in KS:
    ci.loc[("random", k), ["point", "lo", "hi"]] = [random_recall_at_k(val, k)] * 3
tbl = (ci.mul(100).round(1).reset_index().pivot(index="method", columns="k"))
tbl.columns = [f"{a}@{b}" for a, b in tbl.columns]
tbl = tbl[[f"{c}@{k}" for k in KS for c in ("point", "lo", "hi")]].sort_values(f"point@{K_MAIN}", ascending=False)
display(tbl); tbl.to_csv(TAB / "04_val_recall_at_k_ci.csv")

piv = boot.pivot_table(index=["rep", "k"], columns="method", values="recall")
for other in ["phrase, then CVSS (CNA)", "generous CVSS (fallback) ranking"]:
    for k in KS:
        dlt = (piv[other] - piv["CVSS (CNA) ranking"]).xs(k, level="k") * 100
        print(f"{other} minus CVSS (CNA) @K={k}: {dlt.mean():+.1f} pts, 95% CI {dlt.quantile(.025):+.1f} to "
              f"{dlt.quantile(.975):+.1f}, P(better) = {(dlt > 0).mean():.0%}")"""),
takeaway("samek"),

md("""## 4. Recall vs capacity
How many exploited CVEs does each ranking catch as the monthly budget grows? Dots = each rule followed
literally (section 2); vertical lines = the two capacities fixed in section 1."""),
code("""ks = np.unique(np.r_[np.arange(10, 100, 10), np.arange(100, 1151, 25)])
fig, ax = plt.subplots(figsize=(9, 4.4))
styles = {"CVSS (CNA) ranking": BLUE, "phrase, then CVSS (CNA)": ORANGE, "generous CVSS (fallback) ranking": INK2}
curves = {}
for m, s in SCORES.items():
    curves[m] = recall_curve(val, s, ks)
    ax.plot(ks, 100 * curves[m], color=styles[m], lw=2, ls="--" if "generous" in m else "-", label=m)
ax.plot(ks, [100 * random_recall_at_k(val, k) for k in ks], color=GRID, lw=2, label="random")
LABEL_POS = {"CVSS>=9 (CNA)": (8, -16), "CVSS>=7 (CNA)": (-150, 8), "phrase OR CVSS>=9 (CNA)": (-60, 10),
             "generous CVSS>=9 (fallback)": (10, -14)}
for name, color in [("CVSS>=9 (CNA)", BLUE), ("CVSS>=7 (CNA)", BLUE), ("phrase OR CVSS>=9 (CNA)", ORANGE), ("generous CVSS>=9 (fallback)", INK2)]:
    x, y = own.loc[name, "patches/month"], own.loc[name, "recall_%"]
    ax.scatter(x, y, s=60, color=color, zorder=5, edgecolor=SURF, linewidth=2)
    ax.annotate(f"{name} rule", (x, y), fontsize=8, color=INK2, xytext=LABEL_POS[name], textcoords="offset points")
for k in KS:
    ax.axvline(k, color=INK, lw=0.8, ls=":")
    ax.annotate(f"K={k}", (k, 2), fontsize=8, color=INK, xytext=(3, 0), textcoords="offset points")
ax.set_xlabel("emergency patches per month (K)"); ax.set_ylabel("% of exploited CVEs caught (recall@K)")
ax.set_title("Validation: recall vs monthly patching capacity", loc="left"); ax.set_ylim(0, 100); ax.set_xlim(0, 1150)
ax.legend(frameon=False, loc="upper left")
save(fig, "04_01_val_recall_vs_k")"""),
takeaway("curve"),

md("""## 5. Where the rules fail
Which exploited CVEs does the best rule still miss at K = K_MAIN? This is the gap a model has to close."""),
code("""from features import vendor_or_cna
s = SCORES["phrase, then CVSS (CNA)"]
val["_rank_in_month"] = s.groupby(val["published"].dt.to_period("M")).rank(ascending=False, method="average")
miss = val[(val["label"] == 1) & (val["_rank_in_month"] > K_MAIN)].copy()
miss["vendor_key"] = vendor_or_cna(miss)
print(f"exploited CVEs ranked below K={K_MAIN} by 'phrase, then CVSS (CNA)': {len(miss)} of {int(val['label'].sum())}")
print(f"  without any CNA CVSS score: {miss['cvss_cna_score'].isna().sum()}")
print("  top vendors:", miss["vendor_key"].value_counts().head(8).to_dict())
display(miss[["cve_id", "vendor_key", "cvss_cna_score", "cwe", "days_to_kev"]].sort_values("vendor_key").head(15))"""),
takeaway("miss"),

md("""## 6. Why we split by date: the random-split comparison (run once)
A random stratified split was proposed and rejected (Decision 6). Here we measure what it would have hidden.
One simple model — logistic regression on day-0 tabular features (vendor, CNA, CWE, CNA CVSS parts, phrase
flag; encoders fit inside the Pipeline) — is trained twice with identical settings:
**(a) date split**: train → val; **(b) random split** of the same train+val pool into sets of the same sizes.
Because the random validation set spans 30 months, capacity is expressed as the same **share of each month's
CVEs** (K_MAIN / val CVEs per month). *This is only a preview model; Phase 5 builds and tunes the real ones.*"""),
code("""from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from features import day0_tabular, CATEGORICAL, NUMERIC

def make_lr():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(class_weight="balanced", C=0.1, max_iter=3000))])

FRAC = K_MAIN / (len(val) / months_val)
pool = pd.concat([train, val], ignore_index=True)
Xpool = day0_tabular(pool)
res = {}
# (a) date split
is_tr = pool["split"].eq("train").to_numpy()
m = make_lr().fit(Xpool[is_tr], pool.loc[is_tr, "label"])
ev = pool[~is_tr].copy(); ev["p"] = m.predict_proba(Xpool[~is_tr])[:, 1]
res["date split (train -> val)"] = ev
# (b) random stratified split, same sizes
idx_tr, idx_ev = train_test_split(np.arange(len(pool)), test_size=len(val), stratify=pool["label"], random_state=42)
m2 = make_lr().fit(Xpool.iloc[idx_tr], pool.loc[idx_tr, "label"])
ev2 = pool.iloc[idx_ev].copy(); ev2["p"] = m2.predict_proba(Xpool.iloc[idx_ev])[:, 1]
res["random split"] = ev2

def twins(train_part, eval_part):
    key = lambda d: vendor_or_cna(d) + "|" + d["published"].dt.strftime("%Y-%m")
    pos_keys = set(key(train_part[train_part["label"] == 1]))
    e = eval_part[eval_part["label"] == 1]
    return key(e).isin(pos_keys).mean()

rows = []
for name, ev_ in res.items():
    trp = pool[is_tr] if name.startswith("date") else pool.iloc[idx_tr]
    rows.append({"split": name, "model": "preview LR (C=0.1, balanced)", "eval CVEs": len(ev_),
                 "eval positives": int(ev_["label"].sum()),
                 "budget": f"{FRAC:.2%} of each month's CVEs",
                 f"recall@{FRAC:.1%} of monthly CVEs": round(100 * recall_at_budget(ev_, "p", FRAC), 1),
                 "PR-AUC": round(pr_auc(ev_, "p"), 3),
                 "eval positives with a 'twin' in train (same vendor+month, exploited) %": round(100 * twins(trp, ev_), 1)})
rs = pd.DataFrame(rows).set_index("split"); display(rs); rs.to_csv(TAB / "04_random_vs_date_split.csv")
LR_DATE_R110 = recall_at_k(res["date split (train -> val)"], "p", K_MAIN)
LR_DATE_R306 = recall_at_k(res["date split (train -> val)"], "p", K_GEN)
print(f"Budget note: {FRAC:.2%} of each month's CVEs = {K_MAIN} on an average val month, but busier months get more slots.")
print(f"Same preview LR on the date split with a flat budget: recall@{K_MAIN}/month = {LR_DATE_R110:.1%}, "
      f"recall@{K_GEN}/month = {LR_DATE_R306:.1%}  (vs {recall_at_budget(res['date split (train -> val)'], 'p', FRAC):.1%} "
      f"with the share-of-month budget)")

fig, ax = plt.subplots(figsize=(6.5, 3))
col = f"recall@{FRAC:.1%} of monthly CVEs"
ax.barh(rs.index[::-1], rs[col][::-1], color=[BLUE, ORANGE], height=0.5)
for y, v in enumerate(rs[col][::-1]):
    ax.annotate(f"{v:.1f}%", (v, y), va="center", fontsize=9, color=INK2, xytext=(4, 0), textcoords="offset points")
ax.set_xlim(0, 100); ax.set_xlabel("% of exploited CVEs caught at the same capacity"); ax.grid(axis="y", visible=False)
ax.set_title("Same model, same capacity: random split looks better", loc="left")
save(fig, "04_02_random_vs_date_split")"""),
takeaway("split"),

md("## 7. Comparison table (validation) — all methods so far"),
code("""rows = []
own_map = {"CVSS (CNA) ranking": "CVSS>=9 (CNA)", "phrase, then CVSS (CNA)": "phrase OR CVSS>=9 (CNA)",
           "generous CVSS (fallback) ranking": "generous CVSS>=9 (fallback)"}
def fmt(m, k):
    r = ci.loc[(m, k)]
    return f"{100*r['point']:.1f} [{100*r['lo']:.0f}–{100*r['hi']:.0f}]" if r["lo"] != r["hi"] else f"{100*r['point']:.1f}"
for m in ["random", "CVSS (CNA) ranking", "phrase, then CVSS (CNA)", "generous CVSS (fallback) ranking"]:
    o = own.loc[own_map[m]] if m in own_map else None
    rows.append({"method": m, "type": "benchmark" if "generous" not in m else "benchmark (hindsight, upper bound)",
                 "rule patches/month": "" if o is None else int(o["patches/month"]),
                 "rule recall %": "" if o is None else o["recall_%"],
                 f"recall@{K_MAIN} % [95% CI]": fmt(m, K_MAIN), f"recall@{K_GEN} % [95% CI]": fmt(m, K_GEN),
                 "PR-AUC": "" if m == "random" else round(pr_auc(val, SCORES[m]), 3)})
rows.append({"method": "CVSS>=7 (CNA) rule", "type": "benchmark", "rule patches/month": int(own.loc["CVSS>=7 (CNA)", "patches/month"]),
             "rule recall %": own.loc["CVSS>=7 (CNA)", "recall_%"], f"recall@{K_MAIN} % [95% CI]": "= CVSS (CNA) ranking",
             f"recall@{K_GEN} % [95% CI]": "= CVSS (CNA) ranking", "PR-AUC": ""})
rows.append({"method": "preview: logistic regression (day-0 tabular)", "type": "model preview (Phase 5)",
             "rule patches/month": "", "rule recall %": "", f"recall@{K_MAIN} % [95% CI]": f"{100*LR_DATE_R110:.1f}",
             f"recall@{K_GEN} % [95% CI]": f"{100*LR_DATE_R306:.1f}", "PR-AUC": round(pr_auc(res['date split (train -> val)'], 'p'), 3)})
comp = pd.DataFrame(rows).set_index("method"); display(comp); comp.to_csv(TAB / "04_val_comparison.csv")"""),
takeaway("final"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "notebooks" / "04_baselines.ipynb")
print("written")
