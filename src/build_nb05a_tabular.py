"""Builds notebooks/05a_tabular_models.ipynb (Phase 5a). Takeaways in src/nb05a_takeaways.json."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TKF = ROOT / "src" / "nb05a_takeaways.json"
TK = json.loads(TKF.read_text()) if TKF.exists() else {}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
def takeaway(key):
    return md("**Takeaway**: " + TK.get(key, "_(filled after the first run)_"))

cells = [
md("""# Phase 5a — Tabular models (no text yet)

**Goal:** beat the best day-0 rule ("phrase, then CVSS (CNA)": 28.1% of exploited CVEs at K = 110/month, 42.7% at
K = 305 on validation) using only structured publication-day information.
**Selection metric:** recall@K per calendar month (never AUC). The test set is not touched."""),
code("""import os, sys, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl")); os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from metrics import recall_at_k, recall_curve, random_recall_at_k, bootstrap_recall_at_k, ci_table, pr_auc, recall_at_budget
from description_leak import PHRASES, phrase_flags
from features import load_kev_raw, clean_vendor, kev_history_features
from tabular_models import build_X, make_lr, make_hgb, evaluate, grid_search, views

FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
BLUE, ORANGE, AQUA, VIOLET, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "font.size": 9.5})
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")

dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
kev = load_kev_raw(sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1])
K_MAIN, K_GEN = 110, 305                       # fixed from train in Phase 4
val_months = dev.loc[dev["split"] == "val", "published"].dt.to_period("M").nunique()
FRAC = K_MAIN / (dev["split"].eq("val").sum() / val_months)
dev["phrase_day0"] = phrase_flags(dev["description_day0"], PHRASES).any(axis=1)
print(f"dev: {len(dev):,} CVEs ({int(dev['label'].sum())} exploited); KEV snapshot rows: {len(kev):,}")"""),

md("""## 1. Features: what is known on publication day
| group | features | source |
|---|---|---|
| CVSS (CNA only, Decision 9) | score, has-score flag, base-metric letters (AV, AC, PR, UI, S, C, I, A, AT) | the CNA's own vector |
| weakness (Decision 10a) | CNA CWE, number of CNA CWEs | CNA weakness entry |
| who / what | vendor (cleaned; CNA if missing), CNA | CNA "affected" block |
| text-derived (no NLP yet) | exploitation phrase in `description_day0`, description length | Decision 10c |
| **KEV history (new)** | vendor's KEV entries all-time and last 365 days (log), days since its last KEV entry, any-KEV flag, product's KEV entries | KEV entries with dateAdded **strictly before** publication, **never the CVE's own entry** |

KEV `vendorProject` names are mapped with the same `clean_vendor` rule plus four spelling aliases
(e.g. KEV "Juniper" = CNA "Juniper Networks")."""),
code("""X = build_X(dev, kev)
vk = set(clean_vendor(dev["vendor"]).dropna())
print(f"KEV rows whose vendor matches a dev CVE vendor: {kev['vkey'].isin(vk).mean():.1%} "
      f"({kev['vkey'].isin(vk).sum():,} of {len(kev):,}); KEV vendors matched: {pd.Series(kev['vkey'].unique()).isin(vk).mean():.1%}")
neg_gap = dev[dev["days_to_kev"] < 0]
print(f"leakage guard: {len(neg_gap)} dev CVEs were in KEV before NVD published them; their own entry is excluded "
      f"(checked in features._history: dateAdded < publication AND cve_id != own)")
tr = dev["split"].eq("train")
t = pd.DataFrame({"vendor has earlier KEV entries": X.loc[tr, "vendor_kev_any"].map({1: "yes", 0: "no"}), "label": dev.loc[tr, "label"]})
g = t.groupby("vendor has earlier KEV entries")["label"].agg(CVEs="size", exploited="sum")
g["rate_%"] = (100 * g["exploited"] / g["CVEs"]).round(2); g["share_of_exploited_%"] = (100 * g["exploited"] / g["exploited"].sum()).round(1)
print("train only:"); display(g)
print(f"feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")"""),
takeaway("features"),

md("""## 2. Evaluation protocol
Validation alone has only 68 exploited CVEs (CI ≈ ±10 points), so every candidate is scored on **three views
inside dev**, each fitting on the past and scoring the future:
* **val** — fit on train (2022-01 → 2024-06), score 2024-H2 (recall@110 and @305 per month)
* **walk-forward** — fit 2022 → score 2023; fit 2022–2023 → score 2024-H1 (recall@110 per month)
* **grouped CV** inside train — 5 folds grouped by `cv_group` (vendor|month), so twin CVEs never sit on both
  sides; recall at the same share-of-month budget (3.4%) because folds mix months. Reported, not used to select.

**Selection score = mean recall@110 over the three periods (WF 2023, WF 2024-H1, val), equal weights** — the
selection rule for every Phase 5 model. The worst fold is reported too. The rules are scored on exactly the same views:"""),
code("""rule_scores = {"CVSS (CNA) ranking": dev["cvss_cna_score"].fillna(-1),
               "phrase, then CVSS (CNA)": 100 * dev["phrase_day0"] + dev["cvss_cna_score"].fillna(-1)}
rows = []
for name, s in rule_scores.items():
    r = {"method": name}
    for v, (_, ev) in views(dev).items():
        r[f"{v}@{K_MAIN}"] = recall_at_k(dev[ev], s[ev], K_MAIN)
    ev = dev["split"].eq("val").to_numpy()
    r[f"val@{K_GEN}"] = recall_at_k(dev[ev], s[ev], K_GEN)
    r["wf_mean"] = np.mean([r[f"wf_2023@{K_MAIN}"], r[f"wf_2024h1@{K_MAIN}"]]); r["wf_worst"] = min(r[f"wf_2023@{K_MAIN}"], r[f"wf_2024h1@{K_MAIN}"])
    r["selection"] = np.mean([r[f"wf_2023@{K_MAIN}"], r[f"wf_2024h1@{K_MAIN}"], r[f"val@{K_MAIN}"]])
    rows.append(r)
rules_tbl = pd.DataFrame(rows).set_index("method")
for v, (_, ev) in views(dev).items():
    print(f"{v:10s}: {ev.sum():>6,} CVEs, {int(dev.loc[ev, 'label'].sum()):>3} exploited, random recall@{K_MAIN} = {random_recall_at_k(dev[ev], K_MAIN):.1%}")
display((100 * rules_tbl).round(1))"""),
takeaway("protocol"),

md("""## 3. Logistic regression
One-hot encoding of vendor, CNA, CWE and CVSS letters (categories seen fewer than `min_freq` times in the
training part are grouped as "infrequent"), standardised numeric features, **class_weight = balanced** for the
1-in-300 imbalance. Grid: regularisation C × rare-category threshold."""),
code("""LR_GRID = {"C": [0.01, 0.03, 0.1, 0.3, 1.0], "min_freq": [5, 20]}
lr_res = grid_search("lr", LR_GRID, dev, X, K_MAIN, K_GEN, FRAC, log=lambda s: None)
show = ["C", "min_freq", f"val@{K_MAIN}", f"val@{K_GEN}", f"wf_2023@{K_MAIN}", f"wf_2024h1@{K_MAIN}", "wf_worst", f"gcv@{FRAC:.1%}", "val_pr_auc", "selection"]
display(lr_res[show].round(3)); lr_res.to_csv(TAB / "05a_lr_grid.csv", index=False)
best_lr = lr_res.iloc[0][["C", "min_freq"]].to_dict(); best_lr["min_freq"] = int(best_lr["min_freq"])
print("selected LR:", best_lr)"""),
takeaway("lr"),

md("""## 4. Gradient boosting (HistGradientBoostingClassifier, Decision 7)
Trees can combine signals ("Microsoft **and** network **and** vendor had a KEV entry last month") that a linear
model cannot. High-cardinality columns (vendor, CNA, CWE) are encoded two ways, chosen by the grid:
**target encoding** (smoothed exploitation rate of the category, cross-fitted with a seeded 5-fold split inside each training part) or
**one-hot of frequent categories** (≥ 100 CVEs). CVSS letters use native categorical splits.
**class_weight = balanced**; fixed 300 boosting rounds."""),
code("""HGB_GRID = {"encoding": ["target", "onehot"], "learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 15], "min_samples_leaf": [20, 100]}
hgb_res = grid_search("hgb", HGB_GRID, dev, X, K_MAIN, K_GEN, FRAC, log=lambda s: None)
show_h = list(HGB_GRID) + show[2:]
display(hgb_res[show_h].round(3)); hgb_res.to_csv(TAB / "05a_hgb_grid.csv", index=False)
best_hgb = hgb_res.iloc[0][list(HGB_GRID)].to_dict()
best_hgb = {k: (int(v) if k in ("max_leaf_nodes", "min_samples_leaf") else v) for k, v in best_hgb.items()}
print("selected HGB:", best_hgb)"""),
takeaway("hgb"),

md("""## 5. The selected models on validation, with bootstrap CIs
Refit each selected configuration on the full train set, score val, and compare with the best rule using a
**paired** bootstrap (2,000 within-month resamples; the same resample for every method)."""),
code("""is_tr, is_val = dev["split"].eq("train").to_numpy(), dev["split"].eq("val").to_numpy()
val = dev[is_val].reset_index(drop=True)
y = dev["label"].to_numpy()
lr_model = make_lr(**best_lr).fit(X[is_tr], y[is_tr])
hgb_model = make_hgb(**best_hgb).fit(X[is_tr], y[is_tr])
SC = {"CVSS (CNA) ranking": rule_scores["CVSS (CNA) ranking"][is_val].reset_index(drop=True),
      "phrase, then CVSS (CNA)": rule_scores["phrase, then CVSS (CNA)"][is_val].reset_index(drop=True),
      "logistic regression": pd.Series(lr_model.predict_proba(X[is_val])[:, 1]),
      "gradient boosting": pd.Series(hgb_model.predict_proba(X[is_val])[:, 1])}
point = {(m, k): recall_at_k(val, s, k) for m, s in SC.items() for k in (K_MAIN, K_GEN)}
boot = bootstrap_recall_at_k(val, SC, [K_MAIN, K_GEN], n_boot=2000, seed=0)
ci = ci_table(boot, point)
tbl = ci.mul(100).round(1).reset_index().pivot(index="method", columns="k")
tbl.columns = [f"{a}@{b}" for a, b in tbl.columns]
display(tbl[[f"{c}@{k}" for k in (K_MAIN, K_GEN) for c in ("point", "lo", "hi")]].sort_values(f"point@{K_MAIN}", ascending=False))
piv = boot.pivot_table(index=["rep", "k"], columns="method", values="recall")
for mname in ["logistic regression", "gradient boosting"]:
    for k in (K_MAIN, K_GEN):
        d = 100 * (piv[mname] - piv["phrase, then CVSS (CNA)"]).xs(k, level="k")
        print(f"{mname} minus best rule @K={k}: {d.mean():+.1f} pts [{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}], P(better) = {(d > 0).mean():.0%}")
d = 100 * (piv["gradient boosting"] - piv["logistic regression"]).xs(K_MAIN, level="k")
print(f"gradient boosting minus logistic regression @K={K_MAIN}: {d.mean():+.1f} pts [{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}], P(better) = {(d > 0).mean():.0%}")
ci.to_csv(TAB / "05a_val_ci.csv")"""),
takeaway("ci"),

md("""## 6. Recall vs capacity (validation)"""),
code("""ks = np.unique(np.r_[np.arange(10, 100, 10), np.arange(100, 1001, 25)])
fig, ax = plt.subplots(figsize=(9, 4.4))
style = {"CVSS (CNA) ranking": (BLUE, "--"), "phrase, then CVSS (CNA)": (ORANGE, "--"),
         "logistic regression": (AQUA, "-"), "gradient boosting": (VIOLET, "-")}
for m, s in SC.items():
    c, ls = style[m]
    ax.plot(ks, 100 * recall_curve(val, s, ks), color=c, ls=ls, lw=2, label=m)
ax.plot(ks, [100 * random_recall_at_k(val, k) for k in ks], color=GRID, lw=2, label="random")
for k in (K_MAIN, K_GEN):
    ax.axvline(k, color=INK, lw=0.8, ls=":"); ax.annotate(f"K={k}", (k, 2), fontsize=8, xytext=(3, 0), textcoords="offset points")
ax.set_xlabel("emergency patches per month (K)"); ax.set_ylabel("% of exploited CVEs caught (recall@K)")
ax.set_ylim(0, 100); ax.set_xlim(0, 1000); ax.legend(frameon=False, loc="upper left")
ax.set_title("Validation: tabular models vs day-0 rules", loc="left")
save(fig, "05a_01_val_recall_vs_k")"""),
takeaway("curve"),

md("""## 7. What the models rely on
Permutation importance on validation (drop in average precision when one feature is shuffled; interpretation
only, not used for any choice). SHAP explanations come in Phase 7."""),
code("""from sklearn.inspection import permutation_importance
imp = {}
for name, mdl in [("logistic regression", lr_model), ("gradient boosting", hgb_model)]:
    r = permutation_importance(mdl, X[is_val], y[is_val], scoring="average_precision", n_repeats=5, random_state=0)
    imp[name] = pd.Series(r.importances_mean, index=X.columns)
imp = pd.DataFrame(imp).sort_values("gradient boosting", ascending=False)
display(imp.head(10).round(4)); imp.to_csv(TAB / "05a_permutation_importance.csv")
top = imp.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4))
yy = np.arange(len(top))
ax.barh(yy + 0.2, top["gradient boosting"], height=0.38, color=VIOLET, label="gradient boosting")
ax.barh(yy - 0.2, top["logistic regression"], height=0.38, color=AQUA, label="logistic regression")
ax.set_yticks(yy, top.index); ax.grid(axis="y", visible=False); ax.legend(frameon=False, loc="lower right")
ax.set_xlabel("drop in average precision when shuffled"); ax.set_title("Top 10 features (validation)", loc="left")
save(fig, "05a_02_permutation_importance")"""),
takeaway("importance"),

md("## 8. Comparison table — all methods so far (validation + walk-forward)"),
code("""def fmt(m, k):
    r = ci.loc[(m, k)]; return f"{100*r['point']:.1f} [{100*r['lo']:.0f}–{100*r['hi']:.0f}]"
lr_row, hgb_row = lr_res.iloc[0], hgb_res.iloc[0]
rows = [{"method": "random", "type": "reference", f"val recall@{K_MAIN} [CI]": f"{100*random_recall_at_k(val, K_MAIN):.1f}",
         f"val recall@{K_GEN} [CI]": f"{100*random_recall_at_k(val, K_GEN):.1f}", "WF 2023": "", "WF 2024-H1": "", "WF worst": "", "val PR-AUC": ""}]
for m in ["CVSS (CNA) ranking", "phrase, then CVSS (CNA)"]:
    r = rules_tbl.loc[m]
    rows.append({"method": m, "type": "day-0 rule", f"val recall@{K_MAIN} [CI]": fmt(m, K_MAIN), f"val recall@{K_GEN} [CI]": fmt(m, K_GEN),
                 "WF 2023": round(100*r[f"wf_2023@{K_MAIN}"], 1), "WF 2024-H1": round(100*r[f"wf_2024h1@{K_MAIN}"], 1),
                 "WF worst": round(100*r["wf_worst"], 1), "val PR-AUC": round(pr_auc(val, SC[m]), 3)})
rows.append({"method": "generous CVSS (fallback) ranking", "type": "rule with hindsight (upper bound)",
             f"val recall@{K_MAIN} [CI]": "15.1 [6–25]", f"val recall@{K_GEN} [CI]": "38.5 [27–51]", "WF 2023": "", "WF 2024-H1": "", "WF worst": "", "val PR-AUC": "0.011"})
for m, r in [("logistic regression", lr_row), ("gradient boosting", hgb_row)]:
    rows.append({"method": m, "type": "model (tabular)", f"val recall@{K_MAIN} [CI]": fmt(m, K_MAIN), f"val recall@{K_GEN} [CI]": fmt(m, K_GEN),
                 "WF 2023": round(100*r[f"wf_2023@{K_MAIN}"], 1), "WF 2024-H1": round(100*r[f"wf_2024h1@{K_MAIN}"], 1),
                 "WF worst": round(100*r["wf_worst"], 1), "val PR-AUC": round(pr_auc(val, SC[m]), 3)})
comp = pd.DataFrame(rows).set_index("method"); display(comp); comp.to_csv(TAB / "05a_comparison.csv")"""),
takeaway("final"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "notebooks" / "05a_tabular_models.ipynb")
print("written")
