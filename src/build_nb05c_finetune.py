"""Builds notebooks/05c_finetuned_transformer.ipynb (Phase 5c). Takeaways in src/nb05c_takeaways.json.
Reads the fine-tuning outputs written by `src/gpu_run.py finetune ...` (data/finetune/); no GPU needed here."""
import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TKF = ROOT / "src" / "nb05c_takeaways.json"
TK = json.loads(TKF.read_text()) if TKF.exists() else {}
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
def takeaway(key):
    return md("**Takeaway**: " + TK.get(key, "_(filled after the first run)_"))

cells = [
md("""# Phase 5c — Fine-tuning a small transformer on CVE descriptions

In 5b the sentence encoder was **frozen**: only a small model on top learned from our labels (transfer learning).
Here the whole encoder (MiniLM-L6, 22M parameters; the 6b winner) is **fine-tuned end-to-end** on the supervised
task, so its internal representation can adapt to CVE language. The risk is overfitting: 22M parameters and only
a few hundred exploited CVEs per training period.

**Protocol (fixed in advance, run by `src/gpu_run.py finetune` on the Apple GPU, dev only):**
* class-weighted BCE; AdamW, weight decay 0.01, dropout 0.1, max sequence length 128, warm-up 10% then linear decay
* per period: epoch search (max **3** epochs) on the training period minus its last 6 months, stopping set = those
  6 months; then **refit on the full period** for the best epoch count (same seed); score the evaluation rows
* grid of **4 settings**: learning rate {1e-5, 2e-5} × {all layers trained, embeddings + lower 3 of 6 layers frozen}
* deviations, for the ~1.5 h GPU budget: negatives in training/stopping sets **subsampled to 25%** (all positives
  kept; one fixed subsample seed, so every setting sees the same CVEs); **2 seeds** for the chosen setting (not 3);
  frozen settings use "eager" attention (the fused MPS attention kernel has no dropout when layers are frozen)
* same selection rule as all Phase 5 models: mean recall@110 per month over WF 2023, WF 2024-H1, val"""),
code("""import os, sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "mpl")); os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from metrics import recall_at_k, recall_curve, random_recall_at_k, bootstrap_recall_at_k, ci_table, pr_auc
from description_leak import PHRASES, phrase_flags
from features import load_kev_raw
from tabular_models import build_X, make_hgb, views
import text_models as txm

FIG, TAB, FT = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables", ROOT / "data" / "finetune"
BLUE, ORANGE, AQUA, VIOLET, MAGENTA, GREEN, INK, INK2, GRID, SURF = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
    "#e87ba4", "#008300", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb")
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 10, "font.size": 9.5})
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
def save(fig, name): fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
K_MAIN, K_GEN = 110, 305
PERIODS = ["wf_2023", "wf_2024h1", "val"]
SETTINGS = ["lr1e-5_full", "lr2e-5_full", "lr1e-5_freeze", "lr2e-5_freeze"]
HGB_BEST = dict(encoding="onehot", learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100)

dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
y = dev["label"].to_numpy()
pos = pd.Series(np.arange(len(dev)), index=dev["cve_id"])

def ft_scores(setting, seed):
    \"\"\"{period: scores aligned to that period's evaluation rows (dev order)}\"\"\"
    out = {}
    for p, (_, ev) in views(dev).items():
        s = pd.read_parquet(FT / f"{setting}_s{seed}_{p}.parquet").set_index("cve_id")["score"]
        out[p] = s.reindex(dev.loc[ev, "cve_id"]).to_numpy()
        assert not np.isnan(out[p]).any()
    return out

def summarize(scores):
    r = {}
    for p, (_, ev) in views(dev).items():
        part = dev[ev]
        r[f"{p}@{K_MAIN}"] = recall_at_k(part, pd.Series(scores[p], index=part.index), K_MAIN)
        if p == "val": r[f"val@{K_GEN}"] = recall_at_k(part, pd.Series(scores[p], index=part.index), K_GEN)
    pp = [r[f"{p}@{K_MAIN}"] for p in PERIODS]
    r["worst"], r["selection"] = min(pp), float(np.mean(pp))
    return r
print("fine-tuning outputs:", len(list(FT.glob("*.parquet"))), "score files")"""),

md("""## 1. The grid (seed 0)"""),
code("""rows = []
for st in SETTINGS:
    info = [json.loads((FT / f"{st}_s0_{p}.json").read_text()) for p in PERIODS]
    r = summarize(ft_scores(st, 0))
    rows.append({"setting": st, **{k: round(100 * v, 1) for k, v in r.items()},
                 "best epochs (WF23, WF24H1, val)": str([i["best_epoch"] for i in info]),
                 "GPU minutes": round(sum(i["minutes"] for i in info), 1)})
grid = pd.DataFrame(rows).sort_values("selection", ascending=False).reset_index(drop=True)
display(grid); grid.to_csv(TAB / "05c_finetune_grid.csv", index=False)
CHOSEN = grid.loc[0, "setting"]
print(f"selected setting: {CHOSEN} | grid GPU time: {grid['GPU minutes'].sum():.1f} min")"""),
takeaway("grid"),

md("""## 2. Overfitting check: training vs stopping-set loss per epoch
Each panel = one period; one colour per setting; solid = training loss, dashed = stopping-set loss (same
class weight). **Overfitting** = the stopping-set loss rises after its minimum while the training loss keeps
falling. (Training loss is averaged over the epoch while the model changes; stopping-set loss is measured at
the end of the epoch, so the curves are not exactly comparable in level, only in direction.)"""),
code("""colors = dict(zip(SETTINGS, [BLUE, ORANGE, AQUA, VIOLET]))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
diag = []
titles = {"wf_2023": "WF 2023\\nsearch 2022-H1, stop 2022-H2", "wf_2024h1": "WF 2024-H1\\nsearch 2022–23-H1, stop 2023-H2",
          "val": "val\\nsearch 2022–23, stop 2024-H1"}
for ax, p in zip(axes, PERIODS):
    for st in SETTINGS:
        h = pd.DataFrame(json.loads((FT / f"{st}_s0_{p}.json").read_text())["search_history"])
        ax.plot(h["epoch"], h["train_loss"], color=colors[st], lw=2, label=f"{st} train")
        ax.plot(h["epoch"], h["es_loss"], color=colors[st], lw=2, ls="--", label=f"{st} stop")
        best = int(h["es_loss"].idxmin())
        rises = bool((h["es_loss"].iloc[best + 1:] > h["es_loss"].iloc[best] + 1e-3).any())
        diag.append({"period": p, "setting": st, "stop-loss min at epoch": best + 1,
                     "stop-loss rise after min": round(float(h["es_loss"].iloc[-1] - h["es_loss"].iloc[best]), 3),
                     "train loss fell": bool(h["train_loss"].iloc[-1] < h["train_loss"].iloc[0]),
                     "train-stop gap at epoch 3": round(float(h["es_loss"].iloc[-1] - h["train_loss"].iloc[-1]), 3),
                     "overfitting": "yes" if rises else "no"})
    ax.set_title(titles[p], loc="left"); ax.set_xlabel("epoch"); ax.set_xticks([1, 2, 3])
axes[0].set_ylabel("class-weighted BCE")
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.18), fontsize=8)
fig.suptitle("Fine-tuning MiniLM: training (solid) vs stopping-set loss (dashed), seed 0", x=0.01, y=1.08, ha="left", fontweight="bold")
save(fig, "05c_01_finetune_loss_curves")
diag = pd.DataFrame(diag); display(diag); diag.to_csv(TAB / "05c_overfitting_diagnostics.csv", index=False)"""),
takeaway("overfit"),

md("""## 3. The chosen setting over 2 seeds vs the frozen-encoder text models
The 5b reference models are recomputed here on the same three periods so every number is directly comparable."""),
code("""kev = load_kev_raw(sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1])
X_tab = build_X(dev, kev)
E = txm.load_embeddings("minilm", dev)
REF = {}
rule = 100 * phrase_flags(dev["description_day0"], PHRASES).any(axis=1).to_numpy() + dev["cvss_cna_score"].fillna(-1).to_numpy()
REF["phrase, then CVSS (CNA)"] = txm.eval_periods(dev, lambda f, e: rule[e])
REF["gradient boosting (05a)"] = txm.eval_periods(dev, lambda f, e: make_hgb(**HGB_BEST).fit(X_tab[f], y[f]).predict_proba(X_tab[e])[:, 1])
REF["text LR (05b)"] = txm.eval_periods(dev, txm.text_lr_scorer(E, y, 1.0))
mlp_runs = [txm.eval_periods(dev, txm.mlp_scorer(dev, E, y, (64,), 1e-3, s)) for s in range(5)]
REF["text MLP (05b, 5 seeds)"] = {**{k: np.mean([r[k] for r in mlp_runs]) for k in mlp_runs[0] if not k.startswith("_")},
                                  "_scores": {p: np.mean([r["_scores"][p] for r in mlp_runs], axis=0) for p in PERIODS},
                                  "_worst_seed": min(r["selection"] for r in mlp_runs)}
ft_runs = [summarize(ft_scores(CHOSEN, s)) for s in (0, 1)]
ft_ens = {p: np.mean([ft_scores(CHOSEN, s)[p] for s in (0, 1)], axis=0) for p in PERIODS}
REF[f"fine-tuned MiniLM ({CHOSEN}, 2 seeds)"] = {**{k: np.mean([r[k] for r in ft_runs]) for k in ft_runs[0]},
                                                "_scores": ft_ens, "_worst_seed": min(r["selection"] for r in ft_runs)}
seed_tbl = pd.DataFrame([{"seed": s, **{k: round(100 * v, 1) for k, v in r.items()}} for s, r in zip((0, 1), ft_runs)])
print(f"fine-tuned {CHOSEN}, per seed:"); display(seed_tbl)
cmp_ = pd.DataFrame([{"model": m, **{k: round(100 * v, 1) for k, v in r.items() if not k.startswith("_")},
                      "worst seed (selection)": round(100 * r["_worst_seed"], 1) if "_worst_seed" in r else ""} for m, r in REF.items()]).set_index("model")
display(cmp_)
FT_NAME = f"fine-tuned MiniLM ({CHOSEN}, 2 seeds)"
BEATS = REF[FT_NAME]["selection"] > REF["text MLP (05b, 5 seeds)"]["selection"]
print(f"Decision 11b: fine-tuned beats the text MLP on the selection rule? {BEATS} "
      f"({100*REF[FT_NAME]['selection']:.1f} vs {100*REF['text MLP (05b, 5 seeds)']['selection']:.1f})")"""),
takeaway("seeds"),

md("""## 4. Late fusion and the final-model rule (Decision 11)
Late fusion = average of within-month percentile ranks of gradient boosting and the text model. Decision 11b:
the fine-tuned model replaces the text MLP inside the fusion **only if** it beats the text MLP on the selection
rule. Both fusions are shown for transparency; only the rule decides."""),
code("""def fuse(a, b):
    out = {}
    for p, (_, ev) in views(dev).items():
        part = dev[ev]
        out[p] = (txm.month_pct_rank(part, a[p]) + txm.month_pct_rank(part, b[p])) / 2
    return out
REF["late fusion: boosting + text MLP"] = {**summarize(fuse(REF["gradient boosting (05a)"]["_scores"], REF["text MLP (05b, 5 seeds)"]["_scores"])),
    "_scores": fuse(REF["gradient boosting (05a)"]["_scores"], REF["text MLP (05b, 5 seeds)"]["_scores"])}
REF["late fusion: boosting + fine-tuned (info only)"] = {**summarize(fuse(REF["gradient boosting (05a)"]["_scores"], ft_ens)),
    "_scores": fuse(REF["gradient boosting (05a)"]["_scores"], ft_ens)}
fusion_text = "late fusion: boosting + fine-tuned (info only)" if BEATS else "late fusion: boosting + text MLP"
if BEATS:
    REF["late fusion: boosting + fine-tuned"] = REF.pop("late fusion: boosting + fine-tuned (info only)"); fusion_text = "late fusion: boosting + fine-tuned"
# Decision 11a: highest selection wins; within 2 points -> better worst period
cand = {m: REF[m] for m in ["gradient boosting (05a)", fusion_text]}
best_sel = max(c["selection"] for c in cand.values())
close = {m: c for m, c in cand.items() if best_sel - c["selection"] <= 0.02}
FINAL = max(close, key=lambda m: (close[m]["worst"], close[m]["selection"]))
for m, c in cand.items():
    print(f"{m:40s} selection {100*c['selection']:.1f} | worst period {100*c['worst']:.1f}")
print(f"-> Decision 11 final model: {FINAL}")"""),
takeaway("fusion"),

md("""## 5. Validation with bootstrap CIs and recall vs capacity
Neural-net scores on val are seed ensembles (fine-tuned: 2 seeds; text MLP: 5 seeds)."""),
code("""val_mask = dev["split"].eq("val").to_numpy(); val = dev[val_mask].reset_index(drop=True)
cands = ["phrase, then CVSS (CNA)", "gradient boosting (05a)", "text MLP (05b, 5 seeds)", FT_NAME,
         "late fusion: boosting + text MLP", [k for k in REF if k.startswith("late fusion: boosting + fine-tuned")][0]]
SC = {c: pd.Series(REF[c]["_scores"]["val"]) for c in cands}
point = {(m, k): recall_at_k(val, s, k) for m, s in SC.items() for k in (K_MAIN, K_GEN)}
boot = bootstrap_recall_at_k(val, SC, [K_MAIN, K_GEN], n_boot=2000, seed=0)
ci = ci_table(boot, point)
tbl = ci.mul(100).round(1).reset_index().pivot(index="method", columns="k"); tbl.columns = [f"{a}@{b}" for a, b in tbl.columns]
display(tbl[[f"{c}@{k}" for k in (K_MAIN, K_GEN) for c in ("point", "lo", "hi")]].sort_values(f"point@{K_MAIN}", ascending=False))
piv = boot.pivot_table(index=["rep", "k"], columns="method", values="recall")
for base in ["text MLP (05b, 5 seeds)", "phrase, then CVSS (CNA)"]:
    d = 100 * (piv[FT_NAME] - piv[base]).xs(K_MAIN, level="k")
    print(f"fine-tuned minus {base} @110: {d.mean():+.1f} [{d.quantile(.025):+.1f}, {d.quantile(.975):+.1f}], P(better) = {(d > 0).mean():.0%}")
ci.to_csv(TAB / "05c_val_ci.csv")

ks = np.unique(np.r_[np.arange(10, 100, 10), np.arange(100, 1001, 25)])
style = {"phrase, then CVSS (CNA)": (ORANGE, "--"), "gradient boosting (05a)": (VIOLET, "-"), "text MLP (05b, 5 seeds)": (MAGENTA, "-"),
         FT_NAME: (BLUE, "-"), "late fusion: boosting + text MLP": (AQUA, "-")}
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for m in cands:
    if m not in style: continue
    c, ls = style[m]; ax.plot(ks, 100 * recall_curve(val, SC[m], ks), color=c, ls=ls, lw=2, label=m)
ax.plot(ks, [100 * random_recall_at_k(val, k) for k in ks], color=GRID, lw=2, label="random")
for k in (K_MAIN, K_GEN):
    ax.axvline(k, color=INK, lw=0.8, ls=":"); ax.annotate(f"K={k}", (k, 2), fontsize=8, xytext=(3, 0), textcoords="offset points")
ax.set_xlabel("emergency patches per month (K)"); ax.set_ylabel("% of exploited CVEs caught (recall@K)")
ax.set_ylim(0, 100); ax.set_xlim(0, 1000); ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.set_title("Validation: fine-tuned transformer vs frozen-encoder text model", loc="left")
save(fig, "05c_02_val_recall_vs_k")"""),
takeaway("ci"),

md("## 6. Comparison table — all Phase 5 methods (three periods + validation CIs)"),
code("""prev = pd.read_csv(TAB / "05b_comparison.csv")
def fmt(m, k):
    if (m, k) not in ci.index: return ""
    r = ci.loc[(m, k)]; return f"{100*r['point']:.1f} [{100*r['lo']:.0f}–{100*r['hi']:.0f}]"
rows = []
for m in ["phrase, then CVSS (CNA)", "gradient boosting (05a)", "text LR (05b)", "text MLP (05b, 5 seeds)", FT_NAME,
          "late fusion: boosting + text MLP", [k for k in REF if k.startswith("late fusion: boosting + fine-tuned")][0]]:
    r = REF[m]
    rows.append({"method": m, "selection (3-period mean)": round(100 * r["selection"], 1), "WF 2023": round(100 * r["wf_2023@110"], 1),
                 "WF 2024-H1": round(100 * r["wf_2024h1@110"], 1), "worst period": round(100 * r["worst"], 1),
                 "worst seed": round(100 * r["_worst_seed"], 1) if "_worst_seed" in r else "",
                 "val recall@110 [CI]": fmt(m, K_MAIN) or f"{100*r['val@110']:.1f}", "val recall@305 [CI]": fmt(m, K_GEN) or f"{100*r['val@305']:.1f}",
                 "final model (Decision 11)": "<- headline" if m == FINAL else ("simpler alternative" if m == "gradient boosting (05a)" else "")})
final = pd.DataFrame(rows).set_index("method").sort_values("selection (3-period mean)", ascending=False)
display(final); final.to_csv(TAB / "05c_comparison.csv")
print("other 05b models (unchanged): (a) tabular+text MLP 44.8, (b) boosting + OOF text score 45.3 - see 05b")"""),
takeaway("final"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, ROOT / "notebooks" / "05c_finetuned_transformer.ipynb")
print("written")
