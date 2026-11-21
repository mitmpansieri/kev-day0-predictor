"""Phase 7 (after Phase 6): explanations of the FROZEN Phase 6 gradient-boosting model + ransomware coverage.
Uses outputs/models/dev_hgb.joblib (sha256-checked) and the cached Phase 6 test scores. Nothing is refit.

SHAP: shap.TreeExplainer does NOT support the model's native categorical splits (the 9 CVSS letters): its values
did not add up to the model output (error up to 4.6 log-odds; exact on a no-categorical control). So we use
PERMUTATION SHAP (shap.PermutationExplainer, max_evals 500) on the 22 raw input features with the full pipeline
as the function, in log-odds (raw score) units; background = 100 random dev CVEs (the training distribution).
Additivity is checked (sum of SHAP + base = model output). Rows explained (~0.45 s each):
  random  : 2,000 random test CVEs (seed 0)                         -> global importance, beeswarm, 2025 vs 2026
  pos     : all 294 exploited test CVEs                             -> what drives caught / missed
  flagged : 300 random non-exploited CVEs inside the monthly top 110 -> false alarms
  example : 3 worked examples (rules below)
Outputs: data/shap_test.parquet, outputs/tables/07_*.csv; log outputs/logs/step10_phase7.log
Run: .venv/bin/python src/phase7_explain.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_model as fm                                                     # noqa: E402
from description_leak import PHRASES, phrase_flags                           # noqa: E402
from features import load_kev_raw, vendor_or_cna                             # noqa: E402
from metrics import bootstrap_recall_at_k, ci_table, recall_at_k             # noqa: E402
from tabular_models import build_X                                           # noqa: E402

K, K_RULE = 110, 385                    # 385 = CVSS>=9 (CNA) rule's own test capacity (06_own_capacity.csv)
TAB = ROOT / "outputs" / "tables"


class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.o = sys.stdout
    def write(self, s):
        self.o.write(s); self.f.write(s)
    def flush(self):
        self.o.flush(); self.f.flush()


def main():
    t0 = time.time()
    meta = json.loads((ROOT / "outputs" / "models" / "dev_models.json").read_text())
    assert fm._sha(ROOT / "outputs" / "models" / "dev_hgb.joblib") == meta["sha256"]["hgb"]
    model = joblib.load(ROOT / "outputs" / "models" / "dev_hgb.joblib")
    kev = load_kev_raw(fm.KEV_FILE)
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    test = pd.read_parquet(ROOT / "data" / "test_locked.parquet")
    sc = pd.read_parquet(ROOT / "data" / "phase6_scores_test.parquet")
    assert (sc["cve_id"].to_numpy() == test["cve_id"].to_numpy()).all()
    ev = test.assign(**{c: sc[c].to_numpy() for c in ["s_hgb", "fused", "epss", "epss_lag_days"]})
    ev["phrase_day0"] = phrase_flags(ev["description_day0"], PHRASES).any(axis=1)
    ev["year"] = ev["published"].dt.year
    ev["vendor_key"] = vendor_or_cna(ev)
    month = ev["published"].dt.to_period("M")
    ev["month_pct"] = ev.groupby(month)["s_hgb"].rank(pct=True)
    top = pd.Series(False, index=ev.index)
    for _, idx in ev.groupby(month).indices.items():
        top.iloc[idx[np.argsort(-ev["s_hgb"].to_numpy()[idx], kind="stable")[:K]]] = True
    ev["in_top110"] = top

    X = build_X(ev, kev)
    Xd = build_X(dev, kev)
    raw = model["clf"]._raw_predict(model["pre"].transform(X)).ravel()
    assert np.allclose(1 / (1 + np.exp(-raw)), ev["s_hgb"]), "rebuilt features must reproduce the Phase 6 scores"

    # ------------------------------------------------------------ rows to explain
    rng = np.random.default_rng(0)
    pos = (ev["label"] == 1)
    ex_caught = ev[pos & ev["in_top110"] & ~ev["phrase_day0"]]["s_hgb"].idxmax()
    miss = ev[pos & ~ev["in_top110"] & (ev["year"] == 2026) & ev["vendor_key"].eq("microsoft")]
    ex_missed = miss["s_hgb"].idxmax()
    ex_false = ev[~pos & ev["in_top110"]]["s_hgb"].idxmax()
    groups = {
        "random": rng.choice(len(ev), 2000, replace=False),
        "pos": np.flatnonzero(pos),
        "flagged": rng.choice(np.flatnonzero(~pos & ev["in_top110"]), 300, replace=False),
        "example": np.array([ex_caught, ex_missed, ex_false]),
    }
    rows = np.unique(np.concatenate(list(groups.values())))
    print(f"explaining {len(rows):,} test CVEs: " + ", ".join(f"{g} {len(v)}" for g, v in groups.items()), flush=True)

    # ------------------------------------------------------------ permutation SHAP on the 22 raw features
    cols = list(X.columns)
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    both = pd.concat([X, Xd], ignore_index=True)
    cats = {c: pd.Index(both[c].astype(object).unique()) for c in cat}
    dtypes = X.dtypes.to_dict()

    def enc(df):
        A = df.copy()
        for c in cat:
            A[c] = cats[c].get_indexer(df[c].astype(object))
        return A.astype(float).to_numpy()

    def dec(A):
        df = pd.DataFrame(A, columns=cols)
        for c in cat:
            df[c] = cats[c].to_numpy()[df[c].astype(int)]
        return df.astype(dtypes)

    f = lambda A: model["clf"]._raw_predict(model["pre"].transform(dec(A))).ravel()
    bg = enc(Xd.sample(100, random_state=0))
    explainer = shap.PermutationExplainer(f, shap.maskers.Independent(bg, max_samples=100))
    E = enc(X.iloc[rows])
    vals, base = np.zeros((len(rows), len(cols))), np.zeros(len(rows))
    t = time.time()
    for i in range(0, len(rows), 100):
        r = explainer(E[i:i + 100], max_evals=500, silent=True)
        vals[i:i + 100], base[i:i + 100] = r.values, r.base_values
        print(f"  {min(i + 100, len(rows))}/{len(rows)} rows ({time.time() - t:.0f}s)", flush=True)
    err = np.abs(vals.sum(1) + base - raw[rows]).max()
    print(f"additivity check: max |sum(SHAP) + base - model log-odds| = {err:.2e}", flush=True)
    assert err < 1e-6

    out = pd.DataFrame(vals, columns=[f"shap__{c}" for c in cols])
    out.insert(0, "base_value", base)
    info = ev.iloc[rows][["cve_id", "published", "year", "label", "s_hgb", "month_pct", "in_top110", "vendor_key",
                          "cvss_cna_score", "phrase_day0", "kev_ransomware"]].reset_index(drop=True)
    info["log_odds"] = raw[rows]
    for g, idx in groups.items():
        info[f"in_{g}"] = np.isin(rows, idx)
    feat = X.iloc[rows].reset_index(drop=True).add_prefix("x__")
    for c in cat:
        feat[f"x__{c}"] = feat[f"x__{c}"].astype(str)
    res = pd.concat([info, feat, out], axis=1)
    res.to_parquet(ROOT / "data" / "shap_test.parquet", index=False)

    ex = ev.loc[[ex_caught, ex_missed, ex_false], ["cve_id", "published", "vendor", "product", "cna", "cwe",
                                                    "cvss_cna_score", "cvss_cna_vector", "description_day0",
                                                    "label", "days_to_kev", "kev_ransomware", "s_hgb", "month_pct", "in_top110"]]
    ex.insert(0, "example", ["caught exploited CVE", "missed exploited CVE", "false alarm"])
    ex.to_csv(TAB / "07_examples.csv", index=False)

    # ------------------------------------------------------------ ransomware coverage (KEV knownRansomwareCampaignUse)
    S = {"gradient boosting (headline)": ev["s_hgb"], "late fusion (alternative)": ev["fused"],
         "phrase, then CVSS (CNA)": 100 * ev["phrase_day0"] + ev["cvss_cna_score"].fillna(-1),
         "CVSS (CNA) ranking": ev["cvss_cna_score"].fillna(-1),
         "EPSS (first score, <=7 d after pub)": ev["epss"].fillna(-1),
         "EPSS (score by day 1)": ev["epss"].where(ev["epss_lag_days"] <= 1).fillna(-1)}
    rrows = []
    for name, lab in [("all exploited", ev["label"]), ("ransomware-linked exploited", ev["label"] & ev["kev_ransomware"].fillna(False)),
                      ("exploited, not ransomware-linked", ev["label"] & ~ev["kev_ransomware"].fillna(False))]:
        sub = ev.assign(label=lab.astype(int))
        point = {(m, k): recall_at_k(sub, s, k) for m, s in S.items() for k in (K, K_RULE)}
        c = ci_table(bootstrap_recall_at_k(sub, S, [K, K_RULE], n_boot=2000, seed=0), point)
        for m in S:
            rrows.append({"subset": name, "positives": int(sub["label"].sum()), "method": m,
                          **{f"recall@{k}": 100 * c.loc[(m, k), "point"] for k in (K, K_RULE)},
                          **{f"recall@{k} [95% CI]": f"{100 * c.loc[(m, k), 'point']:.1f} [{100 * c.loc[(m, k), 'lo']:.0f}–{100 * c.loc[(m, k), 'hi']:.0f}]"
                             for k in (K, K_RULE)}})
    ransom = pd.DataFrame(rrows)
    flag = ev["cvss_cna_score"] >= 9
    rp = ev["label"].eq(1) & ev["kev_ransomware"].fillna(False)
    ransom.attrs = {}
    ransom.to_csv(TAB / "07_ransomware.csv", index=False)
    rule_line = {"rule": "CVSS>=9 (CNA) at own capacity (385/month)", "ransomware positives": int(rp.sum()),
                 "caught": int((flag & rp).sum()), "recall %": 100 * (flag & rp).sum() / rp.sum()}
    pd.DataFrame([rule_line]).to_csv(TAB / "07_ransomware_rule.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    imp = pd.Series(np.abs(vals[info["in_random"]]).mean(0), index=cols).sort_values(ascending=False)
    print("\n== mean |SHAP| (log-odds), 2,000 random test CVEs ==\n" + imp.round(3).to_string())
    print("\n== examples ==\n" + ex.drop(columns=["description_day0"]).to_string(index=False))
    print("\n== ransomware ==\n" + ransom.pivot(index="method", columns="subset", values=f"recall@{K} [95% CI]").to_string())
    print(rule_line)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.stdout = Tee(ROOT / "outputs" / "logs" / "step10_phase7.log")
    main()
