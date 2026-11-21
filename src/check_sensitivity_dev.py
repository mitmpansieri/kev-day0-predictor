"""Step 7 planned checks, sensitivity runs on DEV (three selection periods, frozen Phase 5 settings).

  base       : headline late fusion + gradient boosting as in 05b (cached OOT scores, data/oot_fused_scores.parquet)
  uncertain  : exploitation phrases REMOVED from the 21 dev CVEs whose day-0 text is not proven
               (desc_day0_uncertain; Decision 10c). Changes phrase_day0, desc_len and the MiniLM embedding
               (data/embeddings/variants/dev_uncertain_removed_minilm.npy, made by gpu_run.py embed-variant).
               Both models are refit (training rows change too).
  cna_first  : CNA CVSS set to the CNA's FIRST known vector (from NVD history) where it changed later
               (outputs/tables/05d_cna_cvss_stability_rows.csv, v3.x only: v4 changes were supplemental 'U:' only).
               History covers ALL dev positives but only a random 600 negatives -> conservative (lower bound).
               Gradient boosting is refit; the text model does not use CVSS (scores reused).
Run: .venv/bin/python src/check_sensitivity_dev.py prepare   -> data/embeddings/variants/dev_uncertain_removed_input.parquet
     .venv/bin/python src/gpu_run.py embed-variant dev_uncertain_removed   (GPU)
     .venv/bin/python src/check_sensitivity_dev.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from description_leak import _phrase_names, drop_sentences_with   # noqa: E402
from features import load_kev_raw                                   # noqa: E402
from metrics import recall_at_k                                     # noqa: E402
from tabular_models import build_X, make_hgb, views                 # noqa: E402
import text_models as txm                                           # noqa: E402
from calibration import EMB_NAME, HGB_BEST, NN_H, NN_LR, SEEDS, PERIODS   # noqa: E402

VAR = ROOT / "data" / "embeddings" / "variants"


def score_periods(dev, X_tab, E, y, text_scores=None):
    out = {}
    for name, (fit_idx, ev_idx) in views(dev).items():
        part = dev.loc[ev_idx, ["published", "label"]].copy()
        part["s_hgb"] = make_hgb(**HGB_BEST).fit(X_tab[fit_idx], y[fit_idx]).predict_proba(X_tab[ev_idx])[:, 1]
        part["s_text"] = (text_scores[name] if text_scores is not None else
                          np.mean([txm.mlp_scorer(dev, E, y, NN_H, NN_LR, s)(fit_idx, ev_idx) for s in SEEDS], axis=0))
        part["fused"] = (txm.month_pct_rank(part, part["s_hgb"].to_numpy())
                         + txm.month_pct_rank(part, part["s_text"].to_numpy())) / 2
        out[name] = part
    return out


def summarize(variant, parts):
    rows = []
    for model, col in [("late fusion (headline)", "fused"), ("gradient boosting", "s_hgb")]:
        r = {p: 100 * recall_at_k(parts[p], col, 110) for p in PERIODS}
        rows.append({"variant": variant, "model": model, **{f"{p}@110": round(v, 1) for p, v in r.items()},
                     "selection": round(np.mean(list(r.values())), 1), "worst": round(min(r.values()), 1),
                     "val@305": round(100 * recall_at_k(parts["val"], col, 305), 1)})
    return rows


def prepare_uncertain_input(out=VAR / "dev_uncertain_removed_input.parquet"):
    """Input for gpu_run.py embed-variant: day-0 text of the dev CVEs with desc_day0_uncertain, phrases removed."""
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet", columns=["cve_id", "description_day0", "desc_day0_uncertain"])
    u = dev[dev["desc_day0_uncertain"].fillna(False)]
    inp = pd.DataFrame({"cve_id": u["cve_id"].to_numpy(),
                        "text": [drop_sentences_with(t, _phrase_names(t)) for t in u["description_day0"]]})
    VAR.mkdir(parents=True, exist_ok=True)
    inp.to_parquet(out, index=False)
    return inp


def main():
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    y = dev["label"].to_numpy()
    kev = load_kev_raw(sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1])
    E = txm.load_embeddings(EMB_NAME, dev)
    oot = pd.read_parquet(ROOT / "data" / "oot_fused_scores.parquet")
    base = {p: oot[oot["period"] == p].reset_index(drop=True) for p in PERIODS}
    rows = summarize("base (as in 05b)", base)

    # --- uncertain phrases removed
    inp = pd.read_parquet(VAR / "dev_uncertain_removed_input.parquet")
    Ev = np.load(VAR / "dev_uncertain_removed_minilm.npy").astype(np.float32)
    du, Eu = dev.copy(), E.copy()
    pos = dev.reset_index().set_index("cve_id").loc[inp["cve_id"], "index"].to_numpy()
    assert dev.loc[pos, "desc_day0_uncertain"].all()
    du.loc[pos, "description_day0"] = inp["text"].to_numpy()
    Eu[pos] = Ev
    assert not any(_phrase_names(t) for t in du.loc[pos, "description_day0"])
    print(f"uncertain: {len(pos)} CVEs edited ({int(y[pos].sum())} positives)", flush=True)
    rows += summarize("uncertain phrases removed (21 CVEs)", score_periods(du, build_X(du, kev), Eu, y))

    # --- CNA CVSS = first known CNA vector
    st = pd.read_csv(ROOT / "outputs" / "tables" / "05d_cna_cvss_stability_rows.csv")
    ch = st[st["first_vs_current_differ"].eq(True) & st["first_score"].notna()]
    dc = dev.copy()
    idx = dev.reset_index().set_index("cve_id").loc[ch["cve_id"], "index"].to_numpy()
    ver = dc.loc[idx, "cvss_cna_version"].astype(str).to_numpy()
    dc.loc[idx, "cvss_cna_vector"] = [f"CVSS:{v}/{fv}" for v, fv in zip(ver, ch["first_vector"])]
    dc.loc[idx, "cvss_cna_score"] = ch["first_score"].to_numpy()
    print(f"cna_first: {len(idx)} CVEs reset to first CNA vector ({int(y[idx].sum())} positives)", flush=True)
    rows += summarize(f"CNA CVSS = first known vector ({len(idx)} CVEs)",
                      score_periods(dc, build_X(dc, kev), E, y, {p: base[p]["s_text"].to_numpy() for p in PERIODS}))

    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "outputs" / "tables" / "05d_sensitivity_dev.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["prepare"]:
        print(f"wrote {len(prepare_uncertain_input())} rows")
    else:
        main()
