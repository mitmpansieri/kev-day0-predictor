# kev-day0-predictor: which new CVEs will attackers exploit?

On the day a CVE is published, rank it by the probability that CISA lists it in **KEV within 90 days**. A
vulnerability team then emergency-patches the top of each month's list instead of "everything with CVSS ≥ 9".

"Patch everything with CVSS ≥ 9" is an *assumed* baseline policy: a common practice, not verified for any specific
organisation. Results are therefore also reported against other rules, against EPSS, and at every capacity K.

## Results
**Test set:** 2025-01 → 2026-08-20, 102,561 CVEs, 294 exploited. The test set was evaluated once, with a protocol
fixed in advance.

| method | recall@110 per month [95% CI] | recall@305 | not yet in KEV when EPSS scored @110 |
|---|---|---|---|
| **gradient boosting (day-0 features)** | **33.7 [28–39]** | **49.3** | **30.4** |
| late fusion (boosting + MiniLM text MLP) | 29.3 [24–35] | 48.0 | 23.8 |
| EPSS, first score ≤ 7 days after publication | 29.6 [24–35] | 45.4 | 14.6 |
| best day-0 rule (exploitation phrase, then CNA CVSS) | 25.2 [20–30] | 42.2 | 29.4 |
| CNA CVSS ranking | 18.6 [14–23] | 36.1 | 22.1 |

**Comparisons**
- Gradient boosting beats the best rule by **+8.5 points [+1.7, +15.3]**.
- It leads every day-0 rule at every K from 10 to 1,000.
- At the CVSS ≥ 9 rule's own 385 patches a month, it catches 51.7% vs 41.2%.
- EPSS is ahead only at ≤ 50 patches a month.

**Business impact (frozen model)**
- It matches the CVSS ≥ 9 rule's coverage with **198 instead of 385 patches a month**.
- That is about $1.35M a year at 6 h per patch and $100/h, and $0.45–2.24M over 2–10 h per patch.

**Drift**
- Recall was 41.9% in 2025 but 23.9% in 2026.
- Retraining each quarter (post-test, exploratory) gives 31.3% in 2026, and 148 patches a month (about $1.7M) for the same coverage.

## Repo
```
notebooks/1_data.ipynb             data collection, evolution of cybersecurity (descriptive), leakage hunt
notebooks/2_train_validate.ipynb   task/metric/splits, baselines, tabular ML vs text deep learning, model selection
notebooks/3_test.ipynb             the single test evaluation, EPSS comparison, "not yet in KEV"
notebooks/4_results_ciso.ipynb     SHAP, drift (post-test), business impact (editable assumptions), recommendations, limitations
notebooks/5_ciso_tool.ipynb        demo of ciso_tool.py on a live NVD day (2026-09-08)
ciso_tool.py                       the tool: one CVE (NVD id or typed in) or a daily NVD feed -> ranked patch list
appendix/                          the detailed working notebooks (00 main report, 03-07), unchanged copies
src/                               pipeline code (data, features, models, evaluation, notebook builders)
docs/decision_log.md               dated decisions, the test protocol (fixed before test), progress log
data/raw/kev_20261120.csv          dated CISA KEV snapshot (labels + KEV-history features)
data/shap_test.parquet             SHAP values of the frozen model on test (notebook 4)
data/nvd_tool_cache/               NVD responses used by the tool demo (offline re-runs)
outputs/tables, outputs/models     every result table; frozen Phase 6 models (dev_*), deployment model (deploy_hgb), calibrators
outputs/logs                       run logs of each step
```

## Quick start: notebooks and tool (no GPU, no API key, about 1 minute)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_notebooks.py                              # (re)generate notebooks 1-5
.venv/bin/python src/run_notebook.py notebooks/1_data.ipynb          # or open in Jupyter / VS Code and Run All
.venv/bin/python ciso_tool.py feed 2026-09-08 --top 20               # daily ranked patch list
.venv/bin/python ciso_tool.py cve CVE-2026-12744                     # one CVE from NVD
.venv/bin/python ciso_tool.py manual --vendor Fortinet --cna psirt@fortinet.com \
    --vector "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H" --cwe CWE-288 --description "Authentication bypass ..."
```

**How the tool reads data**
- It reads NVD **without an API key** and never reads `.env`. The limit is about 5 requests per 30 s.
- Responses are cached in `data/nvd_tool_cache/`.

**Day-0 rules the tool applies**
- Only day-0 information is used: the CNA's own CVSS, CWE and vendor.
- Exploitation wording counts only if the record was fetched within 7 days of publication.
- Vendor/product KEV history uses only KEV entries added **before** publication, from the newest local `data/raw/kev_*.csv`.

**Output per CVE**
- Percentile vs the July 2026 reference month.
- An indicative probability.
- PATCH NOW if the CVE is in the top 4.2% of a month, otherwise MONITOR.
- The top 3 SHAP reasons, in plain words.

## Full reproduction (in order)
Install `requirements-full.txt`; it adds PyTorch and sentence-transformers for the text models. GPU steps need Apple
MPS (`src/gpu_run.py`). An NVD key is optional and makes downloads about 10× faster. To use one, create `.env` with the
line `NVD_API_KEY=<your key>`. It is git-ignored, and the code never prints it.

| # | step | needs | time |
|---|---|---|---|
| 1 | `bash scripts/download_all.sh`: NVD → change history → dataset with `description_day0` → EPSS (uses the committed KEV snapshot) | network | ~3.5 h without key |
| 2 | `src/gpu_run.py embed` (MiniLM @1110a243, BGE-small @5c38ec7c) | GPU | ~7 min |
| 3 | `src/build_nb03_exploration.py`, `build_nb04_baselines.py`, `build_nb05a_tabular.py`, `build_nb05b_text.py`, then `src/run_notebook.py notebooks/<name>.ipynb` each (writes the 03–05 tables) | CPU | 1–2 h |
| 4 | `src/gpu_run.py finetune grid`, `src/gpu_run.py finetune seeds lr1e-5_full`, `build_nb05c_finetune.py` + run | GPU | ~95 min |
| 5 | `src/calibration.py fusion`, `src/calibration.py hgb`, `src/check_cna_cvss_stability.py`, `src/check_sensitivity_dev.py prepare`, `src/gpu_run.py embed-variant dev_uncertain_removed`, `src/check_sensitivity_dev.py` | CPU + GPU | ~20 min |
| 6 | `src/final_model.py fit-dev` (frozen models + sha256) | CPU | ~5 min |
| 7 | `src/phase6_test.py --dry-run` (must reproduce val 47.1 / 42.6) | CPU | ~10 min |
| 8 | `src/gpu_run.py embed-test`, then `src/phase6_test.py --test --confirm-phase6`. This **refuses to run if test outputs exist**: the test set is evaluated once. | GPU, CPU | ~15 min |
| 9 | `src/post_test_analysis.py`, `build_nb06_test.py` + run | CPU | ~4 min |
| 10 | `src/final_model.py fit-all`, `src/final_model.py calibrate-deploy`, `src/phase7_explain.py`, `build_nb07_explain_impact.py` + run | CPU | ~25 min |
| 11 | `src/build_main_tables.py`, then `src/build_notebooks.py` and run notebooks 1–5 | CPU | ~3 min |

Prefix each `src/...` command with `.venv/bin/python`. Seeds are fixed, so results are deterministic. Fine-tuning
results can vary slightly across MPS/PyTorch versions.

## Leakage fixes
- **CVSS:** CNA scores only. CISA-ADP scores were back-filled for exploited CVEs.
- **CWE:** CNA CWEs only.
- **Descriptions:** exploitation phrases absent from the first recorded version are removed.
- **KEV history:** counts only entries added before publication.
- **Splits:** by date. A random split inflates recall from 38% to 59%.

Details are in notebook 1.

## Limitations
- The CVSS ≥ 9 baseline is assumed.
- The label is CISA's KEV selection.
- CNAs sometimes revise their own CVSS later: 5.5% of exploited vs 1.9% of other CVEs.
- EPSS comparison caveats: lookahead, a different target, and EPSS's use of KEV is assumed.
- Drift: the model needs quarterly retraining and a capacity sized to CVE volume.
- Small numbers: 294 test positives.
- Probabilities are indicative.
