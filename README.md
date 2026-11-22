# kev-day0-predictor: which new vulnerabilities will attackers exploit?

### ▶ Live demo: [ciso-patch-advisor.streamlit.app](https://ciso-patch-advisor.streamlit.app/)

**On the day a vulnerability (CVE) is published, this project ranks it by the chance that attackers exploit it within 90
days, so a security team can spend its limited emergency-patching time on the ones that matter.**

> **Research prototype for an MIT course project. Not security advice; always follow your organisation's patching process.**

## Glossary (one sentence each)
| term | meaning |
|---|---|
| **CVE** | Common Vulnerabilities and Exposures: the public ID of one software flaw, e.g. CVE-2026-12744. |
| **NVD** | National Vulnerability Database (US NIST): the public catalogue where every CVE is published with its details. |
| **CNA** | CVE Numbering Authority: the organisation that publishes a CVE, often the vendor itself (e.g. Microsoft, Fortinet). |
| **CVSS** | Common Vulnerability Scoring System: a 0–10 severity score of how bad a flaw would be *if* exploited. |
| **CVSS vector** | The short code behind a CVSS score (e.g. AV:N = attackable over the network, PR:N = no login needed). |
| **CWE** | Common Weakness Enumeration: the type of bug, e.g. CWE-288 "authentication bypass". |
| **CISA** | US Cybersecurity and Infrastructure Security Agency. |
| **KEV** | CISA's Known Exploited Vulnerabilities catalogue: CVEs with evidence of real-world exploitation; our label. |
| **ADP** | Authorized Data Publisher: an organisation (e.g. CISA) that adds data to CVE records after publication. |
| **EPSS** | Exploit Prediction Scoring System (FIRST): a public daily score of exploitation likelihood, our main benchmark. |
| **day-0** | Information available on the CVE's publication day; anything added later is excluded from the model. |
| **label** | 1 if CISA added the CVE to KEV within 90 days of NVD publication, else 0. |
| **base rate** | The share of all CVEs with label 1: 0.31% (about 1 in 320). |
| **K / capacity** | How many CVEs a team can emergency-patch per month (here 110, the monthly volume of "CVSS ≥ 9" in 2022–24). |
| **recall@K** | Share of exploited CVEs caught when patching the top K of each month; our main metric. |
| **PR-AUC** | Area under the precision-recall curve: a summary of ranking quality across all thresholds (secondary metric). |
| **percentile** | Where a CVE's score sits among all CVEs of a reference month (99.7 = higher than 99.7% of them). |
| **bootstrap CI** | A 95% confidence interval obtained by re-sampling the data 2,000 times. |
| **logistic regression** | A simple linear classifier, used as a first model. |
| **gradient boosting** | Many small decision trees added one after another, each fixing the previous ones' errors; our chosen model. |
| **embeddings** | Numeric vectors that capture the meaning of a text, here produced by pretrained language models (MiniLM, BGE). |
| **MLP** | Multi-layer perceptron: a small neural network, here trained on the description embeddings. |
| **fine-tuning** | Further training a pretrained language model on our own labels. |
| **late fusion** | Combining two models by averaging their within-month ranks. |
| **calibration** | Turning model scores into probabilities that match observed frequencies (Platt scaling = a logistic fit). |
| **walk-forward validation** | Train on the past and test on the following period, repeated over time, as in real use. |
| **data leakage** | Using information during training that would not be known at prediction time, which inflates results. |
| **drift** | The data changing over time, so a model trained on the past gets worse. |
| **SHAP** | SHapley Additive exPlanations: splits one prediction into the contribution of each input, giving the "reasons". |
| **PATCH NOW / WATCH / MONITOR** | The tool's three tiers: within this month's patching capacity / elevated risk but outside the budget (recheck when news appears) / the rest. |
| **frozen vs deployment model** | The frozen model (trained on 2022–24) produced every test result; the deployment model (trained on all data to 2026-08) powers the tool and app. |
| **MPS** | Apple's GPU backend, used only to compute text embeddings and fine-tune (full reproduction). |

## Today vs our approach
- **Today (assumed policy):** emergency-patch every CVE with CVSS ≥ 9. This is a common severity-based practice; it has not been verified for any specific organisation.
- **The problem:** CVSS measures how severe a flaw would be *if* exploited, not how *likely* it is to be exploited. Under 1% of CVEs are ever exploited, and many critical ones never are.
- **Our approach:** on publication day, predict the likelihood that the CVE is added to CISA KEV within 90 days. Then patch the K most likely each month.
- **Not tied to CVSS ≥ 9:** results are also compared with other rules, with EPSS, and at every capacity K, so the conclusion does not depend on the assumed policy.

## Start here
Read in this order.

| # | where | what it is for |
|---|---|---|
| 1 | this README | the problem, the glossary, the results |
| 2 | `notebooks/1_data.ipynb` | where the data comes from, how cybersecurity evolved since 2022, and the leaks we found and fixed |
| 3 | `notebooks/2_train_validate.ipynb` | the metric, the baseline rules, rules vs ML vs text deep learning, how the model was chosen *before* testing |
| 4 | `notebooks/3_test.ipynb` | the single test on 20 unseen months, and the comparison with EPSS |
| 5 | `notebooks/4_results_ciso.ipynb` | why the model decides as it does (SHAP), drift, money and time saved, recommendations, limitations |
| 6 | `notebooks/5_ciso_tool.ipynb` | the tool scoring a real day of new CVEs |
| 7 | `app.py` | **the web app** (`streamlit run app.py`): new CVE, daily feed, past CVEs |
| – | `appendix/` | the detailed working notebooks of each phase (long, for reviewers) |
| – | `docs/decision_log.md` | every decision with its date, and the test protocol written before the test |
| – | `src/`, `scripts/` | the code: download, features, models, evaluation, notebook builders |
| – | `data/`, `outputs/` | the small data files, result tables, figures, trained models and run logs |

> **Run the app locally (5 min).** You need Python 3.13 and git. Repository: https://github.com/mitmpansieri/kev-day0-predictor.git (private repository, access on request).
>
> macOS / Linux:
> ```bash
> git clone https://github.com/mitmpansieri/kev-day0-predictor.git kev-day0-predictor && cd kev-day0-predictor
> python3.13 -m venv .venv
> .venv/bin/pip install -r requirements.txt
> .venv/bin/streamlit run app.py
> ```
> Windows (PowerShell):
> ```powershell
> git clone https://github.com/mitmpansieri/kev-day0-predictor.git kev-day0-predictor; cd kev-day0-predictor
> py -3.13 -m venv .venv
> .venv\Scripts\pip install -r requirements.txt
> .venv\Scripts\streamlit run app.py
> ```
> The browser opens at http://localhost:8501. Stop the app with **Ctrl+C** in the terminal.

The app has three tabs:
- **New CVE:** score a CVE by its ID (with three "Try these" examples), or type one in.
- **Daily feed:** a ranked list for one day, tiers highlighted, downloadable as CSV.
- **Past CVEs:** what the frozen model said on day 0 for any CVE of 2025 to Aug 2026, at your capacity and at the evaluated K = 110, and whether it was exploited.

The sidebar **"Your capacity"** sets everything else. You give patches per month, or analyst hours per month and hours per patch. The app turns that into a percentile threshold, using the expected monthly CVE volume: 8,216 a month, the mean of the last 3 complete labelled months, 2026-05 to 2026-07. You also set the expected new CVEs per month (editable) and the Watch list size (default about 3.5x capacity, 1,214 a month). A "Model freshness" badge shows how old the training data is. The sidebar then shows the historical coverage of that capacity on the test period for the model, the best rule and CVSS, with hours and dollars. These are historical estimates; only the 4.2% and K = 110 policies were formally evaluated.

It runs locally. It reads NVD without an API key and sends nothing else anywhere. Usage statistics are switched off in `.streamlit/config.toml`.

**Browsing cannot break anything.**
- Notebooks, tool and app only *read* the committed data.
- Running them rewrites only the notebooks' own outputs and figures, and caches NVD replies.
- The one-time test script refuses to run twice.

## Results
**Test set:** 2025-01 → 2026-08-20, 102,561 CVEs, 294 exploited. The test set was evaluated once, with the protocol
fixed in advance.

| method | exploited CVEs caught with 110 patches/month [95% CI] | with 305/month | on CVEs not yet in KEV when EPSS scored |
|---|---|---|---|
| **gradient boosting (day-0 features)** | **33.7% [28–39]** | **49.3%** | **30.4%** |
| late fusion (boosting + text model) | 29.3% [24–35] | 48.0% | 23.8% |
| EPSS, first score ≤ 7 days after publication | 29.6% [24–35] | 45.4% | 14.6% |
| best day-0 rule (exploitation wording, then CNA CVSS) | 25.2% [20–30] | 42.2% | 29.4% |
| CNA CVSS ranking | 18.6% [14–23] | 36.1% | 22.1% |

**Comparisons**
- Gradient boosting beats the best rule by +8.5 points [+1.7, +15.3].
- It leads every day-0 rule at every K from 10 to 1,000.
- At the CVSS ≥ 9 rule's own 385 patches a month, it catches 51.7% vs 41.2%.
- EPSS is ahead only at ≤ 50 patches a month.

**Business impact (frozen model)**
- It matches the CVSS ≥ 9 rule's coverage with **198 instead of 385 patches a month**.
- That is about $1.35M a year at 6 h per patch and $100/h, and $0.45–2.24M over 2–10 h per patch.

**Drift**
- Recall was 41.9% in 2025 but 23.9% in 2026.
- Retraining every quarter (post-test, exploratory) gives 31.3% in 2026, and 148 patches a month (about $1.7M).
- **Operating recommendation:** retrain every quarter and re-check capacity against CVE volume; the app's "Model freshness" badge turns orange 90 days after the training data end and red after 180.

## Quick start (no GPU, no API key; about 1 minute)
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/build_notebooks.py                     # (re)generate notebooks 1-5
.venv/bin/python src/run_notebook.py notebooks/1_data.ipynb # or open in Jupyter / VS Code and Run All
.venv/bin/streamlit run app.py                              # web app
.venv/bin/python ciso_tool.py cve CVE-2026-12744            # command line: one CVE as a card
.venv/bin/python ciso_tool.py feed 2026-09-08 --top 20      # one day as a ranked table
.venv/bin/python ciso_tool.py feed 2026-09-08 --capacity 110 --watch 385   # same day at your own capacity
.venv/bin/python ciso_tool.py manual --vendor Fortinet --cna psirt@fortinet.com \
    --vector "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H" --cwe CWE-288 --description "Authentication bypass ..."
.venv/bin/python ciso_tool.py sources                       # refresh CNA names (data/nvd_sources.csv)
```

**What the tool and app use**
- **Model and data:** the deployment model and day-0 information only. That is the CNA's own CVSS, CWE and vendor; exploitation wording only if fetched within 7 days of publication; KEV history only before publication, from the local snapshot.
- **Output per CVE:** a percentile vs the July 2026 reference month; an indicative probability and its multiple of the base rate; a tier; and 3 plain-word reasons.
- **Tiers:** PATCH NOW (within your monthly capacity: by default the evaluated 4.2% of a month's CVEs, about 347 a month), WATCH (the next 3.5x capacity by default), MONITOR (the rest). The `--capacity` and `--watch` options, or the app sidebar, change them.
- **Past CVEs tab:** uses only the frozen model's day-0 test scores (`data/past_cves_test_frozen.parquet`), never the deployment model, which was trained on those CVEs.
- **CNA names:** come from `data/nvd_sources.csv` and are for display only.

## Full reproduction (in order)
Install `requirements-full.txt`; it adds PyTorch and sentence-transformers. GPU steps need Apple MPS. An NVD key is
optional and makes downloads about 10× faster. To use one, put `NVD_API_KEY=<key>` in `.env`; that file is
git-ignored and the key is never printed. Prefix each `src/...` command with `.venv/bin/python`. Seeds are fixed, so
results are deterministic; fine-tuning can vary slightly across PyTorch versions.

| # | step | needs | time |
|---|---|---|---|
| 1 | `bash scripts/download_all.sh`: NVD → change history → dataset → EPSS (uses the committed KEV snapshot) | network | ~3.5 h without key |
| 2 | `src/gpu_run.py embed` (MiniLM @1110a243, BGE-small @5c38ec7c) | GPU | ~7 min |
| 3 | `src/build_nb03_exploration.py`, `build_nb04_baselines.py`, `build_nb05a_tabular.py`, `build_nb05b_text.py`, then `src/run_notebook.py notebooks/<name>.ipynb` for each | CPU | 1–2 h |
| 4 | `src/gpu_run.py finetune grid`, `src/gpu_run.py finetune seeds lr1e-5_full`, `build_nb05c_finetune.py`, then run it | GPU | ~95 min |
| 5 | `src/calibration.py fusion`, `src/calibration.py hgb`, `src/check_cna_cvss_stability.py`, `src/check_sensitivity_dev.py prepare`, `src/gpu_run.py embed-variant dev_uncertain_removed`, `src/check_sensitivity_dev.py` | CPU + GPU | ~20 min |
| 6 | `src/final_model.py fit-dev` (frozen models + sha256) | CPU | ~5 min |
| 7 | `src/phase6_test.py --dry-run` (must reproduce val 47.1 / 42.6) | CPU | ~10 min |
| 8 | `src/gpu_run.py embed-test`, then `src/phase6_test.py --test --confirm-phase6` (refuses to run twice) | GPU, CPU | ~15 min |
| 9 | `src/post_test_analysis.py`, `build_nb06_test.py`, then run it | CPU | ~4 min |
| 10 | `src/final_model.py fit-all`, `src/final_model.py calibrate-deploy`, `src/phase7_explain.py`, `build_nb07_explain_impact.py`, then run it | CPU | ~25 min |
| 11 | `src/build_main_tables.py`, then `src/build_notebooks.py` and run notebooks 1–5 | CPU | ~3 min |

## Leakage fixes and limitations
**Leakage fixes** (details in notebook 1):
- CVSS comes from the CNA only, because CISA-ADP scores were back-filled onto exploited CVEs.
- CWE comes from the CNA only.
- Exploitation wording added after publication is removed.
- KEV history counts only entries added before publication.
- The data are split by date; a random split inflates recall from 38% to 59%.

**Limitations:**
- The CVSS ≥ 9 baseline is assumed.
- The label is CISA's KEV selection.
- CNAs revise their own CVSS later for 5.5% of exploited vs 1.9% of other CVEs.
- EPSS comparison caveats: lookahead, a different target, and EPSS's use of KEV is assumed.
- The model needs quarterly retraining and a capacity sized to CVE volume.
- Small numbers: 294 test positives.
- Probabilities are indicative.
- Publishers (CNAs) with fewer than 100 training CVEs, or never seen, share one "rare/unseen" group that the model treats as higher risk. The tool and app show a warning for them, and the manual form offers the 121 publishers the model learned individually.
