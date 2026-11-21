# Decision log, test protocol and progress log

Copied from the project's working log (CLAUDE.md), sections about the local machine setup removed. Decisions are dated; the TEST PROTOCOL was fixed before the test set was opened.

# THE PROJECT: predicting which vulnerabilities attackers will exploit
On the day a new CVE is published, predict the probability that attackers will exploit it within 90 days,
so the security team spends its limited emergency-patching capacity on the flaws that matter.
- Stakeholder: Head of Vulnerability Management, reporting to the CISO
- Problem: ~40,000 new CVEs a year, under 1% ever exploited, capacity for only K emergency patches a month
- Current rule: emergency-patch everything with CVSS >= 9
- Our solution: emergency-patch the K CVEs with the highest predicted probability of exploitation
- Impact metric: share of exploited CVEs caught at the same capacity, plus analyst hours saved
Data: NVD API 2.0 (population + inputs), CISA KEV catalog (label), FIRST EPSS (benchmark, Phase 6).
Machine: Mac, Apple M4, 16 GB. Everything runs locally; the final notebook must also run in Google Colab (Phase 8).

# DECISIONS ALREADY MADE (don't reopen them; explain them as we go)
1. Population = CVEs published from 2022-01-01 (NVD publication date). Keep the FULL KEV file; don't filter
   KEV by its dateAdded. The KEV launch batch (2021-11-03) and old CVEs CISA adds late drop out through the join.
2. Label = 1 if KEV dateAdded - NVD published <= 90 days. A negative gap (KEV listed it before NVD published it)
   counts as positive. A CVE added to KEV after 90 days = 0.
3. Right-censoring: drop CVEs published less than 90 days before the KEV snapshot date (label not knowable yet).
4. LEAKAGE: only information available on publication day. NVD returns today's version of each record, so DROP:
   - cisaExploitAdd / cisaActionDue / cisaRequiredAction / cisaVulnerabilityName (this is the label itself)
   - metrics.ssvcV203 (CISA "exploitation: active/poc/none" rating = label leakage)
   - references and their tags (e.g. "Exploit"), added after publication
   - configurations / CPEs added by NVD analysts later
   - CVSS: prefer the score from the CNA (the metric whose source == the CVE's sourceIdentifier);
     order of preference v3.1 > v3.0 > v4.0 > v2. Fallback: another non-NVD source ("other"), then nvd@nist.gov ("nvd").
     Record cvss_source = cna / other / nvd / none. Later, test whether "other"/"nvd" scores inflate results
     (e.g. Android CVEs often get CVSS from a third party, possibly CISA-ADP id 134c704f-9b21-4f2e-91b3-4a467353bcc0).
   - CWE: prefer the CNA's; otherwise any; default "NVD-CWE-noinfo".
   - Vendor/product: from the "affected" block whose source == the CNA (day-0 info). Some CNAs (VulDB) put "n/a".
5. KEV: download the latest from https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv
   (send a User-Agent header) and save it as data/raw/kev_YYYYMMDD.csv. A newer KEV doesn't change labels;
   it only moves the censoring cutoff forward.
6. SPLIT BY DATE, not random. (A random stratified 50/30/20 split was proposed and rejected: twin CVEs
   (same vendor/month/text) would leak across sets, the model would learn from the future, and drift would be hidden.)
   - train: 2022-01-01 -> 2024-06-30
   - val:   2024-07-01 -> 2024-12-31 (tuning, model choice)
   - test:  2025-01-01 -> censoring cutoff (touched ONCE, in Phase 6)
   - cv_group = vendor (lowercased; CNA if vendor is missing or "n/a") + "|" + YYYY-MM, for grouped,
     label-stratified CV inside train
   - Walk-forward folds for drift: 2022->2023, 2022-23->2024, 2022-24->2025
     DECIDED 2026-11-20: folds 2022->2023 and 2022-23->2024 run in Phase 5. Fold 2022-24->2025 tests on
     test-period rows, so it runs ONLY in Phase 6, alongside the final evaluation. Split stays as is.
   - DECIDED 2026-11-20: after Phase 6, retrain the final model (same pipeline + hyperparameters, no re-tuning)
     on ALL labeled data (train + val + test) before building the demo. Reported metrics stay the Phase 6 ones.
   - Random stratified split: run ONCE as a comparison only; the gap shows how much leakage and drift
     inflate results (good slide).
7. GRADIENT BOOSTING = scikit-learn HistGradientBoostingClassifier, NOT LightGBM. lightgbm fails to import here
   (needs Homebrew libomp, outside Project1). HGB is the same histogram-based boosting, handles NaN and
   categoricals natively, supports class_weight, needs no extra install and runs in Colab.
8. GPU: the sandbox blocks Metal. All GPU (MPS) work lives in src/gpu_run.py, run exactly as
   `.venv/bin/python src/gpu_run.py ...` from the project folder (no `cd` in the same command); that command
   is in sandbox.excludedCommands. Everything else runs sandboxed on CPU.
9. CVSS FEATURES = CNA ONLY (decided 2026-11-20, after the cvss_source_id check). Model features may use ONLY the
   cvss_cna_* columns (cvss_cna_score, cvss_cna_severity, cvss_cna_vector, cvss_cna_version) and the flag
   has_cna_cvss (the user's "cna_has_cvss"). NEVER features: cvss_source, cvss_source_id, has_adp_cvss,
   has_nvd_cvss, and every column built from the fallback metric (cvss_score, cvss_version, cvss_vector,
   attack_vector, attack_complexity, privileges_required, user_interaction, scope, conf/integ/avail_impact,
   exploitability_score, impact_score). CVSS components (AV, AC, PR, UI, ...) must be parsed from
   cvss_cna_vector inside the Pipeline. The fallback score (cna > other > nvd) is used ONLY as a "generous
   baseline" sensitivity check in Phase 6. Why: CISA-ADP ("other") scores were added after the fact, mostly
   to exploited CVEs (2022Q1: 3.4% of all CVEs vs 46% of exploited CVEs have one; ADP did not exist in 2022).
10. DAY-0 TEXT AND CWE (decided 2026-11-20).
   a) CWE = the CNA's weakness entry only (default NVD-CWE-noinfo); n_cwes counts CNA CWEs only. Old rule
      ("CNA else any") took 55% of positives' CWE from NVD analysts (added later). cwe_fallback and
      cwe_fallback_source are kept for analysis only, NEVER features.
   b) Exploitation phrases (src/description_leak.py PHRASES): "exploited in the wild", "actively exploited",
      "may have been exploited", "known exploited", "KEV", "aware of a report". Train: 36/251 positives (14%)
      vs 38/74,017 negatives contain one (label rate 49% when present). Mostly Apple ("Apple is aware of a
      report..."), a few MITRE-CNA CVEs. A phrase present at publication is a LEGITIMATE day-0 signal.
   c) description_day0 = today's description, except phrases NOT in the earliest recorded version are removed
      (the clause containing them; whole sentence only if needed; never a mask token, which would itself leak).
      Label-independent: NVD change history was fetched for EVERY CVE (all splits) whose current description
      matches a phrase (147 CVEs) - cache data/nvd_history_cache/. ALL TEXT FEATURES USE description_day0,
      never description. desc_day0_status / desc_day0_uncertain are diagnostics, NEVER features.
      desc_day0_uncertain = phrase kept but the earliest NVD history event is > 7 days after publication
      (day-0 text not proven); use it for a Phase 5 sensitivity run that also drops those phrases.
   d) THE ONE EXCEPTION to "never open test": to choose which histories to fetch, the phrase match read
      cve_id + description of ALL rows incl. test (no labels, no other columns, nothing summarized).
   e) Evidence (history sample = all 319 train+val positives + 600 random train+val negatives): phrases added
      after publication: 1 real case (CVE-2024-39891 Authy, +1 day, before KEV) + HAProxy CVE-2024-45506 (val,
      label 0); 8 Apple "edits" on 2023-07-27 were a wrong-description swap reverted the same day (phrase was
      there at publication). Of 43 positives with a phrase at publication, 37 day-0 (<=7 d) / 6 uncertain.
      DESCRIPTION EDITS ARE LABEL-CORRELATED: edited 30.4% of positives vs 12.3% of negatives; edits bigger for
      positives (median 53 vs 22 chars, 16% vs 10% of the text; >200 chars 20.6% vs 8.1%).
      -> PLANNED CHECK (Phase 5, see below).
   f) Extra Phase 4 baseline: "patch if description_day0 has a phrase OR CVSS >= 9 (CNA)". Train: 112 patches/
      month, recall 36.3% (vs CNA CVSS>=9 alone 110/month, 21.9%; phrase alone 2/month, 14.3%, precision 49%).
      A careful team could follow this rule today, so the model must beat it at the same K.
11. FINAL MODEL RULE (decided 2026-11-20, BEFORE Step 6c results).
   a) The highest selection score (mean recall@110 over WF 2023, WF 2024-H1, val) wins; if models are within
      2 points, choose the one with the better WORST period. Under the 6b results that is LATE FUSION (gradient
      boosting + text model: 46.7 vs 47.4, worst period 42.6 vs 39.7). Gradient boosting alone is reported as
      the simpler alternative. BOTH are scored on test in Phase 6, but ONLY the pre-chosen model is the headline.
   b) If the 6c fine-tuned model beats the text MLP (selection rule), it replaces the text MLP inside the late
      fusion and the rule in (a) is rerun.
   c) K stays at 110 (headline) and 305; the full recall-vs-K curve is shown. No change of K after seeing results.
   d) Late fusion outputs a rank, not a probability: plan a calibration step (isotonic or Platt, fit on val) to
      turn the fused score into a probability for the demo.
      DONE 2026-11-20 (src/calibration.py): fit on the POOLED out-of-time fused scores of WF 2023 + WF 2024-H1 +
      val (68,769 CVEs, 234 positives; cached data/oot_fused_scores.parquet; reproduces 46.3/51.1/42.6, sel 46.7).
      Chosen by mean LEAVE-ONE-PERIOD-OUT Brier (in-sample Brier always favours isotonic): isotonic 0.003213 vs
      Platt 0.003219 (log loss the other way, 0.01839 vs 0.01815) -> ISOTONIC, a near-tie. Constant base rate:
      0.003391. Saved outputs/models/calibrator_fusion.joblib/.json, table 05d_calibration.csv, fig 05d_calibration.png.
      Plain isotonic's top step rested on 2 CVEs (both positive) -> p = 1.0. USER DECISION: merge steps (top down)
      until each holds >= 10 positives; use Platt instead if the merged version's LOPO Brier is worse.
      FINAL CHOICE = MERGED ISOTONIC (calibration.MergedIsotonic, step function): LOPO Brier 0.003217 vs Platt
      0.003219 (plain isotonic 0.003213); LOPO log loss 0.01817 vs 0.01815. 10 steps, top step fused >= 0.9887 ->
      p = 0.111 (26/235); lowest step p = 0.00043. Saved outputs/models/calibrator_fusion.joblib (+ .json with steps).

12. HEADLINE CHANGE (decided 2026-11-21 by the user, on DEV evidence only, BEFORE any test label or result was
   seen; at that point only test cve_id + description_day0 had been embedded (step D) and EPSS was being
   downloaded for test cve_id + publication dates (step E)).
   HEADLINE = GRADIENT BOOSTING (05a settings); late fusion = reported alternative. Reasons (all dev, pre-test):
   tie on the selection score (47.4 vs 46.7, within 2 pts); better on val (47.1 vs 42.6); better on the val
   "not yet in KEV" subset (52.5 vs 35.0); robust to the uncertain-phrase check 7b (+0.4 vs -2.0, where the
   Decision 11a rule would already pick it); simpler to deploy and explain (no within-month ranking, no text
   model). Supersedes Decision 11a's outcome; both models are still scored on test.
   Calibration (same rule as 11d, src/calibration.py hgb): input = logit(s_hgb). LOPO Brier: Platt 0.003228,
   merged isotonic 0.003235 (worse than Platt -> rule picks PLATT), plain isotonic 0.003224. Platt chosen ->
   outputs/models/calibrator_hgb.joblib/.json; p at pooled score quantiles 50% 0.0010, 90% 0.0096, 99% 0.031,
   max 0.094. Table 05d_calibration_hgb.csv, fig 05d_calibration_hgb.png. Dry run on val (in-sample): Brier
   0.003393 vs 0.003462 constant.

# TEST PROTOCOL (fixed 2026-11-20, BEFORE test_locked.parquet is opened; changes only with the user's OK, never after)
1. MODELS (settings frozen from Phase 5; nothing is tuned, re-chosen or re-weighted after seeing test).
   - DECISION 12 (2026-11-21, pre-test): HEADLINE = GRADIENT BOOSTING (fitted as part (i) below, dev_hgb.joblib);
     LATE FUSION = REPORTED ALTERNATIVE. Calibrated headline probability = Platt on logit(s_hgb)
     (calibrator_hgb.joblib). Where this section says "headline" it means gradient boosting from now on.
   - (Originally headline, now alternative) late fusion: mean of the within-month percentile ranks of (i) gradient boosting
     make_hgb(encoding="onehot", learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=100; max_iter 300,
     no early stopping, random_state 0) on build_X features, and (ii) text MLP on minilm @1110a243 embeddings of
     description_day0, hidden (64,), lr 1e-3, wd 1e-4, dropout 0.2, batch 512, class-weighted BCE, averaged over
     seeds 0-4 (5-seed ensemble, as in 05b). Within-month rank = among the test CVEs of the same calendar month.
   - (Superseded by Decision 12) simpler alternative = gradient boosting alone.
   - USER DECISION 2026-11-21: late fusion stays the headline (rule frozen) despite check 7b. The write-up reports
     check 7b next to the headline and notes that the fusion leans slightly more on the uncertain day-0 phrases
     than gradient boosting (fusion -2.0 pts, gradient boosting +0.4 when they are removed).
   - Both REFIT ON ALL OF DEV (train + val, 2022-01 -> 2024-12). The MLP follows the exact Phase 5 recipe:
     early-stopping set = last 6 months of dev (2024-07 -> 2024-12), best epoch per seed, then refit on all of dev
     for that many epochs (same seed). These 5 epoch counts are computed on dev and WRITTEN HERE before test opens.
     Epoch counts (FROZEN 2026-11-20, src/final_model.py fit-dev, ES set 2024-07..12, 19,574 CVEs / 68 pos):
     seeds 0-4 = [3, 6, 3, 4, 4]. Models saved outputs/models/dev_hgb.joblib + dev_text_mlp_s{0-4}.pt, sha256 in
     dev_models.json; phase6_test.py --test loads these files (hash-checked), it does not refit.
   - KEV snapshot = data/raw/kev_20261120.csv (labels + KEV-history features). No newer KEV download.
   - Text = description_day0 everywhere. Fine-tuned transformer (6c) not scored (not adopted, Decision 11b).
2. METRICS (test = 2025-01-01 -> 2026-08-20, all months).
   - Primary: recall@110 per calendar month (pooled), headline model. Also recall@305. Paired within-month
     Poisson bootstrap, 2,000 resamples, seed 0 (src/metrics.py) -> 95% CIs, plus paired differences
     (headline - each comparator) with CI and P(better).
   - Recall-vs-K curve (K = 10..1000/month) for all methods; PR-AUC secondary.
   - Same-capacity comparison: each rule at its OWN test capacity (patches/month) vs the model at that same K.
   - Calibration (descriptive only): Brier, log loss and reliability bins of the calibrated headline probability
     on test, vs the constant base rate. The calibrator is NOT refit on test for any reported number.
3. COMPARATORS: CVSS>=9 (CNA), CVSS>=7 (CNA), "phrase OR CVSS>=9 (CNA)" (as rule and as phrase-then-CVSS
   ranking), generous fallback CVSS>=9 and fallback ranking (Decision 9, sensitivity), random, gradient boosting
   alone, EPSS.
   - EPSS = first available score ON OR AFTER the NVD publication date, searching up to 7 days later
     (api.first.org, ?cve=...&date=YYYY-MM-DD). Record epss_date and lag (days) per CVE; report the lag distribution
     and the share with a day-0 score. No score within 7 days -> ranked last (random tie-break, expected value)
     and counted. THREE EPSS LINES (user, 2026-11-21): first score <= 7 d after publication; score by day 1
     (lag <= 1, what a team has the morning after); day-0 score only (lag 0). The up-to-7-day LOOKAHEAD FAVOURS EPSS (our model uses publication-day information only);
     say so wherever EPSS is compared, and also report EPSS restricted to its lag-0 (day-0) scores. Note: EPSS predicts exploitation activity in the next 30 days (different target); EPSS v4
     replaced v3 on 2025-03-17, inside the test period - report recall before/after that date as well.
   - ASSUMPTION (FIRST documentation not reachable: api.first.org serves no model description and /epss
     redirects to www.first.org, which is outside the allow-list): "EPSS v3/v4 list CISA KEV among their data
     sources (per FIRST's published model description), so EPSS scores dated after a CVE's KEV addition may
     reflect the label."
   - "NOT YET IN KEV" ANALYSIS (pre-registered 2026-11-21): for EACH EPSS line, recall@110/@305 (CIs) of EVERY
     method counting only exploited CVEs whose KEV dateAdded is STRICTLY AFTER the date of the EPSS score that
     line uses. Exploited CVEs failing this are DROPPED from that evaluation (no credit, no capacity slot);
     exploited CVEs without a score in that line are kept (EPSS cannot benefit). Table 06_not_yet_in_kev.csv.
4. ROBUSTNESS.
   - Walk-forward fold 2022-24 -> 2025: the training set is all of dev, so this fold is the headline model scored
     on the 2025 test months. Reported as such (year 2025 vs 2026-Jan..Aug), next to WF 2023 / 2024-H1 / val.
   - Results by test quarter (2025Q1 .. 2026Q3; Q3 = Jul 1 - Aug 20), recall@110 with CIs and positives per quarter.
   - Sensitivities: generous fallback CVSS; uncertain day-0 phrases removed where desc_day0_uncertain is True
     (Decision 10c; on test = evaluation rows edited, frozen models; embeddings from gpu_run.py embed-test, which
     embeds the phrase-removed text of EVERY test row with a phrase, so it reads only cve_id + description_day0); 5 seeds of the text MLP individually
     (mean + worst) next to the ensemble.
5. EXECUTION.
   - One script, src/phase6_test.py, writing outputs/tables/06_*.csv, outputs/figures/06_*.png and
     outputs/logs/step8_phase6.log. It is first DRY-RUN on val: the SAME recipe fit on TRAIN ONLY (MLP stopping
     set 2024-01..06), val treated as "test"; must reproduce 05b (fusion val@110 42.6, gradient boosting 47.1);
     only then is it run ONCE on test (--test --confirm-phase6; refuses a second run).
   - If the test run crashes, fix only the code error and rerun; record it in the log. No change of model, K,
     metric, comparator or period definition after test rows have been seen.
6. AFTER PHASE 6 (Decision 6): retrain the headline (same recipe; ES set = last 6 months of all labeled data,
   2026-03 -> 2026-08) on train+val+test for the demo. Reported metrics stay the Phase 6 ones. Demo probability
   = calibrator applied to the fused score; a single new CVE is ranked against a reference month of scores
   (the most recent labeled month), since there is no "current month" batch in the demo.

7. PRE-TEST CHECKS (dev only, done 2026-11-20, recorded before test opened):
   a) CNA CVSS stability (src/check_cna_cvss_stability.py, table 05d_cna_cvss_stability.csv; history sample
      319 pos / 641 neg). CVSS 3.x formula reproduces 100% of 5,000 dev CNA scores. CNA vector today differs
      from the CNA's first known vector: 11/199 positives with CNA CVSS (5.5%) vs 7/369 negatives (1.9%) ->
      LABEL-CORRELATED, like description edits. v3 score up 7 / down 2 (pos; mean +1.6 when changed) vs up 5 /
      down 2 (neg; +0.4). Crossed 9.0 upward: 2 positives (CVE-2023-36845 Juniper 5.3->9.8, CVE-2024-4040
      CrushFTP 7.7->9.8), 0 negatives. 2 positives changed only v4 supplemental U:Red->Amber (no base change).
      CNA CVSS first added > 7 d after publication: 1 pos / 12 neg.
      Sensitivity (first known CNA vector for the 16 v3 changes, 9 pos; models refit): fusion selection 46.7 ->
      45.9 (worst 42.6 -> 41.2); gradient boosting 47.4 -> 46.6. Small; a lower bound for negatives (sample).
   b) Uncertain day-0 phrases removed (21 dev CVEs, all train, 5 pos; new MiniLM embeddings via gpu_run.py
      embed-variant; both models refit): fusion selection 46.7 -> 44.7 (45.5 / 48.9 / 39.7; worst 39.7);
      gradient boosting 47.4 -> 47.8 (37.2 / 57.8 / 48.5; worst 37.2). Table 05d_sensitivity_dev.csv.
      NOTE: under this variant the Decision 11a rule would pick gradient boosting (gap 3.1 > 2 pts). The
      headline stays as frozen unless the user decides otherwise BEFORE test; both models are scored on test.

8. LIMITATIONS to report (write-up + README):
   - CNA CVSS is not fully day-0 (check 7a): CNAs changed their own vector after publication for 5.5% of exploited
     vs 1.9% of other CVEs; 2 exploited CVEs crossed 9.0 only later. Using the first known vector costs ~0.8 pts
     of selection score (lower bound: history only for a sample of negatives). Same for description edits (10e).
   - Uncertain day-0 phrases (check 7b): fusion -2.0 pts when removed, gradient boosting (headline) +0.4.
   - EPSS comparison: lookahead favours EPSS; EPSS uses KEV (assumption above) -> "not yet in KEV" analysis.

# LEAKAGE RULES (always; no exceptions without the user's OK)
- After building the dataset, train+val is saved to data/dev.parquet and test to data/test_locked.parquet.
  NEVER open, load or summarize test_locked.parquet (or the test rows of cve_dataset.parquet) before Phase 6,
  except the Phase 3 global counts allowed below (CVEs/positives per year).
- Phase 3: global counts (e.g. positives per year) may use all data. Any analysis that guides model choices
  (rates by vendor or CWE, feature ideas, vendor-name cleaning rules) uses TRAIN only.
- Fit ALL preprocessing (encoders, rare-category grouping, target encoding, scalers, text vectorizers) on train
  only, inside scikit-learn Pipelines, so it is re-fit inside each CV fold.
- Any history feature (e.g. a vendor's past KEV entries) uses only KEV entries with dateAdded BEFORE the CVE's
  publication date.
- Tune thresholds, K and calibration on val. Run test ONCE, in Phase 6.

# FINAL DELIVERABLE (follow from Step 5 on)
1. NOTEBOOK STYLE (like the course homework): numbered sections; each = short markdown intro (why) -> code ->
   at least one figure where useful -> a bold "**Takeaway**:" paragraph with the actual numbers. Ends with a
   side-by-side comparison table of ALL methods. Runs top to bottom from cached data, locally and in Colab.
2. Section "The evolution of cybersecurity" - DESCRIPTIVE ONLY, all years allowed, with a note that no modeling
   choice uses it: CVEs vs KEV additions per month, time-to-KEV by year, vendor share of exploited CVEs over
   time, CWE trends, text topic clusters over time, ransomware share.
3. Section "Leakage we found and fixed": CVSS source (CISA-ADP, Decision 9), CWE source (NVD, Decision 10a),
   description edits (Decision 10c/e) - with before/after numbers.
4. MODEL LADDER. SELECTION RULE FOR ALL PHASE 5 MODELS (decided 2026-11-20): choose each model's setting by the
   MEAN recall@110 (per month) over the three periods WF 2022->2023, WF 2022-23->2024-H1 and val (train->val),
   equal weights - never val alone, never AUC (PR-AUC secondary only). Report the worst period too.
   NEURAL NETS (decided 2026-11-20): the last 6 months of each training period = early-stopping set (never val)
   used only to find the best epoch count; then REFIT on the full training period for that many epochs (same
   seed). Report 5 seeds (mean + worst). Embeddings come only from src/gpu_run.py (pinned HF revisions).
   Main K = monthly capacity of the CVSS >= 9 (CNA) rule (~110/month); also report K matching the generous
   fallback rule (~306/month). recall@K is computed per calendar month (top-K of that month's CVEs), pooled.
   Benchmarks: CVSS>=9 (CNA), CVSS>=7 (CNA), phrase OR CVSS>=9 (CNA), generous fallback CVSS>=9, EPSS (Phase 6).
   Models: logistic regression -> gradient boosting (HGB) -> frozen text embeddings + LR -> small neural net on
   embeddings (early stopping; small grid architectures x learning rates) -> combined tabular + text neural net
   -> optional: fine-tuned small transformer on MPS -> optional: domain-adaptive MLM pretraining on
   train-period descriptions only. Explain: the core task is SUPERVISED; the pretrained encoder is
   SELF-SUPERVISED transfer learning (and MLM pretraining is self-supervised adaptation to our domain).
5. ROBUSTNESS: walk-forward folds + bootstrap CIs on recall@K; report the worst case as well as the average.
6. REPO: one-page README for a data scientist (problem, data, step-by-step reproduction, results table,
   leakage fixes, limitations) + requirements.txt + the dated KEV snapshot (data/raw/kev_YYYYMMDD.csv). Solo project.

# PLANNED CHECKS
- Phase 5, description edits (Decision 10e): on the 919-CVE history sample, score each CVE with the text model
  twice - today's description_day0 vs the reconstructed earliest description (nvd_history.description_timeline)
  - and compare scores and ranking (e.g. Spearman, top-K overlap, per label). If they barely move, edits don't
  matter. If they move a lot: train and evaluate on the reconstructed text where available and report it as a
  limitation (history exists only for the sample + phrase CVEs).
- Phase 5, uncertain day-0 phrases: rerun with phrases removed where desc_day0_uncertain is True.
- Phase 6: fallback-score CVSS rules as the "generous baseline" (Decision 9); walk-forward fold 2022-24 -> 2025.
- CNA CVSS stability: did CNAs change their OWN CVSS score after publication? Use the cached history files
  (data/nvd_history_cache, 'CVSS V3.1'/'CVSS V4.0' details whose change source == the CNA), by label.
  If CNA scores move toward 'exploited', cvss_cna_* is not fully day-0 either.

# FACTS FROM EARLIER TESTS (use them to check work)
- NVD API 2.0: https://services.nvd.nist.gov/rest/json/cves/2.0, params pubStartDate/pubEndDate
  (format 2024-04-01T00:00:00.000, max 120 days per query; use 90-day windows), resultsPerPage=2000, startIndex,
  noRejected. Header "apiKey". Rate limit: 50 requests/30 s with a key (sleep ~0.7 s), 5/30 s without.
  Record fields: id, sourceIdentifier, published, vulnStatus, descriptions, affected, metrics, weaknesses,
  configurations, references, and cisa* on KEV entries.
- Test window 2024-04-01 -> 2024-04-20 gave 2,819 CVEs and 6 positives (0.21%): CVE-2024-3272 and CVE-2024-3273
  (D-Link, 7 days), CVE-2024-29745 and CVE-2024-29748 (Android, -1 day), CVE-2024-29988 (Microsoft, 21 days),
  CVE-2024-3400 (Palo Alto PAN-OS, 0 days). CVSS source counts: 2,007 cna / 636 other / 174 nvd / 2 none.
- KEV on 2026-11-19 had 1,786 rows, latest dateAdded 2026-11-18 -> labels complete up to 2026-08-20.
- Expected full data: about 200k CVEs since 2022, a few hundred positives.

# THE STEPS (one at a time, wait for the user's OK after each)
1. Environment: create .venv with Python 3; install pandas, requests, python-dotenv, pyarrow, matplotlib,
   seaborn, scikit-learn, shap, sentence-transformers, torch (Apple MPS), jupyter, ipykernel.
   Write requirements.txt. Explain what each library is for.
2. Data code, src/build_cve_dataset.py: download in 90-day windows with pagination, retries/backoff and a
   .json.gz cache per finished window in data/nvd_cache/ (re-runs are free, crashes resume); parse_cve with
   publication-day features only (cve_id, published, cna, description, cwe, n_cwes, vendor, product,
   cvss_version, cvss_source, cvss_score, cvss_vector, attack_vector, attack_complexity, privileges_required,
   user_interaction, scope, conf/integ/avail impact, exploitability_score, impact_score); download_latest_kev;
   build_dataset (KEV join, label, days_to_kev, in_kev_ever, kev_ransomware, censoring); add_splits
   (split + cv_group); walk_forward_folds; summarize. First test on 2024-04-01 -> 2024-04-20 and check it
   matches the facts above.
3. Full run (Phases 1-2): the whole NVD download (~5-10 min with the key; run it in the background and show
   progress) -> data/cve_dataset.parquet. Show: total CVEs, positives by year, CVEs added to KEV after 90 days,
   the split table (CVEs and positives per set), walk-forward fold sizes, missing-value checks.
4. Phase 3 exploration (notebooks/03_exploration.ipynb + figures in outputs/figures/): positives per year;
   label rate by vendor and by CWE; CVSS of exploited vs not exploited; how many exploited CVEs the CVSS >= 9
   rule misses; cvss_source check; days-to-KEV distribution; clean vendor names. Summarize for the "problem" slide.
5. Phase 4 baselines: on val, rules CVSS >= 9 and >= 7 (CNA score, day-0) and "description_day0 phrase OR
   CVSS >= 9 (CNA)" -> recall, precision, number of patches.
   Define K (monthly capacity) and the main metric: recall@K (share of exploited CVEs caught at fixed capacity),
   plus PR-AUC. Run the random-split comparison here.
6. Phase 5 models (fit on train, compare on val): logistic regression -> HistGradientBoostingClassifier on tabular features ->
   text embeddings of the description (sentence-transformers on MPS) + simple model -> combined model.
   Grouped CV inside train, class imbalance handled, probability calibration. Walk-forward drift check
   (folds ->2023 and ->2024 only; the ->2025 fold waits for Phase 6).
   Results table.
7. Phase 6 final evaluation: ONE run on test; recall@K with bootstrap confidence intervals; EPSS comparison
   (EPSS score on/near the publication date, from api.first.org); coverage-vs-capacity chart;
   walk-forward fold 2022-24 -> 2025. Then retrain the final model on train+val+test for the demo.
8. Phase 7 explain + impact: SHAP explanations; analyst hours saved vs CVSS >= 9 at the same coverage;
   ransomware coverage (kev_ransomware). The "does it matter?" slide.
9. Phase 8 package (uses the model retrained on all labeled data): demo (paste a CVE description + basic fields -> probability + top reasons), slide content,
   one clean notebook that also runs in Google Colab (key from Colab secret NVD_API_KEY, data downloaded or cloned).
10. Git (when the user says so): the user initializes the repo and pushes to a private GitHub repo from VS Code;
    Claude checks that .env and the large data are ignored.

# Progress log
- 2026-11-19 — Step 0 (0a–0e) done: .claude/settings.json, .gitignore, .env.example, folder structure, CLAUDE.md.
  Pending (user): create .env, restart claude so sandbox loads, verify ~/Documents listing is refused.
- 2026-11-19 — Step 0f done: user confirmed .env created; sandbox active (reads outside Project1 denied).
  Python available: 3.13.15 (/Library/Frameworks/Python.framework/Versions/3.13).
- 2026-11-20 — Step 1 done: .venv (Python 3.13.15) with all libraries; requirements.txt written (pinned versions).
  GPU SETUP: the sandbox blocks Metal, so MPS shows False inside it. The user added
  "excludedCommands": [".venv/bin/python src/gpu_run.py*"] to .claude/settings.json. ALL GPU work (embeddings
  on MPS) goes in src/gpu_run.py, run exactly as `.venv/bin/python src/gpu_run.py ...` from the project folder,
  never with `cd` in the same command. Verified: it prints True. Everything else stays CPU/sandboxed.
  MODEL DECISION: lightgbm imports fail (needs Homebrew libomp, outside Project1). Use scikit-learn's
  HistGradientBoostingClassifier instead of LightGBM wherever the plan says LightGBM (same histogram-based
  gradient boosting; native NaN/categorical support; class_weight; no extra dependency; runs in Colab).
  lightgbm is left out of requirements.txt.
- 2026-11-20 — Step 2 done: src/build_cve_dataset.py (download_nvd/fetch_window with 90-day windows, pagination,
  retries, atomic .json.gz cache; parse_cve; download_latest_kev; build_dataset; add_splits; walk_forward_folds;
  summarize; CLI --start/--end/--out/--no-censor). Windows include the full last day (T23:59:59.999); the
  "2,819" fact = 2024-04-01..2024-04-19 inclusive (5 more CVEs on 04-20). Test reproduced all facts exactly:
  2,819 CVEs, same 6 positives and gaps, cvss_source 2,007/636/174/2, cutoff 2026-08-20 (kev_20261120.csv).
  Vendor/product: "n/a" stored as missing (16% vendor missing in test window); cv_group falls back to CNA.
  API KEY: the sandbox denies reading .env (from the Read(./.env) deny rule), so the code runs WITHOUT the key
  (6.5 s between requests). DECIDED (user, option 1): keep it that way, no sandbox change; full run ~12–15 min.
- 2026-11-20 — Step 3 done (full run, no API key, ~25 min). Pre-checks: make_windows asserts no gap/overlap
  (20 windows, 2022-01-01 -> 2026-11-20); build_dataset drops duplicate cve_ids and reports the count (0 found).
  Added LEAKAGE RULES section. Outputs: data/cve_dataset.parquet (196,403 rows), data/dev.parquet (train+val,
  93,842), data/test_locked.parquet (102,561; DO NOT OPEN before Phase 6). Log: outputs/logs/step3_full_run.log.
  Results: 243,749 raw records, 0 duplicates, 47,346 dropped by censoring; KEV snapshot 2026-11-18, labels
  complete up to 2026-08-20. 613 positives (0.31%); 193 KEV-after-90-days (label 0); 59 negative gaps (label 1).
  By year (cves/pos): 2022 25,073/85; 2023 28,816/121; 2024 39,953/113; 2025 48,150/159; 2026* 54,411/135.
  Splits: train 74,268/251 (0.338%), val 19,574/68 (0.347%), test 102,561/294 (0.287%) -> no <50 warning.
  Walk-forward: ->2023 25,073/85 vs 28,816/121; ->2024 53,889/206 vs 39,953/113; ->2025 93,842/319 vs 48,150/159.
  cvss_source: cna 136,203 / other 36,449 / nvd 22,193 / none 1,558. cvss_version: 3.1 175,384; 4.0 12,683;
  3.0 6,765; 2.0 13. Missing: description 0, cvss_score 0.79%, vendor 19.2%; scope/exploitability/impact
  scores 7.3% missing because CVSS v4.0 has no such fields (expected, not a bug).
  Window 2026-06-09..09-06 has 30k CVEs: checked, genuine 2026 growth (unique ids, 2026 ids, rising by month).
  RESOLVED (user, option 1): the ->2025 fold runs only in Phase 6; final model retrained on all labeled data
  after Phase 6, before the demo. Was: walk-forward fold "2022-24 -> 2025" tests on 2025 = test-period rows, conflicts
  with "never open test before Phase 6". Options: run that fold only in Phase 6, or replace it.
- 2026-11-20 — Step 4 (Phase 3 exploration) done, pending user review. notebooks/03_exploration.ipynb
  (generated by src/build_nb03_exploration.py, executed by src/run_notebook.py because the sandbox blocks
  Jupyter kernel ports), 5 figures in outputs/figures/03_*.png, tables in outputs/tables/03_*.csv.
  src/features.py: clean_vendor (lowercase, punctuation, company suffixes; train-derived) + vendor_or_cna.
  Train findings: 251/74,268 positives (1 in 296). CVSS>=9: 306 patches/month, recall 35.1%, precision 0.96%,
  misses 163/251 (126 of them are 7-8.9; mostly Microsoft 47, Apple 31, Google 20). CVSS>=7: recall 85.3%,
  1,204/month. Top 10 vendors = 70% of positives, 33% of CVEs; 54 vendors have any positive. Vendor missing
  33% in train (cve@mitre.org CNA: 17,120 CVEs, 25 positives). CWE-843 type confusion 7.1% rate (184 CVEs).
  KEV speed: 28% listed by day 0, 47% by 7 d, 58% by 30 d, 68% by 90 d; median 12 d.
  RED FLAG cvss_source: "other" = 20% of CVEs but 42.6% of positives (rate 0.70% vs cna 0.34%), mostly
  Apple/Google/mitre; "nvd" 17,179 CVEs, 0 positives. Suspect CISA-ADP scores added because of exploitation.
  Proposed next: store the metric's source id (cvss_source_id) and re-parse from cache (no re-download) to verify.
- 2026-11-20 — cvss_source_id check done. build_cve_dataset.py now adds cvss_source_id, has_cna_cvss,
  has_adp_cvss, has_nvd_cvss, cvss_cna_score/severity/vector/version. Rebuilt from cache with --end 2026-09-06
  (last cached window; everything later is censored anyway): 0 downloads, same 196,403 rows / 613 positives.
  test_locked.parquet regenerated, not opened. Log outputs/logs/step4b_rebuild.log; v1 log kept.
  Train results: "other" = 100% CISA-ADP (107/107 positives, 15,160/15,160 negatives). Every positive without
  a CNA score (107/251) has an ADP score. has_adp_cvss rate 0.69% vs 0.24%; has_cna_cvss 0.34% vs 0.33% (no
  signal, safe). nvd@nist.gov-sourced: 17,179 CVEs, 0 positives (ADP outranks NVD in the fallback order).
  CNA score missing for 43.7% of train CVEs. ADP is retroactive: 2022Q1 3.4% of CVEs vs 46% of positives.
  Baselines train (patches/month, recall): CVSS>=9 day-0 CNA-only 110/mo 21.9% vs fallback 306/mo 35.1%;
  CVSS>=7 CNA-only 614/mo 47.0% vs fallback 1,204/mo 85.3%. Tables outputs/tables/03b_*.csv. Decision 9 added.
  NEXT: Step 5 (Phase 4 baselines on val; day-0 CNA-only rule = the honest baseline, fallback = generous).
- 2026-11-20 — Pre-Step-5 leakage checks done (Decision 10 + PLANNED CHECKS section added).
  Phase 3 notebook updated: day-0 CNA numbers are the headline, fallback shown as "generous"; runner keeps
  text/table order. CWE: switched to CNA-only (cwe_fallback kept for analysis); rebuilt from cache.
  NVD change history: src/nvd_history.py (sample: 319 positives + 600 negatives, seed 42; plus --phrase-cves:
  147 CVEs incl. test, chosen by phrase match on cve_id+description only). Cache data/nvd_history_cache/
  (gitignored). src/description_leak.py: PHRASES, phrase_flags, build_description_day0 (clause removal).
  src/history_report.py -> outputs/tables/03e_*.csv. Dataset rebuilt from cache (0 NVD downloads): new columns
  description_day0, desc_day0_status, desc_day0_uncertain. Dev: 93,758 no phrase; 82 phrase kept (61 certain,
  36 pos; 21 uncertain, 5 pos); 2 phrase removed (Authy label 1, HAProxy label 0); 0 empty texts.
  Same 196,403 rows / 613 positives; test_locked.parquet regenerated, not opened.
  (Step 5 started after the user's go, see next entry.)
- 2026-11-20 — FINAL DELIVERABLE section added (notebook style, evolution section, leakage section, model ladder
  on val recall@K, robustness, README); PLANNED CHECKS + CNA CVSS stability check.
- 2026-11-20 — Step 5 (Phase 4 baselines) done: notebooks/04_baselines.ipynb (built by src/build_nb04_baselines.py,
  takeaways in src/nb04_takeaways.json), src/metrics.py (monthly recall@K with expected-value tie-breaks, paired
  within-month Poisson bootstrap, recall_at_budget), src/features.py day0_tabular + cvss_components (CNA vector).
  Figures 04_01/04_02, tables outputs/tables/04_*.csv.
  K fixed from TRAIN: K_MAIN = 110/month (CVSS>=9 CNA), K_GEN = 305/month (fallback). Val (68 pos, 3,262 CVEs/mo):
  rules at own capacity: CVSS>=9 CNA 196/mo 33.8% [23-45]; CVSS>=7 CNA 1,089/mo 64.7%; phrase OR CVSS>=9 CNA
  197/mo 38.2%; generous fallback 313/mo 41.2%. recall@110 / @305 [95% CI]: CVSS CNA ranking 23.7 [13-33] /
  38.3; phrase-then-CVSS 28.1 [17-38] / 42.7 [31-55] (+4.3 pts, P(better) 94%); generous ranking 15.1 / 38.5;
  random 3.5 / 9.6. Best rule misses 50/68 at K=110 (10 without CNA CVSS; Microsoft 17).
  Random-split comparison (same preview LR, same 3.4% monthly budget): date 38.2% vs random 59.1% (+21 pts);
  51% of random-val positives have an exploited same-vendor+month twin in train (date split 0%); PR-AUC 0.108 vs
  0.126. Preview LR (date split): recall@110 36.8%, @305 67.6%, PR-AUC 0.108.
  BAR FOR PHASE 5: beat 28.1% @110 and 42.7% @305 on val.
- 2026-11-20 — Clarifications: recall@K IS per calendar month (top-K per month, summed); ties = expected value
  under random tie-break (matches the mean of 50 random seeds within noise: 23.74 vs 23.41, 38.34 vs 38.41).
  Random-split table now names the model (preview LR C=0.1 balanced) and budget (3.37% of each month's CVEs;
  38.2%) vs flat 110/month (36.8%).
- 2026-11-20 — Step 6a (tabular models) done: notebooks/05a_tabular_models.ipynb (src/build_nb05a_tabular.py,
  src/nb05a_takeaways.json), src/tabular_models.py (build_X, make_lr, make_hgb, evaluate, grid_search),
  src/features.py KEV history (load_kev_raw, kev_history_features: dateAdded < publication AND not own entry;
  KEV_VENDOR_ALIASES juniper/craft cms/php/vmware tanzu). KEV match: 89.8% of KEV rows, 61.4% of KEV vendors.
  Train: vendor with earlier KEV entries -> 1.03% vs 0.10% exploited, 77% of positives.
  Views: val (train->val), WF 2022->2023, WF 2022-23->2024-H1 (recall@110 per month), grouped CV 5-fold by
  cv_group inside train (share-of-month budget, reported only). Selection = mean(val@110, WF mean@110).
  Best rule on the same views: val 28.1, WF 39.3 / 34.9 (worst 34.9).
  LR grid (C x min_freq, 10): best C=0.03, min_freq=20 -> val@110 39.7 [26-49], @305 70.6, WF 48.8 / 24.4.
  HGB grid (encoding x lr x leaves x min_leaf, 16): best onehot, lr 0.03, 15 leaves, min_leaf 100 -> val@110
  47.1 [36-60], @305 67.6 [57-79], WF 39.7 / 55.6 (worst 39.7 > rule's 34.9), gcv 46.6, PR-AUC 0.049.
  HGB - best rule @110: +19.9 [+6.7, +33.3], P=100%; best rule needs 520/month to match 47%.
  desc_len is a top feature: ablation -9 pts val; sensitivity check with ORIGINAL description length on the
  919 sample: val recall unchanged (47.1 -> 47.1, Spearman 0.98) -> genuine day-0 signal.
- 2026-11-20 — 6a re-selected with the THREE-PERIOD rule (mean recall@110 of WF 2023, WF 2024-H1, val), now the
  selection rule for all Phase 5 models (FINAL DELIVERABLE 4). Bug fixed: TargetEncoder(cv=5) shuffled its
  internal folds WITHOUT a seed (results changed between runs) -> cv=StratifiedKFold(5, shuffle=True,
  random_state=0). HGB choice UNCHANGED (onehot, lr 0.03, 15 leaves, min_leaf 100; selection 47.4; val 47.1 /
  67.6; WF 39.7 / 55.6). LR choice CHANGED to C=0.3, min_freq=5 (selection 37.8 vs 37.6 for old pick - a tie):
  val@110 30.9 [21-43], @305 58.8, WF 53.7 / 28.9 (worst below the rule's 34.9). HGB - LR @110: +15.5 [+1.6,
  +28.6], P=98%. Best rule selection score (3-period mean): 34.1.
- 2026-11-20 — Step 6b (text models) done: src/gpu_run.py embed (caches forced to .cache/hf; revision pinned and
  recorded): minilm = all-MiniLM-L6-v2 @1110a243 (146 s MPS), bge_small = BAAI/bge-small-en-v1.5 @5c38ec7c
  (267 s) -> data/embeddings/ (gitignored). src/text_models.py (text LR, PyTorch MLP with ES on last 6 months of
  each training period + refit on the full period, tabular+text MLP, HGB + OOF text score grouped by cv_group,
  within-month rank fusion). notebooks/05b_text_models.ipynb (src/build_nb05b_text.py, src/nb05b_takeaways.json).
  Selection (3-period mean recall@110): HGB 05a 47.4 (still best) | (c) late fusion HGB + text MLP 46.7 (worst
  period 42.6 > HGB 39.7) | (b) HGB + OOF text 45.3 | (a) tabular+text MLP (256,64) lr 1e-3 44.8 (worst seed
  42.6; best val@305 75.0 [63-84], PR-AUC 0.112) | text MLP (64,) 39.5 (worst seed 38.7) | text LR minilm C=1
  39.2 (bge_small 38.2) | best rule 34.1. Refit vs stop-set only: text MLP 33.9 -> 39.5 (WF2023 28.3 -> 43.0);
  tab+text MLP 37.3 -> 44.8. Val @110 vs HGB: (b)/(c) 42.6, -4.4/-5.1 (P 18-19%); vs rule +15.5 (P 99%).
  Sensitivity (919 sample, earliest vs today's text): Spearman 0.977 (edited only 0.936); val@110 text LR
  23.5 -> 26.5, HGB+text 42.6 -> 45.6 -> edits do not inflate results. Runs are deterministic (rerun identical).
- 2026-11-20 — Decision 11 (final model rule, K fixed, calibration plan) recorded before Step 6c.
- 2026-11-20 — Step 6c (fine-tuned transformer) done. src/gpu_run.py finetune {grid|one|seeds}: MiniLM @1110a243
  (6b winner) + linear head, class-weighted BCE, AdamW wd 0.01, dropout 0.1, max_len 128, max 3 epochs, epoch
  search on period minus last 6 months then refit; negatives subsampled to 25% with ONE fixed subsample seed
  (12345); HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE set; refuses to run without MPS; frozen settings use eager
  attention (MPS sdpa has no dropout with frozen layers). Outputs data/finetune/ (gitignored; per-epoch losses in
  .json). Mishaps (fixed): first run went to CPU because of a shell redirect (exemption needs the exact command);
  float64 class weight on MPS; sdpa crash on frozen settings. GPU time: grid 72.5 min + seed 1 22.2 min ~= 95 min.
  Grid (seed 0, selection): lr1e-5_full 36.8 | lr2e-5_freeze 36.7 | lr2e-5_full 36.5 | lr1e-5_freeze 30.2.
  Chosen lr1e-5_full; 2 seeds (user: seed 1 only): 36.8 / 32.4 -> mean 34.6 < text MLP 39.5 -> Decision 11b: NOT
  adopted into fusion. Overfitting: clear for lr2e-5_full on WF 2024-H1 (stop loss 1.19 -> 1.53, train 1.23 ->
  0.42); mild on val for 3/4 settings; freezing reduces it (+0.003 vs +0.33) but lr1e-5 + freeze underfits.
  Info only: fusion with fine-tuned 47.2 (worst 44.1) vs fusion with text MLP 46.7 (worst 42.6) = 1 CVE on val.
  DECISION 11 OUTCOME: final model = late fusion (gradient boosting + text MLP), selection 46.7, worst 42.6,
  val@110 42.6 [30-55], @305 64.7; simpler alternative = gradient boosting 47.4, val@110 47.1 [36-60].
  notebooks/05c_finetuned_transformer.ipynb (src/build_nb05c_finetune.py, src/nb05c_takeaways.json).
  NEXT: calibration of the fused score on val (Decision 11d) and Phase 6 (test, once) - wait for the user's go.
- 2026-11-20 — Step 7 (Phase 5d, test prep) in progress: TEST PROTOCOL section written (before opening test).
  Calibration done on dev (Decision 11d): isotonic chosen by leave-one-period-out Brier (near-tie with Platt);
  open issue = top isotonic step on 2 CVEs -> p 1.0 (guard proposed). src/calibration.py, log
  outputs/logs/step7_calibration.log. test_locked.parquet NOT opened. Waiting for the user's OK on the protocol.
- 2026-11-20 — Step 7 A-C done (dev/val only; test_locked.parquet NOT opened). User decisions: epoch rule = same
  recipe on dev; merged isotonic (>= 10 pos per step) unless worse than Platt; both pending checks run now.
  A) src/final_model.py fit-dev: epochs [3, 6, 3, 4, 4] (ES 2024-07..12), dev models + sha256 in outputs/models/.
  Calibration: MERGED ISOTONIC (LOPO Brier 0.003217 vs Platt 0.003219), top step 11.1%. Checks: see TEST
  PROTOCOL 7 (CNA CVSS changes label-correlated 5.5% vs 1.9%, sensitivity -0.8 pts; uncertain phrases -2.0 pts
  fusion / +0.4 HGB -> rule would flip to HGB under that variant; headline unchanged pending user).
  gpu_run.py: new `embed-variant <name>` (dev CVEs only, offline, pinned MiniLM).
  B) src/epss.py: first EPSS score on/after publication, <= 7 d, 100 CVEs/query, cached data/epss_cache/
  (gitignored). Bug fixed: API answers 422 "listNoResults" when no CVE in the batch has a score that day (e.g.
  2024-12-01 has no EPSS data at all) -> treated as an empty answer. Val: 604 queries; lag 0 14.4%, lag 1 80.9%,
  within 7 d 99.3% (143 none); positives: 5 lag 0, 63 lag 1.
  C) src/phase6_test.py --dry-run (train-only fit, ES 2024-01..06, epochs [5,7,7,7,7] = 05b): REPRODUCES 05b/04:
  fusion 42.6 [30-55] / 64.7; HGB 47.1 / 67.6; phrase-then-CVSS 28.1 / 42.7; CVSS CNA 23.7 / 38.3; generous
  15.1 / 38.5. EPSS (<=7 d) 36.8 [25-48] / 38.5, PR-AUC 0.208 (ours 0.037); EPSS day-0 only 2.9 / 7.5.
  Fusion - EPSS@110 +5.9 [-10, +21], P 70%; @305 +26.2 [+12, +42]. Same capacity: CVSS>=9 (CNA) 196/mo 33.8%
  vs fusion 55.9% at 196. Calibration on val (in-sample): Brier 0.003387 vs 0.003462 base. Seeds 41.2-44.1.
  Log outputs/logs/step7_phase6_dryrun.log; tables outputs/tables/06dry_*.csv.
  NEXT (needs explicit OK): D) gpu_run.py embed-test (+ uncertain-removed test variant), E) EPSS for test,
  F) phase6_test.py --test --confirm-phase6.
- 2026-11-21 — User decisions: fusion stays headline (report check 7b beside it); EPSS lag<=1 line added;
  "not yet in KEV" analysis pre-registered (EPSS-uses-KEV = ASSUMPTION, FIRST docs not reachable via
  api.first.org); limitations list (TEST PROTOCOL 8). Dry run rerun on val (outputs/logs/step7_phase6_dryrun.log),
  all earlier numbers unchanged. EPSS by day 1: 36.8 [25-49] / 38.5, PR-AUC 0.252. NOT YET IN KEV (val, cut by
  the <=7 d EPSS dates): 28/68 exploited CVEs were in KEV on/before their EPSS score date -> 40 kept; EPSS
  10.0 [2-20] @110 vs fusion 35.0 [20-51], gradient boosting 52.5 [38-70], phrase-then-CVSS 41.6. Lag<=1 cut:
  identical. Day-0 cut: 1 dropped, EPSS day-0 1.4. -> EPSS's val performance comes mostly from CVEs already in KEV.
- 2026-11-21 — Step D done (gpu_run.py embed-test: 102,561 test texts, MPS, offline, rev 1110a243, 173 s; read
  cve_id + description_day0 only; 57 phrase-removed variants). Step E started (EPSS test). DECISION 12 recorded
  (headline = gradient boosting, fusion = alternative; calibrator = Platt on logit(s_hgb), LOPO Brier 0.003228 vs
  merged isotonic 0.003235). phase6_test.py switched; dry run on val reproduces: headline 47.1 [36-60] / 67.6,
  fusion 42.6 / 64.7; headline - EPSS(<=7 d) @110 +10.3 [-6, +28]; not-yet-in-KEV: headline 52.5, fusion 35.0,
  EPSS 10.0. Log outputs/logs/step8_phase6_dryrun_decision12.log. No test label or result seen yet.
- 2026-11-21 — PHASE 6 DONE (step E: EPSS test, 2,468 queries, 43 min; lag 0 29.9%, lag 1 69.5%, 9 CVEs none.
  Step F: phase6_test.py --test --confirm-phase6 ran ONCE, no crash, no code change after it; log
  outputs/logs/step8_phase6.log, tables outputs/tables/06_*.csv, scores data/phase6_scores_test.parquet).
  Test 2025-01-01 -> 2026-08-20: 102,561 CVEs, 294 positives, 20 months. recall@110 / @305 [95% CI], PR-AUC:
  HEADLINE gradient boosting 33.7 [28-39] / 49.3 [44-55], 0.053 | fusion 29.3 / 48.0, 0.048 | phrase-then-CVSS
  25.2 / 42.2 | CVSS CNA 18.6 / 36.1 | generous 17.8 / 31.6 | EPSS <=7 d 29.6 / 45.4, 0.069 | EPSS by day 1
  29.6 / 45.1 | EPSS day-0 5.1 / 9.9. Headline minus: best rule +8.5 [+1.7, +15.3] @110; CVSS CNA +15.1;
  EPSS(<=7 d) +4.1 [-2.7, +10.5] P 90%; fusion +4.3 [-1.0, +9.1].
  Same capacity: CVSS>=9 (CNA) now 385/mo, recall 41.2% vs headline 51.7% at 385 (fusion 55.4%).
  NOT YET IN KEV (EPSS <=7 d cut: 185 of 294 kept, 109 in KEV on/before the EPSS date): headline 30.4 [24-37],
  phrase-then-CVSS 29.4, fusion 23.8, EPSS 14.6 [9-20] @110. Day-0 cut (280 kept): headline 34.2, EPSS day-0 4.3.
  DRIFT: by year 2025 41.9 [34-50] vs 2026 23.9 [17-32] (EPSS 29.6 both years); by quarter headline 39.1 / 44.4
  / 38.9 / 45.7 / 37.8 / 25.3 / 7.7 (2026Q3 = Jul-Aug 20, 39 pos); fusion 2026Q2 5.4, 2026Q3 0.0. CVE volume
  2026 ~6,800/month vs K = 110 fixed. EPSS v3 period: headline 43.2 vs EPSS 40.9; v4: 32.8 vs 30.8.
  Calibration (Platt, out-of-sample): Brier 0.002816 vs 0.002858 constant; mean predicted 0.42% vs observed
  0.29% (over-predicts; base rate fell); top 0.5% bin under-predicts (4.4% vs 7.6%).
  Seeds (fusion) 28.2-29.9. Uncertain-phrase sensitivity: N/A, test has 0 desc_day0_uncertain rows (57 phrase
  kept, 23 pos; 6 removed, 2 pos).
  NEXT (wait for user): Phase 6 notebook; then retrain on all labeled data (TEST PROTOCOL 6) for the demo.
- 2026-11-21 — POST-TEST EXPLORATORY ANALYSIS (user's go; designed AFTER test results; changes NO Phase 6 number,
  model, K, metric or comparator). src/post_test_analysis.py -> outputs/tables/06x_*.csv, log
  outputs/logs/step9_post_test.log, data/post_test_rolling_scores.parquet.
  DRIFT (recall@110, 2025 -> 2026): headline 41.9 -> 23.9, fusion 39.1 -> 17.8, phrase-then-CVSS 29.5 -> 20.0,
  CVSS CNA 19.2 -> 17.8, EPSS 29.6 -> 29.6. CVEs/month dev 2,607 / 2025 4,013 / 2026 6,801 -> K=110 = 4.2% / 2.7% /
  1.6% of a month. At a constant 4.2% share: headline 49.7 -> 34.1, EPSS 38.9 -> 39.1, CVSS CNA 23.4 -> 29.1.
  Exploited 2026 CVEs NOT from new vendors (88.9% vendor seen in dev; 70.4% vendor with earlier KEV entry vs 68.6%),
  but CVEs/month from KEV-history vendors 696 -> 1,246 -> 2,376 and CNA CVSS>=9/month 124 -> 238 -> 606; headline
  catch rate for KEV-vendor positives 53.2% -> 31.6% (Microsoft 16/28 -> 5/23); day-0 phrases 11.3% -> 3.7% of positives.
  ROLLING QUARTERLY RETRAIN (frozen 05a settings; fit on CVEs published < quarter start - 90 d, dev + test; check:
  frozen model on rebuilt features reproduces Phase 6 scores; 2025Q2 rolling = dev fit, identical): all test 34.7
  [29-40] vs frozen 33.7 (+1.0 [-3.3, +5.6]); 2025 -4.4 [-9.7, +0.7]; 2026 +7.4 [+0.0, +15.2] (31.3 vs 23.9);
  by quarter rolling 39.1 / 44.4 / 25.0 / 40.0 / 35.6 / 37.5 / 17.9 vs frozen 39.1 / 44.4 / 38.9 / 45.7 / 37.8 / 25.3 / 7.7.
  NOTEBOOK notebooks/06_test_evaluation.ipynb (src/build_nb06_test.py, takeaways src/nb06_takeaways.json; reads
  cached tables only, never re-scores test): sections 1-6 pre-registered, 7-8 marked post-test exploratory,
  9 limitations, 10 side-by-side table (outputs/tables/06_final_comparison.csv). Figures outputs/figures/06_0[1-6]_*.png.
  NEXT (wait for user): retrain on all labeled data (TEST PROTOCOL 6), then Phase 7.
- 2026-11-21 — STEP H done (src/final_model.py fit-all): DEPLOYMENT / DEMO model = gradient boosting, Decision 12
  settings, refit on ALL labelled data (dev + test, 2022-01-01 -> 2026-08-20; 196,403 CVEs / 613 positives, 22 s) ->
  outputs/models/deploy_hgb.joblib, sha256 627441c1e8435f73... in deploy_models.json (also records the frozen
  dev_hgb.joblib hash, re-checked unchanged). NEVER used for a reported number; all Phase 6/7 results = dev_hgb.joblib.
  Open for Phase 8: the Platt calibrator was fit on dev-model scores; the demo probability needs recalibration
  (test showed over-prediction on average) - decide with the user before Phase 8.
- 2026-11-21 — PHASE 7 done (frozen Phase 6 model dev_hgb.joblib only; deploy model not used). src/phase7_explain.py
  (log outputs/logs/step10_phase7.log, data/shap_test.parquet, tables 07_examples/07_ransomware*.csv) + notebook
  notebooks/07_explain_and_impact.ipynb (src/build_nb07_explain_impact.py, src/nb07_takeaways.json; assumptions cell
  at top: 6 h/patch, $100/h, SHARE_OF_FEED 1.0, K 110, budget share 4.22%). Figures outputs/figures/07_0[1-7]_*.png.
  SHAP METHOD: shap.TreeExplainer is WRONG for this model (native categorical CVSS splits; additivity error up to
  4.6 log-odds, exact on a no-categorical control) -> PERMUTATION SHAP on the 22 raw features (max_evals 500,
  background 100 random dev CVEs), additivity 1.6e-14. Rows: 2,000 random test + 294 exploited + 300 flagged + 3 examples.
  Global mean |SHAP| (random): CNA 0.79, vendor KEV entries all-time 0.65, CNA CVSS 0.41, desc length 0.36, CVSS PR
  0.35; exploited CVEs: vendor KEV 1.14 first. KEV-history push on NON-exploited CVEs 2025 0.00 -> 2026 +0.23; share
  with push > +1: 18.8% -> 26.6% (exploited unchanged ~+1.05) = mechanism of the 2026 drift.
  Examples: caught CVE-2026-24858 (Fortinet, 9.8); missed CVE-2026-56155 (Microsoft AD FS, 7.8, top 6.4% of month);
  false alarm CVE-2026-20079 (Cisco FMC, 10.0; added to KEV after 189 d). Top-10 false alarms later in KEV: 3% (5/164).
  BUSINESS (176 exploited/yr): @110 GB +15/yr vs best rule, +27/yr vs CVSS ranking; at CVSS>=9's own 385/mo +19/yr.
  Same recall as CVSS>=9: 198 vs 385 patches/mo -> ~13,500 h, ~$1.35M/yr (2025 83 vs 238; 2026 frozen 569 vs 606,
  rolling 302); vs phrase OR CVSS>=9: 265 vs 387 (2026 frozen 630 vs 607 = WORSE; rolling 330).
  Ransomware (43 pos): GB 42.2 [27-57] @110 vs EPSS 27.9, best rule 26.7, CVSS 24.6; @385 GB 65.1 vs CVSS>=9 rule 58.1.
  Policy (post-test): frozen K=110 33.7 | scaled 4.22% 42.5 | rolling+scaled 46.9 (2026: 23.9 -> 44.4); fixed K with
  the same total patches ~equal (41.8 / 48.0) -> gain = capacity sized to volume, cost ~+7,700 h/yr.
  NEXT (wait for user): Phase 8 (demo uses deploy_hgb.joblib; calibration for the demo to decide).
- 2026-11-21 — PHASE 8a done (local only: no git, no push, nothing sent off the machine).
  0) DEMO CALIBRATION (src/final_model.py calibrate-deploy): demo PRIMARY output = within-month percentile vs reference
  month 2026-07 (deploy-model scores, 9,771 CVEs -> outputs/models/demo_reference_2026-07.csv). INDICATIVE probability
  = Platt on logit(score) fit on time-ordered out-of-fold scores of the deploy recipe (8 half-years 2023H1..2026H2, each
  fit on all labelled CVEs published before it; 171,330 CVEs / 528 pos). Leave-one-half-year-out Brier 0.002992 vs
  constant 0.003072; log loss 0.0175 vs 0.0209. p at OOF score quantiles 50/90/99/99.9% = 0.08% / 0.82% / 3.3% / 7.1%.
  -> outputs/models/calibrator_deploy.joblib/.json.
  1) MAIN NOTEBOOK notebooks/00_main_report.ipynb (src/build_nb00_main.py, src/nb00_takeaways.json; 12 sections, runs
  from committed outputs/tables/*.csv + KEV snapshot + model jsons only; figures 00_01..00_09). New summary tables by
  src/build_main_tables.py (00_evo_*, 00_kev_*, 00_leak_adp_by_quarter, 00_leak_kev_history). USER ADDITION: "patch
  CVSS>=9" stated as an ASSUMED stakeholder policy (common practice, not verified for an organisation) in Section 1 and
  Limitations; evidence it doesn't matter: model leads every day-0 rule at every K 10..1000 (checked: min lead +1.2 pts
  vs phrase-then-CVSS at K=30) and each rule at its own capacity; only EPSS (lookahead) is ahead, at K <= 50.
  Leak check added: vendor KEV-history guard did NOT move val numbers (day-0 47.1 vs own-entry leak 45.6 vs naive
  today's-catalogue 35.0) -> reported as a correctness guarantee, not an inflation fix.
  Business section: savings presented CONDITIONAL ON QUARTERLY RETRAINING (148 vs 385 patches/mo -> ~17,000 h, ~$1.7M/yr
  at 6 h, $100/h; sensitivity 2-10 h: $0.57M-2.84M; 2026 rate $0.73M-3.65M; frozen 2026 only $0.09M-0.44M).
  2) README.md (one page: problem, data, results, layout, ordered reproduction table with GPU/key/run time, rerun main
  notebook). 3) requirements.txt pinned from .venv (+numpy, scipy, joblib, transformers, tokenizers, safetensors,
  nbformat); scripts/download_all.sh (NVD -> history -> rebuild -> EPSS, uses committed KEV snapshot);
  build_cve_dataset.py new --kev option (default unchanged: download latest); check_sensitivity_dev.py new `prepare`
  mode (reproduces the 7b variant input exactly; was an unsaved one-off); .gitignore rewritten (caches, parquet,
  embeddings, *.npy ignored; KEV snapshot, tables, figures, notebooks, models (all < 2 MB) kept; verified with a
  gitignore matcher); .vscode/settings.json (python.analysis.extraPaths ["src"], .venv interpreter).
  4) SECRETS AUDIT: 171 text files scanned outside .env* / .venv / .cache / download caches. .env and .env.example NOT
  opened (sandbox-protected) -> key value not compared; the code never had the key (runs were keyless since Step 2).
  Key/token-like lines: 2, both placeholders in CLAUDE.md (lines 25 "...", 36 "paste-your-key-here"). UUID-shaped
  strings: 48 = public NVD CNA ids; 3 = public Ubiquiti advisory URLs in KEV notes. .env, .env.* ignored (except
  .env.example). Nothing found.
  NEXT (wait for user): Phase 8b demo; git init/push only when the user says so.
