#!/usr/bin/env bash
# Re-download every external input of the project (everything except the committed KEV snapshot).
# Run from anywhere:  bash scripts/download_all.sh
# Needs network to services.nvd.nist.gov and api.first.org. Optional: NVD_API_KEY in .env (about 10x faster NVD).
# All downloads are cached (data/nvd_cache, data/nvd_history_cache, data/epss_cache): a crash resumes, a re-run is free.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
KEV=data/raw/kev_20261120.csv          # dated snapshot used for every published number (committed)
END=2026-09-06                          # last NVD window used; CVEs after the label cutoff (2026-08-20) are censored anyway

echo "1/5 NVD CVEs 2022-01-01 -> $END (~25 min without API key, ~5-10 min with)"
$PY src/build_cve_dataset.py --end "$END" --kev "$KEV"

echo "2/5 NVD change history: audit sample (319 positives + 600 negatives of dev, seed 42; ~100 min without key)"
$PY -u src/nvd_history.py

echo "3/5 NVD change history: CVEs whose description has an exploitation phrase (147; ~16 min without key)"
$PY -u src/nvd_history.py --phrase-cves

echo "4/5 rebuild the dataset from cache with description_day0 (no downloads)"
$PY src/build_cve_dataset.py --end "$END" --kev "$KEV"

echo "5/5 FIRST EPSS for validation and test (~10 + ~45 min)"
$PY src/epss.py val
$PY src/epss.py test --confirm-phase6
echo "done: data/cve_dataset.parquet, data/dev.parquet, data/test_locked.parquet, data/epss_{val,test}.parquet"
