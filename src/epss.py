"""EPSS benchmark scores (TEST PROTOCOL 3): FIRST EPSS on or after the NVD publication date.

For each CVE: query api.first.org/data/v1/epss?cve=<up to 100 ids>&date=D for D = publication date (UTC), then
D+1 ... D+7 for the CVEs still without a score. Keep the FIRST score found: epss, epss_percentile, epss_date,
epss_lag_days (0..7). No score within 7 days -> NaN (ranked last in evaluation, counted and reported).
NOTE: the up-to-7-day lookahead FAVOURS EPSS (our model uses publication-day information only).

Every API response is cached as data/epss_cache/<date>_<hash of ids>.json.gz -> re-runs are free, crashes resume.
CLI:
  .venv/bin/python src/epss.py val                      -> dev rows with split == val  -> data/epss_val.parquet
  .venv/bin/python src/epss.py test --confirm-phase6    -> test_locked.parquet (cve_id + published ONLY)
                                                           -> data/epss_test.parquet   [needs the user's OK]
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://api.first.org/data/v1/epss"
CACHE = ROOT / "data" / "epss_cache"
MAX_LAG, BATCH, PAUSE = 7, 100, 0.4
HEADERS = {"User-Agent": "cve-exploit-research/0.1 (academic project)"}


def _query(session: requests.Session, ids: list[str], date: str) -> list[dict]:
    key = hashlib.sha1(",".join(sorted(ids)).encode()).hexdigest()[:12]
    path = CACHE / f"{date}_{key}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))["data"]
    for attempt in range(6):
        try:
            r = session.get(URL, params={"cve": ",".join(ids), "date": date}, headers=HEADERS, timeout=60)
            if r.status_code == 422 and "listNoResults" in r.text:
                # none of these CVEs has a score that day (not scored yet, or an EPSS day with no data,
                # e.g. 2024-12-01) -> valid empty answer; cached like any other
                r_json = {"status": "OK", "total": 0, "limit": 100, "data": [], "note": "422 listNoResults"}
                r.json = lambda r_json=r_json: r_json
                r.status_code = 200
            if r.status_code == 200:
                js = r.json()
                assert js.get("status") == "OK" and js["total"] <= js["limit"], "unexpected EPSS response"
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(json.dumps(js).encode()))
                tmp.replace(path)                       # atomic: a crash never leaves a half-written cache file
                time.sleep(PAUSE)
                return js["data"]
            wait = 2 ** attempt * (5 if r.status_code == 429 else 1)
        except requests.RequestException:
            wait = 2 ** attempt
        time.sleep(wait)
    raise RuntimeError(f"EPSS query failed after retries: date={date}, {len(ids)} ids")


def fetch_epss(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """df: cve_id, published. Returns one row per CVE with the first EPSS score within MAX_LAG days."""
    CACHE.mkdir(parents=True, exist_ok=True)
    pub_day = df["published"].dt.normalize()
    found: dict[str, dict] = {}
    session = requests.Session()
    days = sorted(pub_day.unique())
    t0, n_calls = time.time(), 0
    for i, d in enumerate(days):
        ids = sorted(df.loc[pub_day == d, "cve_id"])
        for lag in range(MAX_LAG + 1):
            todo = [c for c in ids if c not in found]
            if not todo:
                break
            qd = (pd.Timestamp(d) + pd.Timedelta(days=lag)).strftime("%Y-%m-%d")
            for j in range(0, len(todo), BATCH):
                for row in _query(session, todo[j:j + BATCH], qd):
                    if row["cve"] in todo and row["cve"] not in found:
                        found[row["cve"]] = {"epss": float(row["epss"]), "epss_percentile": float(row["percentile"]),
                                             "epss_date": row["date"], "epss_lag_days": lag}
                n_calls += 1
        if (i + 1) % 20 == 0 or i + 1 == len(days):
            log(f"  {i + 1}/{len(days)} publication days | {len(found):,}/{len(df):,} CVEs scored | "
                f"{n_calls} queries | {time.time() - t0:.0f}s", flush=True) if log is print else log(i)
    out = df[["cve_id", "published"]].copy()
    rec = pd.DataFrame.from_dict(found, orient="index")
    out = out.join(rec, on="cve_id")
    return out


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "val":
        df = pd.read_parquet(ROOT / "data" / "dev.parquet", columns=["cve_id", "published", "split"])
        df = df[df["split"] == "val"].drop(columns="split")
    elif which == "test" and "--confirm-phase6" in sys.argv:
        df = pd.read_parquet(ROOT / "data" / "test_locked.parquet", columns=["cve_id", "published"])  # ids + dates only
    else:
        sys.exit(__doc__)
    print(f"EPSS for {which}: {len(df):,} CVEs, {df['published'].dt.normalize().nunique()} publication days")
    out = fetch_epss(df)
    out.to_parquet(ROOT / "data" / f"epss_{which}.parquet", index=False)
    lag = out["epss_lag_days"]
    print("lag (days) distribution:", lag.value_counts(dropna=False).sort_index().to_dict())
    print(f"scored on publication day: {(lag == 0).mean():.1%} | within 7 days: {lag.notna().mean():.1%}")


if __name__ == "__main__":
    main()
