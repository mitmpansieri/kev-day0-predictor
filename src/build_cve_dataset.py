"""Build the CVE exploitation dataset (Phases 1-2).

Population : CVEs published on NVD from 2022-01-01 (NVD API 2.0).
Label      : 1 if the CVE was added to CISA KEV within 90 days of NVD publication (negative gap counts as 1).
Features   : ONLY information available on publication day (see CLAUDE.md, decision 4 - leakage).

Usage (from the project folder):
    .venv/bin/python src/build_cve_dataset.py --start 2024-04-01 --end 2024-04-20 --out data/test_window.parquet
    .venv/bin/python src/build_cve_dataset.py                       # full run -> data/cve_dataset.parquet
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "nvd_cache"
RAW_DIR = ROOT / "data" / "raw"

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
KEV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
USER_AGENT = "cve-exploit-prediction-research/0.1 (python-requests)"
NVD_SOURCE = "nvd@nist.gov"
CISA_ADP_SOURCE = "134c704f-9b21-4f2e-91b3-4a467353bcc0"   # CISA-ADP (Vulnrichment)
POP_START = date(2022, 1, 1)
WINDOW_DAYS = 90          # NVD allows max 120 days per query
PAGE_SIZE = 2000
LABEL_DAYS = 90
DEFAULT_OUT = "data/cve_dataset.parquet"

SPLITS = [  # (name, first day, last day) - test runs up to the censoring cutoff
    ("train", date(2022, 1, 1), date(2024, 6, 30)),
    ("val", date(2024, 7, 1), date(2024, 12, 31)),
    ("test", date(2025, 1, 1), date(2100, 1, 1)),
]


# ----------------------------------------------------------------------------------------------
# 1. Download NVD in 90-day windows, with pagination, retries and a per-window cache
# ----------------------------------------------------------------------------------------------
def _api_key() -> str | None:
    load_dotenv(ROOT / ".env")          # the key is never printed or logged
    return os.getenv("NVD_API_KEY") or None


def _get_with_retries(session: requests.Session, params: dict, max_tries: int = 8, url: str = NVD_URL) -> dict:
    """GET one NVD page. Retries on rate limits (403/429), server errors and network errors."""
    for attempt in range(1, max_tries + 1):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code not in (403, 429, 500, 502, 503, 504):
                raise RuntimeError(f"NVD returned HTTP {r.status_code}: {r.text[:200]}")
            reason = f"HTTP {r.status_code}"
        except (requests.ConnectionError, requests.Timeout, json.JSONDecodeError) as e:
            reason = type(e).__name__
        wait = min(60, 2 ** attempt)
        print(f"    retry {attempt}/{max_tries} after {reason}, sleeping {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError("NVD request failed after retries")


def fetch_window(session: requests.Session, start: date, end: date, has_key: bool) -> list[dict]:
    """All CVE records published between start and end (both days inclusive). Cached once finished."""
    cache = CACHE_DIR / f"nvd_{start:%Y%m%d}_{end:%Y%m%d}.json.gz"
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return json.load(f)

    params = {
        "pubStartDate": f"{start:%Y-%m-%d}T00:00:00.000",
        "pubEndDate": f"{end:%Y-%m-%d}T23:59:59.999",
        "resultsPerPage": PAGE_SIZE,
        "startIndex": 0,
        "noRejected": "",
    }
    pause = 0.7 if has_key else 6.5          # 50 req/30 s with a key, 5 req/30 s without
    records, total = [], None
    while total is None or params["startIndex"] < total:
        page = _get_with_retries(session, params)
        total = page["totalResults"]
        records += [v["cve"] for v in page.get("vulnerabilities", [])]
        params["startIndex"] += PAGE_SIZE
        print(f"  {start} -> {end}: {len(records):,}/{total:,}", flush=True)
        time.sleep(pause)

    if len(records) != total:
        raise RuntimeError(f"window {start}->{end}: got {len(records)} records, expected {total}")
    # cache only windows that are fully in the past (today's window may still grow)
    if end < datetime.now(timezone.utc).date():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(records, f)
        tmp.replace(cache)                    # atomic: a crash never leaves a half-written cache
    return records


def make_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Consecutive, non-overlapping windows of at most WINDOW_DAYS days (both ends inclusive)."""
    windows, w_start = [], start
    while w_start <= end:
        w_end = min(w_start + timedelta(days=WINDOW_DAYS - 1), end)
        windows.append((w_start, w_end))
        w_start = w_end + timedelta(days=1)
    # safety checks: no gap, no overlap, full coverage
    assert windows[0][0] == start and windows[-1][1] == end
    for (_, prev_end), (nxt_start, _) in zip(windows, windows[1:]):
        assert nxt_start == prev_end + timedelta(days=1), f"gap/overlap at {prev_end} -> {nxt_start}"
    return windows


def download_nvd(start: date, end: date) -> list[dict]:
    key = _api_key()
    print(f"NVD download {start} -> {end} ({'with' if key else 'WITHOUT'} API key)", flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if key:
        session.headers["apiKey"] = key
    windows = make_windows(start, end)
    out = []
    for i, (w_start, w_end) in enumerate(windows, 1):
        out += fetch_window(session, w_start, w_end, has_key=bool(key))
        print(f"[progress] windows {i}/{len(windows)} done | CVEs so far: {len(out):,}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------
# 2. Parse one CVE record -> publication-day features only
# ----------------------------------------------------------------------------------------------
# Deliberately NOT read (leakage, see CLAUDE.md decision 4): cisa* fields, metrics.ssvcV203,
# references (+ tags), configurations/CPEs. They describe what happened AFTER publication.
CVSS_KEYS = ["cvssMetricV31", "cvssMetricV30", "cvssMetricV40", "cvssMetricV2"]   # preference order
V2_LEVEL = {"NONE": "NONE", "PARTIAL": "LOW", "COMPLETE": "HIGH"}
V2_AUTH = {"NONE": "NONE", "SINGLE": "LOW", "MULTIPLE": "HIGH"}


def _missing(x) -> bool:
    return x is None or str(x).strip().lower() in ("", "n/a", "na", "none", "unknown", "-")


def _pick_cvss(metrics: dict, cna: str) -> tuple[dict | None, str, str]:
    """Return (metric, version_key, cvss_source). Source order cna > other > nvd; within it v3.1>v3.0>v4.0>v2."""
    for source_kind in ("cna", "other", "nvd"):
        for key in CVSS_KEYS:
            for m in metrics.get(key, []):
                src = m.get("source", "")
                kind = "cna" if src == cna else ("nvd" if src == NVD_SOURCE else "other")
                if kind == source_kind:
                    return m, key, source_kind
    return None, "", "none"


def _cvss_features(m: dict | None, key: str) -> dict:
    """Flatten the chosen metric to one common vocabulary (v3 names; v2/v4 mapped onto them)."""
    f = dict.fromkeys(["cvss_version", "cvss_score", "cvss_vector", "attack_vector", "attack_complexity",
                       "privileges_required", "user_interaction", "scope", "conf_impact", "integ_impact",
                       "avail_impact", "exploitability_score", "impact_score"])
    if m is None:
        return f
    d = m["cvssData"]
    f.update(cvss_version=d.get("version"), cvss_score=d.get("baseScore"), cvss_vector=d.get("vectorString"),
             exploitability_score=m.get("exploitabilityScore"), impact_score=m.get("impactScore"))
    if key in ("cvssMetricV31", "cvssMetricV30"):
        f.update(attack_vector=d.get("attackVector"), attack_complexity=d.get("attackComplexity"),
                 privileges_required=d.get("privilegesRequired"), user_interaction=d.get("userInteraction"),
                 scope=d.get("scope"), conf_impact=d.get("confidentialityImpact"),
                 integ_impact=d.get("integrityImpact"), avail_impact=d.get("availabilityImpact"))
    elif key == "cvssMetricV40":    # no scope / sub-scores in v4; UI is NONE/PASSIVE/ACTIVE
        av = d.get("attackVector")
        ui = d.get("userInteraction")
        f.update(attack_vector="ADJACENT_NETWORK" if av == "ADJACENT" else av,
                 attack_complexity=d.get("attackComplexity"), privileges_required=d.get("privilegesRequired"),
                 user_interaction=None if ui is None else ("NONE" if ui == "NONE" else "REQUIRED"),
                 conf_impact=d.get("vulnConfidentialityImpact"), integ_impact=d.get("vulnIntegrityImpact"),
                 avail_impact=d.get("vulnAvailabilityImpact"))
    else:                           # v2
        uir = m.get("userInteractionRequired")
        f.update(attack_vector=d.get("accessVector"), attack_complexity=d.get("accessComplexity"),
                 privileges_required=V2_AUTH.get(d.get("authentication")),
                 user_interaction=None if uir is None else ("REQUIRED" if uir else "NONE"),
                 conf_impact=V2_LEVEL.get(d.get("confidentialityImpact")),
                 integ_impact=V2_LEVEL.get(d.get("integrityImpact")),
                 avail_impact=V2_LEVEL.get(d.get("availabilityImpact")))
    return f


def _cvss_provenance(metrics: dict, cna: str) -> dict:
    """Who supplied CVSS, and the CNA-only score (day-0 safe). Flags/ids are for analysis, NEVER model features."""
    all_m = [m for key in CVSS_KEYS for m in metrics.get(key, [])]
    sources = {m.get("source", "") for m in all_m}
    out = {"has_cna_cvss": cna in sources, "has_adp_cvss": CISA_ADP_SOURCE in sources,
           "has_nvd_cvss": NVD_SOURCE in sources,
           "cvss_cna_score": None, "cvss_cna_severity": None, "cvss_cna_vector": None, "cvss_cna_version": None}
    for key in CVSS_KEYS:                                   # v3.1 > v3.0 > v4.0 > v2, CNA only
        m = next((m for m in metrics.get(key, []) if m.get("source") == cna), None)
        if m:
            d = m["cvssData"]
            out.update(cvss_cna_score=d.get("baseScore"), cvss_cna_vector=d.get("vectorString"),
                       cvss_cna_version=d.get("version"),
                       cvss_cna_severity=d.get("baseSeverity") or m.get("baseSeverity"))   # v2: top level
            break
    return out


def _source_kind(src: str, cna: str) -> str:
    return "cna" if src == cna else "nvd" if src == NVD_SOURCE else "cisa_adp" if src == CISA_ADP_SOURCE else "other"


def _pick_cwe(weaknesses: list, cna: str) -> dict:
    """cwe / n_cwes = CNA's weakness entry ONLY (day-0). cwe_fallback* = old 'CNA else any' pick, analysis only."""
    def cwes(entries):
        out = []
        for w in entries:
            for d in w.get("description", []):
                v = d.get("value")
                if v and v not in out:
                    out.append(v)
        return out
    cna_cwes = cwes([w for w in weaknesses if w.get("source") == cna])
    fb_entry = next((w for w in weaknesses if w.get("source") == cna and cwes([w])), None) \
        or next((w for w in weaknesses if cwes([w])), None)
    return {"cwe": cna_cwes[0] if cna_cwes else "NVD-CWE-noinfo", "n_cwes": len(cna_cwes),
            "cwe_fallback": cwes([fb_entry])[0] if fb_entry else "NVD-CWE-noinfo",
            "cwe_fallback_source": _source_kind(fb_entry.get("source", ""), cna) if fb_entry else "none"}


def _pick_vendor_product(affected: list, cna: str) -> tuple[str | None, str | None]:
    """Vendor/product from the CNA's own 'affected' block (day-0 info). Prefer an entry marked affected."""
    entries = [d for a in affected if a.get("source") == cna for d in a.get("affectedData", [])]
    def is_affected(d):
        return d.get("defaultStatus") == "affected" or any(v.get("status") == "affected"
                                                            for v in d.get("versions", []))
    entries.sort(key=lambda d: not is_affected(d))     # stable: affected entries first
    for d in entries:
        vendor, product = d.get("vendor"), d.get("product")
        if not (_missing(vendor) and _missing(product)):
            return (None if _missing(vendor) else vendor.strip(), None if _missing(product) else product.strip())
    return None, None


def parse_cve(r: dict) -> dict:
    cna = r.get("sourceIdentifier", "")
    desc = next((d["value"] for d in r.get("descriptions", []) if d.get("lang") == "en"), "")
    cwe_info = _pick_cwe(r.get("weaknesses", []), cna)
    vendor, product = _pick_vendor_product(r.get("affected", []), cna)
    m, key, cvss_source = _pick_cvss(r.get("metrics", {}), cna)
    return {"cve_id": r["id"], "published": r["published"], "cna": cna, "description": desc,
            **cwe_info, "vendor": vendor, "product": product,
            "cvss_source": cvss_source, "cvss_source_id": m.get("source") if m else None,
            **_cvss_features(m, key), **_cvss_provenance(r.get("metrics", {}), cna)}


def parse_all(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([parse_cve(r) for r in records])   # duplicates removed in build_dataset
    df["published"] = pd.to_datetime(df["published"])          # NVD timestamps are UTC
    for c in ["cvss_score", "exploitability_score", "impact_score", "cvss_cna_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("published").reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# 3. KEV catalog (the label source)
# ----------------------------------------------------------------------------------------------
def download_latest_kev(force: bool = False) -> Path:
    """Download today's KEV CSV to data/raw/kev_YYYYMMDD.csv (reuses today's file if present)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"kev_{datetime.now(timezone.utc):%Y%m%d}.csv"
    if path.exists() and not force:
        return path
    r = requests.get(KEV_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def load_kev(path: Path) -> pd.DataFrame:
    kev = pd.read_csv(path)
    kev = kev.rename(columns={"cveID": "cve_id", "dateAdded": "kev_date_added"})
    kev["kev_date_added"] = pd.to_datetime(kev["kev_date_added"])
    kev["kev_ransomware"] = kev["knownRansomwareCampaignUse"].astype(str).str.lower().eq("known")
    return kev[["cve_id", "kev_date_added", "kev_ransomware"]].drop_duplicates("cve_id")


# ----------------------------------------------------------------------------------------------
# 4. Label, censoring, splits
# ----------------------------------------------------------------------------------------------
def build_dataset(cves: pd.DataFrame, kev: pd.DataFrame, censor: bool = True) -> tuple[pd.DataFrame, date, dict]:
    """Remove duplicate cve_ids, join KEV (full file, no dateAdded filter), make the label, drop censored CVEs."""
    stats = {"raw_records": len(cves)}
    cves = cves.drop_duplicates("cve_id", keep="first")
    stats["duplicates_removed"] = stats["raw_records"] - len(cves)
    print(f"Duplicate cve_ids removed: {stats['duplicates_removed']:,}")
    df = cves.merge(kev, on="cve_id", how="left")
    pub_day = df["published"].dt.tz_localize(None).dt.normalize() if df["published"].dt.tz is not None \
        else df["published"].dt.normalize()
    df["in_kev_ever"] = df["kev_date_added"].notna()
    df["days_to_kev"] = (df["kev_date_added"] - pub_day).dt.days            # NaN if never in KEV
    df["label"] = (df["in_kev_ever"] & (df["days_to_kev"] <= LABEL_DAYS)).astype(int)  # negative gap -> 1
    df["kev_ransomware"] = df["kev_ransomware"].fillna(False).astype(bool)

    snapshot = kev["kev_date_added"].max().date()
    cutoff = snapshot - timedelta(days=LABEL_DAYS)        # label knowable only if published <= cutoff
    keep = pub_day.dt.date <= cutoff
    stats["censored_dropped"] = int((~keep).sum()) if censor else 0
    if censor:
        df = df[keep]
    stats["kev_snapshot"] = snapshot
    return df.reset_index(drop=True), cutoff, stats


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    day = df["published"].dt.date
    df["split"] = None
    for name, lo, hi in SPLITS:
        df.loc[(day >= lo) & (day <= hi), "split"] = name
    vendor = df["vendor"].fillna("").str.strip().str.lower()
    key = vendor.where(vendor != "", df["cna"].str.lower())       # vendor missing / n/a -> CNA
    df["cv_group"] = key + "|" + df["published"].dt.strftime("%Y-%m")
    return df


def walk_forward_folds(df: pd.DataFrame) -> list[tuple[str, pd.Series, pd.Series]]:
    """Drift check: train on all years up to Y-1, test on year Y (2023, 2024, 2025)."""
    year = df["published"].dt.year
    return [(f"2022-{test_year - 1} -> {test_year}", (year >= 2022) & (year < test_year), year == test_year)
            for test_year in (2023, 2024, 2025)]


def _rate_table(g) -> pd.DataFrame:
    t = g["label"].agg(cves="size", positives="sum")
    t["rate_%"] = (100 * t["positives"] / t["cves"]).round(3)
    return t


def summarize(df: pd.DataFrame, cutoff: date, stats: dict | None = None) -> None:
    n, p = len(df), int(df["label"].sum())
    if stats:
        print(f"\nRaw records {stats['raw_records']:,} | duplicates removed {stats['duplicates_removed']:,} | "
              f"dropped by censoring {stats['censored_dropped']:,} | KEV snapshot {stats['kev_snapshot']}")
    print(f"CVEs: {n:,} | positives (KEV <= {LABEL_DAYS} d): {p:,} ({p / max(n, 1):.2%}) | "
          f"labels complete up to {cutoff}")
    late = df["in_kev_ever"] & (df["days_to_kev"] > LABEL_DAYS)
    print(f"In KEV but added after {LABEL_DAYS} days (label 0): {int(late.sum())}")
    print(f"In KEV before NVD publication (negative gap, label 1): {int((df['days_to_kev'] < 0).sum())}")
    print("\nBy year:\n" + _rate_table(df.groupby(df["published"].dt.year)).to_string())
    sp = _rate_table(df.groupby("split", sort=False))
    rng = df.groupby("split", sort=False)["published"].agg(["min", "max"])
    sp.insert(0, "from", rng["min"].dt.date)
    sp.insert(1, "to", rng["max"].dt.date)
    print("\nSplits:\n" + sp.to_string())
    n_test_pos = int(sp.loc["test", "positives"]) if "test" in sp.index else 0
    if "test" in sp.index and n_test_pos < 50:
        print(f"\n*** WARNING: test set has only {n_test_pos} positives (< 50) - confidence intervals will be wide ***")
    print("\nWalk-forward folds:")
    for name, tr, te in walk_forward_folds(df):
        print(f"  {name}: train {int(tr.sum()):,} ({int(df.loc[tr, 'label'].sum())} pos) | "
              f"test {int(te.sum()):,} ({int(df.loc[te, 'label'].sum())} pos)")
    print("\ncvss_source:", df["cvss_source"].value_counts().to_dict())
    print("cvss_version:", df["cvss_version"].value_counts(dropna=False).to_dict())
    print("\nMissing values (count, share):")
    key_miss = {"description (empty)": df["description"].fillna("").str.strip().eq("").sum(),
                "cvss_score": df["cvss_score"].isna().sum(), "vendor": df["vendor"].isna().sum()}
    for k, v in key_miss.items():
        print(f"  {k:22s} {int(v):>8,}  {v / max(n, 1):.2%}")
    print("All columns with missing values (share):")
    miss = df.isna().mean()
    print(miss[miss > 0].round(4).to_string())
    pos = df[df["label"] == 1]
    if len(pos) <= 20:
        print("\nPositives:\n" + pos[["cve_id", "vendor", "product", "cvss_score", "cvss_source",
                                     "days_to_kev"]].to_string(index=False))


# ----------------------------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=date.fromisoformat, default=POP_START)
    ap.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--no-censor", action="store_true", help="keep CVEs whose label is not knowable yet")
    ap.add_argument("--kev", default=None, help="use this KEV snapshot (e.g. data/raw/kev_20261120.csv) instead of "
                                                 "downloading the latest one (reproduces the published labels)")
    a = ap.parse_args(argv)

    cves = parse_all(download_nvd(a.start, a.end))
    kev_path = ROOT / a.kev if a.kev else download_latest_kev()
    print(f"KEV: {kev_path.name}")
    df, cutoff, stats = build_dataset(cves, load_kev(kev_path), censor=not a.no_censor)
    df = add_splits(df)
    from description_leak import build_description_day0      # Decision 10
    df = pd.concat([df, build_description_day0(df, ROOT / "data" / "nvd_history_cache")], axis=1)
    dev_status = df.loc[df["split"].isin(["train", "val"]), "desc_day0_status"].value_counts().to_dict()
    print(f"description_day0 status (train+val only): {dev_status}")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    summarize(df, cutoff, stats)
    print(f"\nSaved {len(df):,} rows -> {out.relative_to(ROOT)}")
    if a.out == DEFAULT_OUT:   # full run only: separate dev (train+val) and the LOCKED test set
        dev, test = df[df["split"].isin(["train", "val"])], df[df["split"] == "test"]
        dev.to_parquet(ROOT / "data" / "dev.parquet", index=False)
        test.to_parquet(ROOT / "data" / "test_locked.parquet", index=False)
        print(f"Saved dev.parquet ({len(dev):,} rows) and test_locked.parquet ({len(test):,} rows) "
              f"- do not open test_locked before Phase 6")


if __name__ == "__main__":
    main(sys.argv[1:])
