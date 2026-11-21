"""Small summary tables for notebooks/00_main_report.ipynb, so the main notebook runs from committed files only
(outputs/tables/*.csv + data/raw/kev_*.csv), with no large parquet files, no re-scoring and no API calls.

Writes outputs/tables/00_*.csv:
  evolution (DESCRIPTIVE ONLY, all years; no modelling choice uses it):
    00_evo_monthly        CVEs / exploited per publication month (NVD 2022-01 -> 2026-08-20 label cutoff)
    00_evo_cvss_mix       CNA CVSS severity mix and CVSS source per year
    00_evo_vendors        top vendors: share of CVEs and of exploited CVEs per year
    00_evo_days_to_kev    days from NVD publication to KEV (all KEV-listed CVEs published 2022+)
    00_kev_by_month       KEV additions per month (whole catalogue)
    00_kev_by_cve_year    KEV entries by CVE-ID year (2002+), ransomware share, years from CVE year to KEV
  leakage evidence:
    00_leak_adp_by_quarter  share of CVEs vs of exploited CVEs with a CISA-ADP CVSS score, by quarter
    00_leak_kev_history     val recall@110 of gradient boosting with LEAKY KEV history (all entries, own entry
                            included, ignoring dates) vs the day-0 version used (train -> val; dev only)
Run: .venv/bin/python src/build_main_tables.py   (~2 min)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import final_model as fm                                       # noqa: E402
from features import load_kev_raw, vendor_or_cna               # noqa: E402
from metrics import recall_at_k                                # noqa: E402
from tabular_models import build_X, make_hgb                   # noqa: E402

TAB = ROOT / "outputs" / "tables"


def main():
    df = pd.read_parquet(ROOT / "data" / "cve_dataset.parquet")
    df["month"] = df["published"].dt.to_period("M").astype(str)
    df["year"] = df["published"].dt.year

    m = df.groupby("month").agg(cves=("cve_id", "size"), exploited=("label", "sum"))
    m.to_csv(TAB / "00_evo_monthly.csv")

    sev = df["cvss_cna_severity"].fillna("no CNA score").str.upper().replace({"NO CNA SCORE": "no CNA score"})
    mix = pd.crosstab(df["year"], sev, normalize="index").mul(100)
    mix["CNA CVSS >= 9 %"] = df.groupby("year")["cvss_cna_score"].apply(lambda s: 100 * (s >= 9).mean())
    mix["CVEs"] = df.groupby("year").size()
    src = pd.crosstab(df["year"], df["cvss_source"], normalize="index").mul(100).add_prefix("cvss_source=").add_suffix(" %")
    mix.join(src).round(2).to_csv(TAB / "00_evo_cvss_mix.csv")

    df["vendor_key"] = vendor_or_cna(df)
    top = df.loc[df["label"] == 1, "vendor_key"].value_counts().head(10).index
    rows = []
    for y, g in df.groupby("year"):
        for v in top:
            rows.append({"year": y, "vendor": v, "share of CVEs %": 100 * (g["vendor_key"] == v).mean(),
                         "share of exploited %": 100 * ((g["vendor_key"] == v) & (g["label"] == 1)).sum() / max(g["label"].sum(), 1)})
    pd.DataFrame(rows).round(2).to_csv(TAB / "00_evo_vendors.csv", index=False)

    k = df[df["in_kev_ever"]]
    d = k.groupby("year")["days_to_kev"].agg(
        kev_listed="size", median_days="median",
        **{"<= 0 d %": lambda s: 100 * (s <= 0).mean(), "<= 7 d %": lambda s: 100 * (s <= 7).mean(),
           "<= 30 d %": lambda s: 100 * (s <= 30).mean(), "<= 90 d % (= label 1)": lambda s: 100 * (s <= 90).mean()})
    d.round(1).to_csv(TAB / "00_evo_days_to_kev.csv")

    kev_raw = pd.read_csv(fm.KEV_FILE)
    kev_raw["dateAdded"] = pd.to_datetime(kev_raw["dateAdded"])
    kev_raw["cve_year"] = kev_raw["cveID"].str.slice(4, 8).astype(int)
    kev_raw["ransomware"] = kev_raw["knownRansomwareCampaignUse"].eq("Known")
    kev_raw.groupby(kev_raw["dateAdded"].dt.to_period("M").astype(str)).size().rename("kev_additions").to_csv(TAB / "00_kev_by_month.csv")
    kev_raw["years_to_kev"] = kev_raw["dateAdded"].dt.year - kev_raw["cve_year"]
    ky = kev_raw.groupby("cve_year").agg(kev_entries=("cveID", "size"), ransomware_pct=("ransomware", lambda s: 100 * s.mean()),
                                         median_years_cve_to_kev=("years_to_kev", "median"))
    ky.round(1).to_csv(TAB / "00_kev_by_cve_year.csv")

    q = df["published"].dt.to_period("Q").astype(str)
    adp = df.groupby(q).apply(lambda g: pd.Series({
        "CVEs": len(g), "exploited": int(g["label"].sum()),
        "% of CVEs with ADP CVSS": 100 * g["has_adp_cvss"].mean(),
        "% of exploited with ADP CVSS": 100 * g.loc[g["label"] == 1, "has_adp_cvss"].mean() if g["label"].sum() else np.nan}))
    adp.round(1).to_csv(TAB / "00_leak_adp_by_quarter.csv")

    # leaky KEV-history experiment (dev only: train -> val, frozen 05a settings)
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet")
    kev = load_kev_raw(fm.KEV_FILE)
    leaky = kev.assign(date=pd.Timestamp("1900-01-01"), cve_id="LEAK-" + kev["cve_id"])   # every entry counts, incl. own
    y = dev["label"].to_numpy()
    tr, va = dev["split"].eq("train").to_numpy(), dev["split"].eq("val").to_numpy()
    out = []
    for name, kv in [("day-0 KEV history (used): entries added BEFORE publication, own entry excluded", kev),
                     ("LEAKY (own entry): real dates, but the CVE's own KEV entry may count (if added before NVD publication)",
                      kev.assign(cve_id="LEAK-" + kev["cve_id"])),
                     ("NAIVE (today's catalogue): every entry counts regardless of date, own entry included", leaky)]:
        X = build_X(dev, kv)
        s = make_hgb(**fm.HGB_BEST).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        vd = dev[va].reset_index(drop=True)
        out.append({"variant": name, "val recall@110 %": 100 * recall_at_k(vd, pd.Series(s), 110),
                    "val recall@305 %": 100 * recall_at_k(vd, pd.Series(s), 305)})
        print(out[-1], flush=True)
    pd.DataFrame(out).round(1).to_csv(TAB / "00_leak_kev_history.csv", index=False)
    print("written 00_*.csv")


if __name__ == "__main__":
    main()
