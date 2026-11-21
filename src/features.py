"""Shared feature helpers (used by the exploration notebook and the models, so both apply the same rules)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Rules derived from TRAIN vendor spellings only (Phase 3): case, punctuation and company suffixes,
# e.g. "Google LLC" -> google, "Cisco Systems, Inc." -> cisco, "SAP_SE" -> sap, "Jenkins Project" -> jenkins project.
_SUFFIXES = r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|llc|gmbh|ag|sa|se|plc|bv|srl|" \
            r"technologies|technology|software|systems|the)\b"
_MISSING = {"", "n a", "na", "none", "unknown", "not applicable"}


def clean_vendor(s: pd.Series) -> pd.Series:
    """Normalise vendor names to a matching key. Missing / 'n/a' -> NaN."""
    k = (s.astype("string").str.lower()
         .str.replace(r"[^a-z0-9]+", " ", regex=True)
         .str.replace(_SUFFIXES, " ", regex=True)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    return k.mask(k.isna() | k.isin(_MISSING))


def vendor_or_cna(df: pd.DataFrame) -> pd.Series:
    """Clean vendor; when missing, fall back to the CNA (who published the CVE) - same idea as cv_group."""
    return clean_vendor(df["vendor"]).fillna("cna:" + df["cna"].str.lower())


# ----------------------------------------------------------------------------------------------
# Day-0 tabular features (Decisions 9 + 10): CNA CVSS only, CNA CWE, description_day0. Deterministic
# transformations (no fitted state); anything learned (encoders, rare grouping) goes inside the Pipeline.
# ----------------------------------------------------------------------------------------------
_V4_TO_V3 = {"VC": "C", "VI": "I", "VA": "A"}
CVSS_PARTS = ["AV", "AC", "PR", "UI", "S", "C", "I", "A", "AT"]


def cvss_components(vector: pd.Series) -> pd.DataFrame:
    """Split a CVSS vector string (v3.x / v4.0 / v2) into base-metric letters, e.g. AV=N, AC=L. Missing -> 'none'."""
    rows = []
    for v in vector.fillna(""):
        d = {}
        for part in v.split("/"):
            if ":" in part:
                key, val = part.split(":", 1)
                key = _V4_TO_V3.get(key, key)
                if key in CVSS_PARTS and key not in d:
                    d[key] = val
        if v.startswith("AV:") and "Au" in v:            # CVSS v2 has no version prefix; map Au -> PR
            d["PR"] = {"N": "N", "S": "L", "M": "H"}.get(dict(p.split(":") for p in v.split("/")).get("Au"), "none")
        rows.append(d)
    out = pd.DataFrame(rows, index=vector.index).reindex(columns=CVSS_PARTS)
    out["UI"] = out["UI"].replace({"P": "R", "A": "R"})   # v4 passive/active -> required
    return out.fillna("none").add_prefix("cvss_")


def day0_tabular(df: pd.DataFrame) -> pd.DataFrame:
    from description_leak import PHRASES, phrase_flags
    X = pd.DataFrame(index=df.index)
    X["vendor_key"] = vendor_or_cna(df)
    X["cna"] = df["cna"].str.lower()
    X["cwe"] = df["cwe"]
    X["cvss_cna_score"] = df["cvss_cna_score"]
    X["has_cna_cvss"] = df["has_cna_cvss"].astype(int)
    X["n_cwes"] = df["n_cwes"]
    X["phrase_day0"] = phrase_flags(df["description_day0"], PHRASES).any(axis=1).astype(int)
    X["desc_len"] = df["description_day0"].str.len()
    return pd.concat([X, cvss_components(df["cvss_cna_vector"])], axis=1)


CATEGORICAL = ["vendor_key", "cna", "cwe"] + [f"cvss_{p}" for p in CVSS_PARTS]
NUMERIC = ["cvss_cna_score", "has_cna_cvss", "n_cwes", "phrase_day0", "desc_len"]


# ----------------------------------------------------------------------------------------------
# KEV history features (leakage rule): only KEV entries with dateAdded STRICTLY BEFORE the CVE's publication
# date, and never the CVE's own KEV entry (it would leak the label for negative-gap CVEs).
# ----------------------------------------------------------------------------------------------
def clean_product(s: pd.Series) -> pd.Series:
    k = (s.astype("string").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    return k.mask(k.isna() | k.isin(_MISSING))


# KEV spells some vendors differently from the CNAs (spelling only, checked against train vendor names)
KEV_VENDOR_ALIASES = {"juniper": "juniper networks", "craft cms": "craftcms", "php": "php group", "vmware tanzu": "vmware"}


def load_kev_raw(path) -> pd.DataFrame:
    kev = pd.read_csv(path, usecols=["cveID", "vendorProject", "product", "dateAdded"])
    kev["date"] = pd.to_datetime(kev["dateAdded"]).dt.normalize()
    kev["vkey"] = clean_vendor(kev["vendorProject"]).replace(KEV_VENDOR_ALIASES)
    kev["pkey"] = kev["vkey"] + "|" + clean_product(kev["product"])
    return kev.rename(columns={"cveID": "cve_id"})


def _history(keys: pd.Series, pub: pd.Series, cve: pd.Series, kev: pd.DataFrame, kev_key: str) -> pd.DataFrame:
    """For each CVE: # KEV entries of the same key before pub (all-time, last 365 d) and days since the last one."""
    out = pd.DataFrame({"n_all": 0.0, "n_365": 0.0, "days_since": np.nan}, index=keys.index)
    groups = {k: g.sort_values("date") for k, g in kev.dropna(subset=[kev_key]).groupby(kev_key)}
    for k, idx in keys.dropna().groupby(keys.dropna()).groups.items():
        g = groups.get(k)
        if g is None:
            continue
        d = g["date"].to_numpy(dtype="datetime64[ns]")
        ids = g["cve_id"].to_numpy()
        for i in idx:
            p = np.datetime64(pub.at[i], "ns")
            mask = (d < p) & (ids != cve.at[i])               # strictly before, never the CVE's own entry
            if mask.any():
                dd = d[mask]
                out.at[i, "n_all"] = mask.sum()
                out.at[i, "n_365"] = (dd >= p - np.timedelta64(365, "D")).sum()
                out.at[i, "days_since"] = (p - dd.max()) / np.timedelta64(1, "D")
    return out


def kev_history_features(df: pd.DataFrame, kev: pd.DataFrame) -> pd.DataFrame:
    pub = df["published"].dt.normalize()
    vkey = clean_vendor(df["vendor"])
    pkey = vkey + "|" + clean_product(df["product"])
    v = _history(vkey, pub, df["cve_id"], kev, "vkey").add_prefix("vendor_kev_")
    p = _history(pkey, pub, df["cve_id"], kev, "pkey")[["n_all"]].add_prefix("product_kev_")
    return pd.concat([v, p], axis=1)


KEV_HIST = ["vendor_kev_n_all", "vendor_kev_n_365", "vendor_kev_days_since", "product_kev_n_all"]
