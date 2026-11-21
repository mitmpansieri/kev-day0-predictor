"""CISO tool: "I just saw a new NVD entry. Should we emergency-patch it?"

Scores CVEs with the DEPLOYMENT model (gradient boosting refit on all labelled data 2022-01 -> 2026-08-20,
outputs/models/deploy_hgb.joblib) using DAY-0 information only:
  * the CNA's own CVSS, CWE, vendor/product; the description (exploitation phrases only if provably day-0);
  * the vendor's / product's CISA KEV history, counting only KEV entries added BEFORE the CVE's publication,
    from the newest local KEV snapshot (data/raw/kev_*.csv). The tool never downloads KEV.

Output per CVE:
  percentile   : share of July 2026 CVEs (reference month, 9,771 CVEs) that score lower. PRIMARY output.
  probability  : INDICATIVE chance of a CISA KEV listing within 90 days (Platt calibration on time-ordered
                 out-of-fold scores). Approximate; rankings are what was validated.
  decision     : PATCH NOW if the CVE is in the top 4.22% of a month (the share of monthly CVEs that the
                 110-patch capacity represented in 2022-24; volume-scaled budget), else MONITOR.
  reasons      : top 3 SHAP contributions (permutation SHAP vs 100 training CVEs), in plain words.

NVD access: public API without a key (5 requests / 30 s); the tool never reads .env. Responses are cached in
data/nvd_tool_cache/ (a snapshot of NVD at fetch time; delete a file to refetch). CNA identifiers (e-mails or UUIDs)
are shown as organisation names from data/nvd_sources.csv (NVD source API, refresh with `sources`); display only, the
model itself uses the raw identifiers it was trained on.

CLI (run from the repo root):
  python ciso_tool.py cve CVE-2026-12345                      -> one-CVE card
  python ciso_tool.py manual --vendor Fortinet --product FortiOS --cna psirt@fortinet.com \\
      --vector "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H" --cwe CWE-288 --description "An authentication bypass ..."
  python ciso_tool.py feed 2026-09-08 --top 20                -> ranked table for one publication day
  python ciso_tool.py sources                                 -> refresh data/nvd_sources.csv (CNA names)
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from build_cve_dataset import NVD_URL, parse_cve                     # noqa: E402  (same parser as training)
from check_cna_cvss_stability import cvss3_score                      # noqa: E402  (CVSS 3.x base score)
from description_leak import _phrase_names, drop_sentences_with       # noqa: E402
from features import load_kev_raw                                     # noqa: E402
from tabular_models import build_X                                    # noqa: E402

MODELS = ROOT / "outputs" / "models"
CACHE = ROOT / "data" / "nvd_tool_cache"
SOURCES_CSV = ROOT / "data" / "nvd_sources.csv"
NVD_SOURCE_URL = "https://services.nvd.nist.gov/rest/json/source/2.0"
BUDGET_SHARE = 0.0422                 # 110 / (93,842 dev CVEs / 36 months)
DAY0_GRACE_DAYS = 7                   # phrases in descriptions fetched later than this are not provably day-0
USER_AGENT = "kev-day0-predictor/1.0 (research tool; python-requests)"
NVD_PAUSE = 6.5                       # public rate limit without a key: 5 requests per rolling 30 s

LETTERS = {
    "AV": ("attack vector", {"N": "network", "A": "adjacent network", "L": "local", "P": "physical"}),
    "AC": ("attack complexity", {"L": "low", "H": "high"}),
    "PR": ("privileges required", {"N": "none", "L": "low", "H": "high"}),
    "UI": ("user interaction", {"N": "none", "R": "required", "P": "passive", "A": "active"}),
    "S": ("scope", {"U": "unchanged", "C": "changed"}),
    "C": ("confidentiality impact", {"H": "high", "L": "low", "N": "none"}),
    "I": ("integrity impact", {"H": "high", "L": "low", "N": "none"}),
    "A": ("availability impact", {"H": "high", "L": "low", "N": "none"}),
    "AT": ("attack requirements (v4)", {"N": "none", "P": "present"}),
}


# ------------------------------------------------------------------ CNA identifier -> organisation name
_NAMES: dict[str, str] = {}


def update_sources() -> pd.DataFrame:
    """Download every NVD source (CNAs, ADPs) once, keyless, and save identifier -> name to data/nvd_sources.csv."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    rows, start, total = [], 0, None
    while total is None or start < total:
        r = session.get(NVD_SOURCE_URL, params={"resultsPerPage": 1000, "startIndex": start}, timeout=60)
        r.raise_for_status()
        page = r.json()
        total = page["totalResults"]
        for s in page.get("sources", []):
            for ident in s.get("sourceIdentifiers", []):
                rows.append({"identifier": ident.lower(), "name": s.get("name", "").strip()})
        start += 1000
        time.sleep(NVD_PAUSE)
    df = pd.DataFrame(rows).drop_duplicates("identifier").sort_values("identifier")
    df.to_csv(SOURCES_CSV, index=False)
    _NAMES.clear()
    return df


def load_source_names() -> dict[str, str]:
    if not _NAMES and SOURCES_CSV.exists():
        s = pd.read_csv(SOURCES_CSV, keep_default_na=False)
        _NAMES.update(dict(zip(s["identifier"], s["name"])))
    return _NAMES


def org_name(identifier) -> str:
    """'3c1d8aa1-...' -> 'Ivanti'; unknown identifiers are shown as they are."""
    if identifier is None or (isinstance(identifier, float) and np.isnan(identifier)):
        return "(unknown)"
    return load_source_names().get(str(identifier).lower(), str(identifier))


# ------------------------------------------------------------------ model bundle
class Tool:
    def __init__(self):
        meta = json.loads((MODELS / "deploy_models.json").read_text())
        import hashlib
        assert hashlib.sha256((MODELS / "deploy_hgb.joblib").read_bytes()).hexdigest() == meta["sha256"]["hgb"], \
            "deploy model file does not match its recorded sha256"
        self.model = joblib.load(MODELS / "deploy_hgb.joblib")
        self.cal = joblib.load(MODELS / "calibrator_deploy.joblib")
        self.ref = np.sort(pd.read_csv(MODELS / "demo_reference_2026-07.csv")["score"].to_numpy())
        self.threshold = float(np.quantile(self.ref, 1 - BUDGET_SHARE))
        self.kev_path = sorted((ROOT / "data" / "raw").glob("kev_*.csv"))[-1]
        self.kev_date = pd.Timestamp(datetime.strptime(self.kev_path.stem.split("_")[1], "%Y%m%d"))
        self.kev = load_kev_raw(self.kev_path)
        from features import CATEGORICAL
        head = pd.read_csv(MODELS / "shap_background.csv", nrows=0).columns
        self.bg = pd.read_csv(MODELS / "shap_background.csv", dtype={c: str for c in CATEGORICAL}, keep_default_na=False,
                              na_values={c: [""] for c in head if c not in CATEGORICAL})
        self.meta = meta
        self.base_rate = meta["positives"] / meta["n"]       # 613 / 196,403 = 0.31% of CVEs listed in KEV within 90 d
        load_source_names()

    def describe(self) -> str:
        return (f"deploy model: {self.meta['n']:,} labelled CVEs ({self.meta['published_from']} -> {self.meta['published_to']}), "
                f"sha256 {self.meta['sha256']['hgb'][:12]} | KEV snapshot {self.kev_path.name} | reference month 2026-07 "
                f"({len(self.ref):,} CVEs) | PATCH NOW = top {100 * BUDGET_SHARE:.2f}% (score >= {self.threshold:.4f})")

    # -------------------------------------------------------------- scoring
    def score(self, df: pd.DataFrame, explain_top: int | None = None) -> pd.DataFrame:
        """df = parsed CVEs (training columns). Returns one row per CVE, ranked by score."""
        df = df.reset_index(drop=True)
        X = build_X(df, self.kev)
        s = self.model.predict_proba(X)[:, 1]
        x = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7)))
        out = pd.DataFrame({
            "cve_id": df["cve_id"], "published": df["published"].dt.strftime("%Y-%m-%d"),
            "vendor": df["vendor"].fillna("(not stated)"), "product": df["product"].fillna(""),
            "cna": df["cna"].map(org_name), "cna_cvss": df["cvss_cna_score"],
            "cvss_vector": df["cvss_cna_vector"].fillna(""), "score": s,
            "percentile": 100 * np.searchsorted(self.ref, s, side="left") / len(self.ref),
            "probability_%": 100 * self.cal.predict_proba(x.reshape(-1, 1))[:, 1],
            "x_base_rate": self.cal.predict_proba(x.reshape(-1, 1))[:, 1] / self.base_rate,
            "decision": np.where(s >= self.threshold, "PATCH NOW", "MONITOR"),
            "notes": df.get("day0_note", pd.Series("", index=df.index)).fillna(""),
        })
        late = df["published"].dt.normalize() > self.kev_date
        out.loc[late, "notes"] = (out.loc[late, "notes"] + " KEV snapshot older than publication: vendor history may miss "
                                  "the newest KEV entries.").str.strip()
        order = np.argsort(-s, kind="stable")
        n_exp = len(df) if explain_top is None else min(explain_top, len(df))
        idx = order[:n_exp]
        reasons = self.explain(X.iloc[idx]) if n_exp else []
        for j in range(3):
            out[f"reason_{j + 1}"] = ""
            out.loc[idx, f"reason_{j + 1}"] = [r[j] if len(r) > j else "" for r in reasons]
        out = out.iloc[order].reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    # -------------------------------------------------------------- explanations
    def explain(self, X: pd.DataFrame) -> list[list[str]]:
        import shap
        cols = list(X.columns)
        cat = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
        bg = self.bg[cols].copy()
        for c in cat:
            bg[c] = bg[c].astype(str)
        both = pd.concat([X.astype({c: str for c in cat}), bg], ignore_index=True)
        cats = {c: pd.Index(both[c].astype(str).unique()) for c in cat}
        dtypes = X.dtypes.to_dict()

        def enc(df):
            A = df.copy()
            for c in cat:
                A[c] = cats[c].get_indexer(df[c].astype(str))
            return A.astype(float).to_numpy()

        def dec(A):
            df = pd.DataFrame(A, columns=cols)
            for c in cat:
                df[c] = cats[c].to_numpy()[df[c].astype(int)]
            return df.astype(dtypes)

        f = lambda A: self.model["clf"]._raw_predict(self.model["pre"].transform(dec(A))).ravel()
        ex = shap.PermutationExplainer(f, shap.maskers.Independent(enc(bg), max_samples=100))
        r = ex(enc(X), max_evals=500, silent=True)
        out = []
        for i in range(len(X)):
            v = pd.Series(r.values[i], index=cols)
            top = v.reindex(v.abs().sort_values(ascending=False).index[:3])
            out.append([reason_text(c, X.iloc[i][c], val) for c, val in top.items()])
        return out


def _strength(v: float) -> str:
    a = abs(v)
    return ("strongly " if a >= 1 else "moderately " if a >= 0.4 else "slightly ") + ("raises" if v > 0 else "lowers")


def reason_text(feat: str, value, shap_value: float) -> str:
    """One SHAP contribution in plain words: '<fact> -> strongly raises risk'."""
    if feat == "vendor_key":
        fact = f"vendor not stated (publisher {org_name(str(value)[4:])})" if str(value).startswith("cna:") else f"vendor '{value}'"
    elif feat == "cna":
        fact = f"published by CNA {org_name(value)}"
    elif feat == "cwe":
        fact = f"weakness {value}" if value != "NVD-CWE-noinfo" else "no weakness (CWE) given by the CNA"
    elif feat == "cvss_cna_score":
        fact = "no CVSS score from the CNA" if pd.isna(value) else f"CNA CVSS score {float(value):.1f}"
    elif feat == "has_cna_cvss":
        fact = "CNA gave a CVSS score" if value else "CNA gave no CVSS score"
    elif feat == "n_cwes":
        fact = f"{int(value)} CWE(s) given by the CNA"
    elif feat == "phrase_day0":
        fact = "description says it is exploited (day 0)" if value else "no exploitation wording in the description"
    elif feat == "desc_len":
        fact = f"description length {int(value):,} characters"
    elif feat == "vendor_kev_log_all":
        fact = f"vendor has {int(round(np.expm1(value))):,} earlier CISA KEV entries"
    elif feat == "vendor_kev_log_365":
        fact = f"vendor had {int(round(np.expm1(value))):,} KEV entries in the year before publication"
    elif feat == "vendor_kev_days_since":
        fact = "vendor never in KEV before" if value >= 3650 else f"vendor's last KEV entry {int(value)} days before publication"
    elif feat == "vendor_kev_any":
        fact = "vendor already in KEV" if value else "vendor never in KEV"
    elif feat == "product_kev_log_all":
        fact = f"product has {int(round(np.expm1(value))):,} earlier KEV entries"
    elif feat.startswith("cvss_") and feat[5:] in LETTERS:
        name, m = LETTERS[feat[5:]]
        fact = f"no CNA CVSS vector ({name})" if value in ("none", "", None) else f"{name}: {m.get(value, value)}"
    else:
        fact = f"{feat} = {value}"
    return f"{fact} -> {_strength(shap_value)} risk"


# ------------------------------------------------------------------ NVD (public API, no key)
def _nvd_get(params: dict, cache_name: str, refresh: bool = False) -> dict:
    path = CACHE / f"{cache_name}.json.gz"
    if path.exists() and not refresh:
        return json.loads(gzip.decompress(path.read_bytes()))
    records, total, start = [], None, 0
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT          # no apiKey header: the tool never uses a key
    while total is None or start < total:
        for attempt in range(6):
            r = session.get(NVD_URL, params={**params, "startIndex": start, "resultsPerPage": 2000}, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code not in (403, 429, 500, 502, 503, 504):
                raise RuntimeError(f"NVD returned HTTP {r.status_code}")
            time.sleep(min(60, 6.5 * 2 ** attempt))
        else:
            raise RuntimeError("NVD request failed after retries")
        page = r.json()
        total = page["totalResults"]
        records += [v["cve"] for v in page.get("vulnerabilities", [])]
        start += 2000
        time.sleep(NVD_PAUSE)
    snap = {"fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "params": params, "records": records}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(snap).encode()))
    return snap


def fetch_cve(cve_id: str, refresh: bool = False) -> pd.DataFrame:
    snap = _nvd_get({"cveId": cve_id.upper()}, f"cve_{cve_id.upper()}", refresh)
    if not snap["records"]:
        raise ValueError(f"{cve_id} not found in NVD")
    return records_to_df(snap["records"], snap["fetched_at"])


def fetch_day(day: str, refresh: bool = False) -> pd.DataFrame:
    d = date.fromisoformat(day)
    params = {"pubStartDate": f"{d:%Y-%m-%d}T00:00:00.000", "pubEndDate": f"{d:%Y-%m-%d}T23:59:59.999", "noRejected": ""}
    snap = _nvd_get(params, f"day_{d:%Y%m%d}", refresh)
    return records_to_df(snap["records"], snap["fetched_at"])


def records_to_df(records: list[dict], fetched_at: str) -> pd.DataFrame:
    """Same parser as training. Strict day-0 text: NVD returns TODAY's description; if it was fetched more than
    7 days after publication, exploitation phrases cannot be proven day-0 and are removed (Decision 10c rule)."""
    df = pd.DataFrame([parse_cve(r) for r in records])
    df["published"] = pd.to_datetime(df["published"])
    for c in ["cvss_score", "cvss_cna_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    age = (pd.Timestamp(fetched_at).tz_localize(None) - df["published"]).dt.days
    names = df["description"].map(_phrase_names)
    strip = (age > DAY0_GRACE_DAYS) & names.map(bool)
    df["description_day0"] = [drop_sentences_with(t, n) if s else t for t, n, s in zip(df["description"], names, strip)]
    df["day0_note"] = np.where(strip, f"exploitation wording removed (fetched {DAY0_GRACE_DAYS}+ days after publication, "
                                      "not provably day-0).", "")
    return df.sort_values("published").reset_index(drop=True)


def manual_cve(vendor: str | None, cna: str, description: str, vector: str | None = None, score: float | None = None,
               cwe: str | None = None, product: str | None = None, published: str | None = None,
               cve_id: str = "MANUAL-ENTRY") -> pd.DataFrame:
    """A CVE typed in by hand. vector = the CNA's CVSS vector (v3.x scored automatically; v4 needs `score`)."""
    if vector and score is None:
        if "CVSS:4" in vector:
            raise ValueError("CVSS v4 vector: please also pass the CNA's base score")
        score = cvss3_score(vector.split("/", 1)[1] if vector.startswith("CVSS:") else vector)
    pub = pd.Timestamp(published) if published else pd.Timestamp(datetime.now(timezone.utc).date())
    return pd.DataFrame([{
        "cve_id": cve_id, "published": pub, "cna": cna, "description": description, "description_day0": description,
        "cwe": cwe or "NVD-CWE-noinfo", "n_cwes": 1 if cwe else 0, "vendor": vendor, "product": product,
        "cvss_cna_score": score, "cvss_cna_vector": vector, "has_cna_cvss": vector is not None, "day0_note": ""}])


# ------------------------------------------------------------------ one-CVE card
def card(row: pd.Series, tool: Tool) -> str:
    """Readable summary of one scored CVE (row of Tool.score)."""
    import re
    vec = re.sub(r"/[A-Z]+:X(?=/|$)", "", row["cvss_vector"])          # v4: drop components that are 'X' (not set)
    cvss = "no CNA CVSS" if pd.isna(row["cna_cvss"]) else f"{row['cna_cvss']:.1f}" + (f"  ({vec})" if vec else "")
    vp = row["vendor"] + (f" / {row['product']}" if row["product"] else "")
    line = "=" * 78
    out = [line, f"  {row['cve_id']}   published {row['published']}   CNA: {row['cna']}", f"  {vp}", f"  CVSS (CNA): {cvss}", line,
           f"  DECISION:     {row['decision']}" + ("   (top 4.2% of a month's CVEs)" if row["decision"] == "PATCH NOW" else "   (below the top-4.2% patch budget)"),
           f"  Percentile:   {row['percentile']:.1f}  (" + ("in the top 0.1% of" if row["percentile"] >= 99.9 else f"scores higher than {row['percentile']:.1f}% of")
           + f" the {len(tool.ref):,} CVEs of July 2026)",
           f"  Probability:  {row['probability_%']:.1f}% indicative chance of a CISA KEV listing within 90 days",
           f"                = {row['x_base_rate']:.1f}x the base rate ({100 * tool.base_rate:.2f}% of all CVEs)",
           "  Why:"]
    out += [f"    {i}. {row[f'reason_{i}']}" for i in (1, 2, 3) if row[f"reason_{i}"]]
    if row["notes"]:
        out.append(f"  Note: {row['notes']}")
    out.append(line)
    return "\n".join(out)


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("cve"); a.add_argument("cve_id"); a.add_argument("--refresh", action="store_true")
    b = sub.add_parser("feed"); b.add_argument("day", help="YYYY-MM-DD"); b.add_argument("--top", type=int, default=20)
    b.add_argument("--refresh", action="store_true")
    c = sub.add_parser("manual")
    for k in ["vendor", "product", "cna", "vector", "cwe", "description", "published"]:
        c.add_argument(f"--{k}", required=k in ("cna", "description"))
    c.add_argument("--score", type=float)
    sub.add_parser("sources", help="refresh data/nvd_sources.csv (CNA identifier -> organisation name)")
    args = ap.parse_args(argv)
    if args.cmd == "sources":
        df = update_sources()
        print(f"saved {len(df):,} source identifiers -> {SOURCES_CSV.relative_to(ROOT)}")
        return
    tool = Tool()
    print(tool.describe())
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80); pd.set_option("display.max_columns", 20)
    if args.cmd in ("cve", "manual"):
        df = fetch_cve(args.cve_id, args.refresh) if args.cmd == "cve" else manual_cve(
            args.vendor, args.cna, args.description, args.vector, args.score, args.cwe, args.product, args.published)
        for _, row in tool.score(df).iterrows():
            print(card(row, tool))
        return
    res = tool.score(fetch_day(args.day, args.refresh), explain_top=args.top)
    print(f"{args.day}: {len(res)} CVEs published; PATCH NOW: {(res['decision'] == 'PATCH NOW').sum()}")
    cols = ["rank", "cve_id", "vendor", "product", "cna", "cna_cvss", "percentile", "probability_%", "decision",
            "reason_1", "reason_2", "reason_3", "notes"]
    print(res.head(args.top)[cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
