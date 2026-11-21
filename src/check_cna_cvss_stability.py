"""PLANNED CHECK (before Phase 6): did CNAs change their OWN CVSS after publication? DEV ONLY.

Uses the cached NVD change histories (data/nvd_history_cache/) of DEV CVEs only (history sample: all 319
train+val positives + 600 random train+val negatives, plus dev phrase CVEs). For each CVE, CVSS events
('CVSS V3.1' / 'CVSS V3.0' / 'CVSS V4.0') whose change sourceIdentifier == the CVE's CNA:
  first_cna_vector  = the vector in the CNA's earliest CVSS event (of the version used in cvss_cna_vector)
  cna_changed_later = the CNA changed that vector afterwards
  cna_added_late    = the CNA's first CVSS event came > 7 days after publication (score not day-0)
  score_delta       = current cvss_cna_score - score of first_cna_vector (v3.x only; v4 = vector compare only)
Compared by label. If CNA scores move toward 'exploited' (up, more often for positives), cvss_cna_* is not
fully day-0 either. Writes outputs/tables/05d_cna_cvss_stability*.csv.
Run: .venv/bin/python src/check_cna_cvss_stability.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT / "data" / "nvd_history_cache"
TAB = ROOT / "outputs" / "tables"
FAMILY = {"3.1": {"CVSS V3.1", "CVSS V3.0", "CVSS V3"}, "3.0": {"CVSS V3.1", "CVSS V3.0", "CVSS V3"},
          "4.0": {"CVSS V4.0"}}
ALL_TYPES = set().union(*FAMILY.values())

W = {"AV": {"N": .85, "A": .62, "L": .55, "P": .2}, "AC": {"L": .77, "H": .44}, "UI": {"N": .85, "R": .62},
     "CIA": {"H": .56, "L": .22, "N": 0.0}}


def _roundup(x: float) -> float:            # CVSS 3.1 spec Roundup
    i = round(x * 100000)
    return i / 100000.0 if i % 10000 == 0 else (math.floor(i / 10000) + 1) / 10.0


def cvss3_score(vector: str) -> float:
    m = dict(p.split(":") for p in vector.split("/") if ":" in p and not p.startswith("CVSS"))
    try:
        changed = m["S"] == "C"
        pr = {"N": .85, "L": .68 if changed else .62, "H": .5 if changed else .27}[m["PR"]]
        iss = 1 - (1 - W["CIA"][m["C"]]) * (1 - W["CIA"][m["I"]]) * (1 - W["CIA"][m["A"]])
        imp = 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15 if changed else 6.42 * iss
        exp = 8.22 * W["AV"][m["AV"]] * W["AC"][m["AC"]] * pr * W["UI"][m["UI"]]
    except KeyError:
        return np.nan
    if imp <= 0:
        return 0.0
    return _roundup(min(1.08 * (imp + exp), 10)) if changed else _roundup(min(imp + exp, 10))


def _strip(v) -> str | None:
    """'Cisco Systems, Inc. AV:N/...' or 'CVSS:3.1/AV:N/...' -> 'AV:N/...'; v4 'X' (not set) components dropped."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = str(v)
    i = v.find("AV:")
    if i < 0:
        return None
    return "/".join(p for p in v[i:].split("/") if ":" in p and not p.endswith(":X"))


def main():
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet",
                          columns=["cve_id", "published", "cna", "label", "split", "cvss_cna_score",
                                   "cvss_cna_vector", "cvss_cna_version", "has_cna_cvss"])
    # self-test of the score formula on the dataset's own CNA v3.x scores
    v3 = dev[dev["cvss_cna_version"].astype(str).isin(["3.0", "3.1"])].sample(5000, random_state=0)
    calc = v3["cvss_cna_vector"].map(cvss3_score)
    agree = (calc - v3["cvss_cna_score"]).abs().lt(0.05).mean()
    print(f"CVSS 3.x formula self-test: {agree:.2%} of 5,000 dev CNA scores reproduced")

    have = {p.stem for p in HIST.glob("CVE-*.json")}
    d = dev[dev["cve_id"].isin(have)].copy()
    rows = []
    for r in d.itertuples():
        hist = json.loads((HIST / f"{r.cve_id}.json").read_text())
        ev = []
        for c in hist.get("cveChanges", []):
            ch = c["change"]
            if ch.get("sourceIdentifier") != r.cna:
                continue
            for x in ch.get("details", []):
                if x.get("type") in ALL_TYPES:
                    ev.append((pd.Timestamp(ch["created"]), x["type"], x.get("action"), x.get("newValue"), x.get("oldValue")))
        ev.sort(key=lambda e: e[0])          # stable: keeps Removed-before-Added order within one event
        first_any = (min(pd.Timestamp(c["change"]["created"]) for c in hist["cveChanges"])
                     if hist.get("cveChanges") else pd.NaT)
        out = {"cve_id": r.cve_id, "label": r.label, "has_cna_cvss": bool(r.has_cna_cvss),
               "history_starts_late": bool(pd.notna(first_any) and (first_any - r.published).days > 7),
               "n_cna_cvss_events": len(ev)}
        ver = str(r.cvss_cna_version) if pd.notna(r.cvss_cna_version) else None
        ev_v = [e for e in ev if ver in FAMILY and e[1] in FAMILY[ver]]
        if r.has_cna_cvss and ev_v:
            t0, _, act, new, old = ev_v[0]
            if act in ("Removed", "Changed") and old:     # vector existed before the first recorded change
                first_vec, status = _strip(old), "present_before_first_event"
            else:
                first_vec = _strip(new)
                status = "added_late" if (t0 - r.published).days > 7 else "added_day0"
            timeline = [first_vec] + [_strip(e[3]) for e in ev_v if e[2] in ("Added", "Changed") and e[3]]
            cur_vec = _strip(r.cvss_cna_vector)
            out.update({
                "first_status": status, "cna_added_late": status == "added_late",
                "first_vs_current_differ": first_vec != cur_vec,
                "n_distinct_vectors": len(set(timeline) | {cur_vec}),
                "first_vector": first_vec, "current_vector": cur_vec,
                "first_score": cvss3_score(first_vec) if ver in ("3.0", "3.1") and first_vec else np.nan,
                "current_score": r.cvss_cna_score})
            out["score_delta"] = out["current_score"] - out["first_score"]
        elif r.has_cna_cvss:
            out["first_status"] = "no_cna_event_in_history"      # unchanged since publication (no CNA CVSS event)
        else:
            out["cna_cvss_ever_in_history"] = len(ev) > 0
        rows.append(out)
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "05d_cna_cvss_stability_rows.csv", index=False)

    def summ(g):
        s = g[g["first_vs_current_differ"].notna()] if "first_vs_current_differ" in g else g.iloc[:0]
        return pd.Series({
            "CVEs in sample": len(g),
            "with CNA CVSS today": int(g["has_cna_cvss"].sum()),
            "...no CNA CVSS event at all (unchanged)": int((g.get("first_status") == "no_cna_event_in_history").sum()),
            "...with CNA CVSS events": len(s),
            "   first vector present before 1st event": int((s["first_status"] == "present_before_first_event").sum()),
            "   added on day 0-7": int((s["first_status"] == "added_day0").sum()),
            "CNA CVSS first added >7 d after pub": int(s["cna_added_late"].sum()),
            "first vector != today's": int(s["first_vs_current_differ"].sum()),
            "share changed, of all with CNA CVSS %": round(100 * s["first_vs_current_differ"].sum() / max(g["has_cna_cvss"].sum(), 1), 2),
            "ever >1 distinct vector (incl. reverted)": int((s["n_distinct_vectors"] > 1).sum()),
            "score up (v3)": int((s["score_delta"] > 0).sum()),
            "score down (v3)": int((s["score_delta"] < 0).sum()),
            "mean delta (v3, changed only)": round(s.loc[s["score_delta"] != 0, "score_delta"].mean(), 2),
            "crossed 9.0 upward (v3)": int(((s["first_score"] < 9) & (s["current_score"] >= 9)).sum()),
            "history starts >7 d after pub": int(g["history_starts_late"].sum()),
            "no CNA CVSS today but CNA CVSS event in history": int(g.get("cna_cvss_ever_in_history", pd.Series(dtype=bool)).fillna(False).sum()),
        })
    summary = t.groupby("label").apply(summ).T
    summary.columns = [f"label {c}" for c in summary.columns]
    summary.to_csv(TAB / "05d_cna_cvss_stability.csv")
    pd.set_option("display.width", 200)
    print(summary.to_string())
    ch = t[t.get("first_vs_current_differ", False) == True]   # noqa: E712
    if len(ch):
        print("\nchanged CNA vectors (first 30):")
        print(ch[["cve_id", "label", "first_status", "first_vector", "current_vector", "first_score", "current_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
