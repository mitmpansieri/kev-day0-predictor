"""NVD CVE Change History: did a CVE's description change after publication?

Sample (dev data only - train+val; test_locked.parquet is never touched): all positives + 600 random negatives.
One cached JSON per CVE in data/nvd_history_cache/ (re-runs are free, crashes resume).

Usage: .venv/bin/python -u src/nvd_history.py            # download the sample (~6.5 s/request without key)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cve_dataset import ROOT, USER_AGENT, _api_key, _get_with_retries  # noqa: E402

HIST_URL = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"
CACHE = ROOT / "data" / "nvd_history_cache"
SAMPLE_FILE = CACHE / "_sample.csv"
N_NEG, SEED = 600, 42


def make_sample() -> pd.DataFrame:
    if SAMPLE_FILE.exists():
        return pd.read_csv(SAMPLE_FILE)
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet", columns=["cve_id", "label", "split"])
    s = pd.concat([dev[dev["label"] == 1], dev[dev["label"] == 0].sample(N_NEG, random_state=SEED)])
    CACHE.mkdir(parents=True, exist_ok=True)
    s.to_csv(SAMPLE_FILE, index=False)
    return s


def fetch_history(session: requests.Session, cve_id: str, pause: float) -> dict:
    path = CACHE / f"{cve_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    params, changes, total = {"cveId": cve_id, "startIndex": 0}, [], None
    while total is None or params["startIndex"] < total:
        page = _get_with_retries(session, params, url=HIST_URL)
        total = page["totalResults"]
        changes += page.get("cveChanges", [])
        params["startIndex"] += max(page.get("resultsPerPage", 0), 1)
        time.sleep(pause)
    out = {"cveId": cve_id, "cveChanges": changes}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    return out


def phrase_cve_ids() -> list[str]:
    """ALL splits incl. test: CVEs whose current description has one of the 6 phrases.
    Reads ONLY cve_id + description (no label, no split) - the one documented exception in Decision 10."""
    from description_leak import PHRASES, phrase_flags
    d = pd.read_parquet(ROOT / "data" / "cve_dataset.parquet", columns=["cve_id", "description"])
    return d.loc[phrase_flags(d["description"], PHRASES).any(axis=1), "cve_id"].tolist()


def main() -> None:
    if "--phrase-cves" in sys.argv:
        sample = pd.DataFrame({"cve_id": phrase_cve_ids(), "label": 0})   # label unused here
    else:
        sample = make_sample()
    key = _api_key()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    if key:
        session.headers["apiKey"] = key
    pause = 0.7 if key else 6.5
    ids = sample["cve_id"].tolist()
    todo = [c for c in ids if not (CACHE / f"{c}.json").exists()]
    print(f"history list: {len(ids)} CVEs; {len(todo)} to download, "
          f"~{len(todo) * pause / 60:.0f} min", flush=True)
    for i, cve_id in enumerate(todo, 1):
        fetch_history(session, cve_id, pause)
        if i % 25 == 0 or i == len(todo):
            print(f"[progress] {i}/{len(todo)} downloaded", flush=True)
    print("DONE history download", flush=True)




# ------------------------------------# ----------------------------------------------------------------------------------------------
# Analysis: original (publication-day) description vs later edits
# ----------------------------------------------------------------------------------------------
def description_timeline(hist: dict, current: str) -> dict:
    """Reconstruct the original description and list later edits.
    Newer records have a 'New CVE Received' event with the description 'Added'. Older ones (2022-early 2023)
    start at 'Initial Analysis', so the original = oldValue of the earliest description edit, or the current
    text if it was never edited. (Edits before NVD's first analysis, if any, are not recorded.)"""
    events = sorted((c["change"] for c in hist.get("cveChanges", [])), key=lambda ch: ch.get("created", ""))
    added, edits = None, []
    for ch in events:
        for d in ch.get("details", []):
            if d.get("type") != "Description":
                continue
            if added is None and not edits and d.get("action") == "Added":
                added = d.get("newValue") or ""
            else:
                edits.append({"date": ch.get("created"), "event": ch.get("eventName"), "source": ch.get("sourceIdentifier"),
                              "action": d.get("action"), "old": d.get("oldValue") or "", "new": d.get("newValue") or ""})
    if added is not None:
        original, how = added, "new_cve_event"
    elif edits:
        original, how = edits[0]["old"], "first_edit_old_value"
    else:
        original, how = current, "never_edited"
    return {"original": original, "how": how, "edits": edits}


def analyze() -> pd.DataFrame:
    from description_leak import has_phrase
    sample = pd.read_csv(SAMPLE_FILE)
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet",
                          columns=["cve_id", "label", "split", "published", "description", "kev_date_added", "vendor", "cna"])
    df = sample[["cve_id"]].merge(dev, on="cve_id", how="left")
    rows = []
    for r in df.itertuples():
        path = CACHE / f"{r.cve_id}.json"
        if not path.exists():
            continue
        cur = r.description or ""
        tl = description_timeline(json.loads(path.read_text()), cur)
        added_edits = [e for e in tl["edits"] if has_phrase(e["new"]) and not has_phrase(e["old"])]
        first_add = min((e["date"] for e in added_edits), default=None)
        rows.append({"cve_id": r.cve_id, "label": r.label, "split": r.split, "published": r.published,
                     "vendor": r.vendor, "cna": r.cna, "original_from": tl["how"], "n_desc_edits": len(tl["edits"]),
                     "desc_changed": tl["original"].strip() != cur.strip(),
                     "phrase_at_pub": has_phrase(tl["original"]), "phrase_now": has_phrase(cur),
                     "phrase_added_by_edit": bool(added_edits),
                     "phrase_added_date": pd.to_datetime(first_add) if first_add else pd.NaT,
                     "kev_date_added": r.kev_date_added, "original_description": tl["original"]})
    out = pd.DataFrame(rows)
    out["phrase_added_later"] = out["phrase_now"] & ~out["phrase_at_pub"]
    return out


if __name__ == "__main__":
    main()
