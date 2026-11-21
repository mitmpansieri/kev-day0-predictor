"""Decision 10 report: description edits after publication, day-0 certainty of exploitation phrases,
and the 'phrase + CVSS>=9 (CNA)' baseline. Dev data only (train+val); test_locked.parquet is never read."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from description_leak import PHRASES, phrase_flags  # noqa: E402
from nvd_history import CACHE, analyze  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
pd.set_option("display.width", 200)


def first_event_date(cve_id: str):
    h = json.loads((CACHE / f"{cve_id}.json").read_text())
    dates = [c["change"].get("created") for c in h.get("cveChanges", []) if c["change"].get("created")]
    return pd.to_datetime(min(dates)) if dates else pd.NaT


def main() -> None:
    a = analyze()
    n_pos, n_neg = int(a["label"].sum()), int((a["label"] == 0).sum())
    print(f"history sample analysed: {len(a)} CVEs ({n_pos} positives = all of train+val, {n_neg} random negatives)")
    print("original description reconstructed from:", a["original_from"].value_counts().to_dict())

    # --- 2b: description changes by label
    g = a.groupby("label")
    t = pd.DataFrame({"CVEs": g.size(), "desc_changed_%": (100 * g["desc_changed"].mean()).round(1),
                      "phrase_at_publication_%": (100 * g["phrase_at_pub"].mean()).round(1),
                      "phrase_now_%": (100 * g["phrase_now"].mean()).round(1),
                      "phrase_added_later_n": g["phrase_added_later"].sum(),
                      "edit_added_a_phrase_n": g["phrase_added_by_edit"].sum()})
    t.index = t.index.map({0: "negatives (sample)", 1: "positives (all)"})
    print("\n[2b] Description changes after publication, by label:"); print(t.to_string())
    t.to_csv(TAB / "03e_history_description_changes.csv")

    # edit size: characters changed between the earliest recorded description and today's
    import difflib
    cur = pd.read_parquet(ROOT / "data" / "dev.parquet", columns=["cve_id", "description"]).set_index("cve_id")["description"]
    def chars_changed(o, c):
        sm = difflib.SequenceMatcher(None, o or "", c or "", autojunk=False)
        return len(o or "") + len(c or "") - 2 * sum(b.size for b in sm.get_matching_blocks())
    a["chars_changed"] = [chars_changed(o, cur.get(i, "")) for o, i in zip(a["original_description"], a["cve_id"])]
    a["orig_len"] = a["original_description"].str.len()
    e = a[a["desc_changed"]]
    ge = e.groupby("label")
    size = pd.DataFrame({"edited_CVEs": ge.size(),
                         "edited_%_of_label": (100 * ge.size() / g.size()).round(1),
                         "median_chars_changed": ge["chars_changed"].median(),
                         "mean_chars_changed": ge["chars_changed"].mean().round(0),
                         "median_%_of_text": (100 * ge["chars_changed"].median() / ge["orig_len"].median()).round(0),
                         "share_edits_>200_chars_%": (100 * ge["chars_changed"].apply(lambda x: (x > 200).mean())).round(1),
                         "median_n_edits": ge["n_desc_edits"].median()})
    size.index = size.index.map({0: "negatives (sample)", 1: "positives (all)"})
    print("\n[2b] Size of description edits (earliest recorded vs today), edited CVEs only:"); print(size.to_string())
    size.to_csv(TAB / "03e_history_edit_size.csv")
    print("Edited share by year (positives / negatives):")
    yr = a.assign(year=a["published"].dt.year).groupby(["year", "label"])["desc_changed"].mean().unstack().mul(100).round(1)
    print(yr.rename(columns={0: "negatives_%", 1: "positives_%"}).to_string())

    later = a[a["phrase_added_later"] | a["phrase_added_by_edit"]]
    if len(later):
        print("\nCVEs where an edit added a phrase (or phrase present now but not at publication):")
        print(later[["cve_id", "label", "vendor", "published", "phrase_at_pub", "phrase_now",
                     "phrase_added_date", "kev_date_added"]].to_string(index=False))

    # --- addition 1: day-0 certainty of phrases found in the reconstructed original
    p = a[a["phrase_at_pub"]].copy()
    p["first_event"] = p["cve_id"].map(first_event_date)
    p["gap_days"] = (p["first_event"] - p["published"]).dt.total_seconds() / 86400
    p["bucket"] = (p["gap_days"] > 7).map({True: "uncertain day-0 (>7 d)", False: "day-0 (<=7 d)"})
    p["year"] = p["published"].dt.year
    print(f"\n[1] CVEs whose ORIGINAL description has a phrase: {len(p)} "
          f"({int(p['label'].sum())} positives, {int((p['label'] == 0).sum())} negatives)")
    print("Gap publication -> first NVD history event (days):")
    print(p.groupby(["label", "year"])["gap_days"].describe()[["count", "min", "50%", "max"]].round(1).to_string())
    b = pd.crosstab([p["label"], p["year"]], p["bucket"], margins=True)
    print("\nBuckets (rows: label, year):"); print(b.to_string())
    print("\nPositives per bucket:", p.loc[p["label"] == 1, "bucket"].value_counts().to_dict())
    p.drop(columns=["original_description"]).to_csv(TAB / "03e_history_phrase_day0_gap.csv", index=False)

    # --- addition 2: baseline 'phrase OR CNA CVSS >= 9' on train
    dev = pd.read_parquet(ROOT / "data" / "dev.parquet",
                          columns=["cve_id", "split", "label", "published", "description", "cvss_cna_score"])
    tr = dev[dev["split"] == "train"].copy()
    months = (tr["published"].max() - tr["published"].min()).days / 30.44
    P = int(tr["label"].sum())
    tr["phrase_now"] = phrase_flags(tr["description"], PHRASES).any(axis=1)
    orig = a.set_index("cve_id")["phrase_at_pub"]
    tr["phrase_day0"] = tr["cve_id"].map(orig)            # known for all positives (all are in the sample)
    cvss9 = tr["cvss_cna_score"] >= 9
    rows = []
    for name, flag in [("CVSS >= 9 (CNA)", cvss9),
                       ("phrase only (today's text)", tr["phrase_now"]),
                       ("phrase OR CVSS >= 9 (CNA), today's text", tr["phrase_now"] | cvss9)]:
        tp = int((flag & (tr["label"] == 1)).sum())
        rows.append({"rule": name, "patches/month": round(flag.sum() / months), "caught": tp,
                     "recall_%": round(100 * tp / P, 1), "precision_%": round(100 * tp / max(flag.sum(), 1), 2)})
    pos = tr[tr["label"] == 1]
    tp0 = int((pos["phrase_day0"].fillna(False) | (pos["cvss_cna_score"] >= 9)).sum())
    rows.append({"rule": "phrase OR CVSS >= 9 (CNA), publication-day text (recall only)", "patches/month": None,
                 "caught": tp0, "recall_%": round(100 * tp0 / P, 1), "precision_%": None})
    base = pd.DataFrame(rows)
    print(f"\n[2] Baselines on train ({P} positives, {months:.1f} months):"); print(base.to_string(index=False))
    base.to_csv(TAB / "03e_train_baseline_phrase_or_cvss9.csv", index=False)


if __name__ == "__main__":
    main()
