"""Exploitation phrases in CVE descriptions (Decision 10). Used by the leakage checks and, later, the models."""
from __future__ import annotations

import re

import pandas as pd

# Phrases that STATE exploitation. Case-insensitive except KEV (an acronym).
PHRASES = {
    "exploited in the wild": r"exploited in the wild",
    "actively exploited": r"actively exploited",
    "may have been exploited": r"may have been exploited",
    "known exploited": r"known exploited",
    "KEV": r"(?-i:\bKEV\b)",
    "aware of a report": r"aware of a report",
}
EXTRA = {  # close variants, reported separately
    "limited, targeted exploitation": r"limited,? targeted exploitation",
    "under active exploitation": r"under active exploitation",
    "exploitation in the wild": r"exploitation in the wild",
    "has/have been exploited": r"(?<!could )(?<!would )(?<!might )(?<!may )ha(?:s|ve) been exploited",
}   # "could have been exploited" (Mozilla boilerplate) and "being exploited" are hypothetical -> excluded
ALL = {**PHRASES, **EXTRA}
ANY_RE = re.compile("|".join(f"(?:{p})" for p in ALL.values()), re.IGNORECASE)


def phrase_flags(text: pd.Series, phrases: dict = ALL) -> pd.DataFrame:
    t = text.fillna("")
    return pd.DataFrame({name: t.str.contains(p, flags=re.IGNORECASE, regex=True) for name, p in phrases.items()})


def has_phrase(text: str) -> bool:
    return bool(ANY_RE.search(text or ""))


# ----------------------------------------------------------------------------------------------
# description_day0 (Decision 10): today's text, minus exploitation phrases added after publication
# ----------------------------------------------------------------------------------------------
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _phrase_names(text: str) -> set[str]:
    return {n for n, p in PHRASES.items() if re.search(p, text or "", re.IGNORECASE)}


def drop_sentences_with(text: str, names: set[str]) -> str:
    """Remove the CLAUSE containing each named phrase (e.g. ", as exploited in the wild in 2024"); if the phrase
    survives, remove the whole sentence. No mask token: a token would itself leak the label."""
    out = text or ""
    for n in names:
        p = PHRASES[n]
        out = re.sub(r"[,;:]?[^,.;:!?]*" + p + r"[^,.;:!?]*", "", out, flags=re.IGNORECASE)
    if any(re.search(PHRASES[n], out, re.IGNORECASE) for n in names):
        pats = [PHRASES[n] for n in names]
        out = " ".join(s for s in _SENT_SPLIT.split(out) if not any(re.search(q, s, re.IGNORECASE) for q in pats))
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)          # tidy " ." left behind
    out = re.sub(r"(^|[.!?]\s*)[.,;:]+", r"\1", out)     # drop orphan punctuation at sentence start
    return re.sub(r"\s{2,}", " ", out).strip()


def build_description_day0(df: pd.DataFrame, history_dir) -> pd.DataFrame:
    """Label-independent: every CVE whose CURRENT description has one of the 6 phrases is checked against its
    earliest recorded description (NVD change history). Phrases absent from that version are removed.
    Returns description_day0 + desc_day0_status (diagnostic only, NEVER a model feature)."""
    import json
    from pathlib import Path

    from nvd_history import description_timeline
    history_dir = Path(history_dir)
    day0 = df["description"].fillna("").copy()
    status = pd.Series("no_phrase", index=df.index, dtype="object")
    uncertain = pd.Series(False, index=df.index)
    for i, cur in day0.items():
        now = _phrase_names(cur)
        if not now:
            continue
        path = history_dir / f"{df.at[i, 'cve_id']}.json"
        if not path.exists():                          # no history -> cannot prove day-0 -> remove (conservative)
            day0.at[i], status.at[i] = drop_sentences_with(cur, now), "history_missing_removed"
            continue
        hist = json.loads(path.read_text())
        orig = description_timeline(hist, cur)["original"]
        dates = [c["change"].get("created") for c in hist.get("cveChanges", []) if c["change"].get("created")]
        if dates:   # earliest record > 7 days after publication -> day-0 text not proven
            gap = (pd.Timestamp(min(dates)) - pd.Timestamp(df.at[i, "published"])).total_seconds() / 86400
            uncertain.at[i] = gap > 7
        added = now - _phrase_names(orig)
        if added:
            day0.at[i], status.at[i] = drop_sentences_with(cur, added), "phrase_added_later_removed"
        else:
            status.at[i] = "phrase_at_publication_kept"
    return pd.DataFrame({"description_day0": day0, "desc_day0_status": status,
                         "desc_day0_uncertain": uncertain & status.eq("phrase_at_publication_kept")})
