"""CISO web app (local only):  streamlit run app.py

Sidebar "Your capacity": expected new CVEs per month (default = mean of the trailing 3 complete labelled months), emergency
patches per month (or analyst hours / hours per patch) and the Watch list size -> percentile thresholds; what that capacity
would have caught on 2025-26 test data (frozen model); and a "Model freshness" badge (days since the training data end).
Tabs:
  New CVE     : CVE ID (keyless NVD fetch) or a manual form -> ciso_tool.py card (deployment model)
  Daily feed  : one publication day from NVD -> ranked table, tiers highlighted, CSV download (deployment model)
  Past CVEs   : test period 2025-01 -> 2026-08-20 -> FROZEN Phase 6 model's day-0 rank and decision (your capacity and the
                evaluated K = 110) and the KEV outcome (data/past_cves_test_frozen.parquet; never the deployment model)
All model logic, NVD access, CNA names and the card come from ciso_tool.py. The app never reads .env.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import ciso_tool as ct

ROOT = Path(__file__).resolve().parent
PAST = ROOT / "data" / "past_cves_test_frozen.parquet"
CURVE = ROOT / "outputs" / "tables" / "06_recall_vs_k.csv"
DISCLAIMER = ("**Research prototype for an MIT course project. Not security advice; always follow your organisation's "
              "patching process.**")
EXAMPLES = [("CVE-2026-12744", "Ivanti Neurons for ITSM, CVSS 9.8", "PATCH NOW"),
            ("CVE-2026-67631", "Microsoft SQL Server, CVSS 9.8", "WATCH"),
            ("CVE-2026-16502", "WordPress plugin, CVSS 8.8", "MONITOR")]
OTHER = "Other (type it in below)"
TIER_COLOR = {"PATCH NOW": "red", "WATCH": "orange", "MONITOR": "blue"}
ROW_BG = {"PATCH NOW": "background-color: #fde2e2", "WATCH": "background-color: #fff1d6", "MONITOR": ""}

st.set_page_config(page_title="KEV day-0 predictor", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the deployment model ...")
def get_tool() -> ct.Tool:
    return ct.Tool()


@st.cache_data(show_spinner=False)
def get_past() -> pd.DataFrame:
    df = pd.read_parquet(PAST)
    df["cna_name"] = df["cna"].map(ct.org_name)
    return df


@st.cache_data(show_spinner=False)
def get_curve() -> pd.DataFrame:
    return pd.read_csv(CURVE).set_index("K per month")


def k_for_recall(ks: np.ndarray, recall: np.ndarray, target: float) -> float:
    """Inverse of the recall-vs-K curve: the first capacity (patches/month) where recall reaches target, interpolated
    linearly between the tabulated K values (the curve is non-decreasing)."""
    j = int(np.argmax(recall >= target))
    if j == 0:
        return float(ks[0])
    return float(ks[j - 1] + (target - recall[j - 1]) / (recall[j] - recall[j - 1]) * (ks[j] - ks[j - 1]))


def concave_envelope(ks: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Upper concave envelope of the recall-vs-K curve (indices of its corner points). Its slope never increases, so
    the marginal value read from it decreases steadily instead of following the curve's small bumps."""
    h = [0]
    for i in range(1, len(ks)):
        h.append(i)
        while len(h) >= 3 and ((recall[h[-2]] - recall[h[-3]]) * (ks[h[-1]] - ks[h[-3]])
                               <= (recall[h[-1]] - recall[h[-3]]) * (ks[h[-2]] - ks[h[-3]])):
            h.pop(-2)
    return np.array(h)


def marginal_per_100(ks: np.ndarray, recall: np.ndarray, k: float) -> float:
    """Average extra exploited CVEs caught (out of 100) per 100 extra patches/month, over the window k-100 -> k+200
    (clipped to the tabulated range), read from the concave envelope."""
    h = concave_envelope(ks, recall)
    lo, hi = max(ks[0], k - 100), min(ks[-1], k + 200)
    env = lambda x: np.interp(x, ks[h], recall[h])
    return 100 * (env(hi) - env(lo)) / (hi - lo)


tool = get_tool()

# ------------------------------------------------------------------------------------------- sidebar: capacity
LABEL = ct.TIER_LABEL                     # "Emergency patch (PATCH NOW)", "Watch", "Monitor": same names everywhere
OWN = pd.read_csv(ROOT / "outputs" / "tables" / "06_own_capacity.csv").set_index("rule")
META = tool.meta
TRAIN_END = pd.Timestamp(META["published_to"])
TRAIN_MONTHS = (TRAIN_END - pd.Timestamp(META["published_from"])).days / 30.44
TRAIN_AVG = META["n"] / TRAIN_MONTHS     # average new CVEs per month in the training data

with st.sidebar:
    st.header("Your capacity")
    volume = st.number_input("Expected new CVEs per month", 500, 50000, int(round(tool.volume_default)), 100)
    st.caption(f"Average of May-July 2026 ({tool.volume_default:,.0f}); adjust if you expect more or fewer.")
    default_cap = int(round(ct.BUDGET_SHARE * tool.volume_default))
    curve = get_curve()
    ks = curve.index.to_numpy(float)
    model_curve = 100 * curve["gradient boosting (headline)"].to_numpy()
    caught = lambda col, kk: 100 * np.interp(min(kk, ks.max()), ks, curve[col].to_numpy())
    how = st.radio("Set your capacity as", ["Emergency patches per month", "Analyst hours per month", "Plan by target"],
                   help="Plan by target: choose the share of exploited CVEs you want to catch; the app reads the patches "
                        "needed from the model's 2025-26 test results.")
    hours_per_patch = st.number_input("Hours per emergency patch", 1.0, 40.0, 6.0, 0.5)
    if how == "Emergency patches per month":
        capacity = st.slider("Emergency patches per month", 25, 1500, default_cap, 5,
                             help=f"Default {default_cap} = the evaluated policy: 4.2% of a month's CVEs.")
    elif how == "Analyst hours per month":
        hours = st.number_input("Analyst hours per month for emergency patching", 50.0, 20000.0, float(default_cap * hours_per_patch), 50.0)
        capacity = int(max(1, round(hours / hours_per_patch)))
        st.write(f"= **{capacity:,} emergency patches per month**")
    else:
        lo, hi = int(np.ceil(model_curve.min())), int(np.floor(model_curve.max()))
        target = st.slider("I want to catch __% of exploited CVEs", lo, hi, 50, 1,
                           help=f"Possible range {lo}-{hi}%: the model's test results cover {ks.min():.0f}-{ks.max():,.0f} "
                                "patches/month.")
        capacity = int(round(k_for_recall(ks, model_curve, target)))
        st.write(f"To catch **{target}%** of exploited CVEs: about **{capacity:,} emergency patches/month** = "
                 f"**{capacity * hours_per_patch:,.0f} analyst hours/month** ({capacity * hours_per_patch * 12:,.0f} h/year).")
        st.caption("Historical estimate: read from the model's recall-vs-capacity curve on 2025-26 test data (frozen model, "
                   "never retrained in that period). Future months can differ.")
    default_watch = int(round(ct.WATCH_MULT * ct.BUDGET_SHARE * tool.volume_default))
    watch = st.slider("Watch list: extra CVEs to keep an eye on each month", 0, 3000, default_watch, 10,
                      help="Not patched now; recheck if exploit news appears.")
    rate = st.number_input("Cost per analyst hour ($)", 10.0, 1000.0, 100.0, 10.0)
    is_default = capacity == default_cap and watch == default_watch and volume == int(round(tool.volume_default))
    tool.set_capacity(None if is_default else capacity, None if is_default else watch, None if is_default else volume)
    st.markdown(f"→ your **emergency patches** = the top **{100 * tool.share_patch:.1f}%** of each month's CVEs  \n"
                f"→ **Watch** = the next **{tool.watch:,.0f}** riskiest CVEs after your emergency patches "
                f"(top {100 * tool.share_patch:.1f}%-{100 * tool.share_watch:.1f}%)  \n"
                f"→ **Monitor** = everything else")

    st.subheader("What this capacity would have caught (2025-26)")
    st.write("Out of every 100 CVEs later exploited (added to CISA's KEV list within 90 days), how many would have been in "
             "your emergency-patch list:")
    k = float(tool.capacity)
    in_range = k <= ks.max()
    rows =[{"": name, "caught": f"{caught(col, k):.0f} of 100" + ("" if in_range else "+")}
            for name, col in [("Our model", "gradient boosting (headline)"),
                              ("Best simple rule (exploit wording, then CVSS)", "phrase, then CVSS (CNA)"),
                              ("Sort by CVSS score", "CVSS (CNA) ranking")]]
    st.table(pd.DataFrame(rows).set_index(""))
    rule_k, rule_r = OWN.loc["CVSS>=9 (CNA)", "patches/month"], OWN.loc["CVSS>=9 (CNA)", "rule recall %"]
    # Dynamic comparison with the current CVSS >= 9 policy (385 patches/month, 41 of 100) at YOUR capacity
    c_cap = round(k)                                              # capacity as displayed
    r_cap = caught("gradient boosting (headline)", k)
    d_r = round(r_cap) - round(rule_r)
    d_k = round(rule_k) - c_cap                                   # > 0: fewer patches than the rule
    rule_h = abs(d_k) * hours_per_patch * 12
    hs = f"{rule_h:,.0f} h/year, ${rule_h * rate:,.0f}/year"
    n_pt = f"{abs(d_k):,} {'fewer' if d_k > 0 else 'more'} patches"
    if d_r > 0 and d_k > 0:
        verdict = f"🟢 **Better and cheaper than CVSS ≥ 9:** {d_r:+d} of 100 caught with {n_pt} → **{hs} saved**."
    elif d_r > 0 and d_k == 0:
        verdict = f"🟠 **More protection, at no extra cost:** same number of patches, {d_r:+d} of 100 caught."
    elif d_r > 0:
        verdict = f"🟠 **More protection, at extra cost:** {d_r:+d} of 100 caught with {n_pt} → **{hs} extra**."
    elif d_r < 0:
        verdict = (f"🔴 **Cheaper, but less protection than CVSS ≥ 9:** {d_r:+d} of 100 caught, {n_pt} ({hs} "
                   f"{'less spent' if d_k > 0 else 'extra'}).")
    else:
        verdict = (f"**Same protection ({rule_r:.0f} of 100)** with {n_pt} → **{hs} "
                   f"{'saved' if d_k > 0 else 'extra'}**." if d_k else f"**Same protection ({rule_r:.0f} of 100)** with the same number of patches.")
    st.markdown(f"**vs your current rule (CVSS ≥ 9):** {verdict}",
                help=f"CVSS ≥ 9 rule on 2025-26 test data: {rule_k:,.0f} patches/month, {rule_r:.1f} of 100 caught. Model at "
                     f"{c_cap:,}/month: {r_cap:.1f} of 100. |{rule_k:,.0f} - {c_cap:,}| = {abs(d_k):,} patches/month × "
                     f"{hours_per_patch:g} h × 12 = {rule_h:,.0f} h/year; × ${rate:,.0f}/h = ${rule_h * rate:,.0f}/year "
                     + ("(no difference)." if d_k == 0 else "extra." if d_k < 0 else "less spent." if d_r < 0 else "saved."))
    if k < ks.max():
        extra = marginal_per_100(ks, model_curve, k)
        n_txt = "fewer than 1" if round(extra) < 1 else f"about {extra:.0f}"
        st.write(f"**Marginal value:** each extra 100 patches/month (~{100 * hours_per_patch:,.0f} h) catches {n_txt} more "
                 f"exploited CVEs out of 100 at your current setting.")
        st.caption("Averaged from 100 below to 200 above your setting on a smoothed (concave) version of the test curve, "
                   "so it falls steadily as capacity grows.")
    else:
        st.write(f"**Marginal value:** not estimated at {ks.max():,.0f}+ patches/month (test results stop at "
                 f"{ks.max():,.0f}).")
    yearly_h = round(k) * hours_per_patch * 12                     # rounded capacity, as displayed
    st.write(f"Total analyst effort: {round(k):,} patches/month × {hours_per_patch:g} h = **{yearly_h:,.0f} hours/year ≈ "
             f"${yearly_h * rate:,.0f}/year** (cost of the patching work, not a saving).")
    # Saving at YOUR capacity: patches/month that sorting by CVSS score would need to catch as many as the model does
    c_round, r_model = c_cap, r_cap                               # same C and R as the comparison above
    cvss_curve = 100 * curve["CVSS (CNA) ranking"].to_numpy()
    if r_model > cvss_curve.max():
        st.write(f"**Saved by the model (same protection):** sorting by CVSS doesn't reach {r_model:.0f} of 100 within {ks.max():,.0f} patches/month "
                 "in our test data; the saving is larger than we can estimate.")
    else:
        x_cvss = 5 * round(k_for_recall(ks, cvss_curve, r_model) / 5)   # inverse lookup, rounded to 5
        dyn_h = (x_cvss - c_round) * hours_per_patch * 12
        st.markdown(f"**Saved by the model (same protection):** to catch {r_model:.0f} of 100 by sorting on CVSS score you'd need ~{x_cvss:,.0f} "
                    f"patches/month → ({x_cvss:,.0f} - {c_round:,}) × {hours_per_patch:g} h × 12 = **{dyn_h:,.0f} h/year ≈ "
                    f"${dyn_h * rate:,.0f}/year saved**.",
                    help=f"Model at {c_round:,} patches/month catches {r_model:.1f} of 100 (2025-26 test). Sorting by CNA CVSS "
                         f"score reaches that at ~{x_cvss:,.0f}/month. ({x_cvss:,.0f} - {c_round:,}) × {hours_per_patch:g} h × 12 = "
                         f"{dyn_h:,.0f} h/year; × ${rate:,.0f}/h = ${dyn_h * rate:,.0f}/year.")
    st.caption("Historical estimates from the 2025-26 test period. The model wasn't retrained and was weaker in 2026 (savings "
               "smaller); retraining every quarter is recommended. Only the default setting matches the evaluated policy."
               + ("" if in_range else f" Above {ks.max():,.0f} patches/month the estimate is capped at the value for {ks.max():,.0f}."))

    st.subheader("Model freshness")
    age = (pd.Timestamp(dt.date.today()) - TRAIN_END).days
    badge = st.success if age < 90 else st.warning if age <= 180 else st.error
    badge(f"Trained on CVEs up to **{TRAIN_END:%Y-%m-%d}** ({age} days ago).")
    st.write("Recommendation: retrain every quarter. In our post-test analysis, quarterly retraining recovered +7.4 points in 2026.")
    if volume > 1.5 * TRAIN_AVG:
        st.warning(f"CVE volume has grown since training ({volume:,.0f}/month vs {TRAIN_AVG:,.0f} on average in the training data): "
                   "your capacity covers a smaller share and the model's signals get crowded. Consider retraining and "
                   "increasing capacity.")
    st.caption("Thresholds are rules of thumb; labels need 90 days to be final, so the newest 3 months can never be in training.")

# ------------------------------------------------------------------------------------------- header
st.title("🛡️ Which new CVEs will attackers exploit?")
st.warning(DISCLAIMER)
st.caption("Day-0 ranking of CVEs by the chance CISA lists them in KEV (Known Exploited Vulnerabilities) within 90 days. "
           "Three tiers from your capacity (sidebar): **Emergency patch (PATCH NOW)** (within this month's capacity), **Watch** "
           "(elevated risk, not in this month's budget; recheck when exploit news appears), **Monitor** (the rest). Probabilities are indicative; the "
           "percentile is the primary output. Local app: NVD is read without an API key; nothing else leaves this machine.")


HELP_PCT = (f"Ranks this CVE against the {len(tool.ref):,} CVEs of July 2026. 99.8 means riskier than 998 of every 1,000.")
HELP_PROB = ("Approximate chance this CVE is added to CISA's KEV list within 90 days, from a calibrated model. Best used "
             "to compare CVEs, not as an exact number.")
HELP_BASE = (f"On average {100 * tool.base_rate:.2f}% of CVEs (about {1000 * tool.base_rate:.0f} in 1,000) are added to KEV "
             "within 90 days. This number says how many times riskier this CVE is than an average one.")


# ------------------------------------------------------------------------------------------- plain-English card (display only)
CWE_NAMES = {   # short names of common CWEs (MITRE); others are shown by ID only
    "CWE-20": "improper input validation", "CWE-22": "path traversal", "CWE-77": "command injection",
    "CWE-78": "OS command injection", "CWE-79": "cross-site scripting", "CWE-89": "SQL injection",
    "CWE-94": "code injection", "CWE-119": "memory buffer error", "CWE-120": "classic buffer overflow",
    "CWE-121": "stack-based buffer overflow", "CWE-122": "heap-based buffer overflow", "CWE-125": "out-of-bounds read",
    "CWE-134": "format string", "CWE-190": "integer overflow", "CWE-200": "information exposure",
    "CWE-269": "improper privilege management", "CWE-276": "incorrect default permissions",
    "CWE-284": "improper access control", "CWE-285": "improper authorization", "CWE-287": "improper authentication",
    "CWE-288": "authentication bypass via an alternate path", "CWE-290": "authentication bypass by spoofing",
    "CWE-306": "missing authentication for a critical function", "CWE-312": "cleartext storage of sensitive data",
    "CWE-326": "weak encryption", "CWE-352": "cross-site request forgery", "CWE-362": "race condition",
    "CWE-400": "uncontrolled resource consumption", "CWE-401": "memory leak", "CWE-416": "use after free",
    "CWE-434": "unrestricted file upload", "CWE-476": "NULL pointer dereference", "CWE-502": "deserialization of untrusted data",
    "CWE-601": "open redirect", "CWE-611": "XML external entity", "CWE-639": "authorization bypass via a user-controlled key",
    "CWE-693": "protection mechanism failure", "CWE-770": "resource allocation without limits", "CWE-787": "out-of-bounds write",
    "CWE-798": "hard-coded credentials", "CWE-822": "untrusted pointer dereference", "CWE-843": "type confusion",
    "CWE-862": "missing authorization", "CWE-863": "incorrect authorization", "CWE-918": "server-side request forgery",
    "CWE-1321": "prototype pollution",
}
ARROW = {("strongly", "raises"): "⬆️⬆️", ("moderately", "raises"): "⬆️", ("slightly", "raises"): "⬆️",
         ("strongly", "lowers"): "⬇️⬇️", ("moderately", "lowers"): "⬇️", ("slightly", "lowers"): "⬇️"}
CVSS_PLAIN = {   # (metric, value) -> plain words, for reasons and for reading the vector
    ("AV", "N"): "attackable over the internet", ("AV", "A"): "attacker must be on the same local network",
    ("AV", "L"): "needs local access to the device", ("AV", "P"): "needs physical access to the device",
    ("AC", "L"): "low complexity", ("AC", "H"): "high complexity (hard to pull off)",
    ("AT", "N"): "no special conditions needed", ("AT", "P"): "needs specific conditions to be present",
    ("PR", "N"): "no login needed", ("PR", "L"): "needs a basic user login", ("PR", "H"): "needs an admin login",
    ("UI", "N"): "no user action needed", ("UI", "R"): "needs a user to do something (e.g. click a link)",
    ("UI", "P"): "needs a user to do something", ("UI", "A"): "needs a user to actively do something",
    ("S", "U"): "impact limited to the vulnerable component", ("S", "C"): "can affect other components beyond the vulnerable one",
}
IMPACT_NAME = {"C": "data confidentiality", "I": "data integrity", "A": "system availability"}
LEVEL_WORD = {"H": "high", "L": "low", "N": "no"}


def nice_org(name: str) -> str:
    """Display form of a CNA name: bare lowercase short names (e.g. 'ivanti') get a capital; e-mails/UUIDs stay as is."""
    name = str(name)
    return name[:1].upper() + name[1:] if name.islower() and "@" not in name and "." not in name and "-" not in name else name


def vendor_product(row) -> str:
    v = "" if row["vendor"] == "(not stated)" else str(row["vendor"])
    prod = str(row["product"] or "")
    if v and prod.lower().startswith(v.lower()):        # "Microsoft" + "Microsoft SQL Server" -> "Microsoft SQL Server"
        v = ""
    return " ".join(x for x in (v, prod) if x) or "Vendor not stated"


def cvss_plain(vector: str) -> str:
    """CNA CVSS vector (v3.x or v4.0) -> one plain-English phrase."""
    parts = dict(p.split(":", 1) for p in str(vector).split("/")[1:] if ":" in p)
    if not parts:
        return ""
    out = [CVSS_PLAIN[(m, parts[m])] for m in ("AV", "AC", "AT") if (m, parts.get(m)) in CVSS_PLAIN and parts[m] != "X"]
    pr, ui = parts.get("PR"), parts.get("UI")
    if pr == "N" and ui == "N":
        out.append("no login and no user action needed")
    else:
        out += [CVSS_PLAIN[(m, parts[m])] for m in ("PR", "UI") if (m, parts.get(m)) in CVSS_PLAIN]
    imp = {k: parts.get(k, parts.get("V" + k)) for k in ("C", "I", "A")}          # v4 uses VC / VI / VA
    if all(v == "H" for v in imp.values()):
        out.append("full impact on data and systems")
    elif all(v == "N" for v in imp.values()):
        out.append("no direct impact on data or systems")
    elif all(v in LEVEL_WORD for v in imp.values()):
        out.append(", ".join(f"{LEVEL_WORD[v]} impact on {IMPACT_NAME[k]}" for k, v in imp.items()))
    if parts.get("S") == "C":
        out.append(CVSS_PLAIN[("S", "C")])
    return ", ".join(out)


def reason_plain(reason: str, row) -> tuple[str, str]:
    """One tool reason ('<fact> -> strongly raises risk', from ciso_tool.reason_text) -> (arrow, plain sentence).
    Covers every template reason_text produces; anything unexpected is shown as is."""
    import re
    fact, _, eff = reason.rpartition(" -> ")
    words = eff.split()
    arrow = ARROW.get((words[0], words[1]), "•") if len(words) >= 2 else "•"
    up = len(words) >= 2 and words[1] == "raises"
    linked = f"historically linked to {'more' if up else 'less'} exploitation in our data"
    vendor = "The vendor" if row["vendor"] == "(not stated)" else str(row["vendor"])
    n = lambda s: int(s.replace(",", ""))
    kev = "CISA's exploited list (KEV)"
    if m := re.fullmatch(r"CNA CVSS score ([\d.]+)", fact):
        s = float(m.group(1))
        lvl = "Very high" if s >= 9 else "High" if s >= 7 else "Medium" if s >= 4 else "Low"
        text = f"{lvl} severity score from the publisher ({s:.1f} / 10)"
    elif fact == "no CVSS score from the CNA":
        text = "No severity score from the publisher on day 0"
    elif m := re.fullmatch(r"published by CNA (.+)", fact):
        org = nice_org(m.group(1))
        text = f"Published by {org}, {linked}"
    elif m := re.fullmatch(r"vendor not stated \(publisher (.+)\)", fact):
        text = f"No vendor named by the publisher ({nice_org(m.group(1))}), {linked}"
    elif m := re.fullmatch(r"vendor '(.+)'", fact):
        text = f"Vendor {m.group(1)}, {linked}"
    elif m := re.fullmatch(r"weakness (CWE-\d+)", fact):
        name = CWE_NAMES.get(m.group(1))
        text = f"Weakness type {m.group(1)}" + (f" ({name})" if name else "") + f", {linked}"
    elif fact == "weakness NVD-CWE-Other":
        text = f"Weakness type given only as 'Other' (no specific CWE), {linked}"
    elif m := re.fullmatch(r"weakness (.+)", fact):
        text = f"Weakness type {m.group(1)}, {linked}"
    elif fact == "no weakness (CWE) given by the CNA":
        text = "No weakness type (CWE) given by the publisher"
    elif fact in ("CNA gave a CVSS score", "CNA gave no CVSS score"):
        text = "The publisher gave a severity score" if fact.endswith("a CVSS score") else "The publisher gave no severity score"
    elif m := re.fullmatch(r"(\d+) CWE\(s\) given by the CNA", fact):
        k = int(m.group(1))
        text = "No weakness type listed by the publisher" if k == 0 else f"{k} weakness type{'s' if k > 1 else ''} listed by the publisher"
    elif fact == "description says it is exploited (day 0)":
        text = "The description says it is already being exploited"
    elif fact == "no exploitation wording in the description":
        text = "The description doesn't mention exploitation"
    elif m := re.fullmatch(r"description length ([\d,]+) characters", fact):
        text = f"Description is {n(m.group(1)):,} characters long"
    elif m := re.fullmatch(r"vendor has ([\d,]+) earlier CISA KEV entries", fact):
        k = n(m.group(1))
        text = f"{vendor} has no CVEs on {kev} yet" if k == 0 else f"{vendor} already has {k:,} CVE{'s' if k > 1 else ''} on {kev}"
    elif m := re.fullmatch(r"vendor had ([\d,]+) KEV entries in the year before publication", fact):
        k = n(m.group(1))
        text = (f"{vendor} had no CVEs added to {kev} in the year before" if k == 0
                else f"{vendor} had {k:,} CVE{'s' if k > 1 else ''} added to {kev} in the year before")
    elif fact == "vendor never in KEV before":
        text = f"{vendor} has never had a CVE on {kev}"
    elif m := re.fullmatch(r"vendor's last KEV entry (\d+) days before publication", fact):
        d = int(m.group(1))
        text = f"{vendor} had a CVE added to CISA's exploited list {d:,} day{'s' if d != 1 else ''} earlier"
    elif fact in ("vendor already in KEV", "vendor never in KEV"):
        text = f"{vendor} is already on {kev}" if fact == "vendor already in KEV" else f"{vendor} has never had a CVE on {kev}"
    elif m := re.fullmatch(r"product has ([\d,]+) earlier KEV entries", fact):
        k, prod = n(m.group(1)), (f"{row['product']}" if row["product"] else "This product")
        text = f"{prod} has no earlier CVEs on {kev}" if k == 0 else f"{prod} already has {k:,} CVE{'s' if k > 1 else ''} on {kev}"
    elif m := re.fullmatch(r"no CNA CVSS vector \((.+)\)", fact):
        text = "No severity details (CVSS vector) from the publisher"
    elif (m := re.fullmatch(r"(.+?): (.+)", fact)) and any(m.group(1) == nm for nm, _ in ct.LETTERS.values()):
        key = next(k for k, (nm, _) in ct.LETTERS.items() if nm == m.group(1))
        code = next((c for c, w in ct.LETTERS[key][1].items() if w == m.group(2)), None)
        if key in IMPACT_NAME:
            text = f"{LEVEL_WORD.get(code, m.group(2)).capitalize()} impact on {IMPACT_NAME[key]}"
        else:
            text = CVSS_PLAIN.get((key, code), f"{m.group(1)}: {m.group(2)}")
            text = text[0].upper() + text[1:]
    else:
        text = fact[0].upper() + fact[1:] if fact else reason
    return arrow, text


def times(text: str) -> str:
    """'31x riskier' -> '31× riskier' (display only; ciso_tool keeps 'x' for the command line)."""
    import re
    return re.sub(r"(\d)x (riskier|less risky)", r"\1× \2", text)


def copy_summary(row, reasons) -> str:
    """One plain paragraph to paste into a ticket or e-mail."""
    pct, p, x = row["percentile"], row["probability_%"] / 100, row["x_base_rate"]
    riskier = f"Riskier than {min(999, int(np.floor(pct * 10 + 1e-9))):,} of 1,000 CVEs in a typical month"
    odds = "fewer than 1 in 1,000" if p < 0.001 else f"about 1 in {ct._round_n(1 / p):,}"
    fmt = lambda v: f"{v:.1f}×" if v < 10 else f"{v:,.0f}×"
    mult = f"{fmt(x)} average" if x >= 1 else f"about {fmt(1 / x)} less risky than average"
    cvss = "no CVSS" if pd.isna(row["cna_cvss"]) else f"CVSS {row['cna_cvss']:.1f}"
    names = {w.lower() for w in (str(row["vendor"]), str(row["product"] or "").split(" ")[0], nice_org(row["cna"])) if w}
    lower1 = lambda t: t if t.split(" ")[0].lower() in names or t[:2].isupper() else t[0].lower() + t[1:]   # keep names
    why = "; ".join(lower1(t) for _, t in reasons) or "none available"
    return (f"{row['cve_id']} ({vendor_product(row)}, {cvss}): {LABEL[row['decision']]}. {riskier}; {odds} similar CVEs get "
            f"exploited within 90 days ({mult}). Main reasons: {why}. Source: CISO Patch Advisor (research prototype).")


def show_card(row: pd.Series) -> None:
    st.markdown(f"### :{TIER_COLOR[row['decision']]}[{LABEL[row['decision']]}] — {row['cve_id']}")
    st.write(ct.TIER_TEXT[row["decision"]].capitalize() + ". " + tool.budget_line())
    if row.get("cna_status") in ct.CNA_WARNING:
        st.warning(ct.CNA_WARNING[row["cna_status"]])
    c1, c2, c3 = st.columns(3)
    c1.metric("Percentile (vs July 2026)", ct.fmt_percentile(row["percentile"]), help=HELP_PCT)
    c1.caption(ct.plain_percentile(row["percentile"]) + ".")
    c2.metric("Indicative probability", "<0.1%" if round(row["probability_%"], 1) == 0 else f"{row['probability_%']:.1f}%",
              help=HELP_PROB)
    c2.caption(ct.plain_probability(row["probability_%"]) + ".")
    c3.metric(f"vs base rate ({100 * tool.base_rate:.2f}%)", f"{row['x_base_rate']:.1f}×", help=HELP_BASE)
    c3.caption(times(ct.plain_base_rate(row["x_base_rate"], tool.base_rate)) + ".")
    vp = vendor_product(row)
    st.markdown("#### About this CVE")
    st.markdown(f"{vp} · published {row['published']} (UTC) · reported by {nice_org(row['cna'])}")
    if pd.isna(row["cna_cvss"]):
        st.markdown("No CVSS from the publisher (CNA) on day 0; NVD/CISA scores added later are ignored to avoid using "
                    "information not available on publication day.")
    else:
        reading = cvss_plain(row["cvss_vector"])
        st.markdown(f"Severity score from the publisher: **{row['cna_cvss']:.1f} / 10**" + (f" — {reading}" if reading else ""))
    if row["notes"]:
        st.caption(f"Note: {row['notes']}")
    reasons = [reason_plain(row[f"reason_{i}"], row) for i in (1, 2, 3) if row[f"reason_{i}"]]
    if reasons:
        st.markdown("#### " + ("Why the model doesn't prioritise it" if row["decision"] == "MONITOR" else "Why the model flags it"))
        st.markdown("\n".join(f"- {arrow} {text}" for arrow, text in reasons))
        st.caption("⬆️⬆️ strongly raises · ⬆️ raises · ⬇️ lowers · ⬇️⬇️ strongly lowers the model's risk estimate.")
    st.markdown("**Copy summary**")
    st.code(copy_summary(row, reasons), language=None, wrap_lines=True)
    with st.expander("How to read these numbers", expanded=False):
        st.markdown(f"- **Percentile:** {HELP_PCT}\n- **Probability:** {HELP_PROB}\n- **Times base rate:** {HELP_BASE}\n\n"
                    "**Worked example:** Percentile 99.8, probability 9.7%, 31× base rate = riskier than 998 of 1,000 CVEs; "
                    "about 1 in 10 like it get exploited; 31 times the average risk.")


tab_new, tab_feed, tab_past = st.tabs(["New CVE", "Daily feed", "Past CVEs (test period)"])

# ------------------------------------------------------------------------------------------- New CVE
with tab_new:
    st.write("Uses the **deployment model** (trained on all labelled CVEs 2022-01 → 2026-08-20) and day-0 information only.")
    mode = st.radio("Input", ["CVE ID from NVD", "Type it in"], horizontal=True)
    if mode == "CVE ID from NVD":
        with st.container(border=True):
            st.markdown("**Try these** (published 2026-09-08; decision at the default capacity):")
            cols = st.columns(3)
            for c, (cid, what, tier) in zip(cols, EXAMPLES):
                if c.button(f"{cid} · {what} → {LABEL[tier]}", key=f"ex_{cid}"):
                    st.session_state["cve_id"] = cid
                    st.session_state["auto_score"] = True
        st.session_state.setdefault("cve_id", EXAMPLES[0][0])
        cve_id = st.text_input("CVE ID", key="cve_id").strip().upper()
        if st.button("Score this CVE", type="primary") or st.session_state.pop("auto_score", False):
            try:
                with st.spinner("Fetching from NVD (cached after the first time) and explaining ..."):
                    res = tool.score(ct.fetch_cve(cve_id))
                show_card(res.iloc[0])
            except Exception as e:                       # not found / NVD down: show, don't crash
                st.error(f"Could not score {cve_id}: {e}")
    else:
        with st.form("manual"):
            c1, c2 = st.columns(2)
            vendor = c1.text_input("Vendor", "Fortinet")
            product = c2.text_input("Product", "FortiOS")
            known = tool.known_cnas()
            options = [f"{n} ({i})" for n, i in zip(known["name"], known["identifier"])] + [OTHER]
            default = next((k for k, o in enumerate(options) if o.endswith("(psirt@fortinet.com)")), 0)
            pick = c1.selectbox(f"CNA (who published it) - {len(known)} publishers the model learned", options, index=default)
            other = c1.text_input("If 'Other': CNA e-mail or UUID", "", help="Publishers with fewer than 100 training CVEs or "
                                  "never seen are grouped by the model as rare/unseen (treated as higher risk).")
            cwe = c2.text_input("CWE (e.g. CWE-288; empty = none)", "CWE-288")
            vector = c1.text_input("CNA CVSS vector (v3.x scored automatically; empty = no CVSS)", "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
            score = c2.number_input("CVSS score (only needed for v4 vectors; 0 = compute)", 0.0, 10.0, 0.0, 0.1)
            published = c1.date_input("Publication date (UTC)", dt.date.today())
            desc = st.text_area("Description", "An authentication bypass using an alternate path or channel in FortiOS "
                                "SSL-VPN may allow an unauthenticated remote attacker to gain super-admin privileges via crafted requests.")
            go = st.form_submit_button("Score", type="primary")
        if go:
            try:
                cna = other.strip() if pick == OTHER else known["identifier"][options.index(pick)]
                if not cna:
                    raise ValueError("choose a CNA or type one under 'Other'")
                df = ct.manual_cve(vendor or None, cna, desc, vector or None, score or None, cwe or None, product or None,
                                   published.isoformat())
                with st.spinner("Scoring and explaining ..."):
                    show_card(tool.score(df).iloc[0])
            except Exception as e:
                st.error(f"Could not score this entry: {e}")

# ------------------------------------------------------------------------------------------- Daily feed
with tab_feed:
    st.write(f"Every CVE published on NVD on the date you choose (UTC), sorted from most to least likely to be exploited. The "
             f"emergency-patch cut-off is the same every day (the riskiest {100 * tool.share_patch:.1f}% given your capacity), "
             "so busy days like Patch Tuesday get more emergency patches and quiet days fewer; over a month it adds up to "
             "your capacity.")
    day = st.date_input("Publication date (UTC)", dt.date(2026, 9, 8), max_value=dt.date.today())
    if st.button("Fetch and rank", type="primary"):
        try:
            with st.spinner("Fetching the day from NVD (cached after the first time) and scoring ..."):
                st.session_state["feed"] = (day, ct.fetch_day(day.isoformat()))
        except Exception as e:
            st.error(f"Could not fetch {day}: {e}")
    if "feed" in st.session_state:
        d, raw = st.session_state["feed"]
        res = tool.score(raw, explain_top=0)                     # full ranking, current sidebar thresholds; reasons on demand
        n = res["decision"].value_counts()
        st.subheader(f"Published on {d} (UTC): {len(res):,} CVEs — Emergency patch (PATCH NOW) {n.get('PATCH NOW', 0)}, Watch {n.get('WATCH', 0)}, "
                     f"Monitor {n.get('MONITOR', 0)}")
        cols = ["rank", "cve_id", "decision", "vendor", "product", "cna", "cna_cvss", "percentile", "probability_%", "x_base_rate", "notes"]
        view = res[cols].round({"percentile": 1, "probability_%": 2, "x_base_rate": 1})
        bg = view["decision"].map(ROW_BG).tolist()
        view["decision"] = view["decision"].map(LABEL)
        st.dataframe(view.rename(columns={"cve_id": "CVE", "decision": "decision", "cna": "CNA", "cna_cvss": "CNA CVSS",
                                          "probability_%": "probability %", "x_base_rate": "× base rate"})   # display only; CSV keeps x_base_rate
                     .style.apply(lambda r: [bg[r.name]] * len(r), axis=1), hide_index=True, width="stretch", height=520,
                     column_config={"percentile": st.column_config.NumberColumn(
                         "percentile", format="%.1f",
                         help="Rank vs the July 2026 reference month; 100.0 = riskier than every CVE in that month.")})
        st.download_button("Download CSV", res[cols + ["published", "score"]].to_csv(index=False).encode(),
                           file_name=f"kev_day0_ranking_{d}.csv", mime="text/csv")

        st.subheader("Why is this CVE here?")
        labels = [f"{r.rank}. {r.cve_id} {r.vendor} {r.product} — {LABEL[r.decision]}".replace("  ", " ") for r in res.itertuples()]
        pick = st.selectbox("Choose a CVE from this day", labels, index=0)
        sel = res.iloc[labels.index(pick)]
        with st.spinner("Explaining (about 1 s) ..."):
            one = tool.score(raw[raw["cve_id"] == sel["cve_id"]])  # same score and decision; SHAP reasons for this CVE only
        st.caption(f"Rank {sel['rank']:,} of {len(res):,} CVEs published on {d} (UTC).")
        show_card(one.iloc[0])

# ------------------------------------------------------------------------------------------- Past CVEs
with tab_past:
    st.info("This tab uses the **frozen Phase 6 model** (trained on 2022-01 → 2024-12 only) and its scores computed on "
            "publication day for every test CVE (2025-01-01 → 2026-08-20). It **never** uses the deployment model, which was "
            "trained on these CVEs and would look better than it is. The model's decision = the CVE's rank within its month vs "
            "your capacity (Emergency patch if rank ≤ capacity, Watch if rank ≤ capacity + watch list, otherwise Monitor). "
            "Ranks use a fixed tie order, so totals can differ slightly from the reported results (34.0% vs 33.7% at 110 a month). "
            "Results for a single vendor rest on few exploited CVEs (often under 20): a small sample, read them as examples, "
            "not as a performance estimate.")
    past = get_past()
    cap_n, watch_n = tool.capacity, tool.watch
    past = past.assign(your_decision=np.select([past["rank_in_month"] <= cap_n, past["rank_in_month"] <= cap_n + watch_n],
                                               [LABEL["PATCH NOW"], LABEL["WATCH"]], LABEL["MONITOR"]))
    c1, c2 = st.columns([2, 1])
    q_raw = c1.text_input("Search by CVE ID or vendor", "fortinet").strip()
    q = q_raw.lower()
    show_mode = c2.radio("Show", ["Exploited only", "All matching CVEs"], index=0, horizontal=True)
    if q:
        hit = past[past["cve_id"].str.lower().str.contains(q, regex=False) | past["vendor"].fillna("").str.lower().str.contains(q, regex=False)]
        kev, neg = hit[hit["label"] == 1], hit[hit["label"] == 0]
        cnt = lambda d, t: int((d["your_decision"] == LABEL[t]).sum())
        st.markdown(f"**{len(hit):,}** CVEs matching \"{q_raw}\" in 2025-26. **{len(kev)}** were exploited within 90 days: the "
                    f"model flagged **{cnt(kev, 'PATCH NOW')}** as emergency patches, **{cnt(kev, 'WATCH')}** as Watch. Of the "
                    f"**{len(neg):,}** not exploited, it flagged **{cnt(neg, 'PATCH NOW'):,}** as emergency patches (false alarms) "
                    f"and **{cnt(neg, 'WATCH'):,}** as Watch. (Capacity {cap_n:,.0f}/month, Watch list {watch_n:,.0f}.)")
        pool = kev if show_mode == "Exploited only" else hit
        show = pool.sort_values(["label", "published"], ascending=[False, False]).head(300).reset_index(drop=True)
        st.caption("Exploited = added to CISA's KEV list within 90 days of publication (the label the model was trained on). "
                   "\"Later (N days)\" = added to KEV after that window, so it counts as not exploited here."
                   + ("" if len(pool) <= 300 else f" Showing the first 300 of {len(pool):,}."))
        def exploited(r):
            if r["label"] == 1:
                return "Yes"
            return f"Later ({r['days_to_kev']:.0f} days)" if r["in_kev_ever"] and pd.notna(r["days_to_kev"]) else "No"
        if show.empty:
            st.info("No exploited CVEs among these matches. Choose 'All matching CVEs' to see the others.")
        else:
            main = pd.DataFrame({
                "CVE": show["cve_id"], "Published": show["published"], "Vendor": show["vendor"].fillna("(not stated)"),
                "Model's decision on day 0": show["your_decision"],
                "Exploited within 90 days?": show.apply(exploited, axis=1) if len(show) else pd.Series(dtype=str),
                "Added to KEV": show["kev_date_added"].fillna(""),
                "Days after publication": show["days_to_kev"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")})
            inv = {v: k for k, v in LABEL.items()}
            st.dataframe(main.style.apply(lambda r: [ROW_BG[inv[r["Model's decision on day 0"]]]] * len(r), axis=1),
                         hide_index=True, width="stretch", height=420)
            with st.expander("Details: day-0 rank in month and the evaluated 110-a-month policy", expanded=False):
                st.dataframe(pd.DataFrame({
                    "CVE": show["cve_id"], "Product": show["product"].fillna(""), "CNA": show["cna_name"], "CNA CVSS": show["cvss_cna_score"],
                    "Day-0 rank in month": show["rank_in_month"].map("{:,}".format) + " of " + show["cves_in_month"].map("{:,}".format),
                    "Emergency patch at 110/month (evaluated policy)": show["patch_at_k110"].map({True: "yes", False: "no"}),
                    "Ransomware-linked (KEV)": show["kev_ransomware"].map({True: "yes", False: "", None: ""})}),
                    hide_index=True, width="stretch", height=360)
        if len(hit) == 1:
            r = hit.iloc[0]
            outcome = (f"added to KEV on {r['kev_date_added']} ({r['days_to_kev']:.0f} days after publication)" if r["in_kev_ever"]
                       else "not in the KEV snapshot of 2026-11-20")
            st.markdown(f"**{r['cve_id']}** ({r['vendor']} {r['product'] or ''}) ranked **{r['rank_in_month']:,} of {r['cves_in_month']:,}** "
                        f"CVEs in {r['month']} on day 0 → **{r['your_decision']}** at your capacity of {cap_n:,.0f}/month; "
                        f"{'inside' if r['patch_at_k110'] else 'outside'} the evaluated top 110. Outcome: {outcome}.")
