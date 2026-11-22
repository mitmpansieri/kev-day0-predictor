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
    how = st.radio("Set your capacity as", ["Emergency patches per month", "Analyst hours per month"])
    hours_per_patch = st.number_input("Hours per emergency patch", 1.0, 40.0, 6.0, 0.5)
    if how == "Emergency patches per month":
        capacity = st.slider("Emergency patches per month", 25, 1500, default_cap, 5,
                             help=f"Default {default_cap} = the evaluated policy: 4.2% of a month's CVEs.")
    else:
        hours = st.number_input("Analyst hours per month for emergency patching", 50.0, 20000.0, float(default_cap * hours_per_patch), 50.0)
        capacity = int(max(1, round(hours / hours_per_patch)))
        st.write(f"= **{capacity:,} emergency patches per month**")
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
    curve = get_curve()
    ks = curve.index.to_numpy(float)
    k = float(tool.capacity)
    in_range = k <= ks.max()
    caught = lambda col, kk: 100 * np.interp(min(kk, ks.max()), ks, curve[col].to_numpy())
    rows = [{"": name, "caught": f"{caught(col, k):.0f} of 100" + ("" if in_range else "+")}
            for name, col in [("Our model", "gradient boosting (headline)"),
                              ("Best simple rule (exploit wording, then CVSS)", "phrase, then CVSS (CNA)"),
                              ("Sort by CVSS score", "CVSS (CNA) ranking")]]
    st.table(pd.DataFrame(rows).set_index(""))
    rule_k, rule_r = OWN.loc["CVSS>=9 (CNA)", "patches/month"], OWN.loc["CVSS>=9 (CNA)", "rule recall %"]
    model_curve = 100 * curve["gradient boosting (headline)"].to_numpy()
    j = int(np.argmax(model_curve >= rule_r))                       # first capacity where the model matches the rule
    k_match = ks[j] if j == 0 else ks[j - 1] + (rule_r - model_curve[j - 1]) / (model_curve[j] - model_curve[j - 1]) * (ks[j] - ks[j - 1])
    st.write(f"To catch as many as the CVSS ≥ 9 policy ({rule_r:.0f} of 100, using {rule_k:,.0f} patches/month), the model "
             f"needs only **about {5 * round(k_match / 5):,.0f} patches/month**.")
    yearly_h = k * hours_per_patch * 12
    st.write(f"Total analyst effort: {k:,.0f} patches/month × {hours_per_patch:g} h = **{yearly_h:,.0f} hours/year ≈ "
             f"${yearly_h * rate:,.0f}/year** (cost of the patching work, not a saving).")
    st.caption("Estimated on 2025-26 data. The model wasn't retrained during that period and was weaker in 2026; retraining "
               "every quarter is recommended. Only the default setting matches the evaluated policy."
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


def show_card(row: pd.Series) -> None:
    st.markdown(f"### :{TIER_COLOR[row['decision']]}[{LABEL[row['decision']]}] — {row['cve_id']}")
    st.write(ct.TIER_TEXT[row["decision"]].capitalize() + ". " + tool.budget_line())
    if row.get("cna_status") in ct.CNA_WARNING:
        st.warning(ct.CNA_WARNING[row["cna_status"]])
    c1, c2, c3 = st.columns(3)
    c1.metric("Percentile (vs July 2026)", f"{row['percentile']:.1f}")
    c2.metric("Indicative probability", f"{row['probability_%']:.1f}%")
    c3.metric("vs base rate (0.31%)", f"{row['x_base_rate']:.1f}x")
    st.code(ct.card(row, tool), language=None)


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
                                          "probability_%": "probability %", "x_base_rate": "x base rate"})
                     .style.apply(lambda r: [bg[r.name]] * len(r), axis=1), hide_index=True, width="stretch", height=520)
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
