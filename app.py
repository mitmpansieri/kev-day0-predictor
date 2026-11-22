"""CISO web app (local only):  streamlit run app.py

Three tabs:
  New CVE     : CVE ID (keyless NVD fetch) or a manual form -> the ciso_tool.py card (deployment model)
  Daily feed  : one publication day from NVD -> ranked table, PATCH NOW highlighted, CSV download (deployment model)
  Past CVEs   : test period 2025-01 -> 2026-08-20 -> what the FROZEN Phase 6 model said on day 0, and what happened
                (data/past_cves_test_frozen.parquet; never the deployment model, which was trained on these CVEs)
All model logic, NVD access, CNA names and the card come from ciso_tool.py. The app never reads .env.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

import ciso_tool as ct

ROOT = Path(__file__).resolve().parent
PAST = ROOT / "data" / "past_cves_test_frozen.parquet"

st.set_page_config(page_title="KEV day-0 predictor", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the deployment model ...")
def get_tool() -> ct.Tool:
    return ct.Tool()


@st.cache_data(show_spinner=False)
def get_past() -> pd.DataFrame:
    df = pd.read_parquet(PAST)
    df["cna_name"] = df["cna"].map(ct.org_name)
    return df


def show_card(row: pd.Series, tool: ct.Tool) -> None:
    color = "red" if row["decision"] == "PATCH NOW" else "blue"
    st.markdown(f"### :{color}[{row['decision']}] — {row['cve_id']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Percentile (vs July 2026)", f"{row['percentile']:.1f}")
    c2.metric("Indicative probability", f"{row['probability_%']:.1f}%")
    c3.metric("vs base rate (0.31%)", f"{row['x_base_rate']:.1f}x")
    st.code(ct.card(row, tool), language=None)


tool = get_tool()
st.title("🛡️ Which new CVEs will attackers exploit?")
st.caption("Day-0 ranking of CVEs by the chance CISA lists them in KEV (Known Exploited Vulnerabilities) within 90 days. "
           f"PATCH NOW = top {100 * ct.BUDGET_SHARE:.1f}% of a month's CVEs. Probabilities are indicative; the percentile is "
           "the primary output. Local app: NVD is read without an API key; nothing else leaves this machine.")
tab_new, tab_feed, tab_past = st.tabs(["New CVE", "Daily feed", "Past CVEs (test period)"])

# ------------------------------------------------------------------------------------------- New CVE
with tab_new:
    st.write("Uses the **deployment model** (trained on all labelled CVEs 2022-01 → 2026-08-20) and day-0 information only.")
    mode = st.radio("Input", ["CVE ID from NVD", "Type it in"], horizontal=True)
    if mode == "CVE ID from NVD":
        cve_id = st.text_input("CVE ID", value="CVE-2026-12744").strip().upper()
        if st.button("Score this CVE", type="primary"):
            try:
                with st.spinner("Fetching from NVD and explaining ..."):
                    res = tool.score(ct.fetch_cve(cve_id))
                show_card(res.iloc[0], tool)
            except Exception as e:                       # not found / NVD down: show, don't crash
                st.error(f"Could not score {cve_id}: {e}")
    else:
        with st.form("manual"):
            c1, c2 = st.columns(2)
            vendor = c1.text_input("Vendor", "Fortinet")
            product = c2.text_input("Product", "FortiOS")
            cna = c1.text_input("CNA (who published it: e-mail or UUID)", "psirt@fortinet.com")
            cwe = c2.text_input("CWE (e.g. CWE-288; empty = none)", "CWE-288")
            vector = c1.text_input("CNA CVSS vector (v3.x scored automatically)", "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
            score = c2.number_input("CVSS score (only needed for v4 vectors; 0 = compute)", 0.0, 10.0, 0.0, 0.1)
            published = c1.date_input("Publication date", dt.date.today())
            desc = st.text_area("Description", "An authentication bypass using an alternate path or channel in FortiOS "
                                "SSL-VPN may allow an unauthenticated remote attacker to gain super-admin privileges via crafted requests.")
            go = st.form_submit_button("Score", type="primary")
        if go:
            try:
                df = ct.manual_cve(vendor or None, cna, desc, vector or None, score or None, cwe or None, product or None,
                                   published.isoformat())
                with st.spinner("Scoring and explaining ..."):
                    show_card(tool.score(df).iloc[0], tool)
            except Exception as e:
                st.error(f"Could not score this entry: {e}")

# ------------------------------------------------------------------------------------------- Daily feed
with tab_feed:
    st.write("All CVEs NVD published on one day, ranked by the **deployment model**. The 4.2% budget is a *monthly* share, "
             "so busy days (e.g. Patch Tuesday) can have more PATCH NOW entries.")
    c1, c2 = st.columns(2)
    day = c1.date_input("Publication date", dt.date(2026, 9, 8), max_value=dt.date.today())
    top = c2.slider("Explain the top N (SHAP, ~0.5 s each)", 0, 50, 20)
    if st.button("Fetch and rank", type="primary"):
        try:
            with st.spinner("Fetching the day from NVD (cached after the first time) and scoring ..."):
                res = tool.score(ct.fetch_day(day.isoformat()), explain_top=top)
            st.session_state["feed"] = (day, res)
        except Exception as e:
            st.error(f"Could not fetch {day}: {e}")
    if "feed" in st.session_state:
        d, res = st.session_state["feed"]
        n_now = int((res["decision"] == "PATCH NOW").sum())
        st.subheader(f"{d}: {len(res):,} CVEs, {n_now} PATCH NOW")
        cols = ["rank", "cve_id", "decision", "vendor", "product", "cna", "cna_cvss", "percentile", "probability_%",
                "x_base_rate", "reason_1", "reason_2", "reason_3", "notes"]
        view = res[cols].round({"percentile": 1, "probability_%": 2, "x_base_rate": 1})
        st.dataframe(view.style.apply(lambda r: ["background-color: #fde2e2" if r["decision"] == "PATCH NOW" else ""] * len(r), axis=1),
                     hide_index=True, width="stretch", height=520)
        st.download_button("Download CSV", res[cols + ["published", "score"]].to_csv(index=False).encode(),
                           file_name=f"kev_day0_ranking_{d}.csv", mime="text/csv")

# ------------------------------------------------------------------------------------------- Past CVEs
with tab_past:
    st.info("This tab uses the **frozen Phase 6 model** (trained on 2022-01 → 2024-12 only) and its scores computed on "
            "publication day for every test CVE (2025-01-01 → 2026-08-20). It **never** uses the deployment model, which was "
            "trained on these CVEs and would look better than it is. Ranks use a fixed tie order, so totals can differ "
            "slightly from the reported recall (34.0% vs 33.7% at K = 110).")
    past = get_past()
    q = st.text_input("Search by CVE ID or vendor", "fortinet").strip().lower()
    if q:
        hit = past[past["cve_id"].str.lower().str.contains(q, regex=False) | past["vendor"].fillna("").str.lower().str.contains(q, regex=False)]
        st.write(f"{len(hit):,} matching CVEs; {int(hit['label'].sum())} were added to KEV within 90 days. Showing up to 300, KEV-listed first.")
        show = hit.sort_values(["label", "published"], ascending=[False, False]).head(300).assign(
            in_KEV_within_90d=lambda d: d["label"].map({1: "YES", 0: "no"}),
            rank=lambda d: d["rank_in_month"].astype(str) + " / " + d["cves_in_month"].astype(str))
        cols = ["cve_id", "published", "vendor", "product", "cna_name", "cvss_cna_score", "rank", "patch_at_k110", "patch_at_4pct",
                "in_KEV_within_90d", "kev_date_added", "days_to_kev", "kev_ransomware"]
        st.dataframe(show[cols].rename(columns={"rank": "rank in month", "patch_at_k110": "PATCH at K=110", "patch_at_4pct": "PATCH at 4.2%",
                                                 "cna_name": "CNA", "cvss_cna_score": "CNA CVSS", "kev_date_added": "KEV date added"}),
                     hide_index=True, width="stretch", height=420)
        if len(hit) == 1:
            r = hit.iloc[0]
            verdict = "would have been emergency-patched" if r["patch_at_4pct"] else "would NOT have been in the emergency list"
            outcome = (f"added to KEV on {r['kev_date_added']} ({r['days_to_kev']:.0f} days after publication)" if r["in_kev_ever"]
                       else "not in the KEV snapshot of 2026-11-20")
            st.markdown(f"**{r['cve_id']}** ({r['vendor']} {r['product'] or ''}) ranked **{r['rank_in_month']} of {r['cves_in_month']:,}** "
                        f"CVEs in {r['month']} on day 0: it {verdict} at the 4.2% budget "
                        f"({'inside' if r['patch_at_k110'] else 'outside'} the top 110). Outcome: {outcome}.")
