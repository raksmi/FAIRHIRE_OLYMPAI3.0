"""FairHire - Explainable & Bias-Audited Resume Screener (OlympAI Hackathon, Stage 2).

Two ways to run this ONE file:
    streamlit run main.py     -> the interactive dashboard (HR Recruiter + Company Head)
    python main.py            -> a quick terminal demo: ranking, explanation, fairness audit, email

All logic lives in the `fairhire/` package; this file is only the user interface.
"""
import os
import re
import sys

import pandas as pd

from fairhire import emailer, fairness
from fairhire.data_loader import load_jobs, load_resumes_csv, read_resume_file
from fairhire.db import STATUSES, Store
from fairhire.demo import rank_job, seed_store
from fairhire.identity import infer_gender
from fairhire.ranker import DEFAULT_WEIGHTS, FEATURES, precision_at_k, rank_candidates

# Demo log-in PINs (override with environment variables for real use).
PINS = {"HR Recruiter": os.environ.get("FAIRHIRE_HR_PIN", "hr123"),
        "Company Head": os.environ.get("FAIRHIRE_HEAD_PIN", "head123")}


# ======================================================================= terminal demo
def run_demo():
    """Non-interactive tour of the whole pipeline (no Streamlit needed)."""
    print("=" * 70, "\nFairHire terminal demo\n", "=" * 70, sep="")
    store = Store(":memory:")
    ids = seed_store(store)
    jobs = store.get_jobs()

    print("\n1) Ranking quality on the sample data (Precision@10 vs known job category)")
    for job in jobs:
        cands = store.candidates()
        labels = dict(zip(cands["id"], cands["category"]))
        pool = cands[["id", "name", "text"]]
        ranked = rank_candidates(job, pool)
        print(f"   {job['title']:<26} Precision@10 = {precision_at_k(ranked, labels, job['category']):.0%}")

    job = jobs[0]
    job_id = ids[job["category"]]
    top = store.candidates(job_id).sort_values("rank").iloc[0]
    print(f"\n2) Why is {top['name']} ranked #1 for {job['title']}? (score {top['score']:.1f})")
    print(f"   Base value (pool average): {top['base_value']:.1f}")
    for feature, value in top["contribs"].items():
        print(f"   {feature:<12} {value:+6.2f}")
    print("   Matched skills:", ", ".join(top["matched"]) or "-")
    print("   Missing skills:", ", ".join(top["missing"]) or "-")

    print("\n3) Fairness audit - synthetic twins (identity swapped, skills identical)")
    cands = store.candidates(job_id)
    audit_df = cands[["id", "name", "text", "gender"]]
    result = fairness.before_after(job, audit_df)
    for label in ("before", "after"):
        r = result[label]
        print(f"   {'WITHOUT' if label == 'before' else 'WITH'} anonymisation: parity diff {r['demographic_parity_diff']:.3f}, "
              f"disparate impact {r['disparate_impact']:.2f}, mean twin rank shift {r['mean_twin_rank_shift']:.2f}, "
              f"max {r['max_twin_rank_shift']:.0f}")

    print("\n4) Human-in-the-loop workflow")
    cid = int(top["id"])
    store.set_status(cid, "Shortlisted")
    try:
        store.set_status(cid, "Interview")
    except PermissionError as err:
        print("   Blocked as expected:", err)
    draft = store.emails().iloc[0]
    print(f"   AI drafted a '{draft['kind']}' email (status: {draft['status']}) - HR must confirm before sending:\n")
    print("   " + draft["body"].replace("\n", "\n   "))
    print("\nFor the full dashboard run:  streamlit run main.py")


# ======================================================================= Streamlit app
def run_app():
    """Build the dashboard.  Imported lazily so `python main.py` works without Streamlit."""
    import streamlit as st

    st.set_page_config(page_title="FairHire", page_icon="⚖️", layout="wide")

    @st.cache_resource
    def get_store():
        """One database for the whole app; seeded with sample data on first run."""
        store = Store("fairhire.db")
        if not store.get_jobs():
            seed_store(store)
        return store

    store = get_store()
    st.sidebar.title("⚖️ FairHire")
    role = st.sidebar.radio("Role", list(PINS))
    pin = st.sidebar.text_input("PIN", type="password", help="Demo PINs are in the README")
    if pin != PINS[role]:
        st.title("FairHire - fair, explainable hiring")
        st.info("Choose a role and enter its PIN in the sidebar to continue.")
        return

    if role == "HR Recruiter":
        weights = {}
        with st.sidebar.expander("Scoring weights"):
            for f in FEATURES:
                weights[f] = st.slider(f.replace("_", " ").title(), 0.0, 1.0, float(DEFAULT_WEIGHTS[f]), 0.05, key="w_" + f)
        hr_view(st, store, weights)
    else:
        head_view(st, store)


def job_picker(st, store, key, allow_all=False):
    """Select box over jobs; returns a job dict (or None for 'All jobs')."""
    jobs = store.get_jobs()
    options = ([None] if allow_all else []) + jobs
    return st.selectbox("Job", options, key=key, format_func=lambda j: "All jobs" if j is None else j["title"])


def hr_view(st, store, weights):
    st.title("HR Recruiter dashboard")
    tab_jobs, tab_rank, tab_flow = st.tabs(["Jobs & resumes", "Ranked applicants", "Pipeline & emails"])

    # ---------------------------------------------------------------- jobs & upload
    with tab_jobs:
        jobs = store.get_jobs()
        st.dataframe(pd.DataFrame([{"Job": j["title"], "Required skills": ", ".join(j["required"]),
                                    "Min. years": j["min_years"]} for j in jobs]))
        with st.expander("Add a job description"):
            title = st.text_input("Job title")
            desc = st.text_area("Description")
            req = st.text_input("Required skills (comma separated)")
            nice = st.text_input("Nice-to-have skills (comma separated)")
            years = st.number_input("Minimum years of experience", 0, 20, 2)
            if st.button("Save job") and title and desc and req:
                split = lambda s: [x.strip().lower() for x in s.split(",") if x.strip()]
                store.add_job({"title": title, "description": desc, "required": split(req), "nice": split(nice),
                               "min_years": int(years), "min_edu": 1}, actor="HR Recruiter")
                st.success("Job saved")
                st.rerun()
        st.subheader("Upload resumes")
        job = job_picker(st, store, "upload_job")
        files = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        if files and st.button("Add resumes and rank"):
            for f in files:
                text = read_resume_file(f.name, f.getvalue())
                mail = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
                first = text.strip().splitlines()[0].strip() if text.strip() else ""
                name = first if re.fullmatch(r"[A-Za-z .'-]{3,40}", first) else os.path.splitext(f.name)[0]
                store.add_candidate(job["id"], name, mail.group(0) if mail else "", text, "", job["category"], "HR Recruiter")
            rank_job(store, job["id"], weights, "HR Recruiter")
            st.success(f"Added {len(files)} resume(s) and re-ranked.")
            st.rerun()

    # ---------------------------------------------------------------- ranked applicants
    with tab_rank:
        job = job_picker(st, store, "rank_job")
        cands = store.candidates(job["id"])
        if cands.empty:
            st.info("No applicants yet.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            blind = c1.checkbox("Blind review (hide names)", value=True)
            min_score = c2.slider("Minimum score", 0, 100, 0)
            statuses = c3.multiselect("Status", STATUSES, default=STATUSES)
            must_have = c4.multiselect("Must have skills", job["required"])
            if st.button("Re-run ranking with current weights"):
                rank_job(store, job["id"], weights, "HR Recruiter")
                st.rerun()
            shown = cands[(cands["score"].fillna(0) >= min_score) & cands["status"].isin(statuses)]
            shown = shown[shown["matched"].apply(lambda m: all(s in m for s in must_have))].sort_values("rank")
            label = lambda r: f"Candidate #{r['id']}" if blind else r["name"]
            st.dataframe(pd.DataFrame({
                "Rank": shown["rank"], "Candidate": [label(r) for _, r in shown.iterrows()],
                "Score": shown["score"].round(1), "Status": shown["status"], "Head decision": shown["head_decision"],
                "Matched skills": shown["matched"].apply(", ".join), "Missing skills": shown["missing"].apply(", ".join)}))
            if not shown.empty:
                st.subheader("Why this score?")
                pick = st.selectbox("Applicant", shown["id"].tolist(),
                                    format_func=lambda i: label(shown[shown["id"] == i].iloc[0]))
                row = shown[shown["id"] == pick].iloc[0]
                left, right = st.columns(2)
                left.metric("Score", f"{row['score']:.1f}", help=f"Pool average is {row['base_value']:.1f}")
                left.bar_chart(pd.Series(row["contribs"], name="Contribution to score (points)"))
                right.write("**Matched skills:** " + (", ".join(row["matched"]) or "none"))
                right.write("**Missing skills:** " + (", ".join(row["missing"]) or "none"))
                note = right.text_area("Reviewer notes", value=row["notes"] or "", key=f"note{pick}")
                if right.button("Save note"):
                    store.set_note(int(pick), note, "HR Recruiter")
                    st.success("Saved")
                if right.button("Recommend for shortlist"):
                    store.set_status(int(pick), "Shortlisted", "HR Recruiter")
                    st.success("Moved to Shortlisted - an email draft is waiting in 'Pipeline & emails'.")
                    st.rerun()

    # ---------------------------------------------------------------- pipeline & emails
    with tab_flow:
        st.subheader("Move a candidate through the pipeline")
        cands = store.candidates()
        if not cands.empty:
            names = dict(zip(cands["id"], cands["name"] + " (" + cands["status"] + ")"))
            cid = st.selectbox("Candidate", cands["id"].tolist(), format_func=lambda i: names[i])
            new = st.selectbox("New status", STATUSES)
            if st.button("Update status"):
                try:
                    eid = store.set_status(int(cid), new, "HR Recruiter")
                    st.success("Status updated" + (" - email drafted below for your review." if eid else "."))
                    st.rerun()
                except PermissionError as err:
                    st.error(str(err))
        st.subheader("Email drafts (review, edit, confirm, then send)")
        mails = store.emails()
        open_mails = mails[mails["status"].isin(["draft", "confirmed"])]
        if open_mails.empty:
            st.caption("No drafts waiting. Move a candidate to Shortlisted to create one.")
        for _, m in open_mails.iterrows():
            with st.expander(f"{m['kind'].title()} email to {m['candidate']} [{m['status']}]"):
                subj = st.text_input("Subject", m["subject"], key=f"subj{m['id']}")
                body = st.text_area("Body", m["body"], height=220, key=f"body{m['id']}")
                a, b, c = st.columns(3)
                if a.button("Save edits", key=f"save{m['id']}"):
                    store.update_draft(int(m["id"]), subj, body)
                    st.rerun()
                if b.button("Confirm", key=f"conf{m['id']}"):
                    store.update_draft(int(m["id"]), subj, body)
                    try:
                        store.confirm_email(int(m["id"]))
                        st.rerun()
                    except ValueError as err:
                        st.error(str(err))
                if c.button("Send", key=f"send{m['id']}", disabled=(m["status"] != "confirmed")):
                    st.success("Email " + store.send_email(int(m["id"])))
                    st.rerun()
        st.subheader("Communication tracker")
        st.dataframe(mails[["id", "candidate", "kind", "status", "created_at"]])


def head_view(st, store):
    st.title("Company Head dashboard")
    tab_over, tab_app, tab_fair, tab_log = st.tabs(["Overview", "Approvals", "Fairness audit", "Audit log"])

    with tab_over:
        job = job_picker(st, store, "over_job", allow_all=True)
        cands = store.candidates(job["id"] if job else None)
        advanced = cands["status"].isin(["Shortlisted", "Interview", "Offer"]).sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Applicants", len(cands))
        m2.metric("Shortlisted or further", int(advanced))
        m3.metric("Average score", f"{cands['score'].mean():.1f}" if len(cands) else "-")
        st.subheader("Hiring funnel")
        st.bar_chart(store.funnel(job["id"] if job else None))
        st.subheader("Compare vacancies")
        rows = []
        for j in store.get_jobs():
            c = store.candidates(j["id"])
            rows.append({"Job": j["title"], "Applicants": len(c),
                         "Shortlisted or further": int(c["status"].isin(["Shortlisted", "Interview", "Offer"]).sum()),
                         "Average score": round(c["score"].mean(), 1) if len(c) else None})
        st.dataframe(pd.DataFrame(rows))
        mails = store.emails()
        st.subheader("Pending follow-ups")
        st.dataframe(mails[mails["status"].isin(["draft", "confirmed", "awaiting_reply"])][["candidate", "kind", "status"]])

    with tab_app:
        st.subheader("HR shortlist waiting for your decision")
        pending = store.candidates()
        pending = pending[(pending["status"] == "Shortlisted") & (pending["head_decision"] != "approved")]
        if pending.empty:
            st.caption("Nothing waiting for approval.")
        for _, r in pending.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([3, 2, 2])
                c1.write(f"**{r['name']}** - score {r['score']:.1f}")
                c1.caption("Matched: " + ", ".join(r["matched"]) + " | Missing: " + (", ".join(r["missing"]) or "none"))
                if c2.button("Approve", key=f"ap{r['id']}"):
                    store.head_decide(int(r["id"]), "approved")
                    st.rerun()
                if c3.button("Send back", key=f"sb{r['id']}"):
                    store.head_decide(int(r["id"]), "sent back")
                    store.set_status(int(r["id"]), "Screened", "Company Head")
                    st.rerun()

    with tab_fair:
        st.write("Every applicant gets a synthetic **twin** with identical skills but a swapped name, pronouns and "
                 "years. A fair system should not move the twin.")
        job = job_picker(st, store, "fair_job")
        pct = st.slider("Shortlist the top ... percent", 10, 50, 20)
        if st.button("Run fairness audit"):
            df = store.candidates(job["id"])[["id", "name", "text", "gender"]].copy()
            df["gender"] = [g if g in ("female", "male") else infer_gender(t, n)
                            for g, t, n in zip(df["gender"], df["text"], df["name"])]
            result = fairness.before_after(job, df, pct)
            table = pd.DataFrame(result).rename(columns={"before": "Without anonymisation", "after": "With anonymisation"})
            st.dataframe(table.loc[["demographic_parity_diff", "disparate_impact", "mean_twin_rank_shift",
                                    "max_twin_rank_shift", "mean_twin_score_gap", "passes_four_fifths"]].astype(str))
            after = result["after"]
            (st.success if after["passes_four_fifths"] else st.error)(
                f"Disparate Impact with anonymisation: {after['disparate_impact']:.2f} "
                f"(benchmark: at least {fairness.FOUR_FIFTHS:.2f})")
            store.log("Company Head", "fairness_audit", f"{job['title']}: DI={after['disparate_impact']:.2f}")

    with tab_log:
        st.dataframe(store.audit_log())


# ======================================================================= entry point
def _inside_streamlit():
    """True when launched with `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _inside_streamlit():
        run_app()
    else:
        run_demo()
