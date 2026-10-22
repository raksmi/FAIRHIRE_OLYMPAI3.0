"""SQLite persistence and the recruitment workflow rules.

Tables: jobs, candidates, emails, audit_log.  Every change is written to
audit_log so the Company Head can see who did what (human accountability).
"""
import json
import sqlite3
from datetime import datetime

import pandas as pd

from . import emailer

STATUSES = ["Applied", "Screened", "Shortlisted", "Interview", "Offer", "Closed"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, title TEXT, category TEXT, description TEXT,
    required TEXT, nice TEXT, min_years INTEGER, min_edu INTEGER);
CREATE TABLE IF NOT EXISTS candidates(id INTEGER PRIMARY KEY, job_id INTEGER, name TEXT, email TEXT,
    text TEXT, gender TEXT, category TEXT, status TEXT DEFAULT 'Applied', score REAL, rank INTEGER,
    matched TEXT, missing TEXT, contribs TEXT, base_value REAL, notes TEXT DEFAULT '',
    head_decision TEXT DEFAULT 'pending');
CREATE TABLE IF NOT EXISTS emails(id INTEGER PRIMARY KEY, candidate_id INTEGER, kind TEXT, subject TEXT,
    body TEXT, status TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY, ts TEXT, actor TEXT, action TEXT, detail TEXT);
"""


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Store:
    """Thin wrapper around the SQLite database."""

    def __init__(self, path=":memory:", company="Our Company"):
        self.conn = sqlite3.connect(path, check_same_thread=False)  # Streamlit uses several threads
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.company = company

    # ---- helpers ---------------------------------------------------------------
    def _df(self, sql, params=()):
        return pd.read_sql_query(sql, self.conn, params=params)

    def log(self, actor, action, detail=""):
        self.conn.execute("INSERT INTO audit_log(ts, actor, action, detail) VALUES (?,?,?,?)",
                          (_now(), actor, action, detail))
        self.conn.commit()

    # ---- jobs ----------------------------------------------------------------
    def add_job(self, job, actor="system"):
        cur = self.conn.execute(
            "INSERT INTO jobs(title, category, description, required, nice, min_years, min_edu) VALUES (?,?,?,?,?,?,?)",
            (job["title"], job.get("category", ""), job["description"], json.dumps(job["required"]),
             json.dumps(job.get("nice", [])), job.get("min_years", 1), job.get("min_edu", 1)))
        self.conn.commit()
        self.log(actor, "job_created", job["title"])
        return cur.lastrowid

    def get_jobs(self):
        jobs = []
        for r in self.conn.execute("SELECT * FROM jobs ORDER BY id"):
            d = dict(r)
            d["required"], d["nice"] = json.loads(d["required"]), json.loads(d["nice"])
            jobs.append(d)
        return jobs

    def get_job(self, job_id):
        return next(j for j in self.get_jobs() if j["id"] == job_id)

    # ---- candidates ----------------------------------------------------------
    def add_candidate(self, job_id, name, email, text, gender="", category="", actor="system"):
        cur = self.conn.execute(
            "INSERT INTO candidates(job_id, name, email, text, gender, category) VALUES (?,?,?,?,?,?)",
            (job_id, name, email, text, gender, category))
        self.conn.commit()
        self.log(actor, "candidate_added", f"{name} -> job {job_id}")
        return cur.lastrowid

    def candidates(self, job_id=None):
        df = self._df("SELECT * FROM candidates" + (" WHERE job_id = ?" if job_id else ""),
                      (job_id,) if job_id else ())
        for col in ("matched", "missing", "contribs"):
            df[col] = df[col].apply(lambda v: json.loads(v) if v else ([] if col != "contribs" else {}))
        return df

    def save_ranking(self, ranked, actor="system"):
        """Store scores + explanations from ranker.rank_candidates; move Applied -> Screened."""
        for _, r in ranked.iterrows():
            self.conn.execute(
                "UPDATE candidates SET score=?, rank=?, matched=?, missing=?, contribs=?, base_value=?, "
                "status=CASE WHEN status='Applied' THEN 'Screened' ELSE status END WHERE id=?",
                (float(r["score"]), int(r["rank"]), json.dumps(r["matched"]), json.dumps(r["missing"]),
                 json.dumps({k: float(v) for k, v in r["contribs"].items()}), float(r["base_value"]), int(r["id"])))
        self.conn.commit()
        self.log(actor, "ranking_run", f"{len(ranked)} candidates scored")

    def set_note(self, cid, note, actor):
        self.conn.execute("UPDATE candidates SET notes=? WHERE id=?", (note, cid))
        self.conn.commit()
        self.log(actor, "note_saved", f"candidate {cid}")

    # ---- workflow rules --------------------------------------------------------
    def set_status(self, cid, new_status, actor="HR Recruiter"):
        """Move a candidate along the pipeline.

        Rules: an interview needs the Company Head's approval of the shortlist.
        Moving to Shortlisted / Interview / Offer / Closed auto-drafts the matching
        email (status 'draft') for HR to review - it is never sent automatically.
        Returns the new email id or None.
        """
        if new_status not in STATUSES:
            raise ValueError(f"Unknown status {new_status}")
        row = self.conn.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
        if new_status in ("Interview", "Offer") and row["head_decision"] != "approved":
            raise PermissionError("The Company Head must approve this candidate before interview or offer.")
        self.conn.execute("UPDATE candidates SET status=? WHERE id=?", (new_status, cid))
        self.conn.commit()
        self.log(actor, "status_changed", f"{row['name']}: {row['status']} -> {new_status}")
        kind = emailer.STATUS_EMAIL.get(new_status)
        if not kind:
            return None
        job = self.get_job(row["job_id"])
        missing = json.loads(row["missing"]) if row["missing"] else []
        draft = emailer.build_draft(kind, row["name"], job["title"], self.company, missing_skills=missing)
        cur = self.conn.execute(
            "INSERT INTO emails(candidate_id, kind, subject, body, status, created_at) VALUES (?,?,?,?,?,?)",
            (cid, kind, draft["subject"], draft["body"], "draft", _now()))
        self.conn.commit()
        self.log("FairHire AI", "email_drafted", f"{kind} email for {row['name']}")
        return cur.lastrowid

    def head_decide(self, cid, decision, actor="Company Head"):
        """decision: 'approved' or 'sent back'."""
        if decision not in ("approved", "sent back", "pending"):
            raise ValueError("decision must be approved / sent back / pending")
        self.conn.execute("UPDATE candidates SET head_decision=? WHERE id=?", (decision, cid))
        self.conn.commit()
        self.log(actor, "head_decision", f"candidate {cid}: {decision}")

    # ---- emails ----------------------------------------------------------------
    def emails(self):
        return self._df("SELECT e.*, c.name AS candidate, c.email AS address FROM emails e "
                        "JOIN candidates c ON c.id = e.candidate_id ORDER BY e.id DESC")

    def update_draft(self, eid, subject, body):
        self.conn.execute("UPDATE emails SET subject=?, body=? WHERE id=? AND status='draft'", (subject, body, eid))
        self.conn.commit()

    def confirm_email(self, eid, actor="HR Recruiter"):
        body = self.conn.execute("SELECT body FROM emails WHERE id=?", (eid,)).fetchone()["body"]
        left = emailer.unresolved_placeholders(body)
        if left:
            raise ValueError("Fill in these placeholders first: " + ", ".join(left))
        self.conn.execute("UPDATE emails SET status='confirmed' WHERE id=?", (eid,))
        self.conn.commit()
        self.log(actor, "email_confirmed", f"email {eid}")

    def send_email(self, eid, actor="HR Recruiter"):
        e = self.conn.execute("SELECT e.*, c.email AS address FROM emails e JOIN candidates c "
                              "ON c.id=e.candidate_id WHERE e.id=?", (eid,)).fetchone()
        result = emailer.send(e["address"], e["subject"], e["body"], confirmed=(e["status"] == "confirmed"))
        self.conn.execute("UPDATE emails SET status='awaiting_reply' WHERE id=?", (eid,))
        self.conn.commit()
        self.log(actor, "email_" + result, f"email {eid} to {e['address']}")
        return result

    # ---- reporting -------------------------------------------------------------
    def audit_log(self):
        return self._df("SELECT ts, actor, action, detail FROM audit_log ORDER BY id DESC")

    def funnel(self, job_id=None):
        """Candidates per pipeline stage (cumulative stages counted by current status)."""
        df = self.candidates(job_id)
        counts = df["status"].value_counts() if len(df) else pd.Series(dtype=int)
        return pd.Series({s: int(counts.get(s, 0)) for s in STATUSES})
