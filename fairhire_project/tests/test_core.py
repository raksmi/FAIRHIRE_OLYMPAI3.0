"""Unit tests for FairHire's core logic.  Run:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairhire import emailer, fairness  # noqa: E402
from fairhire.anonymizer import anonymize  # noqa: E402
from fairhire.data_loader import load_jobs, load_resumes_csv  # noqa: E402
from fairhire.db import Store  # noqa: E402
from fairhire.demo import seed_store  # noqa: E402
from fairhire.ranker import FEATURES, rank_candidates  # noqa: E402
from fairhire.skills import extract_education_level, extract_skills, extract_years_experience  # noqa: E402

RESUME = ("Priya Sharma\nEmail: priya.sharma@example.com | Phone: 9876543210\nLocation: Chennai\nGender: Female\n"
          "Summary: Priya is a developer. She has 5 years of experience in python, sql and C++. Her work is solid.\n"
          "Education: M.Tech in Computer Science, Anna University, graduated 2018.")


class SkillTests(unittest.TestCase):
    def test_skills_and_symbols(self):
        found = extract_skills("Built apps in JavaScript and C++ with CI/CD on AWS")
        self.assertIn("javascript", found)
        self.assertIn("c++", found)
        self.assertIn("ci/cd", found)
        self.assertNotIn("java", found)  # 'java' must not match inside 'javascript'

    def test_experience_and_education(self):
        self.assertEqual(extract_years_experience(RESUME), 5)
        self.assertEqual(extract_education_level(RESUME), 2)  # M.Tech
        self.assertEqual(extract_education_level("PhD in physics"), 3)


class AnonymizerTests(unittest.TestCase):
    def test_identity_removed(self):
        out = anonymize(RESUME, "Priya Sharma")
        for leak in ["Priya", "Sharma", "@example.com", "9876543210", "Chennai", "Anna University", "2018",
                     "Female", " She ", " Her "]:
            self.assertNotIn(leak, out, f"'{leak}' leaked through anonymisation")
        self.assertIn("python", out)  # skills must survive


class RankerTests(unittest.TestCase):
    def setUp(self):
        self.job = load_jobs()[0]  # Data Scientist
        df = load_resumes_csv().reset_index(drop=True)
        df["id"], df["text"] = df.index + 1, df["resume_text"]
        self.df = df

    def test_explanations_add_up_to_score(self):
        ranked = rank_candidates(self.job, self.df[["id", "name", "text"]])
        for _, r in ranked.head(10).iterrows():
            self.assertAlmostEqual(r["base_value"] + sum(r["contribs"].values()), r["score"], delta=0.05)
        self.assertEqual(set(ranked.iloc[0]["contribs"]), set(FEATURES))

    def test_strong_candidate_beats_weak_one(self):
        strong = "Alex Doe\nHe has 6 years of experience in python, sql, machine learning, pandas, scikit-learn, statistics. M.Sc."
        weak = "Sam Roe\nShe has 0 years of experience in html and css."
        pool = self.df[["id", "name", "text"]].head(5)
        import pandas as pd
        extra = pd.DataFrame([{"id": 900, "name": "Alex Doe", "text": strong}, {"id": 901, "name": "Sam Roe", "text": weak}])
        ranked = rank_candidates(self.job, pd.concat([pool, extra], ignore_index=True))
        self.assertLess(ranked[ranked["id"] == 900]["rank"].iloc[0], ranked[ranked["id"] == 901]["rank"].iloc[0])


class FairnessTests(unittest.TestCase):
    def test_anonymisation_removes_twin_gap(self):
        job = load_jobs()[1]
        df = load_resumes_csv().reset_index(drop=True)
        df["id"], df["text"] = df.index + 1, df["resume_text"]
        df = df[df["category"] == "Web Development"][["id", "name", "text", "gender"]]
        result = fairness.before_after(job, df)
        self.assertEqual(result["after"]["mean_twin_rank_shift"], 0.0)
        self.assertGreaterEqual(result["after"]["disparate_impact"], fairness.FOUR_FIFTHS)
        self.assertGreaterEqual(result["before"]["mean_twin_score_gap"], result["after"]["mean_twin_score_gap"])


class EmailTests(unittest.TestCase):
    def test_closure_has_feedback(self):
        d = emailer.build_draft("closure", "Sam", "Data Scientist", missing_skills=["sql", "statistics"])
        self.assertIn("sql", d["body"])

    def test_cannot_send_unconfirmed(self):
        with self.assertRaises(PermissionError):
            emailer.send("a@b.com", "s", "b", confirmed=False)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        seed_store(self.store, load_resumes_csv().head(20))
        self.cid = int(self.store.candidates().sort_values("score", ascending=False).iloc[0]["id"])

    def test_shortlist_drafts_email_but_does_not_send(self):
        eid = self.store.set_status(self.cid, "Shortlisted")
        self.assertIsNotNone(eid)
        self.assertEqual(self.store.emails().iloc[0]["status"], "draft")

    def test_interview_needs_head_approval_and_filled_placeholders(self):
        self.store.set_status(self.cid, "Shortlisted")
        with self.assertRaises(PermissionError):
            self.store.set_status(self.cid, "Interview")
        self.store.head_decide(self.cid, "approved")
        eid = self.store.set_status(self.cid, "Interview")
        with self.assertRaises(ValueError):  # [DATE] etc. still unfilled
            self.store.confirm_email(eid)
        with self.assertRaises(PermissionError):  # not confirmed -> cannot send
            self.store.send_email(eid)
        self.store.update_draft(eid, "Interview", "Date: 5 Nov\nTime: 10am\nVenue: Zoom")
        self.store.confirm_email(eid)
        self.assertEqual(self.store.send_email(eid), "simulated")


if __name__ == "__main__":
    unittest.main()
