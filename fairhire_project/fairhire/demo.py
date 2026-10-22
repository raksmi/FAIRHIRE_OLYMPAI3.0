"""Helpers that load sample data into a Store and run rankings."""
from .data_loader import load_jobs, load_resumes_csv
from .ranker import rank_candidates


def seed_store(store, resumes=None, jobs=None):
    """Add the sample jobs and route each resume to the job of its category; rank every job."""
    jobs = jobs or load_jobs()
    resumes = resumes if resumes is not None else load_resumes_csv()
    ids = {job["category"]: store.add_job(job) for job in jobs}
    for _, r in resumes.iterrows():
        if r["category"] in ids:
            store.add_candidate(ids[r["category"]], r["name"], r["email"], r["resume_text"],
                                r.get("gender", ""), r["category"])
    for job_id in ids.values():
        rank_job(store, job_id)
    return ids


def rank_job(store, job_id, weights=None, actor="system"):
    """Score every candidate of a job, persist the result and return the ranked table."""
    cands = store.candidates(job_id)
    if cands.empty:
        return cands
    ranked = rank_candidates(store.get_job(job_id), cands[["id", "name", "text"]], weights)
    store.save_ranking(ranked, actor)
    return ranked
