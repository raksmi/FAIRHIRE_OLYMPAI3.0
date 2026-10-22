"""Scoring, ranking and explanations.

Score (0-100) = 100 * sum(weight_i * feature_i) over five features, each in [0, 1]:
  skill_match  share of the job's REQUIRED skills the candidate has
  nice_skills  share of the NICE-TO-HAVE skills
  experience   years of experience / years the job asks for (capped at 1)
  education    degree level / level the job asks for (capped at 1)
  text_match   TF-IDF cosine similarity between the (anonymised) resume and the job text

Explanations: because the model is linear, the exact Shapley (SHAP) value of
feature i for one candidate is  weight_i * (x_i - mean_i) * 100,  where mean_i is
the average over the applicant pool.  Contributions + base value add up exactly to
the score, so every ranking can be explained with zero approximation.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .anonymizer import anonymize
from .skills import extract_education_level, extract_skills, extract_years_experience

FEATURES = ["skill_match", "nice_skills", "experience", "education", "text_match"]
DEFAULT_WEIGHTS = {"skill_match": 0.45, "nice_skills": 0.10, "experience": 0.15,
                   "education": 0.10, "text_match": 0.20}


def _normalise_weights(weights):
    """Make the weights sum to 1 so scores stay on a 0-100 scale."""
    w = {f: max(float((weights or DEFAULT_WEIGHTS).get(f, 0.0)), 0.0) for f in FEATURES}
    total = sum(w.values()) or 1.0
    return {f: v / total for f, v in w.items()}


def rank_candidates(job, candidates, weights=None, anonymise=True):
    """Rank candidates for one job.

    job        dict with title, description, required, nice, min_years, min_edu
    candidates DataFrame with columns id, name, text
    anonymise  True  -> text_match uses identity-masked resumes (FairHire's default)
               False -> raw text (used only by the fairness audit to show 'before')
    Returns a DataFrame sorted best-first with features, matched/missing skills,
    SHAP-style contributions and a 1-based rank.
    """
    w = _normalise_weights(weights)
    cands = candidates.reset_index(drop=True)
    required, nice = set(job["required"]), set(job.get("nice", []))

    # Structured features come from the ORIGINAL text (rules ignore identity words).
    rows = []
    for _, c in cands.iterrows():
        skills = set(extract_skills(c["text"]))
        years = extract_years_experience(c["text"])
        edu = extract_education_level(c["text"])
        rows.append({
            "skill_match": len(skills & required) / max(len(required), 1),
            "nice_skills": len(skills & nice) / max(len(nice), 1) if nice else 0.0,
            "experience": min(years / max(job.get("min_years", 1), 1), 1.0),
            "education": min(edu / max(job.get("min_edu", 1), 1), 1.0),
            "matched": sorted(skills & required), "missing": sorted(required - skills),
            "bonus": sorted(skills & nice), "years": years,
        })
    feat = pd.DataFrame(rows)

    # Text similarity: fit TF-IDF on job + pool, compare each resume with the job text.
    used = [anonymize(t, n) if anonymise else t for t, n in zip(cands["text"], cands["name"])]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    mat = vec.fit_transform([job["description"] + " " + " ".join(job["required"])] + used)
    sim = cosine_similarity(mat[0], mat[1:]).ravel()
    feat["text_match"] = sim / (sim.max() + 1e-9)  # scale to [0, 1] within the pool

    # Weighted score + exact linear Shapley contributions against the pool mean.
    X = feat[FEATURES].to_numpy(dtype=float)
    wv = np.array([w[f] for f in FEATURES])
    scores = X @ wv * 100
    base = float(X.mean(axis=0) @ wv * 100)
    contribs = (X - X.mean(axis=0)) * wv * 100

    out = cands[["id", "name"]].copy()
    out["score"] = np.round(scores, 2)
    for f in FEATURES:
        out[f] = np.round(feat[f], 3)
    for col in ("matched", "missing", "bonus", "years"):
        out[col] = feat[col]
    out["contribs"] = [dict(zip(FEATURES, np.round(row, 2))) for row in contribs]
    out["base_value"] = round(base, 2)
    out = out.sort_values(["score", "id"], ascending=[False, True], kind="stable").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def verify_with_shap(job, candidates, weights=None):
    """Optional cross-check: compare our closed-form values with shap.LinearExplainer.

    Needs the `shap` package; returns the max absolute difference, or None if shap
    is not installed.  (Our own explanations never depend on shap.)
    """
    try:
        import shap
    except ImportError:
        return None
    ranked = rank_candidates(job, candidates, weights)
    w = _normalise_weights(weights)
    X = ranked[FEATURES].to_numpy(dtype=float)
    coef = np.array([w[f] for f in FEATURES]) * 100
    explainer = shap.LinearExplainer((coef, 0.0), X)
    shap_values = explainer.shap_values(X)
    ours = np.array([[c[f] for f in FEATURES] for c in ranked["contribs"]])
    return float(np.abs(shap_values - ours).max())


def precision_at_k(ranked, labels, target, k=10):
    """Share of the top-k ranked candidates whose true category equals `target`.

    labels: dict {candidate id: category}.
    """
    top = ranked.head(k)["id"].tolist()
    return sum(labels.get(i) == target for i in top) / max(len(top), 1)
