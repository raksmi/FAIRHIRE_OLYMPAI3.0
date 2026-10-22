"""Loading resumes: our sample CSV, Kaggle resume datasets, and uploaded files."""
import io
import json
import os

import pandas as pd

from .identity import FEMALE_FIRST, MALE_FIRST, SURNAMES

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_jobs(path=None):
    """Read the sample job descriptions from data/jobs.json."""
    with open(path or os.path.join(DATA_DIR, "jobs.json"), encoding="utf-8") as f:
        return json.load(f)


def load_resumes_csv(path=None):
    """Load a resume CSV into columns name, email, category, gender, resume_text.

    Understands our sample file and the Kaggle 'Resume Dataset' layout
    (columns Category + Resume / Resume_str).  Kaggle resumes carry no names, so
    synthetic identities are assigned (needed by the fairness audit).
    """
    df = pd.read_csv(path or os.path.join(DATA_DIR, "sample_resumes.csv"))
    cols = {c.lower(): c for c in df.columns}
    text_col = next((cols[c] for c in ("resume_text", "resume_str", "resume") if c in cols), None)
    if text_col is None:
        raise ValueError("CSV needs a resume text column (resume_text / Resume_str / Resume)")
    out = pd.DataFrame({"resume_text": df[text_col].astype(str)})
    out["category"] = df[cols["category"]] if "category" in cols else "Unknown"
    if "name" in cols and "gender" in cols:
        out["name"], out["gender"] = df[cols["name"]], df[cols["gender"]]
        out["email"] = df[cols["email"]] if "email" in cols else ""
    else:
        out = assign_synthetic_identity(out)
    return out


def assign_synthetic_identity(df, seed=7):
    """Give identity-free resumes a name/gender (deterministic) and prepend the name."""
    import random
    rng = random.Random(seed)
    names, genders, texts = [], [], []
    for text in df["resume_text"]:
        g = rng.choice(["female", "male"])
        first = rng.choice(FEMALE_FIRST if g == "female" else MALE_FIRST)
        name = f"{first} {rng.choice(SURNAMES)}"
        names.append(name)
        genders.append(g)
        subj = "She" if g == "female" else "He"
        texts.append(f"{name}\n{subj} is applying for this role.\n{text}")
    out = df.copy()
    out["name"], out["gender"], out["resume_text"] = names, genders, texts
    out["email"] = [n.lower().replace(" ", ".") + "@example.com" for n in names]
    return out


def read_resume_file(filename, data):
    """Extract plain text from an uploaded PDF, DOCX or TXT file (bytes)."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    if ext == ".docx":
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")
