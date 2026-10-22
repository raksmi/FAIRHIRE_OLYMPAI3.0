"""Generate data/sample_resumes.csv - synthetic resumes for demos and tests.

Real resumes cannot be shared in a hackathon repo, so we create 15 fictional
candidates for each of 4 job categories (60 total).  Names are gender-coded
(see fairhire/identity.py) so the fairness audit has ground-truth groups.
Run:  python data/generate_sample_data.py
"""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fairhire.identity import FEMALE_FIRST, MALE_FIRST, SURNAMES  # noqa: E402

random.seed(42)  # reproducible dataset

CORE = {
    "Data Science": ["python", "sql", "machine learning", "pandas", "numpy", "scikit-learn", "statistics", "data visualization"],
    "Web Development": ["javascript", "html", "css", "react", "node.js", "rest api", "git", "sql"],
    "Cybersecurity": ["network security", "penetration testing", "linux", "cryptography", "siem", "firewalls",
                      "incident response", "vulnerability assessment", "wireshark"],
    "DevOps": ["docker", "kubernetes", "aws", "linux", "ci/cd", "git", "terraform", "bash", "monitoring"],
}
EXTRAS = {
    "Data Science": ["tensorflow", "nlp", "deep learning", "pytorch"],
    "Web Development": ["typescript", "mongodb", "django", "flask"],
    "Cybersecurity": ["python", "bash"],
    "DevOps": ["python", "monitoring"],
}
CITIES = ["Chennai", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Pune", "London", "Dubai", "Toronto"]
UNIS = ["Anna University", "IIT Madras", "University of Mumbai", "NIT Trichy", "BITS Pilani", "Delhi University", "VIT University"]
COMPANIES = ["Zenith Labs", "BrightPath Tech", "Northwind Systems", "Orbit Analytics", "Lumen Software"]
DEGREES = ["B.Tech", "B.Sc", "M.Tech", "M.Sc", "Bachelor of Engineering", "Master of Science"]


def make_candidate(idx, category):
    gender = random.choice(["female", "male"])
    k = random.randrange(len(FEMALE_FIRST))
    first = FEMALE_FIRST[k] if gender == "female" else MALE_FIRST[k]
    last = random.choice(SURNAMES)
    name = f"{first} {last}"
    subj, poss = ("She", "Her") if gender == "female" else ("He", "His")
    # Skill mix: most core skills, a few extras, occasionally unrelated noise (makes ranking non-trivial).
    core = CORE[category]
    skills = random.sample(core, k=max(3, int(len(core) * random.uniform(0.5, 1.0))))
    skills += random.sample(EXTRAS[category], k=random.randint(0, 2))
    other = random.choice([c for c in CORE if c != category])
    skills += random.sample(CORE[other], k=random.randint(0, 2))
    skills = sorted(set(skills))
    years = random.randint(0, 9)
    grad = random.randint(2008, 2025)
    degree = random.choice(DEGREES)
    start = grad + 1
    lines = [
        name,
        f"Email: {first.lower()}.{last.lower()}{idx}@example.com | Phone: 9{random.randint(100000000, 999999999)}",
        f"Location: {random.choice(CITIES)}",
        f"Gender: {gender.capitalize()}" if random.random() < 0.5 else "",
        f"Summary: {first} is a motivated professional. {subj} has {years} years of experience in "
        f"{', '.join(skills[:3])}. {poss} projects focus on reliable, well-tested solutions.",
        f"Skills: {', '.join(skills)}",
        f"Experience: {years} years of experience. Worked at {random.choice(COMPANIES)} ({start}-{start + max(years, 1)}) "
        f"using {', '.join(skills[:4])}.",
        f"Education: {degree} in Computer Science, {random.choice(UNIS)}, graduated {grad}.",
    ]
    return {"name": name, "email": f"{first.lower()}.{last.lower()}{idx}@example.com", "category": category,
            "gender": gender, "resume_text": "\n".join(l for l in lines if l)}


def main():
    rows, idx = [], 1
    for category in CORE:
        for _ in range(15):
            rows.append(make_candidate(idx, category))
            idx += 1
    random.shuffle(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_resumes.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "email", "category", "gender", "resume_text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} resumes to {out}")


if __name__ == "__main__":
    main()
