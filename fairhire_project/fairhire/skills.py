"""Rule-based extraction of skills, years of experience and education level.

Why rules and not a black-box model?  These three features are the *verified
technical competencies* the problem statement asks us to score on.  Rules are
deterministic, auditable and cannot pick up names, gender or other identity
signals.  If spaCy is installed we use its PhraseMatcher (faster on big pools);
otherwise a regex fallback with identical behaviour is used.
"""
import re

# Canonical skill -> list of aliases found in real resumes.
SKILL_ALIASES = {
    "python": ["python"], "java": ["java"], "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript"], "sql": ["sql", "mysql", "postgresql", "sqlite"],
    "machine learning": ["machine learning", "ml"], "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"], "pandas": ["pandas"], "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"], "tensorflow": ["tensorflow"],
    "pytorch": ["pytorch"], "statistics": ["statistics", "statistical analysis"],
    "data visualization": ["data visualization", "data visualisation", "tableau", "power bi"],
    "html": ["html", "html5"], "css": ["css", "css3"], "react": ["react", "reactjs", "react.js"],
    "node.js": ["node.js", "nodejs"], "django": ["django"], "flask": ["flask"],
    "rest api": ["rest api", "rest apis", "restful"], "mongodb": ["mongodb"],
    "git": ["git", "github"], "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"], "linux": ["linux", "ubuntu"],
    "ci/cd": ["ci/cd", "continuous integration", "jenkins"], "terraform": ["terraform"],
    "bash": ["bash", "shell scripting"], "monitoring": ["monitoring", "prometheus", "grafana"],
    "network security": ["network security"], "penetration testing": ["penetration testing", "pentesting"],
    "cryptography": ["cryptography"], "siem": ["siem", "splunk"], "firewalls": ["firewall", "firewalls"],
    "incident response": ["incident response"], "vulnerability assessment": ["vulnerability assessment"],
    "wireshark": ["wireshark"], "c++": ["c++"], "agile": ["agile", "scrum"],
}

# Compile one regex per canonical skill.  The look-arounds stop "java" matching
# inside "javascript" and let symbols such as "c++" and "ci/cd" work.
_PATTERNS = {
    skill: re.compile(r"(?<![\w+#.])(?:" + "|".join(re.escape(a) for a in aliases) + r")(?![\w+#])", re.I)
    for skill, aliases in SKILL_ALIASES.items()
}

try:  # optional spaCy acceleration -------------------------------------------------
    import spacy
    from spacy.matcher import PhraseMatcher
    _NLP = spacy.blank("en")
    _MATCHER = PhraseMatcher(_NLP.vocab, attr="LOWER")
    for _skill, _aliases in SKILL_ALIASES.items():
        _MATCHER.add(_skill, [_NLP.make_doc(a) for a in _aliases])
    HAVE_SPACY = True
except Exception:  # spaCy missing or model problem -> regex fallback
    HAVE_SPACY = False


def extract_skills(text):
    """Return the sorted list of canonical skills mentioned in `text`."""
    if HAVE_SPACY:
        doc = _NLP.make_doc(text)
        found = {_NLP.vocab.strings[m_id] for m_id, _, _ in _MATCHER(doc)}
        # PhraseMatcher cannot see symbol aliases like "c++" reliably, so top up with regex.
        found |= {s for s, p in _PATTERNS.items() if p.search(text)}
        return sorted(found)
    return sorted(s for s, p in _PATTERNS.items() if p.search(text))


# "5 years of experience", "3+ yrs experience", "7 years of hands-on experience" ...
_EXP = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:[\w-]+\s+){0,3}experience", re.I)


def extract_years_experience(text):
    """Largest 'N years ... experience' figure in the text (0 if none)."""
    values = [int(m.group(1)) for m in _EXP.finditer(text)]
    return max(values) if values else 0


# Education level: 0 none/unknown, 1 bachelor, 2 master, 3 doctorate.
_EDU = [
    (3, re.compile(r"(?<!\w)(?:ph\.?d|doctorate)(?!\w)", re.I)),
    (2, re.compile(r"(?<!\w)(?:master'?s?|m\.?tech|m\.?sc|mba|m\.?s)(?!\w)", re.I)),
    (1, re.compile(r"(?<!\w)(?:bachelor'?s?|b\.?tech|b\.?e|b\.?sc|bca|b\.?s)(?!\w)", re.I)),
]


def extract_education_level(text):
    """Highest degree level mentioned (0-3)."""
    for level, pattern in _EDU:  # checked from highest to lowest
        if pattern.search(text):
            return level
    return 0
