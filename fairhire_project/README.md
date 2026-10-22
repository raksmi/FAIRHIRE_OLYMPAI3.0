# FairHire - Explainable & Bias-Audited Resume Screener

OlympAI Hackathon - Stage 2 (Category 11-12, Advanced Coders)

FairHire ranks job applicants on **verified skills only**, explains every score,
audits itself for bias, and keeps **humans in control**: HR recommends, the
Company Head approves, and no email leaves without HR confirming it.

## 1. Project setup and installation

Requires Python 3.10 or newer.

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

`spacy` and `shap` are optional. Everything works without them (see section 4).

## 2. What each file does

| File | Purpose |
|---|---|
| `main.py` | The main file. Streamlit dashboard (HR Recruiter + Company Head) **or** a terminal demo. |
| `fairhire/skills.py` | Extracts skills, years of experience and education level with rules (spaCy used if installed). |
| `fairhire/anonymizer.py` | Masks names, gender cues, graduation years, institutions, locations and contact details. |
| `fairhire/ranker.py` | Scores and ranks candidates; computes exact SHAP-style explanations; Precision@k. |
| `fairhire/fairness.py` | Fairness audit: synthetic "twin" applicants, Demographic Parity, Disparate Impact, twin rank shift. |
| `fairhire/identity.py` | Gender-coded name pairs and pronoun swapping used to build twins. |
| `fairhire/emailer.py` | Email templates, drafts, placeholder checks, human-confirmed sending (dry-run by default). |
| `fairhire/db.py` | SQLite storage, pipeline rules (head approval before interview) and the audit log. |
| `fairhire/data_loader.py` | Loads sample/Kaggle CSVs and reads PDF, DOCX and TXT resumes. |
| `fairhire/demo.py` | Seeds the database with sample data and runs rankings. |
| `data/jobs.json` | Four sample job descriptions. |
| `data/sample_resumes.csv` | 60 synthetic resumes (15 per job category). Made by `data/generate_sample_data.py`. |
| `tests/test_core.py` | 10 unit tests for the core logic. |

## 3. Running the main code

**Dashboard (recommended):**
```bash
streamlit run main.py
```
Log in from the sidebar. Demo PINs: HR Recruiter `hr123`, Company Head `head123`
(change them with the environment variables `FAIRHIRE_HR_PIN` / `FAIRHIRE_HEAD_PIN`).

- **HR Recruiter:** add jobs, upload resumes (PDF/DOCX/TXT), see AI-ranked applicants with
  matched/missing skills and a score explanation chart, filter, blind-review mode, add notes,
  move candidates through the pipeline, review/edit/confirm/send email drafts.
- **Company Head:** hiring funnel and vacancy comparison, approve or send back HR shortlists,
  run the fairness audit (before/after anonymisation), view the audit log.

**Terminal demo (no Streamlit needed):**
```bash
python main.py
```
**Tests:**
```bash
python -m unittest discover -s tests -v
```
**Email:** by default sending is simulated (dry-run). To send real email set
`FAIRHIRE_SMTP_HOST`, `FAIRHIRE_SMTP_PORT`, `FAIRHIRE_SMTP_USER`, `FAIRHIRE_SMTP_PASSWORD`.

**Using the Kaggle Resume Dataset:** pass its CSV to `load_resumes_csv("path.csv")` in
`fairhire/data_loader.py`. It reads the `Category` and `Resume`/`Resume_str` columns. Kaggle
resumes contain no names, so synthetic names are assigned (needed for the fairness audit).

## 4. Dependencies

Required: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `pdfplumber`, `python-docx`.
Optional: `spacy` (PhraseMatcher for skills), `shap` (`ranker.verify_with_shap` cross-checks our
explanations). Without them a regex matcher and our closed-form explanations are used.

## 5. AI tools used

- **Claude (Anthropic)** helped write and review the code and this README; the team reviewed and tested it.
- **ChatGPT** helped draft the Stage 1 synopsis slides.
- **No pretrained AI model** is used inside FairHire. The scoring model is a transparent weighted
  linear model over five features (required-skill match, nice-to-have skills, experience, education,
  TF-IDF text similarity from scikit-learn). Because it is linear, the SHAP value of each feature is
  `weight x (feature - pool average) x 100` exactly, so explanations add up to the score with no approximation.

## How the fairness audit works

1. Each applicant gets a **twin**: same skills and experience, different name, gender pronouns and years.
2. Originals and twins are ranked together; the top 20% (adjustable) are shortlisted.
3. We report Demographic Parity difference, Disparate Impact ratio (benchmark >= 0.80) and how far twins move in rank.

Example on the bundled sample data (Data Scientist role, `python main.py`): without anonymisation twins move by
about 1 rank on average; with anonymisation the mean twin rank shift is 0.

## Limitations (honest notes)

- The bundled resumes are **synthetic**; the 100% Precision@10 on them says little about real resumes.
  Test on the Kaggle dataset before drawing conclusions.
- Skill extraction uses a fixed taxonomy of about 45 skills in `skills.py`; add more for other roles.
- Anonymisation is rule-based and can miss unusual formats. The audit exists to catch such leaks.
- Fairness metrics use gender-coded twins only; other attributes (caste, religion, disability) are future work.
- Demo PINs are not production authentication.
