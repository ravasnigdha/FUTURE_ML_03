# Resume Screening — Simple ML/NLP Project

This project implements a small resume screening system that:

- Loads job description and candidate resumes (plain text)
- Extracts skills and preprocesses text
- Computes semantic similarity using sentence-transformers embeddings
- Computes a combined score (embedding similarity + skill overlap)
- Ranks candidates and prints results

Getting started

1. Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Run the demo:

```bash
python -m src.main --job samples/job_description.txt --resumes samples/resumes/*.txt
```

Files added

- `src/main.py` — CLI runner
- `src/utils.py` — parsing, preprocessing, scoring helpers
- `samples/` — sample job description and resumes

Notes

- The implementation uses `sentence-transformers` (all-MiniLM-L6-v2) for embeddings.
- You can tweak weights in `src/main.py` to favor skills or semantic match.
