import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# optional imports — we'll fall back if they're not present
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import spacy
except Exception:
    spacy = None

_NLP = None
_MODEL = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        if spacy is None:
            return None
        _NLP = spacy.load('en_core_web_sm')
    return _NLP


def _get_model():
    global _MODEL
    if _MODEL is None:
        if SentenceTransformer is None:
            return None
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL


def load_text_files(paths: List[str]) -> List[str]:
    texts = []
    for p in paths:
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append(f.read())
    return texts


def preprocess_text(text: str) -> str:
    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        tokens = [t.lemma_.lower() for t in doc if not t.is_stop and not t.is_punct and t.is_alpha]
        return ' '.join(tokens)

    # fallback simple preprocessing
    text = text.lower()
    words = re.findall(r"[a-z]+", text)
    return ' '.join(words)


def get_default_skills() -> List[str]:
    # A small, extendable skills list — expand as needed
    skills = [
        'python','machine learning','deep learning','nlp','natural language processing',
        'data analysis','pandas','numpy','scikit-learn','tensorflow','pytorch',
        'sql','excel','communication','teamwork','java','javascript','react','aws',
    ]
    return skills


def extract_skills_from_text(text: str) -> List[str]:
    skills = get_default_skills()
    found = set()
    txt = text.lower()
    for s in skills:
        if re.search(r"\b" + re.escape(s.lower()) + r"\b", txt):
            found.add(s)
    return list(found)


def compute_embeddings(texts: List[str]) -> np.ndarray:
    """Compute embeddings using SentenceTransformer when available,
    otherwise fall back to TF-IDF vectors (dense numpy array).
    """
    model = _get_model()
    if model is not None:
        return model.encode(texts, convert_to_numpy=True)

    # fallback to TF-IDF
    vec = TfidfVectorizer()
    mat = vec.fit_transform(texts)
    return mat.toarray()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def rank_resumes(job_text: str, job_skills: set, resume_paths: List[str], resume_texts: List[str],
                 sim_weight: float = 0.7, skill_weight: float = 0.3) -> List[dict]:
    # Preprocess
    job_pre = preprocess_text(job_text)
    resumes_pre = [preprocess_text(t) for t in resume_texts]

    # Embeddings
    emb_texts = [job_pre] + resumes_pre
    embs = compute_embeddings(emb_texts)
    job_emb = embs[0]
    resume_embs = embs[1:]

    results = []
    for path, text, emb in zip(resume_paths, resume_texts, resume_embs):
        similarity = cosine_sim(job_emb, emb)
        matched = set(extract_skills_from_text(text)) & set(job_skills)
        skill_score = 0.0
        if job_skills:
            skill_score = len(matched) / max(1, len(job_skills))

        score = sim_weight * similarity + skill_weight * skill_score
        results.append({
            'path': path,
            'similarity': similarity,
            'skill_score': skill_score,
            'score': score,
            'matched_skills': list(matched),
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results
