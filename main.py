import argparse
import glob
import sys
from pathlib import Path

from src.utils import load_text_files, compute_embeddings, extract_skills_from_text, rank_resumes


def parse_args():
    p = argparse.ArgumentParser(description="Simple Resume Screening CLI")
    p.add_argument("--job", required=True, help="Path to job description text file")
    p.add_argument("--resumes", required=True, help="Glob for resume text files (e.g. samples/resumes/*.txt)")
    p.add_argument("--top", type=int, default=10, help="How many top candidates to show")
    return p.parse_args()


def main():
    args = parse_args()
    job_path = Path(args.job)
    if not job_path.exists():
        print(f"Job description not found: {job_path}")
        sys.exit(1)

    resume_paths = sorted(glob.glob(args.resumes))
    if not resume_paths:
        print("No resumes found for the given glob.")
        sys.exit(1)

    # Load texts
    job_text = load_text_files([str(job_path)])[0]
    resumes = load_text_files(resume_paths)

    # Extract skills from job and resumes
    job_skills = set(extract_skills_from_text(job_text))

    # Compute embeddings and scores
    results = rank_resumes(job_text, job_skills, resume_paths, resumes)

    # Print ranked results
    print("\nRanked Candidates:\n")
    for i, r in enumerate(results[: args.top], start=1):
        print(f"{i}. {r['path']}")
        print(f"   score: {r['score']:.4f}  (sim={r['similarity']:.4f}, skills={r['skill_score']:.4f})")
        if r['matched_skills']:
            print(f"   matched skills: {', '.join(sorted(r['matched_skills']))}")
        print()


if __name__ == '__main__':
    main()
