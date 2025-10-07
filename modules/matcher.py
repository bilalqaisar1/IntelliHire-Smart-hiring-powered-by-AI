import numpy as np
from typing import Dict, Any, List
from .extractors import extract_job_requirements, extract_resume_profile
from .openai_utils import get_embedding, compare_skills_openai   # ✅ new import
from .similarity import cosine_sim


def compute_match(job_text: str, resume_text: str) -> Dict[str, Any]:
    """Computes overall matching score and breakdown using OpenAI for skill comparison."""
    
    # Step 1: Extract structured data
    job_info = extract_job_requirements(job_text)
    resume_info = extract_resume_profile(resume_text)

    # Step 2: Extract skills
    job_skills = [s.strip() for s in (job_info.get("required_skills") or []) if s.strip()]
    resume_skills = [s.strip() for s in (resume_info.get("skills") or []) if s.strip()]

    # ✅ Step 3: Use OpenAI to get skill comparison instead of manual set logic
    try:
        skill_result = compare_skills_openai(resume_skills, job_skills)
        matched = skill_result.get("matching_skills", [])
        missing = skill_result.get("missing_skills", [])
        skill_match = float(skill_result.get("match_percentage", 0))
    except Exception as e:
        # fallback if API fails
        job_set = set(s.lower() for s in job_skills)
        resume_set = set(s.lower() for s in resume_skills)
        matched = sorted([s for s in job_skills if s.lower() in resume_set])
        missing = sorted([s for s in job_skills if s.lower() not in resume_set])
        skill_match = round((len(matched) / max(1, len(job_set))) * 100, 2) if job_skills else 0.0

    # Step 4: Experience matching
    req_exp = job_info.get("required_experience_years")
    usr_exp = resume_info.get("total_experience_years")

    if req_exp is None:
        exp_match = 100.0
    elif usr_exp is None:
        exp_match = 0.0
    else:
        exp_match = round(min(100, (usr_exp / req_exp) * 100), 2) if req_exp > 0 else 100.0

    # Step 5: Semantic similarity (embeddings)
    try:
        emb_job = np.array(get_embedding(job_text))
        emb_resume = np.array(get_embedding(resume_text))
        sem_sim = round(float(cosine_sim(emb_job, emb_resume)), 4)
    except Exception:
        sem_sim = 0.0

    # Step 6: Weighted overall score
    overall = round((0.75 * (skill_match / 100) + 0.25 * (exp_match / 100)) * 100, 2)

    # Step 7: Return full structured result
    return {
        "overall_score": overall,
        "semantic_similarity": sem_sim,
        "required_experience": req_exp,
        "user_experience": usr_exp,
        "exp_match_pct": exp_match,
        "required_skills": job_skills,
        "user_skills": resume_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "experiences": resume_info.get("experiences"),
        "job_title": job_info.get("title"),
    }
