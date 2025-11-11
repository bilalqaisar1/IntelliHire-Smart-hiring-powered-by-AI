from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..extractors import extract_job_requirements, extract_resume_profile
from ..openai_utils import get_embedding, compare_skills_openai


def _compute_experience_match(required_exp, user_exp) -> float:
    if required_exp is None:
        return 100.0
    if user_exp is None:
        return 0.0
    try:
        req = float(required_exp)
        usr = float(user_exp)
        if req <= 0:
            return 100.0
        return round(min(100.0, (usr / req) * 100.0), 2)
    except Exception:
        return 0.0


def _cosine(a: List[float], b: List[float]) -> float:
    # Local small cosine to avoid importing numpy here
    import math
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


@dataclass
class ParallelAnalyzeChain:
    """Parallel chain that preserves the structure of compute_match with lower latency.

    Steps executed (some in parallel):
    - Extract job requirements (LLM)
    - Extract resume profile (LLM)
    - Compare skills (LLM)
    - Compute experience match (deterministic)
    - Get embeddings for full texts (parallel) and cosine similarity
    """

    max_workers: int = 4

    def run(self, job_text: str, resume_text: str) -> Dict[str, Any]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # 1) Parallel LLM extractions
            fut_job = pool.submit(extract_job_requirements, job_text)
            fut_resume = pool.submit(extract_resume_profile, resume_text)

            job_info = fut_job.result()
            resume_info = fut_resume.result()

            job_skills = [s.strip() for s in (job_info.get("required_skills") or []) if s and s.strip()]
            resume_skills = [s.strip() for s in (resume_info.get("skills") or []) if s and s.strip()]

            # 2) Skill comparison (LLM)
            try:
                skill_result = compare_skills_openai(resume_skills, job_skills)
                matched = skill_result.get("matching_skills", [])
                missing = skill_result.get("missing_skills", [])
                skill_match = float(skill_result.get("match_percentage", 0.0))
            except Exception:
                job_set = set(s.lower() for s in job_skills)
                resume_set = set(s.lower() for s in resume_skills)
                matched = sorted([s for s in job_skills if s.lower() in resume_set])
                missing = sorted([s for s in job_skills if s.lower() not in resume_set])
                skill_match = round((len(matched) / max(1, len(job_set))) * 100, 2) if job_skills else 0.0

            # 3) Experience match (deterministic)
            exp_match = _compute_experience_match(
                job_info.get("required_experience_years"),
                resume_info.get("total_experience_years"),
            )

            # 4) Parallel embeddings + cosine
            fut_emb_job = pool.submit(get_embedding, job_text)
            fut_emb_resume = pool.submit(get_embedding, resume_text)
            try:
                emb_job = fut_emb_job.result()
                emb_resume = fut_emb_resume.result()
                sem_sim = round(float(_cosine(emb_job, emb_resume)), 4)
            except Exception:
                sem_sim = 0.0

        # 5) Weighted overall score (same logic)
        overall = round((0.75 * (skill_match / 100.0) + 0.25 * (exp_match / 100.0)) * 100.0, 2)

        return {
            "overall_score": overall,
            "semantic_similarity": sem_sim,
            "required_experience": job_info.get("required_experience_years"),
            "user_experience": resume_info.get("total_experience_years"),
            "exp_match_pct": exp_match,
            "required_skills": job_skills,
            "user_skills": resume_skills,
            "matched_skills": matched,
            "missing_skills": missing,
            "experiences": resume_info.get("experiences"),
            "education": resume_info.get("education"),
            "job_title": job_info.get("title"),
            "candidate_name": resume_info.get("name"),
            "candidate_current_title": resume_info.get("current_title"),
        }


