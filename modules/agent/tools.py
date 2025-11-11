from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# Local imports reuse existing logic to preserve behavior
from ..extractors import (
    extract_job_requirements,
    extract_resume_profile,
)
from ..matcher import compute_match
from ..openai_utils import (
    get_matched_skills,
    generate_questions_for_skills,
    evaluate_answers,
    evaluate_answers_strict,
)


@dataclass
class Tool:
    """Simple callable tool wrapper with a name and a run method.

    Tools receive a dictionary payload and return a dictionary result.
    This minimal interface keeps the application behavior identical
    while enabling agentic orchestration.
    """

    name: str
    description: str
    run: Callable[[Dict[str, Any]], Dict[str, Any]]


def _wrap_result(key: str, value: Any) -> Dict[str, Any]:
    return {key: value}


def tool_extract_job_requirements() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        job_text = payload.get("job_text", "")
        result = extract_job_requirements(job_text)
        return _wrap_result("job_info", result)

    return Tool(
        name="extract_job_requirements",
        description="Extract required experience, skills, and title from job description text",
        run=_run,
    )


def tool_extract_resume_profile() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        resume_text = payload.get("resume_text", "")
        result = extract_resume_profile(resume_text)
        return _wrap_result("resume_info", result)

    return Tool(
        name="extract_resume_profile",
        description="Extract total experience, skills, experiences, education, and identity from resume text",
        run=_run,
    )


def tool_compute_match() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        job_text = payload.get("job_text", "")
        resume_text = payload.get("resume_text", "")
        result = compute_match(job_text, resume_text)
        return _wrap_result("match_result", result)

    return Tool(
        name="compute_match",
        description="Compute overall matching score and breakdown between job and resume",
        run=_run,
    )


def tool_get_matched_skills() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        match_result = payload.get("match_result") or {}
        skills = get_matched_skills(match_result)
        return _wrap_result("matched_skills", skills)

    return Tool(
        name="get_matched_skills",
        description="Return skills present both in required_skills and user_skills",
        run=_run,
    )


def tool_generate_questions_for_skills() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        matched_skills = payload.get("matched_skills") or []
        questions = generate_questions_for_skills(matched_skills)
        return _wrap_result("questions", questions)

    return Tool(
        name="generate_questions_for_skills",
        description="Generate hard interview questions for each matched skill",
        run=_run,
    )


def tool_evaluate_answers() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        questions = payload.get("questions") or []
        answers = payload.get("answers") or {}
        score = evaluate_answers(questions, answers)
        return _wrap_result("legacy_score", score)

    return Tool(
        name="evaluate_answers",
        description="Evaluate answers with legacy scoring rubric",
        run=_run,
    )


def tool_evaluate_answers_strict() -> Tool:
    def _run(payload: Dict[str, Any]) -> Dict[str, Any]:
        questions = payload.get("questions") or []
        answers = payload.get("answers") or {}
        score = evaluate_answers_strict(questions, answers)
        return _wrap_result("strict_score", score)

    return Tool(
        name="evaluate_answers_strict",
        description="Strict evaluation with per-question scores; unanswered yields zero",
        run=_run,
    )


# Optional: Voice-related tools could be added here by wrapping the functions in
# voice_interview.py, but that module loads heavy models at import-time.
# To avoid overhead when not using voice, we keep the core agent tools lightweight.


