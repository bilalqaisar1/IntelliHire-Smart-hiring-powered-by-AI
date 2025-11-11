from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .tools import (
    tool_extract_job_requirements,
    tool_extract_resume_profile,
    tool_get_matched_skills,
    tool_generate_questions_for_skills,
    tool_evaluate_answers_strict,
)
from .chains import ParallelAnalyzeChain


@dataclass
class AgentState:
    """Holds shared state between tool invocations.

    The orchestrator populates this state by merging tool outputs. Keys are kept
    aligned with existing application structures so that the UI can reuse values.
    """

    job_text: str = ""
    resume_text: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def update(self, payload: Dict[str, Any]) -> None:
        self.data.update(payload)


class AgentOrchestrator:
    """Sequences tool calls to preserve current functionality with an agentic API."""

    def __init__(self) -> None:
        self.t_extract_job = tool_extract_job_requirements()
        self.t_extract_resume = tool_extract_resume_profile()
        self.t_get_matched = tool_get_matched_skills()
        self.t_gen_questions = tool_generate_questions_for_skills()
        self.t_eval_strict = tool_evaluate_answers_strict()
        self._parallel_chain = ParallelAnalyzeChain()

    def analyze(self, job_text: str, resume_text: str) -> Dict[str, Any]:
        """Run parallel analysis and return match_result (compatible with current app)."""
        state = AgentState(job_text=job_text, resume_text=resume_text)

        # Use the parallel chain for lower latency while preserving fields
        match_result = self._parallel_chain.run(job_text, resume_text)
        state.update({"match_result": match_result})

        # Derive matched skills (for interview stage)
        out_skills = self.t_get_matched.run({"match_result": match_result})
        state.update(out_skills)

        return match_result

    def prepare_interview(self, match_result: Dict[str, Any]) -> List[str]:
        """Generate interview questions given a match_result (unchanged behavior)."""
        out_skills = self.t_get_matched.run({"match_result": match_result})
        matched_skills = out_skills.get("matched_skills", [])
        out_qs = self.t_gen_questions.run({"matched_skills": matched_skills})
        return out_qs.get("questions", [])

    def score_interview(self, questions: List[str], answers_indexed: Dict[int, str]) -> float:
        """Strict scoring across all questions (compatible with current app)."""
        out_score = self.t_eval_strict.run({
            "questions": questions,
            "answers": answers_indexed,
        })
        return float(out_score.get("strict_score", 0.0))

    def get_matched_skills_from_result(self, match_result: Dict[str, Any]) -> List[str]:
        """Expose matched skills so UI can display them without re-implementing logic."""
        out_skills = self.t_get_matched.run({"match_result": match_result})
        return out_skills.get("matched_skills", [])


