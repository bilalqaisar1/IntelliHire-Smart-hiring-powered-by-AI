"""
Enhanced Interview Evaluation System

This module provides improved interview evaluation with correct answer references
and more accurate scoring based on actual responses.
"""

import json
from typing import Dict, Any, List, Tuple
from modules.openai_utils import client, call_chat_json


def generate_correct_answers(questions: List[str], matched_skills: List[str]) -> List[str]:
    """
    Generate correct answers for interview questions to use as reference for scoring.
    
    Args:
        questions: List of interview questions
        matched_skills: List of skills that the questions are based on
        
    Returns:
        List of correct/reference answers
    """
    if not questions:
        return []
    
    sys_prompt = (
        "You are an expert technical interviewer creating reference answers for evaluation. "
        "Generate comprehensive, accurate answers that demonstrate deep technical knowledge. "
        "Each answer should be detailed enough to serve as a scoring reference."
    )
    
    user_prompt = f"""
    Generate reference answers for the following interview questions. These answers will be used 
    to evaluate candidate responses, so they should be comprehensive and technically accurate.
    
    Questions are based on these skills: {', '.join(matched_skills)}
    
    For each question, provide a detailed answer that includes:
    1. Core concept explanation
    2. Technical details
    3. Practical examples or applications
    4. Best practices or considerations
    
    Questions:
    {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
    
    Return ONLY a JSON array of answers in the same order as the questions.
    Each answer should be a comprehensive string (at least 100 characters).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1  # Low temperature for consistent, accurate answers
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean JSON if wrapped in backticks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        correct_answers = json.loads(raw)
        
        # Ensure we have the right number of answers
        if len(correct_answers) != len(questions):
            # Fallback: generate simple answers
            correct_answers = [f"Reference answer for question {i+1}" for i in range(len(questions))]
        
        return correct_answers
        
    except Exception as e:
        print(f"Error generating correct answers: {e}")
        # Fallback: return simple reference answers
        return [f"Reference answer for question {i+1}" for i in range(len(questions))]


def evaluate_with_reference_answers(questions: List[str], answers: Dict[int, str], 
                                  correct_answers: List[str]) -> Dict[str, Any]:
    """
    Evaluate candidate answers against reference answers for more accurate scoring.
    
    Args:
        questions: List of interview questions
        answers: Dictionary of candidate answers (index -> answer)
        correct_answers: List of reference/correct answers
        
    Returns:
        Dictionary containing detailed scoring information
    """
    if not questions or not correct_answers:
        return {"overall_score": 0.0, "per_question_scores": [], "detailed_feedback": []}
    
    # Prepare candidate answers
    candidate_answers = []
    for i in range(len(questions)):
        ans = (answers.get(i) or "").strip()
        if not ans or len(ans) < 3:
            candidate_answers.append("No answer provided.")
        else:
            candidate_answers.append(ans)
    
    # If all answers are blank, return 0 immediately
    if all(a == "No answer provided." for a in candidate_answers):
        return {
            "overall_score": 0.0,
            "per_question_scores": [0.0] * len(questions),
            "detailed_feedback": ["No substantial answers provided"] * len(questions)
        }
    
    # Create evaluation payload
    evaluation_payload = []
    for i in range(len(questions)):
        evaluation_payload.append({
            "question_index": i,
            "question": questions[i],
            "candidate_answer": candidate_answers[i],
            "reference_answer": correct_answers[i]
        })
    
    sys_prompt = (
        "You are a senior technical interviewer evaluating candidate responses against reference answers. "
        "Score each answer on a 0-10 scale based on technical accuracy, completeness, and depth. "
        "CRITICAL: Give 0 for blank, empty, or 'No answer provided' responses. "
        "Return detailed feedback and scores."
    )
    
    user_prompt = f"""
    Evaluate the following candidate responses against reference answers. For each question:
    
    1. Score the candidate answer (0-10):
       - 0: No answer provided, blank, or meaningless response
       - 1-3: Very poor answer, shows minimal understanding
       - 4-6: Adequate answer, shows basic understanding
       - 7-8: Good answer, shows solid understanding
       - 9-10: Excellent answer, shows deep understanding
    
    2. Provide brief feedback explaining the score
    
    Evaluation data:
    {json.dumps(evaluation_payload, indent=2)}
    
    Return ONLY a JSON object with this structure:
    {{
        "per_question_scores": [list of scores 0-10],
        "detailed_feedback": [list of feedback strings],
        "overall_score": overall_percentage_0_100
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Clean JSON if wrapped in backticks
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("\n", 1)[0]
        
        result = json.loads(raw)
        
        # Validate and enforce strict scoring
        per_question_scores = result.get("per_question_scores", [])
        detailed_feedback = result.get("detailed_feedback", [])
        
        # Ensure we have scores for all questions
        while len(per_question_scores) < len(questions):
            per_question_scores.append(0.0)
        
        # Enforce 0 for blank answers regardless of model output
        for i, answer in enumerate(candidate_answers):
            if answer == "No answer provided." or len(answer.strip()) < 3:
                per_question_scores[i] = 0.0
                if i < len(detailed_feedback):
                    detailed_feedback[i] = "No substantial answer provided"
        
        # Calculate overall score
        total_score = sum(per_question_scores)
        max_possible = len(questions) * 10
        overall_score = round((total_score / max_possible) * 100, 2) if max_possible > 0 else 0.0
        
        return {
            "overall_score": overall_score,
            "per_question_scores": per_question_scores,
            "detailed_feedback": detailed_feedback,
            "candidate_answers": candidate_answers,
            "reference_answers": correct_answers
        }
        
    except Exception as e:
        print(f"Error in reference-based evaluation: {e}")
        # Fallback to simple scoring
        per_question_scores = []
        for answer in candidate_answers:
            if answer == "No answer provided." or len(answer.strip()) < 3:
                per_question_scores.append(0.0)
            else:
                per_question_scores.append(5.0)  # Default middle score
        
        total_score = sum(per_question_scores)
        max_possible = len(questions) * 10
        overall_score = round((total_score / max_possible) * 100, 2) if max_possible > 0 else 0.0
        
        return {
            "overall_score": overall_score,
            "per_question_scores": per_question_scores,
            "detailed_feedback": ["Evaluation error occurred"] * len(questions),
            "candidate_answers": candidate_answers,
            "reference_answers": correct_answers
        }


def display_detailed_evaluation_results(evaluation_result: Dict[str, Any], questions: List[str]):
    """
    Display detailed evaluation results with per-question breakdown.
    
    Args:
        evaluation_result: Result from evaluate_with_reference_answers
        questions: List of interview questions
    """
    import streamlit as st
    
    overall_score = evaluation_result.get("overall_score", 0.0)
    per_question_scores = evaluation_result.get("per_question_scores", [])
    detailed_feedback = evaluation_result.get("detailed_feedback", [])
    candidate_answers = evaluation_result.get("candidate_answers", [])
    reference_answers = evaluation_result.get("reference_answers", [])
    
    # Overall score display
    st.metric("🧠 Interview Score", f"{overall_score:.1f}%")
    st.progress(overall_score / 100)
    
    # Per-question breakdown
    st.subheader("📋 Detailed Question Analysis")
    
    for i, question in enumerate(questions):
        score = per_question_scores[i] if i < len(per_question_scores) else 0.0
        feedback = detailed_feedback[i] if i < len(detailed_feedback) else "No feedback available"
        candidate_answer = candidate_answers[i] if i < len(candidate_answers) else "No answer"
        reference_answer = reference_answers[i] if i < len(reference_answers) else "No reference"
        
        with st.expander(f"Question {i+1}: {question[:50]}... (Score: {score}/10)"):
            st.markdown(f"**Your Answer:** {candidate_answer}")
            st.markdown(f"**Score:** {score}/10")
            st.markdown(f"**Feedback:** {feedback}")
            
            if st.checkbox(f"Show reference answer for Question {i+1}", key=f"ref_{i}"):
                st.markdown(f"**Reference Answer:** {reference_answer}")
