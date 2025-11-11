import os
import re
import json
from typing import Optional, Dict, Any, List
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env or environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

CHAT_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-small"

def call_chat_json(system_prompt: str, user_prompt: str, max_tokens: int = 800) -> Optional[Dict[str, Any]]:
    """Calls OpenAI chat model and returns JSON response."""
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:\w+)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception:
        return None

def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def compare_skills_openai(resume_skills, job_skills):
    """
    Uses OpenAI to intelligently compare skill lists and return structured match info.
    Handles synonymous and related skill mappings (e.g., MySQL → SQL, Power BI → BI Tools).
    """

    sys_prompt = (
        "You are a precise and intelligent skill comparison engine. "
        "Your goal is to compare two skill lists (Resume Skills vs Job Description Skills) "
        "and return a STRICT JSON output that accurately reflects how well they match. "
        "Return ONLY JSON — no text, no commentary, no explanations."
    )

    user_prompt = f"""
    Analyze and compare the following two skill lists to evaluate the skill match between
    a candidate's resume and a job description.

    ### INPUTS ###
    Resume Skills: {resume_skills}
    Job Description Skills: {job_skills}

    ### COMPARISON LOGIC & RULES ###
    1️⃣ **Matching Skills:**
       - List all skills that are present or have clear synonyms / equivalences between both lists.
       - Consider synonymous, equivalent, or hierarchical relationships. Examples:
         - "MySQL", "PostgreSQL", "Oracle" → count all as "SQL".
         - "MS Excel", "Advanced Excel" → count as "Excel".
         - "Power BI", "Tableau", "Looker" → consider them all under "BI / Data Visualization Tools".
         - "Machine Learning", "ML", "Deep Learning", "AI" → consider as related and matching if contextually similar.
         - "Pandas", "NumPy", "Matplotlib" → all are Python data libraries; match if related tools exist.
         - "TensorFlow", "PyTorch", "Keras" → deep learning frameworks; match if any appear across lists.
         - "AWS", "Azure", "Google Cloud" → treat as "Cloud Platforms".
         - "Communication", "Teamwork", "Collaboration" → treat as soft skill matches.
       - Case-insensitive comparison.

    2️⃣ **Missing Skills:**
       - List all skills present in the Job Description but missing or unmatched in the Resume.
       - Apply the same synonym and related-skill logic before deciding a skill is missing.

    3️⃣ **Match Percentage:**
       - Compute based on the ratio of matching skills to total job-required skills:
         match_percentage = (number_of_matching_skills / total_job_skills) * 100
       - Round to 2 decimal places.

    4️⃣ **Output Format:**
       Return STRICT JSON only in this exact format (no additional text):
       {{
         "matching_skills": ["Python","SQL","Machine Learning"],
         "missing_skills": ["AWS","Communication"],
         "match_percentage": 80.0
       }}

    ### Output Guidelines ###
    - Output MUST be valid JSON only.
    - Do not include any explanations or comments.
    - Ensure consistent skill capitalization (e.g., "Python", "SQL", "Power BI").
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0
    )

    result_text = response.choices[0].message.content.strip()

    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from model", "raw_output": result_text}


# ----------------------------------------
# Function 1: Get matched skills
# ----------------------------------------
def get_matched_skills(result: dict):
    """Return a list of skills that are present both in required_skills and user_skills."""
    required = set(map(str.lower, result.get("required_skills", [])))
    user = set(map(str.lower, result.get("user_skills", [])))
    matched = list(required.intersection(user))
    return matched


# ----------------------------------------
# Function 2: Generate 2 hard questions per skill
# ----------------------------------------
def generate_questions_for_skills(matched_skills):
    """Generate 1 hard-level interview questions per matched skill using OpenAI.
    Ensures only actual question sentences are returned (no skill headings).
    """
    if not matched_skills:
        return []

    # Tightened instructions to avoid emitting skill names/headings and enforce questions only
    prompt = (
        "For each skill in the list, write exactly 2 hard technical interview questions "
        "that assess deep understanding and practical application. Do not include the skill "
        "names, sections, headings, or any explanatory text. Output a single flat numbered "
        "list of questions only. Each item must be a single question sentence ending with a question mark.\n\n"
        f"Skills: {', '.join(matched_skills)}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert interviewer who outputs only question lines, no headings or skill names."
            )},
            {"role": "user", "content": prompt},
        ]
    )

    questions_text = response.choices[0].message.content.strip()

    # Split lines and clean numbering
    raw_lines = [line.strip() for line in questions_text.split("\n") if line.strip()]

    cleaned_questions = []
    numeric_prefixes = tuple([f"{i}." for i in range(1, 101)])
    skill_set_lower = {s.strip().lower() for s in matched_skills}

    for line in raw_lines:
        # Drop markdown bullets and numeric prefixes
        if line.startswith(numeric_prefixes):
            line = line.split(".", 1)[1].strip()
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            line = line[2:].strip()

        # Filter out lines that are just skill names or don't look like questions
        lower = line.lower().rstrip("?:!.")
        if lower in skill_set_lower:
            continue
        if "?" not in line:
            continue
        # Avoid lines that start with the skill name followed by a colon (headings)
        if any(line.lower().startswith(f"{s.lower()}:") for s in matched_skills):
            continue

        cleaned_questions.append(line)

    return cleaned_questions


# ----------------------------------------
# Function 3: Evaluate user's answers
# ----------------------------------------
def evaluate_answers(questions, answers):
    """
    Evaluate user's answers using OpenAI and return an overall score (0–100).
    Scoring is based on technical accuracy, completeness, reasoning, and clarity.
    """
    if not questions or not answers:
        return 0

    qa_pairs = "\n\n".join(
        [f"Question {i+1}: {questions[i]}\nCandidate Answer: {answers.get(i, 'No answer provided.')}"
         for i in range(len(questions))]
    )

    prompt = f"""
You are acting as a **senior technical interviewer** for a company evaluating candidates for a data or AI-related role.
Your goal is to **score the candidate's answers** as a human interviewer would — being detailed, fair, and consistent.

Here are the evaluation rules:

1. **Scoring Range:** Return a single numeric score between 0 and 100 (float allowed, e.g., 84.5).
2. **Evaluation Criteria (each 0–10):**
   - **Correctness:** How factually and technically accurate the answer is.
   - **Depth & Understanding:** Does the answer show deep conceptual understanding, not just memorization?
   - **Clarity & Structure:** Is the explanation clear, concise, and logically structured?
   - **Relevance:** Does the answer directly address the question and stay on-topic?
   - **Practical Insight (Bonus):** Does the candidate demonstrate applied or real-world thinking?

3. **Weightage:**
   - Each question should be scored out of 10 (based on the above factors).
   - If the candidate didn’t answer (“No answer provided”), give **0 for that question**.
   - Combine all per-question scores and scale to an overall score out of 100.

4. **Important Constraints:**
   - Be unbiased and consistent across all questions.
   - Do **not** explain or justify scores — only return the **final numeric score**.
   - Do **not** include any text other than the score.

Now evaluate the following candidate responses:

{qa_pairs}

Return only the final numeric total score (0–100).
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are a professional technical interviewer for data science, AI, and software engineering roles. "
                "You assess candidates' answers based on correctness, reasoning, and clarity with human-like judgment."
            )},
            {"role": "user", "content": prompt},
        ],
    )

    score_text = response.choices[0].message.content.strip()

    try:
        score = float(score_text)
        score = min(max(score, 0), 100)  # clamp to 0–100
    except ValueError:
        score = 0

    return round(score, 2)


# ----------------------------------------
# Function 4: Strict evaluation with per-question scores
# ----------------------------------------
def evaluate_answers_strict(questions, answers):
    """
    Strictly evaluate across ALL questions. Any unanswered question is scored 0.
    Returns overall score out of 100 (float, 2 decimals).
    """
    if not questions:
        return 0.0

    # Build list of candidate answers aligned to questions, fill with explicit marker
    candidate_answers = []
    for i in range(len(questions)):
        ans = (answers.get(i) or "").strip()
        # Check for truly blank answers (empty, whitespace, or very short responses)
        if not ans or len(ans) < 3:
            candidate_answers.append("No answer provided.")
        else:
            candidate_answers.append(ans)

    # If all are unanswered, return 0 immediately without calling the model
    if all(a == "No answer provided." for a in candidate_answers):
        return 0.0

    # Ask model to return a strict JSON array of per-question scores (0-10 per item)
    scoring_instructions = (
        "You are a strict technical interviewer. Score EACH question independently on a 0–10 scale. "
        "CRITICAL: If the answer is 'No answer provided.' or contains only whitespace/symbols, you MUST return 0 for that question. "
        "Only give points for substantial, meaningful answers that demonstrate knowledge. "
        "Return ONLY a JSON array of numbers (no extra text), length must equal the number of questions."
    )

    qa_payload = []
    for i in range(len(questions)):
        qa_payload.append({
            "question_index": i,
            "question": questions[i],
            "answer": candidate_answers[i],
        })

    sys_prompt = scoring_instructions
    user_prompt = (
        "Score the following question-answer pairs. Output JSON array of scores (0-10), "
        "one per item, matching the order given. IMPORTANT: Give 0 for any blank, empty, or 'No answer provided' responses. "
        "Only award points for substantial answers that show technical knowledge. Do not include explanations.\n\n"
        f"Items: {json.dumps(qa_payload)}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        # Try to parse as JSON list
        try:
            # Clean backticks if model wraps
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:\w+)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            per_question = json.loads(raw)
            if not isinstance(per_question, list):
                raise ValueError("Expected JSON list of numbers")
            # Normalize and clamp each score
            clean_scores = []
            for idx in range(len(questions)):
                val = 0.0
                if idx < len(per_question):
                    try:
                        val = float(per_question[idx])
                    except Exception:
                        val = 0.0
                # Enforce 0 if unanswered regardless of model output
                if candidate_answers[idx] == "No answer provided." or len(candidate_answers[idx].strip()) < 3:
                    val = 0.0
                # Clamp to 0–10
                val = max(0.0, min(10.0, val))
                clean_scores.append(val)

            total_obtained = sum(clean_scores)
            max_total = float(len(questions) * 10)
            overall = round((total_obtained / max_total) * 100.0, 2) if max_total > 0 else 0.0
            return overall
        except Exception:
            # Fallback: deterministic zeros for unanswered or blank answers
            clean_scores = []
            for a in candidate_answers:
                if a == "No answer provided." or not a.strip():
                    clean_scores.append(0.0)
                else:
                    # For answered questions, give a minimal score only if there's substantial content
                    if len(a.strip()) > 10:  # Only if answer has meaningful content
                        clean_scores.append(2.0)  # Minimal score for any attempt
                    else:
                        clean_scores.append(0.0)  # Too short to be meaningful
            
            total_obtained = sum(clean_scores)
            max_total = float(len(questions) * 10)
            return round((total_obtained / max_total) * 100.0, 2) if max_total > 0 else 0.0
    except Exception:
        # Last-resort fallback: zero for any unanswered, 0 for all if error
        zeros = [0.0 for _ in questions]
        total_obtained = sum(zeros)
        max_total = float(len(questions) * 10)
        return round((total_obtained / max_total) * 100.0, 2) if max_total > 0 else 0.0