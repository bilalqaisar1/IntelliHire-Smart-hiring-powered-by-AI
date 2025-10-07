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
