from typing import Dict, Any
from .openai_utils import call_chat_json
from .regex_utils import find_years_in_text

def extract_job_requirements(job_text: str) -> Dict[str, Any]:
    """Extracts required experience, skills, and title from a job posting."""
    
    sys_prompt = (
        "You are a highly accurate and detail-oriented job information extractor. "
        "Your sole task is to read the given job posting text and return a STRICT JSON output "
        "containing complete and exhaustive information for the requested fields. "
        "Do not summarize, interpret, or omit any relevant skills or keywords. "
        "Return only JSON — no explanations, no extra text."
    )

    user_prompt = (
        "Analyze the following JOB POSTING text and extract the requested information carefully. "
        "Your goal is to extract every possible required skill, tool, technology, and keyword "
        "that represents the requirements or expectations of the employer.\n\n"

        "### Extraction Requirements ###\n"
        "1️⃣ required_experience_years: Extract or estimate the total required professional experience (in years, float or int). "
        "If multiple experience ranges are mentioned, choose the most relevant or general one (e.g., '3–5 years' → 4). "
        "If no duration is found, return null.\n\n"

        "2️⃣ required_skills: Return a JSON array of all explicit and implicit required skills mentioned in the job posting. "
        "Include:\n"
        "   - Programming languages (e.g., Python, R, Java, SQL)\n"
        "   - Frameworks, tools, and technologies (e.g., TensorFlow, Power BI, Tableau, Excel)\n"
        "   - Platforms or environments (e.g., AWS, Azure, Google Cloud)\n"
        "   - Data-related, analytical, or engineering terms (e.g., ETL, machine learning, data pipelines)\n"
        "   - Domain or business tools (e.g., Salesforce, SAP, Jira)\n"
        "   - Soft or professional skills (e.g., communication, leadership, problem-solving)\n"
        "   - Abbreviations and alternate spellings (e.g., 'MS Excel' = 'Excel')\n\n"
        "   Be exhaustive — include all variations and separate each unique skill or keyword as its own entry.\n\n"

        "3️⃣ title: Extract the official job title or role (e.g., 'Data Scientist', 'Machine Learning Engineer'). "
        "If multiple titles are mentioned, choose the most prominent or first mentioned one. "
        "If unclear, return null.\n\n"

        "Ensure that the final JSON strictly contains the following keys: "
        "required_experience_years, required_skills, title.\n\n"

        "### Example Output ###\n"
        '{"required_experience_years": 3, "required_skills":["Python","SQL","Machine Learning","Power BI","Communication"], '
        '"title":"Data Analyst"}\n\n'

        "### Job Posting Text ###\n"
        f"{job_text}\n\n"
        "Output STRICTLY as valid JSON — no additional explanations, no commentary, and no formatting beyond pure JSON."
    )

    parsed = call_chat_json(sys_prompt, user_prompt, max_tokens=400)
    if parsed is None:
        req_years = find_years_in_text(job_text)
        parsed = {
            "required_experience_years": req_years,
            "required_skills": [],
            "title": None
        }
    return parsed

def extract_resume_profile(resume_text: str) -> Dict[str, Any]:
    """Extracts total experience, skills, and experiences from resume."""
    sys_prompt = (
        "You are a highly accurate and detail-oriented resume information extractor. "
        "Your only task is to read the given resume text and produce a STRICT JSON output "
        "containing complete and exhaustive information for the requested fields. "
        "Do not summarize or omit any skills. Return only JSON, no explanations or text."
    )

    user_prompt = (
        "Analyze the following RESUME text and extract the requested information carefully. "
        "Your goal is to extract every possible skill, tool, technology, and keyword that "
        "represents a professional capability, whether technical, analytical, or soft skill.\n\n"

        "### Extraction Requirements ###\n"
        "1️⃣ total_experience_years: Estimate the total professional experience (in years, float or int). "
        "If uncertain, return null.\n\n"

        "2️⃣ skills: Return a JSON array of all explicit and implicit skills mentioned, including:\n"
        "   - Programming languages (e.g., Python, R, Java, SQL)\n"
        "   - Libraries, frameworks, and tools (e.g., Pandas, TensorFlow, Power BI, Excel)\n"
        "   - Data-related and analytical terms (e.g., ETL, data cleaning, machine learning)\n"
        "   - Soft and professional skills (e.g., teamwork, communication, problem-solving)\n"
        "   - Abbreviations or variants (e.g., 'MS Excel' = 'Excel', 'BI Tools' includes 'Power BI')\n"
        "   - Cloud, DevOps, or platform names (e.g., AWS, Azure, Docker)\n\n"
        "   Be exhaustive. Do NOT skip or merge similar items. "
        "If two different skill names appear, include both as separate entries.\n\n"

        "3️⃣ experiences: Extract an array of objects, each including:\n"
        "   - position: job title or role (e.g., Data Analyst)\n"
        "   - company: company name if available, else null\n"
        "   - start_date: month-year or year text (e.g., 'Jan 2020' or '2019')\n"
        "   - end_date: month-year, 'Present', or null\n"
        "   - duration_years: approximate numeric value if possible, else null\n"
        "   - description: concise summary of responsibilities or achievements\n\n"

        "If no experience details are found, return an empty array for 'experiences'.\n\n"

        "Ensure the JSON keys are exactly as follows: "
        "total_experience_years, skills, experiences.\n\n"

        "### Example Output ###\n"
        '{"total_experience_years": 4.5, "skills":["Python","Pandas","SQL","Power BI","Excel","Data Cleaning"], '
        '"experiences":[{"position":"Data Analyst","company":"ABC Corp","start_date":"Jan 2020","end_date":"Mar 2023","duration_years":3.17,"description":"Worked on ETL, dashboards, and data automation"}]}\n\n'

        "### Resume Text ###\n"
        f"{resume_text}\n\n"
        "Output STRICTLY as valid JSON — no additional comments or formatting."
    )

    parsed = call_chat_json(sys_prompt, user_prompt, max_tokens=800)
    if parsed is None:
        total_exp = find_years_in_text(resume_text)
        parsed = {"total_experience_years": total_exp, "skills": [], "experiences": []}
    return parsed
