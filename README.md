# IntelliHire-Smart-hiring-powered-by-AI
# 🧠 Resume ↔ To ↔ Interview

An intelligent **resume–job matching system** built with **Streamlit**, **OpenAI GPT**, and **LangChain**.  
The app analyzes how well a candidate’s resume matches a given job posting — step by step — based on **skills**, **experience**, and **semantic similarity**.

---

## 🚀 Features

✅ Upload your **resume** (`.pdf`, `.docx`, `.txt`)  
✅ Paste any **job posting / description**  
✅ Automatically extract:
- Required experience & skills from the job posting  
- Candidate’s skills and experiences from the resume  
✅ Compute:
- **Skill match percentage**  
- **Experience match percentage**  
- **Overall match score (0–100)**  
- **Semantic similarity** between job posting and resume  
✅ Display a clean **step-by-step analysis**:
1. Overall matching score  
2. Required experience vs. user experience  
3. Required skills  
4. User’s skills  
5. Missing skills  
6. Experience listing (position, duration, and description)

---

## 🧩 Tech Stack

- **Python 3.9+**
- **Streamlit** – UI framework
- **OpenAI API (GPT + Embeddings)** – for text understanding and scoring
- **LangChain Community** – for PDF parsing (`PyPDFLoader`)
- **python-docx** – for extracting text from Word documents
- **NumPy** – for similarity computation
- **python-dotenv** – for environment variable management

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/resume-to-interview.git
cd resume-to-interview
````

### 2️⃣ Create & activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On macOS/Linux
```

### 3️⃣ Install dependencies

```bash
pip install streamlit python-docx openai langchain langchain-community numpy python-dotenv
```

### 4️⃣ Set up environment variables

Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🧠 Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

Then open the displayed URL (usually `http://localhost:8501`) in your browser.

1. Upload your resume file.
2. Paste the job posting text.
3. Click **Analyze**.
4. View your step-by-step matching report.

---

## 📊 Output Example

**Step-by-step Analysis Example:**

| Step | Description                                                     |
| ---- | --------------------------------------------------------------- |
| 1️⃣  | Overall matching score → `82 / 100`                             |
| 2️⃣  | Required exp: `3 years` → User exp: `4.5 years`                 |
| 3️⃣  | Required skills: `['Python', 'SQL', 'Docker']`                  |
| 4️⃣  | User skills: `['Python', 'Pandas', 'SQL', 'Flask']`             |
| 5️⃣  | Missing skills: `['Docker']`                                    |
| 6️⃣  | Experience listing: Data Analyst @ ABC Corp (Jan 2020–Mar 2023) |

---

## 💡 How It Works

1. **Text Extraction** – Extract text from uploaded resume and job posting.
2. **Information Extraction** – GPT parses job and resume data into structured JSON.
3. **Skill & Experience Matching** – Compare sets and compute match percentages.
4. **Embeddings Similarity** – Generate embeddings and compute cosine similarity.
5. **Weighted Scoring** – Combine metrics into a final overall score:

   ```
   overall_score = 0.6 * skill_match + 0.3 * experience_match + 0.1 * semantic_similarity
   ```

---

## 🛠 Future Improvements

* Integrate **Hugging Face local models** to reduce API cost
* Generate **interview questions** based on missing skills
* Add **visual charts** for match visualization
* Support **multiple resumes and jobs** comparison

---

## 👨‍💻 Author

**Bilal Qaisar**
🎓 AI Engineer @ Primo Dialler | BS Data Science @ PUCIT
📧 [LinkedIn](https://www.linkedin.com/in/bilalqaisar) • [GitHub](https://github.com/bilalqaisar)

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use and modify it.
