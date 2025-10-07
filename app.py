import streamlit as st
from modules.file_utils import extract_text_from_file
from modules.matcher import compute_match

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Resume ↔ AI Interview",
    page_icon="🧠",
    layout="wide"
)

# -------------------------------
# Header & Sidebar
# -------------------------------
st.title("🧠 Resume ↔ AI Interview")
st.caption("AI-powered Resume Screening — Compare Resume with Job Description and get Match Score!")

with st.sidebar:
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. Upload your **resume (PDF/DOCX/TXT)**.  
    2. Paste the **job description** in the box.  
    3. Click **Analyze** to get results.
    """)

# -------------------------------
# Main Layout
# -------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_resume = st.file_uploader(
        "📂 Upload your Resume",
        type=["pdf", "docx", "doc", "txt"]
    )
    analyze_btn = st.button("🚀 Analyze Resume")

with col2:
    job_post = st.text_area(
        "💼 Paste Job Description",
        height=300,
        placeholder="Paste the full job posting text here..."
    )

# -------------------------------
# Resume & Job Matching Logic
# -------------------------------
if analyze_btn:
    if not uploaded_resume:
        st.error("❌ Please upload a resume file first.")
    elif not job_post.strip():
        st.error("❌ Please paste the job posting text.")
    else:
        try:
            with st.spinner("📄 Extracting text from resume..."):
                resume_text = extract_text_from_file(uploaded_resume)

            if not resume_text.strip():
                st.error("⚠️ Could not extract text from this resume. Try a text-based version.")
            else:
                with st.spinner("🧮 Computing AI-based match score..."):
                    result = compute_match(job_post, resume_text)

                # -------------------------------
                # Display Results
                # -------------------------------
                st.header("📊 Match Analysis Results")

                st.metric("✅ Overall Match Score", f"{result['overall_score']} / 100")
                st.progress(result['overall_score'] / 100)

                st.subheader("🧩 Required Skills (from Job Posting)")
                st.write(result.get("required_skills") or "No specific skills found in job description.")

                st.subheader("💪 Skills Found in Resume")
                st.write(result.get("user_skills") or "No skills found in resume.")

                st.subheader("⚠️ Missing Skills")
                st.write(result.get("missing_skills") or "No missing skills — Great Match!")

                st.subheader("🧠 Experience Summary")
                experiences = result.get("experiences", [])
                if experiences:
                    for exp in experiences:
                        st.markdown(
                            f"**{exp.get('position', 'Unknown Role')}** @ {exp.get('company', '')} "
                            f"({exp.get('start_date', '?')} - {exp.get('end_date', '?')})"
                        )
                        st.write(f"- 🕒 Duration: {exp.get('duration_years', '?')} years")
                        st.write(f"- 🧾 Summary: {exp.get('description', '')}")
                        st.markdown("---")
                else:
                    st.write("No work experience found in the resume.")

                st.info(f"🔍 Semantic Similarity: **{result.get('semantic_similarity', 0):.2f}**")
                st.success("🎯 Analysis complete!")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
