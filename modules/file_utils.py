import os
import io
import docx
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF using PyPDFLoader."""
    temp_path = "temp_resume.pdf"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
    finally:
        os.remove(temp_path)
    return text.strip()

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extracts text from a DOCX file."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def extract_text_from_file(uploaded_file) -> str:
    """Detects file type and extracts text."""
    content = uploaded_file.read()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(content)
    elif name.endswith((".docx", ".doc")):
        return extract_text_from_docx(content)
    else:
        try:
            return content.decode("utf-8")
        except Exception:
            return ""
