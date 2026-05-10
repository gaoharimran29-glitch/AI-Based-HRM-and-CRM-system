import os
from langchain_community.document_loaders import PyPDFLoader

def extract_text_from_pdf(file_path: str) -> str:
    """Extract and returns clean text from jd or resume"""

    if isinstance(file_path, str) and not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        text = "\n\n".join([doc.page_content for doc in docs])

    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    return text.strip()
