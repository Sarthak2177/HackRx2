import requests
from PyPDF2 import PdfReader
import tempfile
import os

def extract_text_from_pdf(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.strip() + "\n"
        return text.strip()
    finally:
        os.remove(tmp_path)  # Clean up temp file
