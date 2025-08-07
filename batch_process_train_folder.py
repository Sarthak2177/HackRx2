import os
import hashlib
from dotenv import load_dotenv
from utils.chunk_utils import chunk_text, store_chunks_to_pinecone
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

TRAIN_DIR = "./train"

def extract_text_from_pdf(path):
    with open(path, "rb") as f:
        reader = PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

def process_file(path):
    print(f"📄 Processing: {path}")
    try:
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text)
        namespace = hashlib.md5(path.encode()).hexdigest()
        print(f"🧠 Sending {len(chunks)} chunks to Pinecone from: {path}")
        store_chunks_to_pinecone(chunks, namespace=namespace)
        print("✅ Done.\n")
    except Exception as e:
        print(f"❌ Failed to process {path}: {e}")

if __name__ == "__main__":
    pdf_files = [os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(".pdf")]
    print(f"📄 Found {len(pdf_files)} PDF files")
    for file_path in pdf_files:
        process_file(file_path)
