from utils.chunk_utils import read_pdf_text, chunk_text, store_chunks_to_pinecone
import os

TRAIN_FOLDER = "./train"

for file_name in os.listdir(TRAIN_FOLDER):
    if file_name.lower().endswith(".pdf"):
        file_path = os.path.join(TRAIN_FOLDER, file_name)
        print(f"📄 Processing {file_name}...")
        text = read_pdf_text(file_path)
        chunks = chunk_text(text)
        store_chunks_to_pinecone(chunks, file_name)
