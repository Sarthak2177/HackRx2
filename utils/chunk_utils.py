import uuid
from PyPDF2 import PdfReader
from utils.index import pc, PINECONE_INDEX_NAME

# 🔒 Always use one namespace for this project
NAMESPACE = "hackrx"

def read_pdf_text(file_path):
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text.strip() + "\n"
    return text.strip()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def store_chunks_to_pinecone(chunks, file_name):
    """
    Store text chunks in Pinecone using server-side embeddings.
    file_name is stored in metadata for future filtering.
    """
    index = pc.Index(PINECONE_INDEX_NAME)
    records = [
        {
            "_id": str(uuid.uuid4()),
            "chunk_text": chunk,
            "source_doc": file_name  # keep track of which file this chunk came from
        }
        for chunk in chunks
    ]
    batch_size = 96
    uploaded = 0
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            index.upsert_records(NAMESPACE, batch)
            uploaded += len(batch)
        except Exception as e:
            print(f"❌ Pinecone upsert_records failed for batch starting at {i}: {e}")
    print(f"✅ Finished upserting {uploaded}/{len(records)} chunks to namespace '{NAMESPACE}'")

def get_relevant_chunks(query: str, top_k: int = 15):
    """Search Pinecone for most relevant chunks for a query."""
    index = pc.Index(PINECONE_INDEX_NAME)
    results = index.search(
        namespace=NAMESPACE,
        query={
            "top_k": top_k,
            "inputs": {
                "text": query
            }
        }
    )

    chunks = []
    hits = results.get("result", {}).get("hits", [])
    for hit in hits:
        fields = hit.get("fields", {})
        if "chunk_text" in fields:
            chunks.append(fields["chunk_text"])
    return chunks
