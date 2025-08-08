import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load .env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx")
NAMESPACE = "hackrx-docs"

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing. Please check your .env file")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Function to chunk text
def chunk_text(text, max_length=1000):
    sentences = text.split(".")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_length:
            current += sentence + "."
        else:
            chunks.append(current.strip())
            current = sentence + "."
    if current:
        chunks.append(current.strip())
    return chunks

# ✅ Fixed: Upsert records using Pinecone built-in text-index format
def store_chunks_to_pinecone(chunks, namespace=NAMESPACE):
    BATCH_SIZE = 96
    total_chunks = len(chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        records = [
            {
                "id": f"chunk-{i+j}",
                "values": None,
                "metadata": {
                    "chunk_text": chunk
                }
            }
            for j, chunk in enumerate(batch)
        ]

        try:
            index.upsert(vectors=records, namespace=namespace)
            print(f"✅ Upserted batch {i // BATCH_SIZE + 1}")
        except Exception as e:
            print(f"❌ Pinecone upsert failed at batch {i // BATCH_SIZE + 1}: {e}")

# ✅ Fixed: Search using Pinecone built-in text input
def get_relevant_chunks(query, namespace=NAMESPACE, top_k=8):
    try:
        results = index.query(
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            vector=None,
            filter=None,
            id=None,
            text=query  # 🔥 Key line: use text directly (Pinecone inbuilt embed model)
        )
        return [match["metadata"]["chunk_text"] for match in results["matches"]]
    except Exception as e:
        print(f"❌ Error during Pinecone search: {e}")
        return []
