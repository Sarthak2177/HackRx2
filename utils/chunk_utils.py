import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load .env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx")  # match what you created
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")  # not needed for new text-index
NAMESPACE = "hackrx-docs"  # you can rename as needed

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing. Please check your .env file")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
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

# Function to upsert chunks using Pinecone's built-in embedding
def store_chunks_to_pinecone(chunks, namespace=NAMESPACE):
    BATCH_SIZE = 96
    total_chunks = len(chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]

        # ✅ Each record must have _id and "chunk_text"
        records = [
            {
                "_id": f"chunk-{i+j}",
                "chunk_text": chunk
            }
            for j, chunk in enumerate(batch)
        ]

        try:
            index.upsert_records(namespace=namespace, records=records)
            print(f"✅ Upserted batch {i // BATCH_SIZE + 1}")
        except Exception as e:
            print(f"❌ Pinecone upsert failed at batch {i // BATCH_SIZE + 1}: {e}")

# Function to retrieve relevant chunks via semantic search
def get_relevant_chunks(query, namespace=NAMESPACE, top_k=8):
    try:
        results = index.search(
            namespace=namespace,
            query={
                "top_k": top_k,
                "inputs": {
                    "text": query
                }
            }
        )
        hits = results["result"]["hits"]
        return [hit["fields"]["chunk_text"] for hit in hits]
    except Exception as e:
        print(f"❌ Error during Pinecone search: {e}")
        return []
