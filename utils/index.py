# utils/index.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load .env variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx")

if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY is missing in .env")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index with integrated model
if not pc.has_index(PINECONE_INDEX_NAME):
    print(f"📦 Creating index '{PINECONE_INDEX_NAME}' with integrated model llama-text-embed-v2...")
    pc.create_index_for_model(
        name=PINECONE_INDEX_NAME,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}  # match upsert field name
        }
    )
    print(f"✅ Index '{PINECONE_INDEX_NAME}' created.")
else:
    print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists.")
