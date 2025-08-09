from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os
import re
import asyncio
from dotenv import load_dotenv
from utils.dynamic_decision import DynamicDecisionEngine
from utils.extract_text_from_pdfs import extract_text_from_pdf as download_pdf_and_extract_text
from utils.chunk_utils import chunk_text, store_chunks_to_pinecone, get_relevant_chunks
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx")
index = pc.Index(index_name)

# FastAPI app setup
app = FastAPI()
security = HTTPBearer()
decision_engine = DynamicDecisionEngine()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    documents: str
    questions: List[str] = []

class QueryResponse(BaseModel):
    answers: List[str]
    success: bool

# Extract questions from text
def extract_questions_from_text(text: str, max_q: int = 10) -> List[str]:
    question_words = (
        "what", "how", "why", "can", "does", "is", "are", "do",
        "should", "could", "when", "who", "where", "which", "will", "would"
    )
    lines = re.findall(r"[^\n\r]+?[?]", text)
    questions = [
        line.strip() for line in lines
        if line.strip().lower().startswith(question_words) and len(line.strip()) > 20
    ]
    return questions[:max_q]

# Format answers into clean readable sentences
def format_answers(raw_answers: list[str]) -> list[str]:
    formatted = []
    for ans in raw_answers:
        if not ans or ans.strip().lower() in ["no", "yes"]:
            if ans.strip().lower() == "yes":
                formatted.append("Yes, this is covered under the policy with specific conditions.")
            elif ans.strip().lower() == "no":
                formatted.append("No, this is not covered under the policy.")
            else:
                formatted.append("Information not available in the policy document.")
        else:
            ans_clean = ans.strip()
            if not ans_clean.endswith("."):
                ans_clean += "."
            formatted.append(ans_clean)
    return formatted

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_decision_engine(
    payload: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    try:
        # Extract raw text from the provided PDF document
        raw_text = download_pdf_and_extract_text(payload.documents)
        chunks = chunk_text(raw_text)

        # Always store chunks into hackrx namespace, passing file_name for metadata
        file_name = os.path.basename(payload.documents)
        store_chunks_to_pinecone(chunks, file_name)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    # If no questions provided, auto-extract them from text
    if not payload.questions:
        payload.questions = extract_questions_from_text(raw_text)

    try:
        batch_size = 2
        tasks = []
        for i in range(0, len(payload.questions), batch_size):
            batch_questions = payload.questions[i:i+batch_size]

            # Get relevant chunks from Pinecone for the question batch
            relevant_chunks = get_relevant_chunks("\n\n".join(batch_questions))

            task = asyncio.create_task(
                process_question_batch(batch_questions, relevant_chunks)
            )
            tasks.append(task)

        all_answers = await asyncio.gather(*tasks)
        answers = [ans for batch in all_answers for ans in batch]

    except Exception as e:
        print("❌ Error during question processing:", str(e))
        answers = [f"LLM processing failed: {str(e)}"] * len(payload.questions)

    return {
        "answers": answers,
        "success": True
    }

async def process_question_batch(batch_questions: List[str], relevant_chunks: List[str]) -> List[str]:
    max_chunks = 15
    trimmed_chunks = [chunk[:1000] for chunk in relevant_chunks[:max_chunks]]

    # Send the question batch + chunks to the decision engine (LLM)
    result = decision_engine.make_decision_from_context(
        "\n".join(batch_questions), {}, trimmed_chunks
    )

    print("🧠 Raw LLM response:\n", result)

    if result is None or (isinstance(result, str) and not result.strip()):
        return ["LLM returned empty response"] * len(batch_questions)

    try:
        parsed_result = result if isinstance(result, dict) else json.loads(result)

        if isinstance(parsed_result, dict) and "answers" in parsed_result:
            return format_answers([str(a).strip() for a in parsed_result["answers"]])

        return format_answers([str(parsed_result).strip()])

    except Exception as e:
        print("❌ Parsing error:", str(e))
        print("❗Failed content:\n", result)
        return [f"LLM parsing failed: {str(e)}"] * len(batch_questions)
