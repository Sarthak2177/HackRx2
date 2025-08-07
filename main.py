from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Any
import json
import os
import re
import time
import hashlib
import asyncio
from dotenv import load_dotenv
from utils.dynamic_decision import DynamicDecisionEngine
from utils.extract_text_from_pdfs import extract_text_from_pdf as download_pdf_and_extract_text
from utils.chunk_utils import chunk_text, store_chunks_to_pinecone, get_relevant_chunks

load_dotenv()

app = FastAPI()
security = HTTPBearer()
decision_engine = DynamicDecisionEngine()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str] = []

class QueryResponse(BaseModel):
    answers: List[str]
    response_time_seconds: float
    success: bool

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

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_decision_engine(
    payload: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    start_time = time.time()

    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    try:
        raw_text = download_pdf_and_extract_text(payload.documents)
        chunks = chunk_text(raw_text)
        namespace = hashlib.md5(payload.documents.encode()).hexdigest()

        existing_chunks = get_relevant_chunks("check", namespace=namespace)
        if len(existing_chunks) < 5:
            store_chunks_to_pinecone(chunks, namespace=namespace)
        else:
            print("✅ Chunks already exist, skipping upsert.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    if not payload.questions:
        payload.questions = extract_questions_from_text(raw_text)

    try:
        batch_size = 2
        tasks = []

        for i in range(0, len(payload.questions), batch_size):
            batch_questions = payload.questions[i:i+batch_size]
            relevant_chunks = get_relevant_chunks("\n\n".join(batch_questions), namespace=namespace)
            task = asyncio.create_task(
                process_question_batch(batch_questions, relevant_chunks)
            )
            tasks.append(task)

        all_answers = await asyncio.gather(*tasks)
        answers = [ans for batch in all_answers for ans in batch]

    except Exception as e:
        print("❌ Error during question processing:", str(e))
        answers = [f"LLM processing failed: {str(e)}"] * len(payload.questions)

    response_time = round(time.time() - start_time, 2)
    return {"answers": answers, "response_time_seconds": response_time, "success": True}


async def process_question_batch(batch_questions: List[str], relevant_chunks: List[str]) -> List[str]:
    max_chunks = 15
    trimmed_chunks = [chunk[:600] for chunk in relevant_chunks[:max_chunks]]

    result = decision_engine.make_decision_from_context("\n".join(batch_questions), {}, trimmed_chunks)
    print("🧠 Raw LLM response:\n", result)

    if result is None or (isinstance(result, str) and not result.strip()):
        return ["LLM returned empty response"] * len(batch_questions)

    try:
        parsed_result = result if isinstance(result, dict) else json.loads(result)

        def safe_strip(val):
            if isinstance(val, str):
                return val.strip()
            elif isinstance(val, dict):
                return json.dumps(val)
            return str(val).strip()

        if isinstance(parsed_result, dict):
            if 'answers' in parsed_result:
                return [
                    safe_strip(a.get("justification") or a.get("answer") or a)
                    for a in parsed_result['answers']
                ]
            elif 'questions_analysis' in parsed_result:
                return [safe_strip(qa.get('justification') or qa.get('answer')) for qa in parsed_result['questions_analysis']]
            elif 'decision' in parsed_result:
                return [safe_strip(parsed_result.get("justification") or parsed_result.get("answer"))]
            else:
                return [safe_strip(parsed_result)]

        elif isinstance(parsed_result, list):
            return [safe_strip(a.get("justification") or a.get("answer") or a) for a in parsed_result]

        return [safe_strip(parsed_result)]

    except Exception as e:
        print("❌ Parsing error:", str(e))
        print("❗Failed content:\n", result)
        return [f"LLM parsing failed: {str(e)}"] * len(batch_questions)
