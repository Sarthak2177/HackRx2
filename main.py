from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os
import re
import hashlib
import asyncio
from dotenv import load_dotenv
from utils.dynamic_decision import DynamicDecisionEngine
from utils.extract_text_from_pdfs import extract_text_from_pdf as download_pdf_and_extract_text
from utils.chunk_utils import chunk_text
import tiktoken
from pinecone import Pinecone

load_dotenv()

# Pinecone init
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "hackrx")
index = pc.Index(index_name)

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

def get_relevant_chunks(query: str, namespace: str, top_k: int = 25):
    embed_model = tiktoken.get_encoding("cl100k_base")
    input_ids = embed_model.encode(query)
    vector = [float(x) for x in input_ids[:768]] + [0.0] * (768 - len(input_ids[:768]))

    results = index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )
    return [match["metadata"]["text"] for match in results.get("matches", [])]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_decision_engine(
    payload: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    if not token:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    try:
        raw_text = download_pdf_and_extract_text(payload.documents)
        chunks = chunk_text(raw_text)
        namespace = hashlib.md5(payload.documents.encode()).hexdigest()
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
        print("\u274c Error during question processing:", str(e))
        answers = [f"LLM processing failed: {str(e)}"] * len(payload.questions)

    return {
        "answers": answers,
        "success": True
    }

async def process_question_batch(batch_questions: List[str], relevant_chunks: List[str]) -> List[str]:
    max_chunks = 20
    trimmed_chunks = relevant_chunks[:max_chunks]

    result = decision_engine.make_decision_from_context("\n".join(batch_questions), {}, trimmed_chunks)
    print("\U0001f9e0 Raw LLM response:\n", result)

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
                return [
                    safe_strip(qa.get('justification') or qa.get('answer'))
                    for qa in parsed_result['questions_analysis']
                ]
            elif 'decision' in parsed_result:
                return [safe_strip(parsed_result.get("justification") or parsed_result.get("answer"))]
            else:
                return [safe_strip(parsed_result)]

        elif isinstance(parsed_result, list):
            return [
                safe_strip(a.get("justification") or a.get("answer") or a)
                for a in parsed_result
            ]

        return [safe_strip(parsed_result)]

    except Exception as e:
        print("\u274c Parsing error:", str(e))
        print("\u2757Failed content:\n", result)
        return [f"LLM parsing failed: {str(e)}"] * len(batch_questions)


