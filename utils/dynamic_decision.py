import json
import re
from typing import List, Dict, Any
from collections import defaultdict
from groq import Groq
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

# Create HTTP client for Groq
custom_http_client = httpx.Client()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    http_client=custom_http_client
)

class DynamicDecisionEngine:
    def __init__(self):
        pass

    def make_decision_from_context(self, joined_questions: str, metadata: Dict[str, Any], context_chunks: List[str]) -> str:
        context_text = "\n\n".join(context_chunks)

        system_role = "You are a senior insurance policy assistant. Only answer based on provided context. Never assume."

instruction = '''
Use ONLY the provided context to answer each question as accurately and specifically as possible.

IMPORTANT GUIDELINES:
- Extract and include exact figures like days, months, or years.
- Include the full clause or section numbers that justify your answers.
- Never guess. Say “Not mentioned in context” if unsure.

Respond in this JSON format:
{
  "answers": [
    {
      "question": "...",
      "answer": "...",
      "justification": "...",
      "referenced_clauses": [...]
    }
  ]
}
'''


        # Convert joined_questions to list of strings
        if isinstance(joined_questions, str):
            questions = [q.strip() for q in re.split(r'(?<=[?])\s+', joined_questions) if len(q.strip()) > 5]
        elif isinstance(joined_questions, list):
            questions = joined_questions
        else:
            return json.dumps({
                "answers": [
                    {
                        "question": "Invalid input",
                        "answer": "Input must be a string or list of strings.",
                        "justification": "",
                        "referenced_clauses": []
                    }
                ]
            })

        # Final formatted prompt
        prompt = f"""Context:
{context_text}

Questions:
{json.dumps(questions, indent=2)}

{instruction}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=2048
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print("❌ LLM call failed:", str(e))
            return json.dumps({
                "answers": [
                    {
                        "question": questions[0] if questions else "Unknown",
                        "answer": "LLM failed to generate response.",
                        "justification": str(e),
                        "referenced_clauses": []
                    }
                ]
            })

