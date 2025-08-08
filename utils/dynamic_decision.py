import json
import re
from typing import List, Dict, Any
from groq import Groq
import os
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()
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

        system_role = (
            "You are a senior insurance policy assistant. Only use the provided policy clauses for answers. "
            "Never assume anything outside the context."
        )

        instruction = '''
Use ONLY the provided context to answer each question accurately.

IMPORTANT:
- Quote exact numbers (e.g., 24 months, 1%) from the context.
- If clauses or section numbers are present, include them in "justification" and "referenced_clauses".
- Be concise and objective. Avoid filler language.

Respond in this exact JSON format:
{
  "answers": [
    {
      "question": "...",
      "answer": "Fact-based answer in 1–2 sentences, using wording from context.",
      "justification": "Mention clause reference or quote context that justifies the answer.",
      "referenced_clauses": ["Clause X.Y", "Section 3.2"]
    },
    ...
  ]
}
'''

        # Step 1: Validate and split questions
        if isinstance(joined_questions, str):
            questions = [q.strip() for q in re.split(r'(?<=\?)\s+', joined_questions) if len(q.strip()) > 5]
        elif isinstance(joined_questions, list):
            questions = [q.strip() for q in joined_questions if isinstance(q, str) and len(q.strip()) > 5]
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

        if not questions:
            return json.dumps({
                "answers": [
                    {
                        "question": "No valid questions provided",
                        "answer": "Please provide one or more meaningful questions.",
                        "justification": "",
                        "referenced_clauses": []
                    }
                ]
            })

        # Step 2: Format prompt
        prompt = f"""Context (policy clauses):
{context_text}

Questions to answer:
{json.dumps(questions, indent=2)}

{instruction}
"""

        # Step 3: Call LLM
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
