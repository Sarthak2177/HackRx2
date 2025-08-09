import json
import re
import ast
from typing import List, Dict, Any, Union
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

    def make_decision_from_context(
        self,
        joined_questions: Union[str, List[str]],
        metadata: Dict[str, Any],
        context_chunks: List[str]
    ) -> str:
        context_text = "\n\n".join(context_chunks)

        system_role = (
            "You are a senior insurance policy assistant. Only use the provided policy clauses for answers. "
            "Never assume anything outside the context."
        )

        instruction = '''
Use ONLY the provided context to answer each question accurately.

INTERNAL WORK:
- First, think step-by-step using a structure: {question, answer, justification, referenced_clauses}.
- You MUST reason with clause references, exact numbers, and wording from context.
- Do not skip relevant clauses, and cross-reference multiple ones if needed.
- Try to keep answers in single lines.
OUTPUT FORMAT:
Return a JSON object:
{
  "answers": [
    "Plain answer to question 1",
    "Plain answer to question 2",
    ...
  ]
}
Where each item in "answers" is just the final answer text, **no question, justification, or clause data**.
'''

        # Step 1: Normalize questions
        if isinstance(joined_questions, str):
            questions = [q.strip() for q in re.split(r'(?<=[?.!])\s+(?=[A-Z])', joined_questions) if len(q.strip()) > 5]
        elif isinstance(joined_questions, list):
            questions = [q.strip() for q in joined_questions if isinstance(q, str) and len(q.strip()) > 5]
        else:
            return json.dumps({"answers": []})

        if not questions:
            return json.dumps({"answers": []})

        # Step 2: Build prompt
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

            content = response.choices[0].message.content
            parsed = self._safe_json_load(content)

            # Ensure "answers" is a clean list of strings
            if not parsed or "answers" not in parsed:
                return json.dumps({"answers": []})

            clean_answers = []
            for ans in parsed["answers"]:
                if isinstance(ans, dict) and "answer" in ans:
                    clean_answers.append(ans["answer"].strip())
                elif isinstance(ans, str):
                    # Try parsing stringified dict
                    maybe_dict = self._try_parse_dict_string(ans)
                    if "answer" in maybe_dict:
                        clean_answers.append(maybe_dict["answer"].strip())
                    else:
                        clean_answers.append(ans.strip())
                else:
                    clean_answers.append(str(ans).strip())

            return json.dumps({"answers": clean_answers}, ensure_ascii=False)

        except Exception as e:
            print("❌ LLM call failed:", str(e))
            return json.dumps({"answers": []})

    def _safe_json_load(self, data: Union[str, dict]) -> dict:
        """Safely load JSON/dict content."""
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(data)
                except Exception:
                    return {}
        return {}

    def _try_parse_dict_string(self, text: str) -> dict:
        """Try to parse string that looks like a dict."""
        text = text.strip()
        if (text.startswith("{") and text.endswith("}")):
            try:
                return ast.literal_eval(text)
            except Exception:
                try:
                    return json.loads(text)
                except Exception:
                    return {}
        return {}
