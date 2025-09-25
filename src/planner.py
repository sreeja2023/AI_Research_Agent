"""
Step 1: Planner (Gemini) — Decompose query into sub-questions
"""

import os, json, re
from dotenv import load_dotenv
from google import genai

load_dotenv()

def planner(query, depth="normal", model="gemini-1.5-flash"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are ResearchPlanner. Decompose this high-level query into 6–7 concise, search-friendly sub-questions.

Query: "{query}"

Output ONLY JSON in this schema:
{{ "query":"{query}", "depth":"{depth}", "subquestions":[
  {{ "id":"Q1","category":"scope|stakeholders|causes|impacts|policy|logistics|other",
     "text":"short phrase (<=6 words)",
     "priority":"high|medium|low",
     "search_keywords":["kw1","kw2"] }}
]}}
"""

    resp = client.models.generate_content(model=model, contents=[prompt])
    raw = getattr(resp, "text", None) or str(resp)

    # Extract JSON
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        raise ValueError("No JSON found in Gemini output")
    return json.loads(m.group(0))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="High-level query")
    args = parser.parse_args()
    q = " ".join(args.query)
    out = planner(q)
    print(json.dumps(out, indent=2, ensure_ascii=False))
