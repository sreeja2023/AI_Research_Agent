"""
planner.py

Step 1: Planner — Decompose query into 4–8 concise, search-friendly sub-questions.
This planner outputs a minimal JSON plan containing only:
{
  "query": "<original query>",
  "depth": "<shallow|normal|deep>",
  "subquestions": [
    { "id": "Q1", "category": "scope|stakeholders|causes|impacts|policy|logistics|other",
      "text": "short phrase (<=8 words)", "priority": "high|medium|low" }
  ]
}

Important: NO search_keywords are emitted by this planner (per your request).
If GEMINI_API_KEY is present it will try to call Gemini; otherwise uses deterministic fallback.
"""
import os
import json
import re
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

ALLOWED_CATEGORIES = {"scope", "stakeholders", "causes", "impacts", "policy", "logistics", "other"}
ALLOWED_PRIORITIES = {"high", "medium", "low"}

def extract_query_tokens(query):
    if not query:
        return []
    stopwords = {"what", "is", "in", "the", "of", "and", "for", "new", "a", "an", "to", "on", "by", "with"}
    q = query.strip()
    seen = set()
    tokens = []
    # short alphanum tokens
    for m in re.finditer(r'\b([A-Za-z0-9]{2,4})\b', q):
        tok = m.group(1)
        key = tok.lower()
        if key not in seen and key not in stopwords:
            tokens.append(tok)
            seen.add(key)
    # Proper nouns / capitalized words
    for m in re.finditer(r'\b([A-Z][a-z]{1,})\b', q):
        tok = m.group(1)
        key = tok.lower()
        if key not in seen and key not in stopwords:
            tokens.append(tok)
            seen.add(key)
    # longer tokens
    for m in re.finditer(r'\b([A-Za-z0-9\-]{3,})\b', q):
        tok = m.group(1)
        key = tok.lower()
        if key not in seen and key not in stopwords:
            tokens.append(tok)
            seen.add(key)
    return tokens[:8]

def normalize_subq(subq, idx, original_tokens):
    if not isinstance(subq, dict):
        return None
    sid = subq.get("id") or f"Q{idx+1}"
    cat = (subq.get("category") or "other").lower()
    if cat not in ALLOWED_CATEGORIES:
        cat = "other"
    text = (subq.get("text") or "").strip()
    if not text:
        return None
    # Keep text short (<=8 words)
    if len(text.split()) > 8:
        text = " ".join(text.split()[:8])
    pr = (subq.get("priority") or "medium").lower()
    if pr not in ALLOWED_PRIORITIES:
        pr = "medium"
    return {"id": sid, "category": cat, "text": text, "priority": pr}

# Depth mapping helpers
def depth_to_n(depth):
    d = (depth or "normal").lower()
    if d == "shallow":
        return 4
    if d == "deep":
        return 8
    return 6  # normal

def validate_and_clean_plan(plan, original_tokens, min_q=4, max_q=8):
    if not isinstance(plan, dict):
        raise ValueError("Plan is not a JSON object")
    q = plan.get("query") or ""
    depth = plan.get("depth") or "normal"
    raw_subqs = plan.get("subquestions") or []
    cleaned = []
    for i, s in enumerate(raw_subqs):
        cs = normalize_subq(s, i, original_tokens)
        if cs:
            cleaned.append(cs)
    # Trim if too many
    if len(cleaned) > max_q:
        cleaned = cleaned[:max_q]
    # Pad if too few: use deterministic fallback items (avoid duplicates)
    if len(cleaned) < min_q:
        fallback_plan = deterministic_fallback(q, depth=depth, n=min_q, original_tokens=original_tokens)
        existing_texts = {c['text'].lower() for c in cleaned}
        for fs in fallback_plan['subquestions']:
            if fs['text'].lower() not in existing_texts:
                cleaned.append(fs)
                existing_texts.add(fs['text'].lower())
            if len(cleaned) >= min_q:
                break
    if not cleaned:
        raise ValueError("No valid subquestions produced")
    return {"query": q, "depth": depth, "subquestions": cleaned}

def deterministic_fallback(query, depth="normal", n=None, original_tokens=None):
    q = (query or "").strip()
    if original_tokens is None:
        original_tokens = extract_query_tokens(q)
    if n is None:
        n = depth_to_n(depth)
    base = " ".join(original_tokens[:6]) if original_tokens else q
    templates = [
        f"Overview of {base}",
        f"Key drivers of {base}",
        f"Recent statistics for {base}",
        f"Geographic impacts of {base}",
        f"Policy and regulation for {base}",
        f"Major stakeholders related to {base}",
        f"Data sources about {base}",
        f"Open questions about {base}",
        f"Historical trends for {base}",
        f"Market outlook for {base}"
    ]
    subqs = []
    for i in range(min(n, len(templates))):
        text = templates[i]
        sid = f"Q{i+1}"
        subqs.append({"id": sid, "category": "other", "text": text, "priority": "medium"})
    # If templates shorter than n, repeat with numbered suffixes
    i = len(subqs)
    while len(subqs) < n:
        sid = f"Q{len(subqs)+1}"
        text = (f"Additional question about {base} ({len(subqs)+1})")[:120]
        subqs.append({"id": sid, "category": "other", "text": text, "priority": "medium"})
        i += 1
    return {"query": q, "depth": depth, "subquestions": subqs}

def planner(query, depth="normal", model="gemini-2.5-flash", debug=False):
    query_text = (query or "").strip()
    if not query_text:
        raise ValueError("Query must be non-empty")
    original_tokens = extract_query_tokens(query_text)
    api_key = os.getenv("GEMINI_API_KEY")
    desired_n = depth_to_n(depth)

    prompt = (
        f"You are ResearchPlanner. Decompose this high-level query into {desired_n} concise, search-friendly sub-questions.\n\n"
        f"Query: \"{query_text}\"\n\n"
        "Output ONLY valid JSON matching this schema:\n"
        '{\n'
        '  \"query\":\"<original query>\",\n'
        '  \"depth\":\"<shallow|normal|deep>\",\n'
        '  \"subquestions\":[\n'
        '    { \"id\":\"Q1\",\"category\":\"scope|stakeholders|causes|impacts|policy|logistics|other\",\n'
        '      \"text\":\"short phrase (<=8 words)\",\"priority\":\"high|medium|low\" }\n'
        '  ]\n'
        '}\n\n'
        "Important: DO NOT include search_keywords. Output JSON only.\n"
    )

    # If no API key or client not available, deterministic fallback
    if not api_key:
        if debug:
            print("planner: no GEMINI_API_KEY found — using deterministic fallback")
        return deterministic_fallback(query_text, depth=depth, n=desired_n, original_tokens=original_tokens)

    # Try to call the Generative API (genai) if available
    try:
        from google import genai
    except Exception:
        if debug:
            print("planner: google.genai import failed — using deterministic fallback")
        return deterministic_fallback(query_text, depth=depth, n=desired_n, original_tokens=original_tokens)

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=[prompt])
        raw = None
        if hasattr(resp, "text"):
            raw = resp.text
        elif isinstance(resp, (str, bytes)):
            raw = resp if isinstance(resp, str) else resp.decode("utf-8", errors="ignore")
        else:
            raw = str(resp)

        # find first JSON-like block
        m = re.search(r'(\{[\s\S]*\})', raw)
        jtxt = m.group(1) if m else raw

        # Try parsing, with best-effort cleaning
        parsed = None
        try:
            parsed = json.loads(jtxt)
        except Exception:
            jtxt2 = jtxt.replace("'", '"')
            jtxt2 = re.sub(r",\s*}", "}", jtxt2)
            jtxt2 = re.sub(r",\s*]", "]", jtxt2)
            try:
                parsed = json.loads(jtxt2)
            except Exception:
                # last attempt: extract lines that look like subquestion objects
                parsed = None

        # If parsed is None, fallback
        if parsed is None:
            if debug:
                print("planner: LLM output could not be parsed as JSON — falling back")
            return deterministic_fallback(query_text, depth=depth, n=desired_n, original_tokens=original_tokens)

        # If parsed is a dict, validate and clean; if list assume list of subqs
        try:
            if isinstance(parsed, dict) and parsed.get("subquestions"):
                cleaned = validate_and_clean_plan(parsed, original_tokens, min_q=depth_to_n(depth), max_q=8)
                if debug:
                    print("planner: using LLM plan (validated).")
                return cleaned
            elif isinstance(parsed, list):
                candidate = {"query": query_text, "depth": depth, "subquestions": parsed}
                cleaned = validate_and_clean_plan(candidate, original_tokens, min_q=depth_to_n(depth), max_q=8)
                if debug:
                    print("planner: using LLM list-of-subqs (validated).")
                return cleaned
            else:
                # If dict without subquestions, try to wrap or fall back
                if debug:
                    print("planner: parsed JSON did not contain subquestions — falling back")
                return deterministic_fallback(query_text, depth=depth, n=desired_n, original_tokens=original_tokens)
        except Exception as e:
            if debug:
                print(f"planner: validation failed ({e}) — falling back")
            return deterministic_fallback(query_text, depth=depth, n=desired_n, original_tokens=original_tokens)

    except Exception as e:
        if debug:
            print(f"planner: exception during LLM call ({e}) — falling back")
        return deterministic_fallback(query_text, depth=depth, n=desired_n, original_tokens=original_tokens)

# CLI support
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Planner: decompose research query into subquestions")
    parser.add_argument("query", nargs="+", help="The high-level research query")
    parser.add_argument("--depth", choices=["shallow", "normal", "deep"], default="normal")
    parser.add_argument("--debug", action="store_true", help="Show debug messages about fallback/LLM usage")
    args = parser.parse_args()
    q = " ".join(args.query)
    plan = planner(q, depth=args.depth, debug=args.debug)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
