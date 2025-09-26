"""
Step 4: Writer — Markdown report (LangChain + Gemini)
"""

import os, json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

def writer(findings, contradictions, model="gemini-2.5-flash"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY not set in .env")

    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=api_key   # ✅ use API key instead of ADC
    )

    prompt = PromptTemplate(
        input_variables=["findings","contradictions"],
        template="""
You are WriterAgent. Use findings and contradictions to create a concise Markdown report.
Rules:
- Every fact must be followed by [url].
- Use professional, clear style.
- Mention contradictions if any.

Findings:
{findings}

Contradictions:
{contradictions}

Markdown report:
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run({
        "findings": json.dumps(findings, ensure_ascii=False),
        "contradictions": json.dumps(contradictions, ensure_ascii=False)
    })


if __name__ == "__main__":
    demo_findings = [
        {"subq_id":"Q1","claim":"H1B fee raised to $100,000","url":"https://example.com","nums":["100000"]}
    ]
    print(writer(demo_findings, []))
