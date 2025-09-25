"""
Step 5: Main Orchestrator — Run full pipeline
"""

from planner import planner
from scout import scout
from analyst import analyst
from writer import writer

def run_pipeline(query):
    print("🔹 Step 1: Planner")
    plan = planner(query)
    subqs = plan.get("subquestions", [])
    print(f"Planner -> {len(subqs)} subquestions")

    print("🔹 Step 2: Scout")
    results = scout(subqs)
    print(f"Scout -> {len(results)} items")

    print("🔹 Step 3: Analyst")
    findings, contradictions = analyst(results)
    print(f"Analyst -> {len(findings)} findings, {len(contradictions)} contradictions")

    print("🔹 Step 4: Writer")
    report = writer(findings, contradictions)
    print("\n✅ Final Markdown Report:\n")
    print(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+", help="Research query")
    args = parser.parse_args()
    run_pipeline(" ".join(args.query))
