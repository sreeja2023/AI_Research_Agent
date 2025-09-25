# app.py
import os
import json
import streamlit as st
from datetime import datetime

# Import your pipeline modules (assumes they are in same folder)
from planner import planner
from scout import scout
from analyst import analyst
from writer import writer

st.set_page_config(page_title="Project Galileo — Research Agent", layout="wide")

st.title("Project Galileo — Research Agent")
st.markdown("Enter a research question, run the pipeline, and get an evidence-backed report.")

# Sidebar controls
st.sidebar.header("Settings")
query = st.sidebar.text_area("Research query", value="What is the current status of H1B visa in 2025?", height=80)
depth = st.sidebar.selectbox("Depth", ["normal", "shallow", "deep"], index=1)
top_k = st.sidebar.slider("Top URLs per keyword (Scout)", min_value=1, max_value=4, value=2)
keywords_per_subq = st.sidebar.slider("Keywords per sub-question", min_value=1, max_value=3, value=2)
save_report_file = st.sidebar.checkbox("Save report to file", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ Make sure `.env` contains `GEMINI_API_KEY` before running.")

# Main area
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Run pipeline")
    run_btn = st.button("Run Research Agent")

with col_right:
    st.subheader("Quick actions")
    if st.button("Show latest planner output (planner_output.json)"):
        try:
            with open("planner_output.json","r",encoding="utf-8") as f:
                st.code(json.dumps(json.load(f), indent=2, ensure_ascii=False), language="json")
        except Exception as e:
            st.info("No planner_output.json found yet. Run the pipeline first.")

# placeholders for progress / outputs
pln_ph = st.empty()
scout_ph = st.empty()
analyst_ph = st.empty()
writer_ph = st.empty()

if run_btn:
    if not query.strip():
        st.error("Please enter a research query.")
    else:
        # Step 1: Planner
        pln_ph.info("Step 1 — Planner: contacting Gemini...")
        try:
            plan = planner(query, depth=depth)
        except Exception as e:
            st.error(f"Planner error: {e}")
            raise
        subqs = plan.get("subquestions", [])
        pln_ph.success(f"Planner produced {len(subqs)} sub-questions.")
        # show planner output
        st.subheader("Planner — Sub-questions")
        st.json(plan)

        # save planner output for convenience
        with open("planner_output.json","w",encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        # Step 2: Scout
        scout_ph.info("Step 2 — Scout: searching & scraping (DuckDuckGo)...")
        try:
            results = scout(subqs, top_k=top_k)
        except TypeError:
            # some scout versions accept only one arg; attempt default call
            results = scout(subqs)
        scout_ph.success(f"Scout found {len(results)} items.")
        st.subheader("Scout — top scraped results (first 8)")
        if results:
            for r in results[:8]:
                st.markdown(f"**{r.get('title','(no title)')}**  \n{r.get('url')}  \n_{r.get('subq_id')}: {r.get('search_keyword', r.get('keyword',''))}_")
                snippet = (r.get("text") or "").strip().replace("\n", " ")[:600]
                st.write(snippet + ("…" if len(snippet) >= 600 else ""))
                st.markdown("---")
        else:
            st.info("No scraped results found. Try increasing Top URLs or keywords per sub-question.")

        # Step 3: Analyst
        analyst_ph.info("Step 3 — Analyst: extracting claims & checking contradictions...")
        try:
            findings, contradictions = analyst(results)
        except Exception as e:
            st.error(f"Analyst error: {e}")
            raise
        analyst_ph.success(f"Analyst found {len(findings)} findings and {len(contradictions)} contradictions.")
        st.subheader("Analyst — Top findings (first 10)")
        if findings:
            for f in findings[:10]:
                claim = f.get("claim","").replace("\n"," ")
                st.markdown(f"- **{f.get('subq_id')}**: {claim}  \n  — Source: {f.get('url')}")
        else:
            st.info("No findings.")

        if contradictions:
            st.subheader("Detected contradictions")
            st.json(contradictions)

        # Step 4: Writer
        writer_ph.info("Step 4 — Writer: generating Markdown report (LangChain + Gemini)...")
        try:
            md_report = writer(findings, contradictions)
        except Exception as e:
            writer_ph.error(f"Writer error: {e}")
            raise
        writer_ph.success("Report generated.")

        # Display final report
        st.subheader("Final Markdown Report")
        st.markdown("---")
        st.markdown(md_report)

        # Save report if requested
        if save_report_file:
            fn = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(md_report)
            st.success(f"Saved report to {fn}")
            with open(fn, "rb") as f:
                st.download_button("Download report (.md)", f, file_name=fn)

st.markdown("---")
st.caption("Project Galileo — Streamlit frontend. Uses Gemini for LLM steps and DuckDuckGo for search.")
