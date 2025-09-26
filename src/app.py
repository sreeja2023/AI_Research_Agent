# app.py
import os
import json
import streamlit as st
from datetime import datetime
from pathlib import Path

from generate_json import generate_json

# Import your pipeline modules (assumes they are in same folder)
from planner import planner
from scout import scout
from analyst import analyst
from writer import writer

# Import the RAGChatbot class from chatbot.py
from chatbot import RAGChatbot  # <-- Ensure chatbot.py is accessible and contains RAGChatbot class

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
st.sidebar.markdown("⚠ Make sure .env contains:")
st.sidebar.markdown("- GEMINI_API_KEY")
st.sidebar.markdown("- GROQ_API_KEY") 
st.sidebar.markdown("- SERPER_API_KEY (for web search)")

# Main area columns for controls and quick actions
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Run pipeline")
    run_btn = st.button("Run Research Agent")

with col_right:
    st.subheader("Quick actions")
    if st.button("Show latest planner output (planner_output.json)"):
        try:
            with open("planner_output.json", "r", encoding="utf-8") as f:
                st.code(json.dumps(json.load(f), indent=2, ensure_ascii=False), language="json")
        except Exception:
            st.info("No planner_output.json found yet. Run the pipeline first.")

# Placeholders for progress / outputs
pln_ph = st.empty()
scout_ph = st.empty()
analyst_ph = st.empty()
writer_ph = st.empty()

report_filepath = None  # will hold path if report is saved

# Initialize session state for chatbot and documents
if 'chatbot_ready' not in st.session_state:
    st.session_state['chatbot_ready'] = False

if 'latest_report' not in st.session_state:
    st.session_state['latest_report'] = None

if 'latest_report_content' not in st.session_state:
    st.session_state['latest_report_content'] = None

if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# ---------------- Pipeline Execution ---------------- #
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

        st.subheader("Planner — Sub-questions")
        st.json(plan)

        # save planner output
        with open("planner_output.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        # Step 2: Scout
        scout_ph.info("Step 2 — Scout: searching & scraping (Serper + Firecrawl)...")
        try:
            results = scout(subqs, top_k=top_k)
        except TypeError:
            results = scout(subqs)  # fallback
        scout_ph.success(f"Scout found {len(results)} items.")

        st.subheader("Scout — top scraped results (first 8)")
        if results:
            for r in results[:8]:
                st.markdown(
                    f"{r.get('title','(no title)')}**  \n{r.get('url')}  \n_{r.get('subq_id')}: {r.get('search_keyword', r.get('keyword',''))}_"
                )
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
                claim = f.get("claim", "").replace("\n", " ")
                st.markdown(f"- {f.get('subq_id')}: {claim}  \n  — Source: {f.get('url')}")
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

        st.subheader("Final Markdown Report")
        st.markdown("---")
        st.markdown(md_report)
         # ---------------- Step 3.5: Generate Structured JSON ---------------- #
        json_ph = st.empty()
        json_ph.info("Generating structured JSON from findings...")
        try:
            structured_json, json_filename = generate_json(findings)
            json_ph.success(f"Structured JSON saved to {json_filename}")
        except Exception as e:
            json_ph.error(f"JSON generation error: {e}")
        st.subheader("Structured JSON Output")
        st.json(structured_json)
        try:
            with open(json_filename, "rb") as f:
                st.download_button(
                    label="Download Structured JSON",
                    data=f,
                    file_name=json_filename,
                    mime="application/json"
            )
        except Exception:
            st.warning("JSON file not found for download.")
        # Save report if requested
        if save_report_file:
            fn = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(md_report)
            st.success(f"Saved report to {fn}")
            report_filepath = fn
            st.session_state['latest_report'] = fn
            st.session_state['latest_report_content'] = md_report
            
            with open(fn, "rb") as f:
                st.download_button("Download report (.md)", f, file_name=fn)

        # Initialize and index chatbot with the new report
        st.info("Initializing enhanced chatbot with web search fallback...")
        try:
            groq_api_key = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
            serper_api_key = os.getenv("SERPER_API_KEY", "3cf13dc86ab5a8f0c9bd7dfc8fdb351766aa973a")
            
            if groq_api_key == "your-groq-api-key-here":
                st.error("⚠ GROQ_API_KEY not found in environment variables. Please set it in your .env file.")
                st.session_state['chatbot_ready'] = False
            else:
                # Create new enhanced chatbot instance with web search
                chatbot = RAGChatbot(
                    groq_api_key=groq_api_key,
                    serper_api_key=serper_api_key,
                    memory_type="buffer_window",
                    max_memory_length=15,
                    enable_web_search=True
                )
                
                # Index the report file if it exists
                if report_filepath and Path(report_filepath).exists():
                    chatbot.load_and_index_documents([report_filepath])
                    st.session_state['chatbot'] = chatbot
                    st.session_state['chatbot_ready'] = True
                    st.success("✅ Enhanced chatbot successfully indexed the report and is ready for questions!")
                    
                    # Show web search status
                    if hasattr(chatbot, 'web_search') and chatbot.web_search and chatbot.web_search.search_enabled:
                        st.success("🌐 Web search fallback is enabled!")
                    else:
                        st.warning("⚠ Web search fallback is disabled (check SERPER_API_KEY)")
                else:
                    st.warning("Report file not found. Chatbot may not have proper context.")
                    st.session_state['chatbot_ready'] = False
                    
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.session_state['chatbot_ready'] = False

# ---------------- Chatbot Section ---------------- #
st.markdown("---")
st.subheader("Step 5 — Ask the Research Agent Chatbot")

# Always show Step 4 report here so it doesn't disappear when interacting with chatbot
if st.session_state.get('latest_report_content'):
    with st.expander("Step 4 — Final Markdown Report (click to expand/collapse)", expanded=True):
        st.markdown("---")
        st.markdown(st.session_state['latest_report_content'])
else:
    st.info("No report available yet. Run the pipeline to generate a report for chatbot context.")

# Check if chatbot is ready
if not st.session_state.get('chatbot_ready', False):
    st.warning("⚠ Chatbot not ready yet. Run the pipeline first to generate a report that the chatbot can use for context.")
    
    # Offer to initialize with existing report if available
    if st.session_state.get('latest_report') and Path(st.session_state['latest_report']).exists():
        if st.button("Initialize Enhanced Chatbot with Latest Report"):
            try:
                groq_api_key = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
                serper_api_key = os.getenv("SERPER_API_KEY", "3cf13dc86ab5a8f0c9bd7dfc8fdb351766aa973a")
                
                if groq_api_key != "your-groq-api-key-here":
                    chatbot = RAGChatbot(
                        groq_api_key=groq_api_key,
                        serper_api_key=serper_api_key,
                        memory_type="buffer_window",
                        max_memory_length=15,
                        enable_web_search=True
                    )
                    chatbot.load_and_index_documents([st.session_state['latest_report']])
                    st.session_state['chatbot'] = chatbot
                    st.session_state['chatbot_ready'] = True
                    st.success("✅ Enhanced chatbot initialized with existing report!")
                    st.rerun()
                else:
                    st.error("GROQ_API_KEY not set in environment variables.")
            except Exception as e:
                st.error(f"Error initializing chatbot: {e}")
else:
    # Chatbot is ready - show enhanced interface
    chatbot = st.session_state['chatbot']
    
    # Create two columns for the chat interface
    chat_col, history_col = st.columns([2, 1])
    
    with chat_col:
        # User question input
        user_question = st.text_input("Ask a question about the research:")
        
        if user_question:
            with st.spinner("Chatbot is generating an answer..."):
                try:
                    response = chatbot.ask(user_question)
                    
                    # Store conversation in session state
                    st.session_state['conversation_history'].append({
                        'question': user_question,
                        'answer': response['answer'],
                        'used_web_search': response.get('used_web_search', False),
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Show if web search was used
                    if response.get('used_web_search'):
                        st.success("🌐 This answer was enhanced with web search results!")
                    
                    st.markdown(f"Answer: {response['answer']}")
                    
                    if response['sources']:
                        source_type = "Web Search Results" if response.get('used_web_search') else "Document Sources"
                        st.markdown(f"{source_type} ({len(response['sources'])}):")
                        for i, src in enumerate(response['sources'], 1):
                            snippet = src['content']
                            source_name = src['metadata'].get('file_name', src['metadata'].get('source', 'unknown'))
                            st.markdown(f"{i}. {source_name}: {snippet}")
                    else:
                        st.info("No sources found for this answer.")
                        
                except Exception as e:
                    st.error(f"Error getting chatbot response: {e}")
    
    with history_col:
        st.subheader("Conversation History")
        
        # Show recent conversation history
        if st.session_state['conversation_history']:
            for i, conv in enumerate(reversed(st.session_state['conversation_history'][-5:]), 1):
                with st.expander(f"{conv['timestamp']} - {conv['question'][:30]}..."):
                    st.write(f"Q: {conv['question']}")
                    st.write(f"A: {conv['answer'][:200]}...")
                    if conv['used_web_search']:
                        st.caption("🌐 Web search used")
        else:
            st.info("No conversation history yet.")
        
        # Clear history button
        if st.button("Clear Conversation History"):
            try:
                chatbot.clear_history()
                st.session_state['conversation_history'] = []
                st.success("History cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing history: {e}")

# Sidebar enhancements
st.sidebar.markdown("---")
st.sidebar.subheader("Chatbot Features")

if st.session_state.get('chatbot_ready', False):
    chatbot = st.session_state['chatbot']
    
    # Show memory statistics
    if st.sidebar.button("Show Memory Stats"):
        try:
            stats = chatbot.get_memory_stats()
            st.sidebar.write("*Memory Statistics:")
            st.sidebar.write(f"Memory Type: {stats['memory_type']}")
            st.sidebar.write(f"Total Exchanges: {stats['total_exchanges']}")
            st.sidebar.write(f"Session ID: {stats['current_session']}")
            st.sidebar.write(f"LangChain Memory: {stats['langchain_memory_size']}")
        except Exception as e:
            st.sidebar.error(f"Error getting stats: {e}")
    
    # Show full conversation history
    if st.sidebar.button("Show Full History"):
        try:
            history = chatbot.get_history_summary(10)
            st.sidebar.text_area("Recent History", history, height=200)
        except Exception as e:
            st.sidebar.error(f"Error getting history: {e}")

# Enhanced status section
st.sidebar.markdown("---")
st.sidebar.markdown("*Chatbot Status:")
if st.session_state.get('chatbot_ready', False):
    st.sidebar.success("✅ Ready")
    if st.session_state.get('latest_report'):
        st.sidebar.info(f"Using: {Path(st.session_state['latest_report']).name}")
    
    # Show web search status
    chatbot = st.session_state['chatbot']
    if hasattr(chatbot, 'web_search') and chatbot.web_search and chatbot.web_search.search_enabled:
        st.sidebar.success("🌐 Web Search: Enabled")
    else:
        st.sidebar.warning("🌐 Web Search: Disabled")
    
    # Show conversation count
    conv_count = len(st.session_state.get('conversation_history', []))
    st.sidebar.info(f"💬 Conversations: {conv_count}")
else:
    st.sidebar.error("❌ Not Ready")

st.markdown("---")
st.caption("Project Galileo — Enhanced Streamlit frontend with web search fallback. Uses Gemini for LLM steps, Groq for chatbot, Serper for web search, and Firecrawl for content extraction.")
