# app_lite.py — minimal, stable Streamlit UI

# ---- path shim (so `src/...` imports work) ----
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- imports ----
import streamlit as st
from src.retrieval.search import hybrid_search
from src.llm.generator import generate_answer

# ---- page config ----
st.set_page_config(page_title="SmartHR Copilot — Lite", layout="wide")
st.title("SmartHR Copilot — Lite")

# ---- simple Q&A form (stable on Windows/Cloud) ----
with st.form("qa"):
    q = st.text_input("Ask a question (e.g., 'What is the PTO carryover rule?')", "")
    style_label = st.radio(
        "Answer style",
        ["Bullets", "Paragraph (no-LLM)", "LLM (OpenAI)"],
        index=1,
        horizontal=True
    )
    submitted = st.form_submit_button("Search")

style = {"Bullets": "bullets", "Paragraph (no-LLM)": "paragraph", "LLM (OpenAI)": "llm"}[style_label]

if submitted and q.strip():
    try:
        hits = hybrid_search(q.strip())
        st.session_state["last_hits"] = hits
        st.markdown(generate_answer(q, hits, style=style))
    except Exception as e:
        st.error("Error while answering:")
        st.exception(e)

with st.expander("🔎 Retrieved snippets (debug)"):
    for i, h in enumerate(st.session_state.get("last_hits", []), 1):
        snippet = h.get("text", "")
        if len(snippet) > 500:
            snippet = snippet[:500] + "…"
        st.markdown(
            f"**{i}. {h.get('source','')}**  \n"
            f"*score:* `{h.get('score', 0):.2f}` • "
            f"*effective:* `{h.get('effective_from','n/a')}`  \n"
            f"{snippet}"
        )
