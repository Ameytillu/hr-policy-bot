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
st.set_page_config(
    page_title="SmartHR Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---- Custom CSS for modern styling ----
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom card styling */
    .main-card {
        background: white;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin: 2rem auto;
        max-width: 900px;
    }
    
    /* Title styling */
    .custom-title {
        font-size: 3rem;
        font-weight: 700;
        color: white !important;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .custom-subtitle {
        text-align: center;
        color: white !important;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Make all form labels visible */
    label, .stMarkdown, h3, h4 {
        color: #2c3e50 !important;
    }
    
    /* Ensure text in main content area is visible */
    .element-container {
        color: #2c3e50 !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e0e0e0;
        padding: 15px 20px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Radio button styling */
    .stRadio > label {
        font-weight: 600;
        color: #2c3e50 !important;
        font-size: 1rem;
    }
    
    .stRadio > div {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
    }
    
    .stRadio > div > label {
        color: #2c3e50 !important;
    }
    
    .stRadio > div > label > div {
        color: #2c3e50 !important;
    }
    
    /* Answer box styling */
    .answer-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 12px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Snippet card styling */
    .snippet-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .snippet-card:hover {
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<h1 class="custom-title">🤖 SmartHR Copilot</h1>', unsafe_allow_html=True)
st.markdown('<p class="custom-subtitle">Your AI-powered HR policy assistant</p>', unsafe_allow_html=True)

# ---- Main content area ----
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    # ---- Q&A form ----
    with st.form("qa", clear_on_submit=False):
        st.markdown('<h3 style="color: #2c3e50 !important;">💬 Ask Your Question</h3>', unsafe_allow_html=True)
        q = st.text_input(
            "Type your HR policy question here...",
            placeholder="e.g., What is the PTO carryover rule?",
            label_visibility="collapsed"
        )
        
        st.markdown('<h3 style="color: #2c3e50 !important;">🎨 Response Style</h3>', unsafe_allow_html=True)
        style_label = st.radio(
            "Choose your preferred answer format:",
            ["Bullets", "Paragraph (no-LLM)", "LLM (OpenAI)"],
            index=1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("🔍 Get Answer", use_container_width=True)

    style = {"Bullets": "bullets", "Paragraph (no-LLM)": "paragraph", "LLM (OpenAI)": "llm"}[style_label]

    if submitted and q.strip():
        with st.spinner('🔍 Searching and generating answer...'):
            try:
                hits = hybrid_search(q.strip())
                st.session_state["last_hits"] = hits
                
                # Display answer in a styled box
                st.markdown("---")
                st.markdown("### ✨ Answer")
                answer = generate_answer(q, hits, style=style)
                st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)
                
                # Success message
                st.success("✅ Answer generated successfully!")
                
            except Exception as e:
                st.error("❌ Error while answering:")
                st.exception(e)

    # ---- Retrieved snippets section ----
    if st.session_state.get("last_hits"):
        st.markdown("---")
        with st.expander("🔎 View Retrieved Snippets & Sources", expanded=False):
            st.markdown(f"**Found {len(st.session_state.get('last_hits', []))} relevant snippets**")
            
            for i, h in enumerate(st.session_state.get("last_hits", []), 1):
                snippet = h.get("text", "")
                if len(snippet) > 500:
                    snippet = snippet[:500] + "…"
                
                # Create styled snippet cards
                st.markdown(f"""
                <div class="snippet-card">
                    <h4>📄 {i}. {h.get('source', 'Unknown Source')}</h4>
                    <p style="color: #666; font-size: 0.9rem;">
                        <strong>Relevance Score:</strong> <code>{h.get('score', 0):.2f}</code> • 
                        <strong>Effective From:</strong> <code>{h.get('effective_from', 'n/a')}</code>
                    </p>
                    <p style="margin-top: 1rem; line-height: 1.6;">{snippet}</p>
                </div>
                """, unsafe_allow_html=True)

# ---- Sidebar with info ----
with st.sidebar:
    st.markdown("### 📚 About SmartHR Copilot")
    st.markdown("""
    This AI assistant helps you quickly find answers to HR policy questions.
    
    **Features:**
    - 🔍 Hybrid search across all policies
    - 🤖 AI-powered responses
    - 📊 Source citations
    - ⚡ Fast and accurate
    
    **Tips:**
    - Ask specific questions
    - Use natural language
    - Try different response styles
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Examples")
    example_questions = [
        "What is the PTO policy?",
        "How do I request time off?",
        "What is the dress code?",
        "Tell me about health insurance",
        "What are the working hours?"
    ]
    
    for eq in example_questions:
        if st.button(f"💡 {eq}", key=eq, use_container_width=True):
            st.session_state["example_q"] = eq
            st.rerun()

# Handle example question clicks
if "example_q" in st.session_state:
    st.info(f"Try asking: {st.session_state['example_q']}")
    del st.session_state["example_q"]
