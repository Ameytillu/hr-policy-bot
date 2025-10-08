# SmartHR Copilot (Streamlit + PostgreSQL)

An AI-powered RAG chatbot that answers HR policy questions with citations and personalization.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OPENAI_API_KEY + DATABASE_URL
streamlit run src/app/main.py
```

### Secrets
- Local: put keys in `.env` (never commit).
- Streamlit Cloud: use **App → Settings → Secrets**. See `streamlit.secrets.example.toml`.

### Structure
See `src/` packages for UI (`app`), retrieval (`retrieval`), LLM calls (`llm`), and storage (`storage`).
