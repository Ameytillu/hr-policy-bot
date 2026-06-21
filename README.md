### SmartHR Copilot (Streamlit + OpenAI API)

SmartHR Copilot is an AI-powered HR assistant built with Streamlit that uses Retrieval-Augmented Generation (RAG) and the OpenAI API to answer company HR policy questions with accuracy, citations, and a personal touch.

It’s designed to make internal HR communication effortless — helping employees instantly check policies, understand benefits, or get leave-related answers without emailing HR.
The project is evolving to include features like checking leave balances, applying for leave, and accessing personalized HR data through natural conversations.

Dataset used- A custom synthetic policy dataset is used for RAG. 

#### Offline mode (default)

The application works without an OpenAI key. It uses BM25 retrieval and extractive,
source-cited answers by default.

```bash
python -m pip install -r requirements.txt
python -m src.data_pipeline.cli_ingest --in data/raw_policies --out data/processed
streamlit run streamlit_app.py
```

The checked-in index can be searched in offline mode. Rebuild it with local
SentenceTransformers only when policy files change:

```bash
set EMBEDDINGS_PROVIDER=st
python scripts/build_index.py
```

To enable OpenAI later, configure `OPENAI_API_KEY`, set `USE_LLM=true`, and optionally
set `EMBEDDINGS_PROVIDER=openai` plus `USE_DENSE=true`. Rebuild the index whenever the
embedding provider or model changes.


