import os
from functools import lru_cache
from typing import List
import numpy as np
from dotenv import load_dotenv

# ensure .env is loaded for ANY entry point (build_index, streamlit, console, etc.)
load_dotenv()

@lru_cache(maxsize=2)
def _get_st_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    vecs = vecs.astype(np.float32, copy=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)

def embed_texts(texts: List[str]) -> np.ndarray:
    provider = os.getenv("EMBEDDINGS_PROVIDER", "st").lower()

    # ---- OpenAI embeddings (preferred when you set EMBEDDINGS_PROVIDER=openai) ----
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
            resp = client.embeddings.create(model=model, input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            return _normalize_rows(vecs)
        except Exception as e:
            # fall through to ST so we don't crash
            print(f"[embeddings] OpenAI failed ({e}); falling back to ST")

    # ---- Local SentenceTransformers fallback ----
    st_name = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
    model = _get_st_model(st_name)
    vecs = np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
    return vecs

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
