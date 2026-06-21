from functools import lru_cache
from typing import List
import numpy as np
from src.core.config import settings

@lru_cache(maxsize=2)
def _get_st_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)

def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    vecs = vecs.astype(np.float32, copy=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return (vecs / norms).astype(np.float32)

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    provider = settings.EMBEDDINGS_PROVIDER.lower()

    # ---- OpenAI embeddings (preferred when you set EMBEDDINGS_PROVIDER=openai) ----
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        resp = client.embeddings.create(model=settings.EMBEDDINGS_MODEL, input=texts)
        vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
        return _normalize_rows(vecs)

    # ---- Local SentenceTransformers fallback ----
    if provider != "st":
        raise ValueError(f"Unsupported embeddings provider: {provider}")
    st_name = settings.ST_MODEL
    model = _get_st_model(st_name)
    vecs = np.array(model.encode(texts, normalize_embeddings=True), dtype=np.float32)
    return vecs

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
