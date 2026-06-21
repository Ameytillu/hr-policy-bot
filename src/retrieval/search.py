import logging
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from src.core.config import settings
from src.retrieval.embeddings import embed_query
from src.retrieval.index_faiss import load_index

logger = logging.getLogger(__name__)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def _searchable_text(item: Dict) -> str:
    """Include policy and section names so direct policy questions rank reliably."""
    policy_name = str(item.get("policy_id", "")).replace("_", " ")
    section_name = str(item.get("section", "")).replace("_", " ")
    return f"{policy_name} {section_name} {item.get('text', '')}"


@lru_cache(maxsize=1)
def _load_meta_corpus() -> Tuple[np.ndarray, List[Dict], BM25Okapi]:
    vectors, meta = load_index()
    if len(vectors) != len(meta):
        raise ValueError("Index is corrupt: vector and metadata counts differ")
    bm25 = BM25Okapi([_tokenize(_searchable_text(item)) for item in meta])
    return vectors, meta, bm25


def clear_search_cache() -> None:
    """Reload index files after rebuilding them in a long-running process."""
    _load_meta_corpus.cache_clear()


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    minimum, maximum = float(arr.min()), float(arr.max())
    if maximum - minimum < 1e-9:
        return np.ones_like(arr) if maximum > 0 else np.zeros_like(arr)
    return (arr - minimum) / (maximum - minimum)


def hybrid_search(query: str) -> List[Dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    vectors, meta, bm25 = _load_meta_corpus()
    top_k = settings.TOP_K

    dense_idxs = np.array([], dtype=int)
    dense_normalized = np.array([], dtype=np.float32)
    if settings.USE_DENSE and vectors.size:
        try:
            query_vector = embed_query(query).astype("float32")
            if vectors.ndim != 2 or query_vector.ndim != 1 or vectors.shape[1] != query_vector.shape[0]:
                raise ValueError(
                    f"Index dimension {vectors.shape} does not match query dimension {query_vector.shape}"
                )
            query_vector /= np.linalg.norm(query_vector) + 1e-12
            dense_scores = vectors @ query_vector
            dense_idxs = np.argsort(dense_scores)[::-1][:top_k]
            dense_normalized = _normalize(dense_scores[dense_idxs])
        except Exception as exc:
            logger.warning("Dense retrieval unavailable; using BM25 only: %s", exc)

    bm25_scores = bm25.get_scores(query_tokens)
    positive_bm25 = np.flatnonzero(bm25_scores > 0)
    bm25_idxs = positive_bm25[np.argsort(bm25_scores[positive_bm25])[::-1]][:top_k]
    bm25_normalized = _normalize(bm25_scores[bm25_idxs])

    candidates = set(dense_idxs.tolist()) | set(bm25_idxs.tolist())
    dense_positions = {idx: pos for pos, idx in enumerate(dense_idxs)}
    bm25_positions = {idx: pos for pos, idx in enumerate(bm25_idxs)}
    results = []

    for idx in candidates:
        dense_score = (
            float(dense_normalized[dense_positions[idx]]) if idx in dense_positions else 0.0
        )
        bm25_score = (
            float(bm25_normalized[bm25_positions[idx]]) if idx in bm25_positions else 0.0
        )
        score = 0.6 * dense_score + 0.4 * bm25_score if dense_normalized.size else bm25_score
        if score < settings.MIN_RELEVANCE_SCORE:
            continue
        item = meta[idx]
        results.append(
            {
                "text": item["text"],
                "source": item.get(
                    "source", f"policy://{item.get('policy_id', 'unknown')}/{item.get('section', '')}"
                ),
                "score": score,
                "policy_id": item.get("policy_id"),
                "section": item.get("section"),
                "effective_from": item.get("effective_from"),
            }
        )

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
