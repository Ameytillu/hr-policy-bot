from typing import List, Dict, Tuple
from functools import lru_cache
import os, numpy as np, json
from rank_bm25 import BM25Okapi
from pathlib import Path

from src.retrieval.index_faiss import load_index
from src.retrieval.embeddings import embed_query

TOP_K = int(os.getenv("TOP_K", "6"))
USE_DENSE = os.getenv("USE_DENSE", "1") == "1"  # set USE_DENSE=0 to force BM25-only

@lru_cache(maxsize=1)
def _load_meta_corpus() -> Tuple[np.ndarray, List[Dict], BM25Okapi]:
    V, meta = load_index()               # V shape could be (N,1) if zeros
    tokenized = [m["text"].lower().split() for m in meta]
    bm25 = BM25Okapi(tokenized)
    return V, meta, bm25

def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def hybrid_search(query: str) -> List[Dict]:
    V, meta, bm25 = _load_meta_corpus()

    # ----- Dense (guarded) -----
    dense_idxs, dnorm = np.array([], dtype=int), np.array([], dtype=np.float32)
    if USE_DENSE and V.size > 0:
        try:
            qv = embed_query(query).astype("float32")
            if qv.ndim == 1 and V.shape[1] == qv.shape[0] and V.shape[1] > 1:
                qv /= (np.linalg.norm(qv) + 1e-12)
                dense_scores_all = V @ qv  # (N,)
                dense_idxs = np.argsort(dense_scores_all)[::-1][:TOP_K]
                dnorm = _normalize(dense_scores_all[dense_idxs])
        except Exception as e:
            # Dense disabled silently; BM25 will still work.
            dense_idxs, dnorm = np.array([], dtype=int), np.array([], dtype=np.float32)

    # ----- BM25 -----
    bm25_scores_all = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores_all)[::-1][:TOP_K]
    bnorm = _normalize(bm25_scores_all[bm25_top])

    # ----- Fusion -----
    cand = set(dense_idxs.tolist()) | set(bm25_top.tolist())
    dense_pos = {idx: i for i, idx in enumerate(dense_idxs)}
    bm25_pos = {idx: i for i, idx in enumerate(bm25_top)}

    results = []
    for idx in cand:
        d = float(dnorm[dense_pos[idx]]) if idx in dense_pos else 0.0
        b = float(bnorm[bm25_pos[idx]]) if idx in bm25_pos else 0.0
        score = (0.6 * d + 0.4 * b) if dnorm.size else b  # BM25-only if dense off
        m = meta[idx]
        results.append({
            "text": m["text"],
            "source": m.get("source", f"policy://{m.get('policy_id','unknown')}/{m.get('section','')}"),
            "score": score,
            "policy_id": m.get("policy_id"),
            "section": m.get("section"),
            "effective_from": m.get("effective_from"),
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:TOP_K]
