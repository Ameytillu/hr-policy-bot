import hashlib
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from src.core.config import settings
from src.retrieval.embeddings import embed_texts

ROOT      = Path(__file__).resolve().parents[2]
DOCS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
INDEX_DIR = ROOT / "data" / "index"
VECS_PATH = INDEX_DIR / "vectors.npy"
META_PATH = INDEX_DIR / "meta.jsonl"
INFO_PATH = INDEX_DIR / "index_info.json"

def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Missing {DOCS_PATH}. Run ingestion first.")
    docs = [json.loads(line) for line in DOCS_PATH.open("r", encoding="utf-8")]
    texts = [d["text"] for d in docs]

    provider = settings.EMBEDDINGS_PROVIDER.lower()
    model = settings.EMBEDDINGS_MODEL if provider == "openai" else settings.ST_MODEL
    print(f"🔧 building index using provider={provider}, model={model}")

    vecs = embed_texts(texts).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    np.save(VECS_PATH, vecs)

    with META_PATH.open("w", encoding="utf-8") as f:
        for document in docs:
            f.write(json.dumps(document, ensure_ascii=False) + "\n")

    info = {
        "provider": provider,
        "model": model,
        "dimension": int(vecs.shape[1]),
        "document_count": len(docs),
        "corpus_sha256": hashlib.sha256(DOCS_PATH.read_bytes()).hexdigest(),
    }
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"✅ Built dense index: {len(docs)} chunks, dim={vecs.shape[1]}")
    print(f"   - {VECS_PATH}\n   - {META_PATH}")

def load_index() -> Tuple[np.ndarray, List[Dict]]:
    vecs = np.load(VECS_PATH)
    meta = [json.loads(line) for line in META_PATH.open("r", encoding="utf-8")]
    if vecs.ndim != 2:
        raise ValueError(f"Invalid vector index shape: {vecs.shape}")
    if len(vecs) != len(meta):
        raise ValueError("Index is corrupt: vector and metadata counts differ")
    return vecs, meta
