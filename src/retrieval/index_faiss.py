import os, json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # make sure EMBEDDINGS_PROVIDER from .env is seen here, too

from src.retrieval.embeddings import embed_texts

ROOT      = Path(__file__).resolve().parents[2]
DOCS_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
INDEX_DIR = ROOT / "data" / "index"
VECS_PATH = INDEX_DIR / "vectors.npy"
META_PATH = INDEX_DIR / "meta.jsonl"

def build_index() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not DOCS_PATH.exists():
        raise FileNotFoundError(f"Missing {DOCS_PATH}. Run ingestion first.")
    docs = [json.loads(l) for l in DOCS_PATH.open("r", encoding="utf-8")]
    texts = [d["text"] for d in docs]

    provider = os.getenv("EMBEDDINGS_PROVIDER", "st").lower()
    model = os.getenv("EMBEDDINGS_MODEL", os.getenv("ST_MODEL", "all-MiniLM-L6-v2"))
    print(f"🔧 building index using provider={provider}, model={model}")

    vecs = embed_texts(texts).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms
    np.save(VECS_PATH, vecs)

    with META_PATH.open("w", encoding="utf-8") as f:
        for d in docs: f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"✅ Built dense index: {len(docs)} chunks, dim={vecs.shape[1]}")
    print(f"   - {VECS_PATH}\n   - {META_PATH}")

def load_index() -> Tuple[np.ndarray, List[Dict]]:
    vecs = np.load(VECS_PATH)
    meta = [json.loads(l) for l in META_PATH.open("r", encoding="utf-8")]
    return vecs, meta
