# src/llm/generator.py
import re
from typing import List, Dict
from textwrap import shorten
from src.core.config import settings

def _sources_numbered(hits: List[Dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h.get('source','')} (effective {h.get('effective_from','n/a')})")
    return "\n".join(f"- {line}" for line in lines)

def _sentences(text: str) -> List[str]:
    # simple sentence splitter
    sents = re.split(r'(?<=[.!?])\s+', (text or "").strip())
    return [s.strip() for s in sents if len(s.strip()) >= 30]

def _paragraph_no_llm(hits: List[Dict], max_sents: int = 5) -> str:
    pool = []
    for source_number, hit in enumerate(hits[:5], 1):
        pool.extend((sentence, source_number) for sentence in _sentences(hit.get("text", "")))
    # keep the first informative, dedupe by lowercase
    out, seen = [], set()
    for sentence, source_number in pool:
        k = sentence.lower()
        if k in seen: 
            continue
        seen.add(k)
        out.append(f"{sentence} [{source_number}]")
        if len(out) >= max_sents:
            break
    return " ".join(out) if out else "No clear policy sentences were found."

def _bullets(hits: List[Dict], max_points: int = 5) -> str:
    points = []
    for source_number, h in enumerate(hits[:max_points], 1):
        t = (h.get("text") or "").strip()
        t = shorten(t, width=300, placeholder="…")
        points.append(f"- {t} [{source_number}]" if t else "- (empty snippet)")
    return "\n".join(points) if points else "- _No clear snippets found._"

def _paragraph_llm_openai(query: str, hits: List[Dict]) -> str:
    """Compose a short grounded paragraph with bracketed citations using OpenAI."""
    from openai import OpenAI
    if not settings.USE_LLM:
        raise RuntimeError("LLM generation is disabled")
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    chunks = [h.get("text","") for h in hits[:6]]
    sources = "\n".join([f"{i+1}) {c}" for i, c in enumerate(chunks)])
    prompt = (
        "You are an HR policy assistant. Write a concise 4–6 sentence answer to the question. "
        "Use only the snippets provided and add bracketed citations like [1], [2] that refer to the numbered snippets. "
        "Do not invent facts. Keep it clear and neutral.\n\n"
        f"Question: {query}\n\nSnippets:\n{sources}"
    )
    resp = client.chat.completions.create(
        model=settings.GEN_MODEL,
        temperature=0.2,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.choices[0].message.content.strip()

def generate_answer(query: str, hits: List[Dict], style: str = "bullets") -> str:
    """
    style: 'bullets' | 'paragraph' | 'llm'
    - bullets    -> extractive bullet list
    - paragraph  -> no-LLM paragraph from snippets
    - llm        -> OpenAI-written paragraph grounded in snippets
    """
    if not hits:
        return f"**Question:** {query}\n\n_Sorry, I couldn't find a matching policy. Try adding role/region or keywords._"

    if style == "paragraph":
        body = _paragraph_no_llm(hits)
    elif style == "llm" and settings.USE_LLM:
        try:
            body = _paragraph_llm_openai(query, hits)
        except Exception:
            body = _paragraph_no_llm(hits) + "\n\n_(LLM unavailable; falling back to snippet-based paragraph.)_"
    elif style == "llm":
        body = _paragraph_no_llm(hits) + "\n\n_(Offline mode: LLM generation is disabled.)_"
    else:
        body = _bullets(hits)

    return f"**Question:** {query}\n\n{body}\n\n**Sources:**\n{_sources_numbered(hits)}"
