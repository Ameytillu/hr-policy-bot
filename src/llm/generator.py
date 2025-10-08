# src/llm/generator.py
import os, re
from typing import List, Dict
from textwrap import shorten
from dotenv import load_dotenv

load_dotenv()

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
    for h in hits[:5]:
        pool.extend(_sentences(h.get("text", "")))
    # keep the first informative, dedupe by lowercase
    out, seen = [], set()
    for s in pool:
        k = s.lower()
        if k in seen: 
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= max_sents:
            break
    return " ".join(out) if out else "No clear policy sentences were found."

def _bullets(hits: List[Dict], max_points: int = 5) -> str:
    points = []
    for h in hits[:max_points]:
        t = (h.get("text") or "").strip()
        t = shorten(t, width=300, placeholder="…")
        points.append(f"- {t}" if t else "- (empty snippet)")
    return "\n".join(points) if points else "- _No clear snippets found._"

def _paragraph_llm_openai(query: str, hits: List[Dict]) -> str:
    """Compose a short grounded paragraph with bracketed citations using OpenAI."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("GEN_MODEL", "gpt-4o-mini")
    chunks = [h.get("text","") for h in hits[:6]]
    sources = "\n".join([f"{i+1}) {c}" for i, c in enumerate(chunks)])
    prompt = (
        "You are an HR policy assistant. Write a concise 4–6 sentence answer to the question. "
        "Use only the snippets provided and add bracketed citations like [1], [2] that refer to the numbered snippets. "
        "Do not invent facts. Keep it clear and neutral.\n\n"
        f"Question: {query}\n\nSnippets:\n{sources}"
    )
    resp = client.chat.completions.create(
        model=model,
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
    elif style == "llm":
        try:
            body = _paragraph_llm_openai(query, hits)
        except Exception as e:
            body = _paragraph_no_llm(hits) + "\n\n_(LLM unavailable; falling back to snippet-based paragraph.)_"
    else:
        body = _bullets(hits)

    return f"**Question:** {query}\n\n{body}\n\n**Sources:**\n{_sources_numbered(hits)}"
