import re
from typing import Dict, List, Tuple

from src.core.config import settings

SYNTHETIC_NOTICE_RE = re.compile(
    r"Disclaimer:\s*This is a synthetic document generated for a model training dataset\.\s*"
    r"It is not a real policy and should not be used as such\.\s*",
    re.IGNORECASE,
)
POLICY_HEADER_RE = re.compile(
    r"^[A-Z][A-Za-z0-9 &()/,'-]+ Policy\s+Effective Date:\s*"
    r"[A-Za-z]+\s+\d{1,2},\s+\d{4}\s+I\.\s+Policy Statement\s+",
)
SECTION_HEADING_RE = re.compile(
    r"\s+[IVX]{1,5}\.\s+[A-Z][A-Za-z &/()'-]{2,40}?"
    r"(?=\s+(?:The|This|Employees|All|Eligible|Any|There|Business|Non|An|A)\b)"
)
SECTION_MARKER_RE = re.compile(r"\s+[IVX]{1,5}\.\s+(?=[A-Z])")


def _policy_title(policy_id: str | None) -> str:
    return (policy_id or "HR policy").replace("_", " ").title()


def _clean_text(text: str) -> str:
    text = SYNTHETIC_NOTICE_RE.sub("", text or "")
    text = POLICY_HEADER_RE.sub("", text)
    text = text.replace("[Company Name]", "the company")
    text = SECTION_HEADING_RE.sub(". ", text)
    text = SECTION_MARKER_RE.sub(". ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> List[str]:
    cleaned = _clean_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [
        sentence.strip()[:1].upper() + sentence.strip()[1:]
        for sentence in sentences
        if sentence.strip()
    ]


def _focused_hits(hits: List[Dict]) -> List[Dict]:
    """Drop weak, unrelated tail results while retaining all chunks of the best policy."""
    if not hits:
        return []
    top_policy = hits[0].get("policy_id")
    top_score = float(hits[0].get("score", 0.0))
    related_score = max(0.35, top_score * 0.65)
    focused = [
        hit
        for hit in hits
        if hit.get("policy_id") == top_policy or float(hit.get("score", 0.0)) >= related_score
    ]
    return focused or hits[:1]


def _evidence(hits: List[Dict], max_sentences: int = 9) -> List[Tuple[str, str]]:
    evidence: List[Tuple[str, str, bool]] = []
    seen = set()
    for source_number, hit in enumerate(hits, 1):
        for sentence in _sentences(hit.get("text", "")):
            is_complete = sentence.endswith((".", "!", "?"))
            if evidence and not evidence[-1][2]:
                previous, previous_source, _ = evidence.pop()
                sentence = f"{previous.rstrip()} {sentence.lstrip()}"
                source = f"{previous_source}, {source_number}"
            else:
                source = str(source_number)
            sentence = sentence.rstrip(" .")
            if len(sentence) < 25:
                continue
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            evidence.append((sentence, source, is_complete))
            if len(evidence) >= max_sentences:
                return [(text, source) for text, source, _ in evidence]
    return [(text, source) for text, source, _ in evidence]


def _sources_numbered(hits: List[Dict]) -> str:
    lines = []
    for number, hit in enumerate(hits, 1):
        title = _policy_title(hit.get("policy_id"))
        section = str(hit.get("section") or "Policy details").replace("_", " ")
        legacy_section = re.fullmatch(r"sec-(\d+)", section)
        if legacy_section:
            section = f"Section {int(legacy_section.group(1)) + 1}"
        effective = hit.get("effective_from")
        date = f" (effective {effective})" if effective else ""
        lines.append(f"{number}. **{title}**: {section}{date}")
    return "\n".join(lines)


def _offline_body(hits: List[Dict], style: str) -> str:
    evidence = _evidence(hits)
    if not evidence:
        return "No clear policy details were found in the retrieved documents."

    summary_sentence, summary_source = evidence[0]
    details = evidence[1:] or evidence
    summary = f"{summary_sentence}. [{summary_source}]"

    if style == "paragraph":
        detail_text = " ".join(f"{sentence}. [{source}]" for sentence, source in details)
        return f"### Overview\n\n{summary}\n\n### Detailed guidance\n\n{detail_text}"

    bullets = "\n".join(f"- {sentence}. **[{source}]**" for sentence, source in details)
    return f"### Overview\n\n{summary}\n\n### Key policy details\n\n{bullets}"


def _paragraph_llm_openai(query: str, hits: List[Dict]) -> str:
    """Compose a detailed grounded answer with bracketed citations using OpenAI."""
    from openai import OpenAI

    if not settings.USE_LLM:
        raise RuntimeError("LLM generation is disabled")
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    sources = "\n".join(
        f"{number}) {_clean_text(hit.get('text', ''))}" for number, hit in enumerate(hits, 1)
    )
    prompt = (
        "You are an HR policy assistant. Answer only from the supplied policy excerpts. "
        "Start with a direct two-sentence overview, then provide 5-8 concise bullet points with practical details. "
        "Use Markdown headings and cite every factual statement with bracketed source numbers such as [1]. "
        "Do not mention synthetic data, invent requirements, or include unrelated policies.\n\n"
        f"Question: {query}\n\nPolicy excerpts:\n{sources}"
    )
    response = client.chat.completions.create(
        model=settings.GEN_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_answer(query: str, hits: List[Dict], style: str = "bullets") -> str:
    if not hits:
        return (
            "### No matching policy found\n\n"
            "I couldn't find reliable information for that question. Try using the policy name or "
            "contact HR for clarification."
        )

    focused_hits = _focused_hits(hits)
    title = _policy_title(focused_hits[0].get("policy_id"))

    if style == "llm" and settings.USE_LLM:
        try:
            body = _paragraph_llm_openai(query, focused_hits)
        except Exception:
            body = _offline_body(focused_hits, "paragraph")
    else:
        body = _offline_body(focused_hits, style)

    return (
        f"## {title}\n\n"
        f"{body}\n\n"
        "---\n\n"
        "### Policy sources\n\n"
        f"{_sources_numbered(focused_hits)}"
    )
