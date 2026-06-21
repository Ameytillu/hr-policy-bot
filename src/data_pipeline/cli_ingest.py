import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterator, Tuple

HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
LINK_RE = re.compile(r"\[([^]]+)]\([^)]+\)")


def clean_text(text: str) -> str:
    text = LINK_RE.sub(r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[`*_]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_sections(content: str, fallback_title: str) -> Iterator[Tuple[str, str]]:
    """Yield Markdown sections while retaining heading names as retrieval metadata."""
    matches = list(HEADING_RE.finditer(content))
    if not matches:
        text = clean_text(content)
        if text:
            yield fallback_title, text
        return

    preamble = clean_text(content[: matches[0].start()])
    if preamble:
        yield fallback_title, preamble
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        section_text = clean_text(content[match.end() : end])
        if section_text:
            yield clean_text(match.group(1)), section_text


def chunk_text(text: str, max_chars: int = 900) -> Iterator[str]:
    """Create sentence-aware chunks without splitting words or policy clauses unnecessarily."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            yield current
            current = ""
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)
            split_at = split_at if split_at > 0 else max_chars
            if current:
                yield current
                current = ""
            yield sentence[:split_at].strip()
            sentence = sentence[split_at:].strip()
        current = f"{current} {sentence}".strip()
    if current:
        yield current


def ingest(input_dir: Path, output_dir: Path, region: str, effective_from: str) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "corpus.jsonl"
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for path in sorted(input_dir.iterdir()):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            policy_id = path.stem
            content = path.read_text(encoding="utf-8", errors="replace")
            for section_number, (section_title, section_text) in enumerate(
                iter_sections(content, policy_id.replace("_", " ").title())
            ):
                for chunk_number, chunk in enumerate(chunk_text(section_text)):
                    if len(chunk) < 40:
                        continue
                    section_id = f"sec-{section_number:02d}-{chunk_number:02d}"
                    record = {
                        "id": hashlib.sha256(
                            f"{policy_id}:{section_title}:{chunk_number}:{chunk}".encode("utf-8")
                        ).hexdigest(),
                        "policy_id": policy_id,
                        "section": section_title,
                        "text": chunk,
                        "region": region,
                        "effective_from": effective_from,
                        "source": f"file://{path.name}#{section_id}",
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input folder with policy files")
    parser.add_argument("--out", required=True, help="Output folder for processed JSONL")
    parser.add_argument("--region", default="GLOBAL")
    parser.add_argument("--effective-from", default="2025-01-01")
    args = parser.parse_args()
    count = ingest(Path(args.inp), Path(args.out), args.region, args.effective_from)
    print(f"Processed {count} chunks into {Path(args.out) / 'corpus.jsonl'}")


if __name__ == "__main__":
    main()
