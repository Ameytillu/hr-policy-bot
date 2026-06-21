from pathlib import Path

from src.data_pipeline.cli_ingest import chunk_text, ingest, iter_sections
from src.llm.generator import generate_answer
from src.core.config import settings
from src.retrieval.search import _tokenize, hybrid_search
from src.storage.models import Base


def test_database_models_are_mappable():
    assert {"users", "chat_sessions", "messages", "policy_chunks"} <= set(Base.metadata.tables)


def test_markdown_sections_are_preserved():
    sections = list(iter_sections("# Leave\nEmployees receive paid leave.\n## Carryover\nFive days carry over.", "Policy"))
    assert sections == [
        ("Leave", "Employees receive paid leave."),
        ("Carryover", "Five days carry over."),
    ]


def test_chunks_respect_maximum_size():
    chunks = list(chunk_text("A policy sentence. " * 100, max_chars=100))
    assert chunks
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_ingest_supports_text_files(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "leave.txt").write_text(
        "Employees may request paid leave after completing the required waiting period.",
        encoding="utf-8",
    )
    assert ingest(source, output, "GLOBAL", "2025-01-01") == 1
    assert "leave.txt" in (output / "corpus.jsonl").read_text(encoding="utf-8")


def test_offline_answer_needs_retrieval_evidence():
    answer = generate_answer("unrelated question", [])
    assert "couldn't find a matching policy" in answer


def test_tokenizer_removes_punctuation():
    assert _tokenize("PTO carry-over policy?") == ["pto", "carry-over", "policy"]


def test_offline_retrieval_uses_policy_evidence(monkeypatch):
    monkeypatch.setattr(settings, "USE_DENSE", False)
    hits = hybrid_search("What is the PTO carryover policy?")
    assert hits
    assert hits[0]["policy_id"] == "paid_time_off_policy"
    assert hybrid_search("zxqv unmatched gibberish") == []
