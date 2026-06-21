from pathlib import Path

from src.data_pipeline.cli_ingest import chunk_text, ingest, iter_sections
from src.llm.generator import generate_answer
from src.core.config import settings
from src.retrieval.index_faiss import load_index
from src.retrieval.search import _searchable_text, _tokenize, clear_search_cache, hybrid_search
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
    assert "couldn't find reliable information" in answer


def test_tokenizer_removes_punctuation():
    assert _tokenize("PTO carry-over policy?") == ["pto", "carry-over", "policy"]


def test_searchable_text_includes_policy_and_section_names():
    text = _searchable_text(
        {"policy_id": "employee_benefits_policy", "section": "Health Insurance", "text": "Details"}
    )
    assert "employee benefits policy" in text
    assert "Health Insurance" in text


def test_offline_retrieval_uses_policy_evidence(monkeypatch):
    monkeypatch.setattr(settings, "USE_DENSE", False)
    clear_search_cache()
    hits = hybrid_search("What is the PTO carryover policy?")
    assert hits
    assert hits[0]["policy_id"] == "paid_time_off_policy"
    assert hybrid_search("zxqv unmatched gibberish") == []


def test_common_policy_questions_retrieve_the_correct_policy(monkeypatch):
    monkeypatch.setattr(settings, "USE_DENSE", False)
    clear_search_cache()
    questions = {
        "What is the attendance policy?": "attendance_and_punctuality_policy",
        "What is the employee dress code?": "dress_code_policy",
        "What employee benefits are available?": "employee_benefits_policy",
        "What is the leave of absence policy?": "leave_of_absence_policy",
        "What is the employee code of conduct?": "code_of_conduct_policy",
        "How is overtime pay handled?": "overtime_policy",
    }

    for question, expected_policy in questions.items():
        hits = hybrid_search(question)
        assert hits, question
        assert hits[0]["policy_id"] == expected_policy, question
        answer = generate_answer(question, hits, style="paragraph")
        assert "couldn't find a matching policy" not in answer, question


def test_every_indexed_policy_is_retrievable_by_name(monkeypatch):
    monkeypatch.setattr(settings, "USE_DENSE", False)
    clear_search_cache()
    _, metadata = load_index()
    policy_ids = sorted({item["policy_id"] for item in metadata})

    assert policy_ids
    for policy_id in policy_ids:
        policy_name = policy_id.replace("_", " ")
        hits = hybrid_search(f"What is the {policy_name}?")
        assert hits, policy_id
        assert hits[0]["policy_id"] == policy_id, policy_id


def test_answer_is_detailed_readable_and_focused(monkeypatch):
    monkeypatch.setattr(settings, "USE_DENSE", False)
    clear_search_cache()
    hits = hybrid_search("What is the dress code?")
    answer = generate_answer("What is the dress code?", hits, style="bullets")

    assert answer.startswith("## Dress Code Policy")
    assert "### Overview" in answer
    assert "### Key policy details" in answer
    assert "### Policy sources" in answer
    assert "synthetic document" not in answer.lower()
    assert "[Company Name]" not in answer
    assert "Social Media Policy" not in answer
    assert answer.count("\n- ") >= 3
