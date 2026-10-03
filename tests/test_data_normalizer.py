"""Tests for core.data_normalizer."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_normalizer import normalize_qa_item, extract_options, validate_qa_item


# ── normalize_qa_item tests ─────────────────────────────────────────

def test_normalize_qa_item_standard_fields():
    raw = {
        "question": "What is 2+2?",
        "answer": "4",
        "category": "math",
        "difficulty": "easy",
    }
    result = normalize_qa_item(raw)
    assert result["question"] == "What is 2+2?"
    assert result["answer"] == "4"
    assert result["category"] == ["math"]
    assert result["difficulty"] == "easy"
    assert result["metadata"] == {}


def test_normalize_qa_item_fallback_keys():
    raw = {
        "problem": "Solve for x",
        "solution": "x=5",
        "subject": "algebra",
        "level": "hard",
    }
    result = normalize_qa_item(raw)
    assert result["question"] == "Solve for x"
    assert result["answer"] == "x=5"
    assert result["category"] == ["algebra"]
    assert result["difficulty"] == "hard"


def test_normalize_qa_item_category_list():
    raw = {
        "question": "Q",
        "answer": "A",
        "category": ["math", "algebra"],
    }
    result = normalize_qa_item(raw)
    assert result["category"] == ["math", "algebra"]


def test_normalize_qa_item_metadata_preserved():
    raw = {
        "question": "Q",
        "answer": "A",
        "source": "exam_2024",
        "year": 2024,
    }
    result = normalize_qa_item(raw)
    assert result["metadata"] == {"source": "exam_2024", "year": 2024}


def test_normalize_qa_item_empty_values_skipped():
    raw = {
        "question": "",
        "answer": None,
        "category": [],
        "difficulty": "",
    }
    result = normalize_qa_item(raw)
    assert result["question"] == ""
    assert result["answer"] == ""
    assert result["category"] == []
    assert result["difficulty"] == ""


# ── extract_options tests ────────────────────────────────────────────

def test_extract_options_direct_keys():
    raw = {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}
    result = extract_options(raw)
    assert result == ["A. Option A", "B. Option B", "C. Option C", "D. Option D"]


def test_extract_options_with_prefix():
    raw = {"option_A": "Opt A", "option_B": "Opt B", "option_C": "Opt C", "option_D": "Opt D"}
    result = extract_options(raw)
    assert result == ["A. Opt A", "B. Opt B", "C. Opt C", "D. Opt D"]


def test_extract_options_partial():
    raw = {"A": "Opt A", "B": "Opt B"}
    result = extract_options(raw)
    assert result == ["A. Opt A", "B. Opt B"]


def test_extract_options_missing():
    raw = {"question": "Q", "answer": "A"}
    result = extract_options(raw)
    assert result == []


# ── validate_qa_item tests ───────────────────────────────────────────

def test_validate_qa_item_valid():
    normalized = {"question": "Q", "answer": "A", "category": [], "difficulty": "", "metadata": {}}
    warnings = validate_qa_item(normalized)
    assert warnings == []


def test_validate_qa_item_missing_question():
    normalized = {"question": "", "answer": "A", "category": [], "difficulty": "", "metadata": {}}
    warnings = validate_qa_item(normalized)
    assert "question 字段为空" in warnings


def test_validate_qa_item_missing_answer():
    normalized = {"question": "Q", "answer": "", "category": [], "difficulty": "", "metadata": {}}
    warnings = validate_qa_item(normalized)
    assert "answer 字段为空" in warnings


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
