import json
from pathlib import Path

import pytest

from app.evaluation_dataset import load_evaluation_cases


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.jsonl"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def test_evaluation_dataset_has_required_scale_and_unique_ids() -> None:
    cases = load_evaluation_cases(DATASET_PATH)

    assert len(cases) >= 32
    assert len({case.case_id for case in cases}) == len(cases)
    assert len({case.query for case in cases}) == len(cases)


def test_expected_sources_exist_in_raw_corpus() -> None:
    cases = load_evaluation_cases(DATASET_PATH)
    available = {path.name for path in RAW_DATA_DIR.iterdir() if path.is_file()}

    for case in cases:
        assert set(case.expected_sources) <= available


def test_every_document_is_covered_by_answerable_cases() -> None:
    cases = load_evaluation_cases(DATASET_PATH)
    expected = {
        source
        for case in cases
        if case.answerable
        for source in case.expected_sources
    }
    available = {path.name for path in RAW_DATA_DIR.iterdir() if path.is_file()}

    assert expected == available


def test_dataset_covers_retrieval_challenges() -> None:
    cases = load_evaluation_cases(DATASET_PATH)
    query_types = {case.query_type for case in cases}

    assert query_types == {"semantic", "exact", "multi_source", "unanswerable"}
    assert sum(not case.answerable for case in cases) >= 6


def test_invalid_json_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    valid = {
        "case_id": "ok",
        "query": "多久可以退货？",
        "answerable": True,
        "expected_sources": ["退换货政策.md"],
        "expected_answer_keywords": ["七个自然日"],
        "category": "退换货",
        "query_type": "semantic",
    }
    path.write_text(
        json.dumps(valid, ensure_ascii=False) + "\nnot-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="第2行"):
        load_evaluation_cases(path)


def test_unanswerable_case_cannot_declare_source(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    payload = {
        "case_id": "invalid",
        "query": "未知问题",
        "answerable": False,
        "expected_sources": ["退换货政策.md"],
        "expected_answer_keywords": [],
        "category": "不可回答",
        "query_type": "unanswerable",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="不可回答问题不能声明"):
        load_evaluation_cases(path)
