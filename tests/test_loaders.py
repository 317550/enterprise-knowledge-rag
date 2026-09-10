from pathlib import Path

import pymupdf
import pytest

from app.loaders import (
    DocumentLoadError,
    UnsupportedDocumentError,
    discover_documents,
    load_document,
    normalize_text,
)


def test_normalize_text_preserves_paragraphs() -> None:
    raw = "标题\r\n\r\n\r\n第一条   适用规则。\u200b"
    assert normalize_text(raw) == "标题\n\n第一条 适用规则。"


def test_load_utf8_markdown(tmp_path: Path) -> None:
    path = tmp_path / "policy.md"
    path.write_text("# 退货政策\n\n签收后七天内可申请。", encoding="utf-8")

    documents = load_document(path, tmp_path)

    assert len(documents) == 1
    assert documents[0].filename == "policy.md"
    assert documents[0].source_path == "policy.md"
    assert documents[0].page is None
    assert documents[0].checksum


def test_pdf_loader_keeps_page_number(tmp_path: Path) -> None:
    path = tmp_path / "manual.pdf"
    pdf = pymupdf.open()
    first = pdf.new_page()
    first.insert_text((72, 72), "Warranty policy page one")
    second = pdf.new_page()
    second.insert_text((72, 72), "Return policy page two")
    pdf.save(path)
    pdf.close()

    documents = load_document(path, tmp_path)

    assert [document.page for document in documents] == [1, 2]
    assert "Warranty" in documents[0].content
    assert "Return" in documents[1].content


def test_unsupported_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentError):
        load_document(path, tmp_path)


def test_file_outside_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "knowledge"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_document(outside, root)


def test_discovery_is_recursive_and_filtered(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "nested" / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("x", encoding="utf-8")

    paths = discover_documents(tmp_path)

    assert [path.name for path in paths] == ["b.txt", "a.md"]
