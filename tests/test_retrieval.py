from pathlib import Path

import pytest

from app.retrieval_service import RetrievalService
from app.schemas import FileType, TextChunk
from app.vector_store import ChromaVectorStore


class ControlledEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "七天内可以申请退货。": [1.0, 0.0, 0.0],
            "标准配送需要三到五天。": [0.0, 1.0, 0.0],
            "办公区禁止吸烟。": [-1.0, 0.0, 0.0],
        }
        return [vectors[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def make_chunk(index: int, content: str, filename: str) -> TextChunk:
    return TextChunk(
        chunk_id=f"chunk-{index}",
        document_id=f"document-{index}",
        filename=filename,
        source_path=f"data/raw/{filename}",
        file_type=FileType.MARKDOWN,
        page=None,
        chunk_index=0,
        content=content,
        checksum=f"checksum-{index}",
    )


def build_retrieval(tmp_path: Path) -> RetrievalService:
    embedder = ControlledEmbeddingProvider()
    store = ChromaVectorStore(tmp_path / "chroma", "retrieval_test")
    chunks = [
        make_chunk(1, "七天内可以申请退货。", "退货政策.md"),
        make_chunk(2, "标准配送需要三到五天。", "物流说明.md"),
        make_chunk(3, "办公区禁止吸烟。", "办公制度.md"),
    ]
    for chunk in chunks:
        store.sync_document([chunk], embedder)
    return RetrievalService(embedder, store)


def test_retrieve_orders_results_and_returns_source(tmp_path: Path) -> None:
    service = build_retrieval(tmp_path)

    result = service.retrieve("  退货   期限  ", top_k=2, min_relevance=0.0)

    assert result.query == "退货 期限"
    assert len(result.hits) == 2
    assert result.hits[0].filename == "退货政策.md"
    assert result.hits[0].content == "七天内可以申请退货。"
    assert result.hits[0].relevance_score == pytest.approx(1.0)
    assert result.hits[0].page is None
    assert result.hits[0].relevance_score >= result.hits[1].relevance_score


def test_retrieve_applies_relevance_threshold(tmp_path: Path) -> None:
    service = build_retrieval(tmp_path)

    result = service.retrieve("退货期限", top_k=3, min_relevance=0.5)

    assert [hit.chunk_id for hit in result.hits] == ["chunk-1"]


def test_empty_collection_returns_no_hits(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "empty", "empty_test")
    service = RetrievalService(ControlledEmbeddingProvider(), store)

    assert service.retrieve("退货期限").hits == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_is_rejected(tmp_path: Path, query: str) -> None:
    store = ChromaVectorStore(tmp_path / "blank", "blank_test")
    service = RetrievalService(ControlledEmbeddingProvider(), store)

    with pytest.raises(ValueError, match="查询内容不能为空"):
        service.retrieve(query)


@pytest.mark.parametrize(
    ("top_k", "threshold", "message"),
    [(0, 0.5, "top_k"), (3, -0.1, "min_relevance"), (3, 1.1, "min_relevance")],
)
def test_invalid_search_options_are_rejected(
    tmp_path: Path, top_k: int, threshold: float, message: str
) -> None:
    service = build_retrieval(tmp_path)

    with pytest.raises(ValueError, match=message):
        service.retrieve("退货", top_k=top_k, min_relevance=threshold)
