from pathlib import Path

from app.chunking import TextChunker
from app.ingestion_service import IngestionService
from app.vector_store import ChromaVectorStore


class FakeEmbeddingProvider:
    """固定维度的可重复向量，测试不下载真实模型。"""

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [
            [float(len(text)), float(sum(map(ord, text)) % 997), 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 997), 1.0]


def build_service(tmp_path: Path):
    embedder = FakeEmbeddingProvider()
    store = ChromaVectorStore(tmp_path / "chroma", "test_knowledge")
    service = IngestionService(
        TextChunker(chunk_size=120, overlap=20),
        embedder,
        store,
    )
    return service, embedder, store


def test_first_ingestion_writes_chunks(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "policy.md").write_text("退货政策。" * 80, encoding="utf-8")
    service, embedder, store = build_service(tmp_path)

    result = service.ingest_directory(raw)

    assert result.files_inserted == 1
    assert result.chunks_written > 1
    assert result.errors == []
    assert store.count() == result.chunks_written
    assert embedder.document_calls == 1


def test_repeated_ingestion_skips_embedding(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "policy.txt"
    path.write_text("物流通常需要三到五个工作日。" * 40, encoding="utf-8")
    service, embedder, store = build_service(tmp_path)

    first = service.ingest_directory(raw)
    second = service.ingest_directory(raw)

    assert first.files_inserted == 1
    assert second.files_skipped == 1
    assert second.chunks_written == 0
    assert embedder.document_calls == 1
    assert store.count() == first.chunks_written


def test_changed_file_replaces_stale_chunks(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    path = raw / "policy.md"
    path.write_text("旧政策。" * 100, encoding="utf-8")
    service, _, store = build_service(tmp_path)
    service.ingest_directory(raw)

    path.write_text("新政策只保留这一条。", encoding="utf-8")
    result = service.ingest_directory(raw)

    assert result.files_replaced == 1
    assert result.chunks_written == 1
    assert store.count() == 1
