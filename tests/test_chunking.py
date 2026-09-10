from app.chunking import TextChunker
from app.schemas import FileType, SourceDocument


def source(content: str) -> SourceDocument:
    return SourceDocument(
        document_id="doc-1",
        filename="policy.md",
        source_path="policy.md",
        file_type=FileType.MARKDOWN,
        content=content,
        checksum="checksum-1",
    )


def test_chunks_respect_maximum_size() -> None:
    text = "。".join(["这是一条用于测试的企业售后政策"] * 80)
    chunks = TextChunker(chunk_size=120, overlap=20).split(text)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 120 for chunk in chunks)


def test_chunk_ids_are_stable() -> None:
    chunker = TextChunker(chunk_size=120, overlap=20)
    first = chunker.build_chunks([source("相同内容" * 50)])
    second = chunker.build_chunks([source("相同内容" * 50)])

    assert [chunk.chunk_id for chunk in first] == [
        chunk.chunk_id for chunk in second
    ]


def test_invalid_chunk_configuration_is_rejected() -> None:
    try:
        TextChunker(chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("应该拒绝overlap等于chunk_size")
