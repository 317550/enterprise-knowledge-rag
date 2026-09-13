from app.context_builder import ContextBuilder
from app.schemas import FileType, RetrievalHit


def make_hit(index: int, content: str, source: str = "制度A.md") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk-{index}",
        content=content,
        relevance_score=0.9 - index / 100,
        document_id=f"doc-{source}",
        filename=source,
        source_path=source,
        file_type=FileType.MARKDOWN,
        page=None,
        chunk_index=index,
    )


def test_context_deduplicates_chunk_id_and_content() -> None:
    original = make_hit(1, "甲" * 100)
    duplicate_id = original.model_copy(update={"content": "乙" * 100})
    duplicate_content = make_hit(2, "甲" * 100, "制度B.md")

    selected = ContextBuilder().select(
        [original, duplicate_id, duplicate_content]
    )

    assert [hit.chunk_id for hit in selected] == ["chunk-1"]


def test_context_limits_chunks_per_source_but_keeps_other_sources() -> None:
    hits = [
        make_hit(1, "甲" * 100),
        make_hit(2, "乙" * 100),
        make_hit(3, "丙" * 100),
        make_hit(4, "丁" * 100, "制度B.md"),
    ]

    selected = ContextBuilder(max_chunks_per_source=2).select(hits)

    assert [hit.chunk_id for hit in selected] == ["chunk-1", "chunk-2", "chunk-4"]


def test_context_removes_exact_overlap_between_adjacent_chunks() -> None:
    overlap = "共同重叠内容" * 10
    first = make_hit(1, "第一段" + overlap)
    second = make_hit(2, overlap + "第二段")

    selected = ContextBuilder(minimum_overlap_chars=20).select([first, second])

    assert selected[1].content == "第二段"


def test_context_respects_budget_and_skips_tiny_remainder() -> None:
    hits = [make_hit(1, "甲" * 300), make_hit(2, "乙" * 300)]

    selected = ContextBuilder(max_context_chars=500).select(hits)

    assert [hit.chunk_id for hit in selected] == ["chunk-1"]
    assert sum(len(hit.content) + len(hit.filename) + 80 for hit in selected) <= 500


def test_oversized_first_chunk_is_truncated_to_budget() -> None:
    selected = ContextBuilder(max_context_chars=500).select(
        [make_hit(1, "甲" * 1000)]
    )

    assert len(selected) == 1
    assert len(selected[0].content) + len(selected[0].filename) + 80 == 500
