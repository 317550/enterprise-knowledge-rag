from app.lexical_retriever import BM25Retriever
from app.schemas import FileType, IndexedChunk


def make_chunk(index: int, content: str) -> IndexedChunk:
    return IndexedChunk(
        chunk_id=f"chunk-{index}",
        content=content,
        document_id=f"doc-{index}",
        filename=f"制度{index}.md",
        source_path=f"data/raw/制度{index}.md",
        file_type=FileType.MARKDOWN,
        page=None,
        chunk_index=0,
    )


def whitespace_tokens(text: str) -> list[str]:
    return text.lower().split()


def test_bm25_returns_exact_business_term_first() -> None:
    retriever = BM25Retriever(
        [
            make_chunk(1, "订单 打包 后 不支持 直接 取消"),
            make_chunk(2, "商品 签收 后 七天 可以 退货"),
            make_chunk(3, "客服 不得 索要 短信 验证码"),
        ],
        tokenizer=whitespace_tokens,
    )

    hits = retriever.search("短信 验证码", top_k=3)

    assert [hit.chunk_id for hit in hits] == ["chunk-3"]
    assert hits[0].bm25_score is not None
    assert hits[0].bm25_score > 0
    assert hits[0].relevance_score == 1.0


def test_bm25_empty_index_returns_no_hits() -> None:
    retriever = BM25Retriever([], tokenizer=whitespace_tokens)

    assert retriever.search("订单", top_k=3) == []
