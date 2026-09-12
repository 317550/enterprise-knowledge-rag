import pytest

from app.hybrid_retriever import HybridRetrievalService
from app.schemas import FileType, RetrievalHit


def make_hit(
    chunk_id: str,
    relevance: float,
    *,
    vector_score: float | None = None,
    bm25_score: float | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        content=f"{chunk_id}的制度正文",
        relevance_score=relevance,
        vector_score=vector_score,
        bm25_score=bm25_score,
        document_id=f"doc-{chunk_id}",
        filename=f"{chunk_id}.md",
        source_path=f"data/raw/{chunk_id}.md",
        file_type=FileType.MARKDOWN,
        page=None,
        chunk_index=0,
    )


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("检索不应生成文档向量")


class FakeVectorStore:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.thresholds: list[float] = []

    def similarity_search(self, query_embedding, top_k, min_relevance):
        self.thresholds.append(min_relevance)
        return self.hits[:top_k]


class FakeLexicalRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        return self.hits[:top_k]


class IdentityReranker:
    def rerank(self, query: str, hits: list[RetrievalHit], top_n: int):
        return [
            hit.model_copy(
                update={"rerank_score": 0.9, "relevance_score": 0.9}
            )
            for hit in hits[:top_n]
        ]


class ScoreReranker:
    def rerank(self, query: str, hits: list[RetrievalHit], top_n: int):
        scores = {"both": 0.8, "vector": 0.4, "lexical": 0.7}
        ranked = [
            hit.model_copy(
                update={
                    "rerank_score": scores[hit.chunk_id],
                    "relevance_score": scores[hit.chunk_id],
                }
            )
            for hit in hits
        ]
        ranked.sort(key=lambda hit: -(hit.rerank_score or 0.0))
        return ranked[:top_n]


def build_service(vector_hits, lexical_hits, reranker=None):
    store = FakeVectorStore(vector_hits)
    service = HybridRetrievalService(
        embedder=FakeEmbedder(),
        vector_store=store,
        lexical_retriever=FakeLexicalRetriever(lexical_hits),
        reranker=reranker or IdentityReranker(),
        default_top_k=3,
        default_min_relevance=0.0,
        candidate_k=5,
        rrf_k=60,
    )
    return service, store


def test_rrf_promotes_chunk_found_by_both_retrievers() -> None:
    vector_hits = [
        make_hit("vector", 0.9, vector_score=0.9),
        make_hit("both", 0.8, vector_score=0.8),
    ]
    lexical_hits = [
        make_hit("both", 1.0, bm25_score=5.0),
        make_hit("lexical", 0.7, bm25_score=3.0),
    ]
    service, _ = build_service(vector_hits, lexical_hits)

    result = service.retrieve("订单取消")

    assert result.hits[0].chunk_id == "both"
    assert len({hit.chunk_id for hit in result.hits}) == 3
    assert result.hits[0].vector_score == 0.8
    assert result.hits[0].bm25_score == 5.0
    assert result.hits[0].rrf_score is not None


def test_vector_threshold_is_not_applied_before_fusion() -> None:
    service, store = build_service(
        [make_hit("vector", 0.2, vector_score=0.2)],
        [make_hit("lexical", 1.0, bm25_score=4.0)],
    )

    service.retrieve("精确术语", min_relevance=0.6)

    assert store.thresholds == [0.0]


def test_final_threshold_uses_reranker_score() -> None:
    service, _ = build_service(
        [
            make_hit("vector", 0.9, vector_score=0.9),
            make_hit("both", 0.8, vector_score=0.8),
        ],
        [
            make_hit("both", 1.0, bm25_score=5.0),
            make_hit("lexical", 0.7, bm25_score=3.0),
        ],
        reranker=ScoreReranker(),
    )

    result = service.retrieve("问题", min_relevance=0.65)

    assert [hit.chunk_id for hit in result.hits] == ["both", "lexical"]


@pytest.mark.parametrize(
    ("top_k", "threshold", "message"),
    [(0, 0.5, "top_k"), (3, -0.1, "min_relevance"), (3, 1.1, "min_relevance")],
)
def test_invalid_hybrid_options_are_rejected(top_k, threshold, message) -> None:
    service, _ = build_service([], [])

    with pytest.raises(ValueError, match=message):
        service.retrieve("问题", top_k=top_k, min_relevance=threshold)
