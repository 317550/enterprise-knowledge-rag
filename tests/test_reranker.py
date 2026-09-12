import pytest

from app.reranker import CrossEncoderReranker
from app.schemas import FileType, RetrievalHit


class ArrayLike:
    def __init__(self, values) -> None:
        self.values = values

    def tolist(self):
        return self.values


class FakeCrossEncoder:
    def predict(self, pairs, batch_size, show_progress_bar, activation_fn):
        assert len(pairs) == 3
        assert batch_size == 2
        assert show_progress_bar is False
        assert activation_fn is not None
        return ArrayLike([-1.0, 3.0, 0.0])


def make_hit(index: int) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk-{index}",
        content=f"正文{index}",
        relevance_score=0.5,
        rrf_score=0.03 - index / 1000,
        document_id=f"doc-{index}",
        filename=f"制度{index}.md",
        source_path=f"data/raw/制度{index}.md",
        file_type=FileType.MARKDOWN,
        page=None,
        chunk_index=0,
    )


def test_cross_encoder_changes_order_and_limits_results() -> None:
    reranker = CrossEncoderReranker("fake-model", batch_size=2)
    reranker._model = FakeCrossEncoder()

    result = reranker.rerank(
        "用户问题", [make_hit(1), make_hit(2), make_hit(3)], top_n=2
    )

    assert [hit.chunk_id for hit in result] == ["chunk-2", "chunk-3"]
    assert result[0].rerank_score == pytest.approx(0.952574, abs=1e-6)
    assert result[0].relevance_score == result[0].rerank_score


def test_reranker_empty_candidates_do_not_load_model() -> None:
    reranker = CrossEncoderReranker("fake-model")

    assert reranker.rerank("问题", [], top_n=3) == []
    assert reranker._model is None
