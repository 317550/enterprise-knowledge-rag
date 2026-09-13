from app.evaluation_dataset import EvaluationCase
from app.evaluation_service import EvaluationService
from app.schemas import (
    AnswerCitation,
    FileType,
    RAGAnswer,
    RetrievalHit,
    RetrievalResult,
)


def case(
    case_id: str,
    query: str,
    answerable: bool,
    sources: list[str],
    keywords: list[str],
    query_type: str = "semantic",
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        query=query,
        answerable=answerable,
        expected_sources=sources,
        expected_answer_keywords=keywords,
        category="测试",
        query_type=query_type,
    )


def hit(index: int, filename: str, score: float = 0.9) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"chunk-{index}",
        content="测试正文",
        relevance_score=score,
        vector_score=score,
        bm25_score=float(index),
        rrf_score=0.03,
        rerank_score=score,
        document_id=f"doc-{index}",
        filename=filename,
        source_path=filename,
        file_type=FileType.MARKDOWN,
        page=None,
        chunk_index=0,
    )


class StubRetrieval:
    def __init__(self, results: dict[str, list[RetrievalHit]]) -> None:
        self.results = results

    def retrieve(self, query, top_k=None, min_relevance=None):
        return RetrievalResult(query=query, hits=self.results[query][:top_k])


class FailingRetrieval:
    def retrieve(self, query, top_k=None, min_relevance=None):
        raise RuntimeError("检索失败")


class StubRAG:
    def __init__(self, answers: dict[str, RAGAnswer]) -> None:
        self.answers = answers

    def answer(self, query, top_k=None, min_relevance=None):
        return self.answers[query]


def citation(index: int, filename: str) -> AnswerCitation:
    return AnswerCitation(
        citation_index=index,
        chunk_id=f"chunk-{index}",
        filename=filename,
        source_path=filename,
        page=None,
        relevance_score=0.9,
        excerpt="测试",
    )


def test_retrieval_metrics_include_multi_source_recall_and_refusal() -> None:
    cases = [
        case("a1", "问题1", True, ["A.md"], ["甲"]),
        case("a2", "问题2", True, ["A.md", "B.md"], ["乙"], "multi_source"),
        case("u1", "问题3", False, [], [], "unanswerable"),
    ]
    retrieval = StubRetrieval(
        {
            "问题1": [hit(1, "A.md")],
            "问题2": [hit(2, "X.md"), hit(3, "A.md"), hit(4, "B.md")],
            "问题3": [],
        }
    )

    report = EvaluationService().evaluate_retrieval(
        cases, retrieval, "test", top_k=3
    )

    assert report.hit_at_1 == 0.5
    assert report.hit_at_3 == 1.0
    assert report.mrr == 0.75
    assert report.source_recall_at_3 == 1.0
    assert report.unanswerable_empty_rate == 1.0
    assert report.failed_cases == 0
    assert report.cases[0].returned_hits[0].filename == "A.md"
    assert report.cases[0].returned_hits[0].rerank_score == 0.9


def test_retrieval_error_is_recorded_without_stopping_batch() -> None:
    cases = [case("a1", "问题", True, ["A.md"], ["甲"])]

    report = EvaluationService().evaluate_retrieval(
        cases, FailingRetrieval(), "test", top_k=3
    )

    assert report.failed_cases == 1
    assert "RuntimeError" in report.cases[0].error


def test_rag_metrics_check_keywords_citations_and_refusal() -> None:
    cases = [
        case("a1", "问题1", True, ["A.md"], ["七天", "退货"]),
        case("u1", "问题2", False, [], [], "unanswerable"),
    ]
    rag = StubRAG(
        {
            "问题1": RAGAnswer(
                query="问题1",
                answer="可以在七天内退货。[资料1]",
                refused=False,
                citations=[citation(1, "A.md")],
            ),
            "问题2": RAGAnswer(
                query="问题2",
                answer="资料不足",
                refused=True,
                citations=[],
            ),
        }
    )

    report = EvaluationService().evaluate_rag(cases, rag, top_k=3)

    assert report.answerable_success_rate == 1.0
    assert report.refusal_accuracy == 1.0
    assert report.citation_precision == 1.0
    assert report.citation_recall == 1.0
    assert report.keyword_coverage == 1.0
    assert report.failed_cases == 0
    assert report.cases[0].answer_text == "可以在七天内退货。[资料1]"
    assert report.cases[0].matched_keywords == ["七天", "退货"]
    assert report.cases[0].missing_keywords == []


def test_threshold_calibration_preserves_best_answerable_recall() -> None:
    cases = [
        case("a1", "问题1", True, ["A.md"], ["甲"]),
        case("a2", "问题2", True, ["B.md"], ["乙"]),
        case("u1", "问题3", False, [], [], "unanswerable"),
    ]
    retrieval = StubRetrieval(
        {
            "问题1": [hit(1, "A.md", 0.31)],
            "问题2": [hit(2, "B.md", 0.72)],
            "问题3": [hit(3, "X.md", 0.20)],
        }
    )
    report = EvaluationService().evaluate_retrieval(
        cases, retrieval, "test", top_k=3, min_relevance=0.0
    )

    calibration = EvaluationService.calibrate_threshold(report, step=0.01)

    assert calibration.recommended_threshold == 0.26
    assert calibration.best_answerable_hit_at_3 == 1.0
    assert calibration.best_answerable_source_recall_at_3 == 1.0
    assert calibration.recommended_unanswerable_empty_rate == 1.0
