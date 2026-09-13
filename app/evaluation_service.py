"""固定评测集上的检索与端到端RAG指标计算。"""

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field

from app.evaluation_dataset import EvaluationCase
from app.schemas import RAGAnswer, RetrievalResult


class RetrievalProvider(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> RetrievalResult:
        """返回检索结果。"""


class AnswerProvider(Protocol):
    def answer(
        self,
        query: str,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> RAGAnswer:
        """返回带引用或拒答的RAG结果。"""


class RetrievalCaseResult(BaseModel):
    case_id: str
    query_type: str
    answerable: bool
    expected_sources: list[str]
    returned_sources: list[str] = Field(default_factory=list)
    returned_hits: list["RetrievalHitAudit"] = Field(default_factory=list)
    first_relevant_rank: int | None = None
    hit_at_1: bool = False
    hit_at_3: bool = False
    reciprocal_rank: float = 0.0
    source_recall_at_3: float = 0.0
    unanswerable_empty: bool | None = None
    error: str | None = None


class RetrievalEvaluationReport(BaseModel):
    system_name: str
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    source_recall_at_3: float
    unanswerable_empty_rate: float
    failed_cases: int
    cases: list[RetrievalCaseResult]


class RetrievalHitAudit(BaseModel):
    rank: int = Field(ge=1)
    chunk_id: str
    filename: str
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    relevance_score: float


class ThresholdPoint(BaseModel):
    threshold: float
    answerable_hit_at_3: float
    answerable_source_recall_at_3: float
    unanswerable_empty_rate: float


class ThresholdCalibrationReport(BaseModel):
    policy: str
    recommended_threshold: float
    best_answerable_hit_at_3: float
    best_answerable_source_recall_at_3: float
    recommended_unanswerable_empty_rate: float
    points: list[ThresholdPoint]


class RAGCaseResult(BaseModel):
    case_id: str
    query_type: str
    answerable: bool
    refused: bool | None = None
    expected_sources: list[str]
    citation_sources: list[str] = Field(default_factory=list)
    answer_text: str | None = None
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)
    answered_correctly: bool = False
    refusal_correct: bool | None = None
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    keyword_coverage: float = 0.0
    error: str | None = None


class RAGEvaluationReport(BaseModel):
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    answerable_success_rate: float
    refusal_accuracy: float
    citation_precision: float
    citation_recall: float
    keyword_coverage: float
    failed_cases: int
    cases: list[RAGCaseResult]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


class EvaluationService:
    def evaluate_retrieval(
        self,
        cases: list[EvaluationCase],
        retrieval: RetrievalProvider,
        system_name: str,
        top_k: int = 5,
        min_relevance: float | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> RetrievalEvaluationReport:
        if not cases:
            raise ValueError("评测用例不能为空")
        if top_k < 3:
            raise ValueError("检索评测的top_k不能小于3")

        results: list[RetrievalCaseResult] = []
        for index, case in enumerate(cases, start=1):
            if progress:
                progress(index, len(cases), case.case_id)
            try:
                response = retrieval.retrieve(
                    case.query,
                    top_k=top_k,
                    min_relevance=min_relevance,
                )
                results.append(self._retrieval_case(case, response))
            except Exception as exc:  # 单个失败不能丢失整批评测结果
                results.append(
                    RetrievalCaseResult(
                        case_id=case.case_id,
                        query_type=case.query_type,
                        answerable=case.answerable,
                        expected_sources=case.expected_sources,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        answerable = [item for item in results if item.answerable]
        unanswerable = [item for item in results if not item.answerable]
        return RetrievalEvaluationReport(
            system_name=system_name,
            total_cases=len(results),
            answerable_cases=len(answerable),
            unanswerable_cases=len(unanswerable),
            hit_at_1=_mean([float(item.hit_at_1) for item in answerable]),
            hit_at_3=_mean([float(item.hit_at_3) for item in answerable]),
            mrr=_mean([item.reciprocal_rank for item in answerable]),
            source_recall_at_3=_mean(
                [item.source_recall_at_3 for item in answerable]
            ),
            unanswerable_empty_rate=_mean(
                [float(item.unanswerable_empty is True) for item in unanswerable]
            ),
            failed_cases=sum(item.error is not None for item in results),
            cases=results,
        )

    @staticmethod
    def _retrieval_case(
        case: EvaluationCase,
        response: RetrievalResult,
    ) -> RetrievalCaseResult:
        sources = [hit.filename for hit in response.hits]
        unique_sources = list(dict.fromkeys(sources))
        expected = set(case.expected_sources)
        first_rank = next(
            (rank for rank, source in enumerate(sources, start=1) if source in expected),
            None,
        )
        first_three = set(sources[:3])
        recall = len(expected & first_three) / len(expected) if expected else 0.0
        return RetrievalCaseResult(
            case_id=case.case_id,
            query_type=case.query_type,
            answerable=case.answerable,
            expected_sources=case.expected_sources,
            returned_sources=unique_sources,
            returned_hits=[
                RetrievalHitAudit(
                    rank=rank,
                    chunk_id=hit.chunk_id,
                    filename=hit.filename,
                    vector_score=hit.vector_score,
                    bm25_score=hit.bm25_score,
                    rrf_score=hit.rrf_score,
                    rerank_score=hit.rerank_score,
                    relevance_score=hit.relevance_score,
                )
                for rank, hit in enumerate(response.hits, start=1)
            ],
            first_relevant_rank=first_rank,
            hit_at_1=bool(first_rank == 1),
            hit_at_3=bool(first_rank is not None and first_rank <= 3),
            reciprocal_rank=round(1.0 / first_rank, 6) if first_rank else 0.0,
            source_recall_at_3=round(recall, 6),
            unanswerable_empty=(not response.hits) if not case.answerable else None,
        )

    @staticmethod
    def calibrate_threshold(
        report: RetrievalEvaluationReport,
        step: float = 0.01,
    ) -> ThresholdCalibrationReport:
        """从零阈值结果扫描阈值；优先保持最佳召回，再提高拒答率。"""

        if not 0.0 < step <= 1.0:
            raise ValueError("threshold step必须在0到1之间")
        count = int(round(1.0 / step))
        thresholds = sorted(
            {round(min(index * step, 1.0), 6) for index in range(count + 1)}
            | {1.0}
        )
        points: list[ThresholdPoint] = []
        answerable = [case for case in report.cases if case.answerable]
        unanswerable = [case for case in report.cases if not case.answerable]

        for threshold in thresholds:
            answerable_hits = []
            answerable_source_recalls = []
            for case in answerable:
                # 与生产检索保持一致：Top-1通过门槛后保留完整Top-K。
                retained = (
                    case.returned_hits
                    if case.returned_hits
                    and case.returned_hits[0].relevance_score >= threshold
                    else []
                )[:3]
                sources = {hit.filename for hit in retained}
                expected = set(case.expected_sources)
                answerable_hits.append(bool(sources & expected))
                answerable_source_recalls.append(
                    len(sources & expected) / len(expected)
                )
            empty_unknown = [
                not case.returned_hits
                or case.returned_hits[0].relevance_score < threshold
                for case in unanswerable
            ]
            points.append(
                ThresholdPoint(
                    threshold=threshold,
                    answerable_hit_at_3=_mean(
                        [float(value) for value in answerable_hits]
                    ),
                    answerable_source_recall_at_3=_mean(
                        answerable_source_recalls
                    ),
                    unanswerable_empty_rate=_mean(
                        [float(value) for value in empty_unknown]
                    ),
                )
            )

        best_source_recall = max(
            point.answerable_source_recall_at_3 for point in points
        )
        source_eligible = [
            point for point in points
            if point.answerable_source_recall_at_3 == best_source_recall
        ]
        best_hit_rate = max(point.answerable_hit_at_3 for point in source_eligible)
        eligible = [
            point for point in source_eligible
            if point.answerable_hit_at_3 == best_hit_rate
        ]
        best_empty_rate = max(
            point.unanswerable_empty_rate for point in eligible
        )
        finalists = [
            point for point in eligible
            if point.unanswerable_empty_rate == best_empty_rate
        ]
        # 在同样达到最佳召回和拒答率的平台区间选择中点，
        # 避免阈值贴近任意一侧的观测边界而缺少余量。
        plateau_midpoint = (
            finalists[0].threshold + finalists[-1].threshold
        ) / 2
        recommended = min(
            finalists,
            key=lambda point: (
                abs(point.threshold - plateau_midpoint),
                point.threshold,
            ),
        )
        return ThresholdCalibrationReport(
            policy=(
                "先保持最佳answerable source recall@3和Hit@3，"
                "再最大化unanswerable empty rate；最后选择最佳平台区间中点"
            ),
            recommended_threshold=recommended.threshold,
            best_answerable_hit_at_3=recommended.answerable_hit_at_3,
            best_answerable_source_recall_at_3=(
                recommended.answerable_source_recall_at_3
            ),
            recommended_unanswerable_empty_rate=(
                recommended.unanswerable_empty_rate
            ),
            points=points,
        )

    def evaluate_rag(
        self,
        cases: list[EvaluationCase],
        rag: AnswerProvider,
        top_k: int = 5,
        min_relevance: float | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> RAGEvaluationReport:
        if not cases:
            raise ValueError("评测用例不能为空")

        results: list[RAGCaseResult] = []
        for index, case in enumerate(cases, start=1):
            if progress:
                progress(index, len(cases), case.case_id)
            try:
                answer = rag.answer(
                    case.query,
                    top_k=top_k,
                    min_relevance=min_relevance,
                )
                results.append(self._rag_case(case, answer))
            except Exception as exc:
                results.append(
                    RAGCaseResult(
                        case_id=case.case_id,
                        query_type=case.query_type,
                        answerable=case.answerable,
                        expected_sources=case.expected_sources,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        answerable = [item for item in results if item.answerable]
        unanswerable = [item for item in results if not item.answerable]
        return RAGEvaluationReport(
            total_cases=len(results),
            answerable_cases=len(answerable),
            unanswerable_cases=len(unanswerable),
            answerable_success_rate=_mean(
                [float(item.answered_correctly) for item in answerable]
            ),
            refusal_accuracy=_mean(
                [float(item.refusal_correct is True) for item in unanswerable]
            ),
            citation_precision=_mean(
                [item.citation_precision for item in answerable]
            ),
            citation_recall=_mean([item.citation_recall for item in answerable]),
            keyword_coverage=_mean(
                [item.keyword_coverage for item in answerable]
            ),
            failed_cases=sum(item.error is not None for item in results),
            cases=results,
        )

    @staticmethod
    def _rag_case(case: EvaluationCase, answer: RAGAnswer) -> RAGCaseResult:
        citation_sources = list(
            dict.fromkeys(citation.filename for citation in answer.citations)
        )
        expected = set(case.expected_sources)
        cited = set(citation_sources)
        if case.answerable:
            precision = len(expected & cited) / len(cited) if cited else 0.0
            recall = len(expected & cited) / len(expected)
            normalized_answer = "".join(answer.answer.lower().split())
            matched_keywords = [
                keyword for keyword in case.expected_answer_keywords
                if "".join(keyword.lower().split()) in normalized_answer
            ]
            missing_keywords = [
                keyword for keyword in case.expected_answer_keywords
                if keyword not in matched_keywords
            ]
            keyword_coverage = (
                len(matched_keywords) / len(case.expected_answer_keywords)
            )
            answered_correctly = not answer.refused and bool(expected & cited)
            refusal_correct = None
        else:
            precision = 0.0
            recall = 0.0
            keyword_coverage = 0.0
            answered_correctly = False
            refusal_correct = answer.refused and not answer.citations
            matched_keywords = []
            missing_keywords = []

        return RAGCaseResult(
            case_id=case.case_id,
            query_type=case.query_type,
            answerable=case.answerable,
            refused=answer.refused,
            expected_sources=case.expected_sources,
            citation_sources=citation_sources,
            answer_text=answer.answer,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            answered_correctly=answered_correctly,
            refusal_correct=refusal_correct,
            citation_precision=round(precision, 6),
            citation_recall=round(recall, 6),
            keyword_coverage=round(keyword_coverage, 6),
        )
