"""BGE稠密召回、BM25词法召回、RRF融合与CrossEncoder精排。"""

from app.embeddings import EmbeddingProvider
from app.lexical_retriever import BM25Retriever
from app.reranker import RerankerProvider
from app.schemas import RetrievalHit, RetrievalResult
from app.vector_store import ChromaVectorStore


class HybridRetrievalService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        lexical_retriever: BM25Retriever,
        reranker: RerankerProvider,
        default_top_k: int = 5,
        default_min_relevance: float = 0.45,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k必须大于0")
        if not 0.0 <= default_min_relevance <= 1.0:
            raise ValueError("default_min_relevance必须在0到1之间")
        if candidate_k < 1:
            raise ValueError("candidate_k必须大于0")
        if rrf_k < 1:
            raise ValueError("rrf_k必须大于0")
        self.embedder = embedder
        self.vector_store = vector_store
        self.lexical_retriever = lexical_retriever
        self.reranker = reranker
        self.default_top_k = default_top_k
        self.default_min_relevance = default_min_relevance
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> RetrievalResult:
        normalized_query, resolved_top_k, threshold = self._resolve_options(
            query, top_k, min_relevance
        )
        candidate_k = max(self.candidate_k, resolved_top_k)
        fused = self._retrieve_fused(normalized_query, candidate_k)
        reranked = self.reranker.rerank(
            normalized_query,
            fused[:candidate_k],
            top_n=resolved_top_k,
        )
        # 阈值用于判断“这个问题是否有可靠候选”，而不是逐块过滤。
        # 多来源问题的第二份必要资料分数可能显著低于Top-1；若逐块过滤，
        # 会在已经确认问题可回答后错误删除补充来源。
        hits = (
            reranked
            if reranked and reranked[0].relevance_score >= threshold
            else []
        )
        return RetrievalResult(query=normalized_query, hits=hits)

    def retrieve_rrf(
        self,
        query: str,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> RetrievalResult:
        """返回RRF融合结果但不执行CrossEncoder，用于消融评测。"""

        normalized_query, resolved_top_k, threshold = self._resolve_options(
            query, top_k, min_relevance
        )
        candidate_k = max(self.candidate_k, resolved_top_k)
        fused = self._retrieve_fused(normalized_query, candidate_k)
        hits = [
            hit for hit in fused[:resolved_top_k]
            if hit.relevance_score >= threshold
        ]
        return RetrievalResult(query=normalized_query, hits=hits)

    def _resolve_options(
        self,
        query: str,
        top_k: int | None,
        min_relevance: float | None,
    ) -> tuple[str, int, float]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("查询内容不能为空")
        resolved_top_k = self.default_top_k if top_k is None else top_k
        threshold = (
            self.default_min_relevance
            if min_relevance is None
            else min_relevance
        )
        if resolved_top_k < 1:
            raise ValueError("top_k必须大于0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("min_relevance必须在0到1之间")
        return normalized_query, resolved_top_k, threshold

    def _retrieve_fused(
        self, normalized_query: str, candidate_k: int
    ) -> list[RetrievalHit]:
        vector_hits = self.vector_store.similarity_search(
            query_embedding=self.embedder.embed_query(normalized_query),
            top_k=candidate_k,
            # 不能在融合前用向量阈值删除BM25可能补回的候选。
            min_relevance=0.0,
        )
        lexical_hits = self.lexical_retriever.search(
            normalized_query, top_k=candidate_k
        )
        return self._reciprocal_rank_fusion(vector_hits, lexical_hits)

    def _reciprocal_rank_fusion(
        self,
        vector_hits: list[RetrievalHit],
        lexical_hits: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        merged: dict[str, RetrievalHit] = {}
        scores: dict[str, float] = {}

        for source_hits, score_name in (
            (vector_hits, "vector_score"),
            (lexical_hits, "bm25_score"),
        ):
            for rank, hit in enumerate(source_hits, start=1):
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )
                if hit.chunk_id not in merged:
                    merged[hit.chunk_id] = hit
                current = merged[hit.chunk_id]
                source_score = getattr(hit, score_name)
                if source_score is not None:
                    merged[hit.chunk_id] = current.model_copy(
                        update={score_name: source_score}
                    )

        if not merged:
            return []
        maximum = max(scores.values())
        fused = []
        for chunk_id, hit in merged.items():
            rrf_score = scores[chunk_id]
            fused.append(
                hit.model_copy(
                    update={
                        "rrf_score": round(rrf_score, 8),
                        "relevance_score": round(rrf_score / maximum, 6),
                    }
                )
            )
        fused.sort(key=lambda hit: (-(hit.rrf_score or 0.0), hit.chunk_id))
        return fused
