"""检索用例：校验问题、生成查询向量并返回带来源的知识片段。"""

from app.embeddings import EmbeddingProvider
from app.schemas import RetrievalResult
from app.vector_store import ChromaVectorStore


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: ChromaVectorStore,
        default_top_k: int = 5,
        default_min_relevance: float = 0.45,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.default_min_relevance = default_min_relevance

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_relevance: float | None = None,
    ) -> RetrievalResult:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("查询内容不能为空")

        resolved_top_k = self.default_top_k if top_k is None else top_k
        resolved_threshold = (
            self.default_min_relevance
            if min_relevance is None
            else min_relevance
        )
        query_embedding = self.embedder.embed_query(normalized_query)
        hits = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=resolved_top_k,
            min_relevance=resolved_threshold,
        )
        return RetrievalResult(query=normalized_query, hits=hits)
