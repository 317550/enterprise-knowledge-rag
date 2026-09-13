"""应用装配入口，保证检索命令和问答命令使用同一套混合检索配置。"""

from app.config import Settings, settings
from app.embeddings import SentenceTransformerEmbeddingProvider
from app.hybrid_retriever import HybridRetrievalService
from app.lexical_retriever import BM25Retriever
from app.reranker import CrossEncoderReranker
from app.retrieval_service import RetrievalService
from app.vector_store import ChromaVectorStore


def build_hybrid_retrieval_service(
    config: Settings = settings,
) -> HybridRetrievalService:
    return build_retrieval_services(config)[1]


def build_retrieval_services(
    config: Settings = settings,
) -> tuple[RetrievalService, HybridRetrievalService]:
    embedder = SentenceTransformerEmbeddingProvider(
        config.embedding_model,
        config.embedding_batch_size,
    )
    store = ChromaVectorStore(
        config.vector_db_dir,
        config.collection_name,
    )
    lexical = BM25Retriever(store.list_chunks())
    reranker = CrossEncoderReranker(
        config.rerank_model,
        config.rerank_batch_size,
    )
    vector_only = RetrievalService(
        embedder=embedder,
        vector_store=store,
        default_top_k=config.retrieval_top_k,
        default_min_relevance=config.retrieval_min_relevance,
    )
    hybrid = HybridRetrievalService(
        embedder=embedder,
        vector_store=store,
        lexical_retriever=lexical,
        reranker=reranker,
        default_top_k=config.retrieval_top_k,
        default_min_relevance=config.retrieval_min_relevance,
        candidate_k=config.hybrid_candidate_k,
        rrf_k=config.rrf_k,
    )
    return vector_only, hybrid
