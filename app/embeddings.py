"""向量模型抽象；生产使用BGE，测试注入轻量假实现。"""

from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """为文档块生成向量。"""

    def embed_query(self, text: str) -> list[float]:
        """为用户查询生成向量，下一阶段检索时使用。"""


class SentenceTransformerEmbeddingProvider:
    """延迟加载SentenceTransformer，避免模块导入时占用大量内存。"""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # trust_remote_code保持False，避免执行未知仓库中的自定义代码。
            self._model = SentenceTransformer(
                self.model_name,
                trust_remote_code=False,
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encode_document = getattr(self.model, "encode_document", None)
        encoder = encode_document or self.model.encode
        vectors = encoder(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) >= self.batch_size,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        encode_query = getattr(self.model, "encode_query", None)
        encoder = encode_query or self.model.encode
        vector = encoder(text, normalize_embeddings=True)
        return vector.tolist()
