"""CrossEncoder精排抽象；生产延迟加载模型，测试可注入假实现。"""

import math
from typing import Protocol

from app.schemas import RetrievalHit


class RerankerProvider(Protocol):
    def rerank(
        self, query: str, hits: list[RetrievalHit], top_n: int
    ) -> list[RetrievalHit]:
        """重新排列候选并返回前top_n条。"""


def _sigmoid(value: float) -> float:
    if value >= 0:
        factor = math.exp(-value)
        return 1.0 / (1.0 + factor)
    factor = math.exp(value)
    return factor / (1.0 + factor)


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        if not model_name.strip():
            raise ValueError("RERANK_MODEL不能为空")
        if batch_size < 1:
            raise ValueError("RERANK_BATCH_SIZE必须大于0")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                trust_remote_code=False,
            )
        return self._model

    def rerank(
        self, query: str, hits: list[RetrievalHit], top_n: int
    ) -> list[RetrievalHit]:
        if top_n < 1:
            raise ValueError("top_n必须大于0")
        if not hits:
            return []

        # 显式取得原始logit，再统一做sigmoid，避免不同模型默认激活函数不同。
        predictions = self.model.predict(
            [(query, hit.content) for hit in hits],
            batch_size=self.batch_size,
            show_progress_bar=False,
            activation_fn=lambda value: value,
        )
        raw_scores = predictions.tolist()
        # 兼容形如[[score], ...]的单标签模型输出。
        raw_scores = [
            score[0] if isinstance(score, list) else score
            for score in raw_scores
        ]
        if len(raw_scores) != len(hits):
            raise ValueError("Reranker分数数量与候选数量不一致")

        scored = []
        for hit, raw_score in zip(hits, raw_scores, strict=True):
            score = _sigmoid(float(raw_score))
            scored.append(
                hit.model_copy(
                    update={
                        "rerank_score": round(score, 6),
                        "relevance_score": round(score, 6),
                    }
                )
            )
        scored.sort(
            key=lambda hit: (
                -(hit.rerank_score or 0.0),
                -(hit.rrf_score or 0.0),
                hit.chunk_id,
            )
        )
        return scored[:top_n]
