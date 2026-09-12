"""基于中文分词和BM25的关键词检索。"""

from collections.abc import Callable

from app.schemas import IndexedChunk, RetrievalHit


Tokenize = Callable[[str], list[str]]


def tokenize_chinese(text: str) -> list[str]:
    """使用jieba搜索模式分词，并移除空白和纯标点token。"""
    import jieba

    return [
        token.lower()
        for token in jieba.lcut_for_search(text)
        if token.strip() and any(character.isalnum() for character in token)
    ]


class BM25Retriever:
    def __init__(
        self,
        chunks: list[IndexedChunk],
        tokenizer: Tokenize = tokenize_chinese,
    ) -> None:
        self.chunks = chunks
        self.tokenizer = tokenizer
        self._index = None
        if chunks:
            from rank_bm25 import BM25Okapi

            corpus = [tokenizer(chunk.content) for chunk in chunks]
            self._index = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k必须大于0")
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("查询内容不能为空")
        if self._index is None:
            return []

        scores = self._index.get_scores(self.tokenizer(normalized_query))
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        positive = [item for item in ranked if float(item[1]) > 0.0][:top_k]
        if not positive:
            return []

        maximum = max(float(score) for _, score in positive)
        hits: list[RetrievalHit] = []
        for index, raw_score in positive:
            chunk = self.chunks[index]
            bm25_score = float(raw_score)
            # relevance_score保持兼容；真正融合使用原始BM25排名而非该归一值。
            normalized_score = bm25_score / maximum if maximum > 0 else 0.0
            hits.append(
                RetrievalHit(
                    **chunk.model_dump(),
                    relevance_score=round(normalized_score, 6),
                    bm25_score=round(bm25_score, 6),
                )
            )
        return hits
