"""Chroma持久化适配器，把第三方API隔离在基础设施层。"""

from pathlib import Path
from typing import Literal

import chromadb

from app.embeddings import EmbeddingProvider
from app.schemas import FileType, IndexedChunk, RetrievalHit, TextChunk


SyncAction = Literal["inserted", "replaced", "skipped"]


class ChromaVectorStore:
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def sync_document(
        self,
        chunks: list[TextChunk],
        embedder: EmbeddingProvider,
    ) -> tuple[SyncAction, int]:
        """
        同步一个完整文件。

        内容未变化则跳过向量计算；内容变化时先写新块，再删除旧块，
        避免嵌入服务失败后把旧知识也删掉。
        """
        if not chunks:
            raise ValueError("不能同步空文档")
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("一次只能同步一个document_id")

        document_id = chunks[0].document_id
        current = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )
        current_ids = set(current.get("ids") or [])
        new_ids = {chunk.chunk_id for chunk in chunks}
        current_metadata = current.get("metadatas") or []
        unchanged = (
            bool(current_ids)
            and current_ids == new_ids
            and all(
                metadata is not None
                and metadata.get("checksum") == chunks[0].checksum
                for metadata in current_metadata
            )
        )
        if unchanged:
            return "skipped", 0

        embeddings = embedder.embed_documents(
            [chunk.content for chunk in chunks]
        )
        if len(embeddings) != len(chunks):
            raise ValueError("向量数量与文本块数量不一致")

        # upsert允许稳定ID重复执行，并保存我们显式计算的BGE向量。
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[chunk.chroma_metadata() for chunk in chunks],
            embeddings=embeddings,
        )

        stale_ids = list(current_ids - new_ids)
        if stale_ids:
            self.collection.delete(ids=stale_ids)

        action: SyncAction = "replaced" if current_ids else "inserted"
        return action, len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def list_chunks(self) -> list[IndexedChunk]:
        """读取全部已入库文本块，供BM25建立与Chroma一致的词法索引。"""
        if self.collection.count() == 0:
            return []
        result = self.collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks: list[IndexedChunk] = []
        for chunk_id, content, metadata in zip(
            ids, documents, metadatas, strict=True
        ):
            if content is None or metadata is None:
                continue
            page_value = int(metadata.get("page", 0))
            chunks.append(
                IndexedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    source_path=str(metadata["source_path"]),
                    file_type=FileType(str(metadata["file_type"])),
                    page=page_value or None,
                    chunk_index=int(metadata["chunk_index"]),
                )
            )
        return chunks

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        min_relevance: float,
    ) -> list[RetrievalHit]:
        if not query_embedding:
            raise ValueError("查询向量不能为空")
        if top_k < 1:
            raise ValueError("top_k必须大于0")
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance必须在0到1之间")
        if self.collection.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[RetrievalHit] = []
        for chunk_id, content, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            if content is None or metadata is None or distance is None:
                continue
            # collection使用cosine空间：distance越小越相似。
            # 将其转成便于业务理解的[0, 1]相关度并处理浮点边界。
            relevance = max(0.0, min(1.0, 1.0 - float(distance)))
            if relevance < min_relevance:
                continue
            page_value = int(metadata.get("page", 0))
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    content=content,
                    relevance_score=round(relevance, 6),
                    vector_score=round(relevance, 6),
                    document_id=str(metadata["document_id"]),
                    filename=str(metadata["filename"]),
                    source_path=str(metadata["source_path"]),
                    file_type=FileType(str(metadata["file_type"])),
                    page=page_value or None,
                    chunk_index=int(metadata["chunk_index"]),
                )
            )
        return hits
