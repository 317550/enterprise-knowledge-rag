"""Chroma持久化适配器，把第三方API隔离在基础设施层。"""

from pathlib import Path
from typing import Literal

import chromadb

from app.embeddings import EmbeddingProvider
from app.schemas import TextChunk


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
