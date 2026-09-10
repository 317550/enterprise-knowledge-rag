"""编排文档发现、加载、分块、向量化和持久化。"""

import logging
from pathlib import Path

from app.chunking import TextChunker
from app.embeddings import EmbeddingProvider
from app.loaders import DocumentLoadError, discover_documents, load_document
from app.schemas import IngestionResult
from app.vector_store import ChromaVectorStore


logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        chunker: TextChunker,
        embedder: EmbeddingProvider,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest_directory(self, root_dir: Path) -> IngestionResult:
        """逐文件处理；单个损坏文件不会让整批任务全部丢失。"""
        result = IngestionResult()
        paths = discover_documents(root_dir)
        result.files_seen = len(paths)

        for path in paths:
            try:
                pages = load_document(path, root_dir)
                chunks = self.chunker.build_chunks(pages)
                action, written = self.vector_store.sync_document(
                    chunks, self.embedder
                )
                result.chunks_written += written
                if action == "inserted":
                    result.files_inserted += 1
                elif action == "replaced":
                    result.files_replaced += 1
                else:
                    result.files_skipped += 1
            except (DocumentLoadError, OSError, ValueError) as exc:
                logger.exception("文档入库失败：%s", path)
                result.errors.append(f"{path.name}: {exc}")
        return result
