"""离线入库入口：python -m scripts.ingest_documents"""

import json
import logging

from app.chunking import TextChunker
from app.config import settings
from app.embeddings import SentenceTransformerEmbeddingProvider
from app.ingestion_service import IngestionService
from app.vector_store import ChromaVectorStore


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = IngestionService(
        chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
        embedder=SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            settings.embedding_batch_size,
        ),
        vector_store=ChromaVectorStore(
            settings.vector_db_dir,
            settings.collection_name,
        ),
    )
    result = service.ingest_directory(settings.raw_data_dir)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    if result.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
