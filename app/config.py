"""集中管理路径与入库配置，避免业务代码散落环境变量读取。"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    raw_data_dir: Path = Path(
        os.getenv("RAW_DATA_DIR", str(BASE_DIR / "data" / "raw"))
    ).expanduser().resolve()
    vector_db_dir: Path = Path(
        os.getenv("VECTOR_DB_DIR", str(BASE_DIR / "data" / "chroma"))
    ).expanduser().resolve()
    collection_name: str = os.getenv(
        "COLLECTION_NAME", "enterprise_knowledge"
    ).strip()
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    ).strip()
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    embedding_batch_size: int = int(
        os.getenv("EMBEDDING_BATCH_SIZE", "32")
    )
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    retrieval_min_relevance: float = float(
        os.getenv("RETRIEVAL_MIN_RELEVANCE", "0.45")
    )
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_base_url: str = os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    ).strip()
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash").strip()
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    rag_max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

    def validate(self) -> None:
        if not self.collection_name:
            raise ValueError("COLLECTION_NAME不能为空")
        if self.chunk_size < 100:
            raise ValueError("CHUNK_SIZE不能小于100")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP不能为负数")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP必须小于CHUNK_SIZE")
        if self.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE必须大于0")
        if self.retrieval_top_k < 1:
            raise ValueError("RETRIEVAL_TOP_K必须大于0")
        if not 0.0 <= self.retrieval_min_relevance <= 1.0:
            raise ValueError("RETRIEVAL_MIN_RELEVANCE必须在0到1之间")
        if not self.deepseek_base_url:
            raise ValueError("DEEPSEEK_BASE_URL不能为空")
        if not self.deepseek_model:
            raise ValueError("DEEPSEEK_MODEL不能为空")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE必须在0到2之间")
        if self.llm_max_tokens < 1:
            raise ValueError("LLM_MAX_TOKENS必须大于0")
        if self.rag_max_context_chars < 500:
            raise ValueError("RAG_MAX_CONTEXT_CHARS不能小于500")


settings = Settings()
settings.validate()
