"""文档入库阶段内部使用的稳定数据契约。"""

from enum import Enum

from pydantic import BaseModel, Field


class FileType(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TEXT = "text"


class SourceDocument(BaseModel):
    """加载器输出的一页或一个文本文件。PDF每页是一条记录。"""

    document_id: str
    filename: str
    source_path: str
    file_type: FileType
    page: int | None = Field(default=None, ge=1)
    content: str
    checksum: str


class TextChunk(BaseModel):
    """写入向量库的最小检索单元。"""

    chunk_id: str
    document_id: str
    filename: str
    source_path: str
    file_type: FileType
    page: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)
    content: str
    checksum: str

    def chroma_metadata(self) -> dict[str, str | int]:
        # Chroma元数据不写None；非PDF使用0表示没有页码。
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "source_path": self.source_path,
            "file_type": self.file_type.value,
            "page": self.page or 0,
            "chunk_index": self.chunk_index,
            "checksum": self.checksum,
        }


class IngestionResult(BaseModel):
    files_seen: int = 0
    files_inserted: int = 0
    files_replaced: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    errors: list[str] = Field(default_factory=list)


class RetrievalHit(BaseModel):
    """通过相关度阈值的单条知识片段。"""

    chunk_id: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    document_id: str
    filename: str
    source_path: str
    file_type: FileType
    page: int | None = Field(default=None, ge=1)
    chunk_index: int = Field(ge=0)


class RetrievalResult(BaseModel):
    query: str
    hits: list[RetrievalHit] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    """大模型必须返回的结构；citation_indices对应检索上下文编号。"""

    answer: str
    citation_indices: list[int] = Field(default_factory=list)
    insufficient_context: bool = False


class AnswerCitation(BaseModel):
    citation_index: int = Field(ge=1)
    chunk_id: str
    filename: str
    source_path: str
    page: int | None = Field(default=None, ge=1)
    relevance_score: float = Field(ge=0.0, le=1.0)
    excerpt: str


class RAGAnswer(BaseModel):
    query: str
    answer: str
    refused: bool
    citations: list[AnswerCitation] = Field(default_factory=list)
