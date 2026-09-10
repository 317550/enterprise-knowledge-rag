"""受控文档加载器：只接受明确支持的本地文件类型。"""

import hashlib
import unicodedata
from pathlib import Path

import pymupdf

from app.schemas import FileType, SourceDocument


SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


class DocumentLoadError(RuntimeError):
    """文件损坏、无法读取或没有可用文本。"""


class UnsupportedDocumentError(ValueError):
    """文件格式不在允许列表。"""


def normalize_text(text: str) -> str:
    """执行保守清洗：保留段落边界，不破坏制度条款结构。"""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")

    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    output: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        output.append(line)
        previous_blank = is_blank
    return "\n".join(output).strip()


def _file_identity(path: Path, root_dir: Path) -> tuple[str, str, str]:
    try:
        relative_path = path.resolve().relative_to(root_dir.resolve())
    except ValueError as exc:
        raise DocumentLoadError(f"文件不在知识库目录中：{path}") from exc

    source_path = relative_path.as_posix()
    document_id = hashlib.sha256(
        source_path.casefold().encode("utf-8")
    ).hexdigest()[:24]
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return document_id, source_path, checksum


def load_document(path: Path, root_dir: Path) -> list[SourceDocument]:
    """加载单个文件；PDF按页返回，文本文件返回一条记录。"""
    if not path.is_file():
        raise DocumentLoadError(f"文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedDocumentError(f"不支持的文件类型：{suffix}")

    try:
        document_id, source_path, checksum = _file_identity(path, root_dir)
        if suffix in {".md", ".txt"}:
            content = normalize_text(path.read_text(encoding="utf-8-sig"))
            file_type = (
                FileType.MARKDOWN if suffix == ".md" else FileType.TEXT
            )
            documents = [
                SourceDocument(
                    document_id=document_id,
                    filename=path.name,
                    source_path=source_path,
                    file_type=file_type,
                    content=content,
                    checksum=checksum,
                )
            ]
        else:
            documents = []
            with pymupdf.open(path) as pdf:
                if pdf.needs_pass:
                    raise DocumentLoadError(f"PDF已加密：{path.name}")
                for page_number, page in enumerate(pdf, start=1):
                    content = normalize_text(page.get_text("text", sort=True))
                    if content:
                        documents.append(
                            SourceDocument(
                                document_id=document_id,
                                filename=path.name,
                                source_path=source_path,
                                file_type=FileType.PDF,
                                page=page_number,
                                content=content,
                                checksum=checksum,
                            )
                        )
    except (OSError, UnicodeError, pymupdf.FileDataError) as exc:
        raise DocumentLoadError(f"无法读取文档：{path.name}") from exc

    if not documents or not any(item.content for item in documents):
        raise DocumentLoadError(f"文档没有可提取文本：{path.name}")
    return documents


def discover_documents(root_dir: Path) -> list[Path]:
    """递归发现支持的文件，并用稳定顺序返回。"""
    if not root_dir.exists():
        raise DocumentLoadError(f"知识库目录不存在：{root_dir}")
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(root_dir).parts)
    )
