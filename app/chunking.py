"""可测试的文本分块器，优先在段落或中文标点附近断开。"""

import hashlib

from app.schemas import SourceDocument, TextChunk


BREAK_MARKS = ("\n\n", "\n", "。", "！", "？", "；")


class TextChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 100) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size不能小于100")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap必须满足 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        """长度受控的滑动分块；在窗口后半段寻找自然边界。"""
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            hard_end = min(start + self.chunk_size, text_length)
            end = hard_end

            if hard_end < text_length:
                search_start = start + self.chunk_size // 2
                best_break = -1
                best_mark_length = 0
                for mark in BREAK_MARKS:
                    position = text.rfind(mark, search_start, hard_end)
                    if position > best_break:
                        best_break = position
                        best_mark_length = len(mark)
                if best_break >= search_start:
                    end = best_break + best_mark_length

            content = text[start:end].strip()
            if content:
                chunks.append(content)
            if end >= text_length:
                break

            next_start = max(0, end - self.overlap)
            if next_start <= start:  # 防止极端配置造成死循环。
                next_start = end
            start = next_start
        return chunks

    def build_chunks(
        self, documents: list[SourceDocument]
    ) -> list[TextChunk]:
        """同一文件跨页使用连续chunk_index，便于引用和排序。"""
        chunks: list[TextChunk] = []
        chunk_index = 0
        for document in documents:
            for content in self.split(document.content):
                raw_id = "|".join(
                    [
                        document.document_id,
                        document.checksum,
                        str(document.page or 0),
                        str(chunk_index),
                        content,
                    ]
                )
                chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        filename=document.filename,
                        source_path=document.source_path,
                        file_type=document.file_type,
                        page=document.page,
                        chunk_index=chunk_index,
                        content=content,
                        checksum=document.checksum,
                    )
                )
                chunk_index += 1
        return chunks
