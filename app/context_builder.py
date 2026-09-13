"""在字符预算内选择、去重并裁剪RAG上下文。"""

from collections import defaultdict

from app.schemas import RetrievalHit


class ContextBuilder:
    def __init__(
        self,
        max_context_chars: int = 6000,
        max_chunks_per_source: int = 2,
        minimum_overlap_chars: int = 40,
        minimum_chunk_chars: int = 80,
    ) -> None:
        if max_context_chars < 500:
            raise ValueError("max_context_chars不能小于500")
        if max_chunks_per_source < 1:
            raise ValueError("max_chunks_per_source必须大于0")
        if minimum_overlap_chars < 1:
            raise ValueError("minimum_overlap_chars必须大于0")
        if minimum_chunk_chars < 1:
            raise ValueError("minimum_chunk_chars必须大于0")
        self.max_context_chars = max_context_chars
        self.max_chunks_per_source = max_chunks_per_source
        self.minimum_overlap_chars = minimum_overlap_chars
        self.minimum_chunk_chars = minimum_chunk_chars

    def select(self, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """保持精排顺序，同时限制重复、单来源占比和上下文长度。"""
        selected: list[RetrievalHit] = []
        seen_ids: set[str] = set()
        seen_contents: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)
        last_content_by_source: dict[str, str] = {}
        used_chars = 0

        for hit in hits:
            normalized_content = " ".join(hit.content.split())
            if (
                hit.chunk_id in seen_ids
                or normalized_content in seen_contents
                or source_counts[hit.source_path] >= self.max_chunks_per_source
            ):
                continue

            content = self._remove_overlap(
                last_content_by_source.get(hit.source_path, ""),
                hit.content,
            )
            if not content.strip():
                continue

            overhead = len(hit.filename) + 80
            remaining = self.max_context_chars - used_chars - overhead
            if remaining < self.minimum_chunk_chars:
                continue
            if len(content) > remaining:
                content = content[:remaining].rstrip()
            if not content:
                continue

            selected_hit = hit.model_copy(update={"content": content})
            selected.append(selected_hit)
            used_chars += len(content) + overhead
            seen_ids.add(hit.chunk_id)
            seen_contents.add(normalized_content)
            source_counts[hit.source_path] += 1
            last_content_by_source[hit.source_path] = hit.content

            if used_chars >= self.max_context_chars:
                break

        return selected

    def _remove_overlap(self, previous: str, current: str) -> str:
        """移除同一来源相邻文本块中完全相同的前后缀重叠。"""
        if not previous:
            return current
        maximum = min(len(previous), len(current), 1000)
        for size in range(maximum, self.minimum_overlap_chars - 1, -1):
            if previous[-size:] == current[:size]:
                return current[size:].lstrip()
        return current
