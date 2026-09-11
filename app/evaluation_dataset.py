"""离线评测集的数据契约与JSONL加载器。"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


QueryType = Literal["semantic", "exact", "multi_source", "unanswerable"]


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answerable: bool
    expected_sources: list[str] = Field(default_factory=list)
    expected_answer_keywords: list[str] = Field(default_factory=list)
    category: str = Field(min_length=1)
    query_type: QueryType

    @model_validator(mode="after")
    def validate_expectations(self):
        if self.answerable:
            if not self.expected_sources:
                raise ValueError("可回答问题必须声明expected_sources")
            if not self.expected_answer_keywords:
                raise ValueError("可回答问题必须声明expected_answer_keywords")
            if self.query_type == "unanswerable":
                raise ValueError("可回答问题不能标记为unanswerable")
        else:
            if self.expected_sources or self.expected_answer_keywords:
                raise ValueError("不可回答问题不能声明预期来源或答案关键词")
            if self.query_type != "unanswerable":
                raise ValueError("不可回答问题必须标记为unanswerable")
        return self


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    if not path.is_file():
        raise FileNotFoundError(f"评测集不存在：{path}")

    cases: list[EvaluationCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"评测集第{line_number}行不是有效JSON") from exc
            cases.append(EvaluationCase.model_validate(payload))

    if not cases:
        raise ValueError("评测集不能为空")
    return cases
