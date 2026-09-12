"""命令行问答入口。"""

import argparse

from app.config import settings
from app.llm import DeepSeekChatModel
from app.rag_service import RAGService
from app.service_factory import build_hybrid_retrieval_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于企业知识库生成带引用答案")
    parser.add_argument("query", help="需要回答的问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-relevance", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retrieval_service = build_hybrid_retrieval_service(settings)
    chat_model = DeepSeekChatModel(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    service = RAGService(
        retrieval_service=retrieval_service,
        chat_model=chat_model,
        max_context_chars=settings.rag_max_context_chars,
    )
    result = service.answer(
        args.query, top_k=args.top_k, min_relevance=args.min_relevance
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
