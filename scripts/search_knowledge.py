"""命令行检索入口：python -m scripts.search_knowledge "退货期限是多久？"""

import argparse

from app.config import settings
from app.service_factory import build_hybrid_retrieval_service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检索企业知识库")
    parser.add_argument("query", help="需要检索的问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-relevance", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = build_hybrid_retrieval_service(settings)
    result = service.retrieve(
        args.query,
        top_k=args.top_k,
        min_relevance=args.min_relevance,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
