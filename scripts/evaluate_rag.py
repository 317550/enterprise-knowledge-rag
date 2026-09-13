"""运行固定评测集并保存可复现的JSON报告。"""

import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.evaluation_dataset import load_evaluation_cases
from app.evaluation_service import EvaluationService
from app.llm import DeepSeekChatModel
from app.rag_service import RAGService
from app.service_factory import build_retrieval_services


class RRFOnlyView:
    """将混合检索器的RRF阶段暴露为标准检索接口。"""

    def __init__(self, hybrid) -> None:
        self.hybrid = hybrid

    def retrieve(self, query, top_k=None, min_relevance=None):
        return self.hybrid.retrieve_rrf(
            query, top_k=top_k, min_relevance=min_relevance
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评测企业知识库RAG")
    parser.add_argument(
        "--mode",
        choices=("retrieval", "rag", "all"),
        default="retrieval",
        help="retrieval不调用大模型；rag会调用DeepSeek；all依次运行两者",
    )
    parser.add_argument("--dataset", type=Path, default=settings.evaluation_dataset_path)
    parser.add_argument("--output-dir", type=Path, default=settings.evaluation_results_dir)
    parser.add_argument("--top-k", type=int, default=settings.retrieval_top_k)
    parser.add_argument("--min-relevance", type=float, default=None)
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="自动阈值扫描步长，默认0.01",
    )
    return parser


def _summary(report: dict) -> dict:
    return {key: value for key, value in report.items() if key != "cases"}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _progress(stage: str):
    def report(index: int, total: int, case_id: str) -> None:
        print(f"[{stage} {index}/{total}] {case_id}", file=sys.stderr)

    return report


def main() -> None:
    args = build_parser().parse_args()
    cases = load_evaluation_cases(args.dataset.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    evaluator = EvaluationService()
    vector_only, hybrid = build_retrieval_services(settings)
    printed: dict[str, object] = {}

    if args.mode in {"retrieval", "all"}:
        # 排名评测默认关闭业务阈值；阈值由同一批零阈值结果离线扫描。
        ranking_threshold = 0.0 if args.min_relevance is None else args.min_relevance
        vector_report = evaluator.evaluate_retrieval(
            cases,
            vector_only,
            system_name="vector_only",
            top_k=args.top_k,
            min_relevance=ranking_threshold,
            progress=_progress("vector"),
        )
        rrf_report = evaluator.evaluate_retrieval(
            cases,
            RRFOnlyView(hybrid),
            system_name="hybrid_rrf",
            top_k=args.top_k,
            min_relevance=ranking_threshold,
            progress=_progress("rrf"),
        )
        hybrid_report = evaluator.evaluate_retrieval(
            cases,
            hybrid,
            system_name="hybrid_reranked",
            top_k=args.top_k,
            min_relevance=ranking_threshold,
            progress=_progress("hybrid"),
        )
        calibration = (
            evaluator.calibrate_threshold(hybrid_report, args.threshold_step)
            if ranking_threshold == 0.0
            else None
        )
        payload = {
            "configuration": {
                "embedding_model": settings.embedding_model,
                "rerank_model": settings.rerank_model,
                "top_k": args.top_k,
                "ranking_min_relevance": ranking_threshold,
                "hybrid_candidate_k": settings.hybrid_candidate_k,
                "rrf_k": settings.rrf_k,
            },
            "vector_only": vector_report.model_dump(mode="json"),
            "hybrid_rrf": rrf_report.model_dump(mode="json"),
            "hybrid_reranked": hybrid_report.model_dump(mode="json"),
            "threshold_calibration": (
                calibration.model_dump(mode="json") if calibration else None
            ),
        }
        path = output_dir / "retrieval_report.json"
        _write_json(path, payload)
        printed["retrieval"] = {
            "vector_only": _summary(payload["vector_only"]),
            "hybrid_rrf": _summary(payload["hybrid_rrf"]),
            "hybrid_reranked": _summary(payload["hybrid_reranked"]),
            "threshold_calibration": (
                {
                    key: value
                    for key, value in payload["threshold_calibration"].items()
                    if key != "points"
                }
                if payload["threshold_calibration"]
                else None
            ),
            "report_path": str(path),
        }

    if args.mode in {"rag", "all"}:
        chat_model = DeepSeekChatModel(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        rag = RAGService(
            retrieval_service=hybrid,
            chat_model=chat_model,
            max_context_chars=settings.rag_max_context_chars,
            max_chunks_per_source=settings.rag_max_chunks_per_source,
        )
        rag_report = evaluator.evaluate_rag(
            cases,
            rag,
            top_k=args.top_k,
            min_relevance=args.min_relevance,
            progress=_progress("rag"),
        )
        payload = rag_report.model_dump(mode="json")
        payload["configuration"] = {
            "embedding_model": settings.embedding_model,
            "rerank_model": settings.rerank_model,
            "llm_model": settings.deepseek_model,
            "top_k": args.top_k,
            "min_relevance": (
                settings.retrieval_min_relevance
                if args.min_relevance is None
                else args.min_relevance
            ),
            "max_context_chars": settings.rag_max_context_chars,
            "max_chunks_per_source": settings.rag_max_chunks_per_source,
        }
        path = output_dir / "rag_report.json"
        _write_json(path, payload)
        printed["rag"] = {
            **_summary(payload),
            "report_path": str(path),
        }

    print(json.dumps(printed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
