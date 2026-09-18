from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from src.config import load_config
from src.evaluation.evaluator import run_evaluation
from src.evaluation.threshold_sweep import run_threshold_sweep
from src.pipeline.rag_pipeline import CostAwareRAGPipeline
from src.rag.vector_store import ingest_corpus
from src.router.domain_policy import DOMAIN_CHOICES


def cmd_ingest() -> int:
    config = load_config()
    chunks = ingest_corpus(config)
    print(f"Ingestion complete. Chunks indexed: {chunks}")
    return 0


def cmd_ask(query: str, domain: str | None = None) -> int:
    config = load_config()
    pipeline = CostAwareRAGPipeline(config, domain=domain)
    response, classification = pipeline.answer(query=query, scenario="interactive")

    print("\n=== Answer ===")
    print(response.answer)
    print("\n=== Routing Metadata ===")
    print(
        json.dumps(
            {
                "classification": classification.label.value,
                "classification_confidence": classification.confidence,
                "domain": response.domain,
                "route_reason": response.route_reason,
                "model": response.model,
                "tier": response.tier,
                "final_confidence": response.final_confidence,
                "escalated": response.escalated,
                "cost_usd": response.cost_usd,
                "latency_ms": response.latency_ms,
                "retrieved_sources": response.retrieved_sources,
            },
            indent=2,
        )
    )
    return 0


def cmd_eval(use_cache: bool = False, domain: str | None = None) -> int:
    config = load_config()
    summary = run_evaluation(config, domain=domain, use_cache=use_cache, show_progress=True)
    print("Evaluation complete. Summary:")
    print(json.dumps(summary, indent=2))
    print(f"Detailed results: {config.eval_results_path}")
    print(f"Summary file: {config.eval_summary_path}")
    return 0


def cmd_sweep(start_threshold: float, end_threshold: float, step: float, domain: str | None = None) -> int:
    config = load_config()
    summary = run_threshold_sweep(config, start=start_threshold, end=end_threshold, step=step, domain=domain)
    print("Threshold sweep complete:")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_run_all(start_threshold: float, end_threshold: float, step: float, skip_sweep: bool, domain: str | None = None) -> int:
    config = load_config()
    active_domain = domain or config.default_domain
    print(f"Domain preset: {active_domain}")

    chunks = ingest_corpus(config)
    print(f"[1/3] Ingestion complete. Chunks indexed: {chunks}")

    eval_summary = run_evaluation(config, domain=active_domain, use_cache=False, show_progress=True)
    print("[2/3] Evaluation complete.")
    print(json.dumps(eval_summary, indent=2))

    if skip_sweep:
        print("[3/3] Threshold sweep skipped.")
    else:
        sweep_summary = run_threshold_sweep(
            config,
            start=start_threshold,
            end=end_threshold,
            step=step,
            domain=active_domain,
        )
        print("[3/3] Threshold sweep complete.")
        print(json.dumps(sweep_summary, indent=2))

    print("Dashboard artifacts are ready in logs/.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cost-aware multi-model router for RAG.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Chunk corpus, embed, and index documents into Chroma.")

    ask_parser = sub.add_parser("ask", help="Run one query through routed RAG.")
    ask_parser.add_argument("--query", required=True, help="Question to ask.")
    ask_parser.add_argument("--domain", choices=DOMAIN_CHOICES, default=None)

    eval_parser = sub.add_parser("eval", help="Run benchmark scenarios and save cost/accuracy reports.")
    eval_parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable semantic cache during evaluation (disabled by default for fair benchmarking).",
    )
    eval_parser.add_argument("--domain", choices=DOMAIN_CHOICES, default=None)

    sweep_parser = sub.add_parser("sweep", help="Run confidence-threshold sweep experiments.")
    sweep_parser.add_argument("--start-threshold", type=float, default=0.60)
    sweep_parser.add_argument("--end-threshold", type=float, default=0.90)
    sweep_parser.add_argument("--step", type=float, default=0.05)
    sweep_parser.add_argument("--domain", choices=DOMAIN_CHOICES, default=None)

    run_all_parser = sub.add_parser("run-all", help="One command: ingest + eval + optional threshold sweep.")
    run_all_parser.add_argument("--start-threshold", type=float, default=0.60)
    run_all_parser.add_argument("--end-threshold", type=float, default=0.90)
    run_all_parser.add_argument("--step", type=float, default=0.05)
    run_all_parser.add_argument("--skip-sweep", action="store_true")
    run_all_parser.add_argument("--domain", choices=DOMAIN_CHOICES, default=None)
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        return cmd_ingest()
    if args.command == "ask":
        return cmd_ask(args.query, domain=args.domain)
    if args.command == "eval":
        return cmd_eval(use_cache=args.use_cache, domain=args.domain)
    if args.command == "sweep":
        return cmd_sweep(
            start_threshold=args.start_threshold,
            end_threshold=args.end_threshold,
            step=args.step,
            domain=args.domain,
        )
    if args.command == "run-all":
        return cmd_run_all(
            start_threshold=args.start_threshold,
            end_threshold=args.end_threshold,
            step=args.step,
            skip_sweep=args.skip_sweep,
            domain=args.domain,
        )
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
