from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from src.config import AppConfig
from src.pipeline.rag_pipeline import CostAwareRAGPipeline


def _normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def score_answer(predicted: str, expected: str) -> float:
    predicted_norm = _normalize_text(predicted)
    expected_norm = _normalize_text(expected)
    if not predicted_norm or not expected_norm:
        return 0.0
    return fuzz.token_set_ratio(predicted_norm, expected_norm) / 100.0


def _summarize(df: pd.DataFrame) -> dict:
    summary = {}
    for scenario in ["cheap_only", "expensive_only", "router"]:
        subset = df[df["scenario"] == scenario]
        if subset.empty:
            continue
        summary[scenario] = {
            "queries": int(len(subset)),
            "accuracy": float(subset["correct"].mean()),
            "total_cost_usd": float(subset["cost_usd"].sum()),
            "avg_cost_usd": float(subset["cost_usd"].mean()),
            "avg_latency_ms": float(subset["latency_ms"].mean()),
            "escalation_rate": float(subset["escalated"].mean()),
            "cache_hit_rate": float(subset["cache_hit"].mean()) if "cache_hit" in subset.columns else 0.0,
            "multi_source_rate": float((subset["source_count"] >= 2).mean()) if "source_count" in subset.columns else 0.0,
        }

        tier_1_subset = subset[subset["tier"] == "tier_1"] if "tier" in subset.columns else pd.DataFrame()
        summary[scenario]["tier_1_p95_latency_ms"] = (
            float(tier_1_subset["latency_ms"].quantile(0.95)) if not tier_1_subset.empty else 0.0
        )

    if "router" in summary and "expensive_only" in summary:
        expensive_cost = summary["expensive_only"]["total_cost_usd"]
        router_cost = summary["router"]["total_cost_usd"]
        summary["router"]["cost_savings_vs_expensive_pct"] = (
            float((1 - router_cost / expensive_cost) * 100.0) if expensive_cost > 0 else 0.0
        )
    if "router" in summary and "cheap_only" in summary:
        summary["router"]["accuracy_delta_vs_cheap"] = float(
            summary["router"]["accuracy"] - summary["cheap_only"]["accuracy"]
        )

    return summary


def run_evaluation(
    config: AppConfig,
    domain: str | None = None,
    eval_set_path: Path | None = None,
    results_output_path: Path | None = None,
    summary_output_path: Path | None = None,
    use_cache: bool = False,
    show_progress: bool = False,
) -> dict:
    eval_path = eval_set_path or config.eval_set_path
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval set at {eval_path}")

    eval_df = pd.read_csv(eval_path)
    pipeline = CostAwareRAGPipeline(config, domain=domain)
    active_domain = pipeline.domain_policy.name.value

    rows: list[dict] = []
    scenarios = {
        "cheap_only": "tier_1",
        "expensive_only": "tier_3",
        "router": None,
    }
    total_queries = len(eval_df) * len(scenarios)
    progress_count = 0

    for scenario, forced_tier in scenarios.items():
        if show_progress:
            print(f"Running scenario: {scenario}", flush=True)
        for _, record in eval_df.iterrows():
            response, classification = pipeline.answer(
                query=str(record["question"]),
                scenario=scenario,
                forced_tier=forced_tier,
                use_cache=use_cache,
            )
            progress_count += 1
            if show_progress and (progress_count == 1 or progress_count % 10 == 0 or progress_count == total_queries):
                print(
                    f"Progress: {progress_count}/{total_queries} | scenario={scenario} | id={int(record['id'])} | model={response.model}",
                    flush=True,
                )
            similarity = score_answer(response.answer, str(record["ground_truth"]))
            rows.append(
                {
                    "id": int(record["id"]),
                    "domain": active_domain,
                    "scenario": scenario,
                    "difficulty": str(record.get("difficulty", "unknown")),
                    "question": str(record["question"]),
                    "ground_truth": str(record["ground_truth"]),
                    "answer": response.answer,
                    "similarity": similarity,
                    "correct": 1 if similarity >= 0.74 else 0,
                    "classification": classification.label.value,
                    "model": response.model,
                    "tier": response.tier,
                    "escalated": int(response.escalated),
                    "cost_usd": response.cost_usd,
                    "latency_ms": response.latency_ms,
                    "final_confidence": response.final_confidence,
                    "route_reason": response.route_reason,
                    "source_count": len(response.retrieved_sources),
                    "cache_hit": int(response.cache_hit),
                    "cache_similarity": response.cache_similarity,
                }
            )

    results_df = pd.DataFrame(rows)
    output_results_path = results_output_path or config.eval_results_path
    output_summary_path = summary_output_path or config.eval_summary_path
    output_results_path.parent.mkdir(parents=True, exist_ok=True)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_results_path, index=False)

    summary = _summarize(results_df)
    summary["domain"] = active_domain
    with output_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary

