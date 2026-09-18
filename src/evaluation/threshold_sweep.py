from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from src.config import AppConfig
from src.evaluation.evaluator import run_evaluation


def _threshold_grid(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be > 0.")
    if end < start:
        raise ValueError("End threshold must be >= start threshold.")

    values = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 4))
        current += step
    return values


def run_threshold_sweep(config: AppConfig, start: float, end: float, step: float, domain: str | None = None) -> dict:
    thresholds = _threshold_grid(start, end, step)
    config.sweep_runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for threshold in thresholds:
        run_slug = f"t_{threshold:.2f}".replace(".", "_")
        run_dir = config.sweep_runs_dir / run_slug
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = replace(
            config,
            confidence_threshold=threshold,
            eval_results_path=run_dir / "eval_results.csv",
            eval_summary_path=run_dir / "eval_summary.json",
            query_log_path=run_dir / "query_log.jsonl",
        )
        summary = run_evaluation(run_config, domain=domain, use_cache=False)
        router = summary.get("router", {})
        rows.append(
            {
            "domain": summary.get("domain", domain or run_config.default_domain),
                "threshold": threshold,
                "router_accuracy": float(router.get("accuracy", 0.0)),
                "router_total_cost_usd": float(router.get("total_cost_usd", 0.0)),
                "router_avg_latency_ms": float(router.get("avg_latency_ms", 0.0)),
                "router_escalation_rate": float(router.get("escalation_rate", 0.0)),
                "router_cost_savings_vs_expensive_pct": float(router.get("cost_savings_vs_expensive_pct", 0.0)),
                "router_accuracy_delta_vs_cheap": float(router.get("accuracy_delta_vs_cheap", 0.0)),
            }
        )

    sweep_df = pd.DataFrame(rows).sort_values("threshold")
    config.threshold_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(config.threshold_sweep_path, index=False)

    best_row = sweep_df.sort_values(["router_accuracy", "router_total_cost_usd"], ascending=[False, True]).iloc[0]
    summary_payload = {
        "domain": domain or config.default_domain,
        "start_threshold": start,
        "end_threshold": end,
        "step": step,
        "num_runs": int(len(sweep_df)),
        "best_threshold": float(best_row["threshold"]),
        "best_router_accuracy": float(best_row["router_accuracy"]),
        "best_router_total_cost_usd": float(best_row["router_total_cost_usd"]),
        "output_csv": str(config.threshold_sweep_path),
        "runs_dir": str(config.sweep_runs_dir),
    }
    with config.threshold_sweep_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload
