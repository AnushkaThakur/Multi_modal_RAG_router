from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from src.router.domain_policy import DOMAIN_KPI_TARGETS
except Exception:
    DOMAIN_KPI_TARGETS = {
        "support": {
            "accuracy_min_pct": 88.0,
            "cost_savings_min_pct": 35.0,
            "tier1_p95_latency_max_ms": 4000.0,
            "escalation_rate_max_pct": 45.0,
            "cache_hit_rate_min_pct": 5.0,
        }
    }


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"
EVAL_RESULTS = LOGS_DIR / "eval_results.csv"
EVAL_SUMMARY = LOGS_DIR / "eval_summary.json"
QUERY_LOG = LOGS_DIR / "query_log.jsonl"
THRESHOLD_SWEEP = LOGS_DIR / "threshold_sweep.csv"


def _load_summary() -> dict:
    if not EVAL_SUMMARY.exists():
        return {}
    return json.loads(EVAL_SUMMARY.read_text(encoding="utf-8"))


def _load_eval_results() -> pd.DataFrame:
    if not EVAL_RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(EVAL_RESULTS)


def _load_query_log() -> pd.DataFrame:
    if not QUERY_LOG.exists():
        return pd.DataFrame()
    rows = []
    with QUERY_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _load_threshold_sweep() -> pd.DataFrame:
    if not THRESHOLD_SWEEP.exists():
        return pd.DataFrame()
    return pd.read_csv(THRESHOLD_SWEEP)


def _scenario_summary(df: pd.DataFrame) -> dict:
    summary: dict = {}
    if df.empty:
        return summary

    for scenario in ["cheap_only", "expensive_only", "router"]:
        subset = df[df["scenario"] == scenario]
        if subset.empty:
            continue
        summary[scenario] = {
            "accuracy": float(subset["correct"].mean()),
            "total_cost_usd": float(subset["cost_usd"].sum()),
            "avg_latency_ms": float(subset["latency_ms"].mean()),
            "escalation_rate": float(subset["escalated"].mean()) if "escalated" in subset.columns else 0.0,
            "cache_hit_rate": float(subset["cache_hit"].mean()) if "cache_hit" in subset.columns else 0.0,
            "multi_source_rate": float((subset["source_count"] >= 2).mean()) if "source_count" in subset.columns else 0.0,
        }

        tier1 = subset[subset["tier"] == "tier_1"] if "tier" in subset.columns else pd.DataFrame()
        summary[scenario]["tier_1_p95_latency_ms"] = float(tier1["latency_ms"].quantile(0.95)) if not tier1.empty else 0.0

    if "router" in summary and "expensive_only" in summary:
        expensive_cost = summary["expensive_only"]["total_cost_usd"]
        router_cost = summary["router"]["total_cost_usd"]
        summary["router"]["cost_savings_vs_expensive_pct"] = (
            float((1 - router_cost / expensive_cost) * 100.0) if expensive_cost > 0 else 0.0
        )

    return summary


def _kpi_rows(router: dict, domain: str) -> list[dict[str, str]]:
    targets = DOMAIN_KPI_TARGETS.get(domain, DOMAIN_KPI_TARGETS.get("support", {}))

    checks = [
        ("Router Accuracy", router.get("accuracy", 0.0) * 100.0, targets.get("accuracy_min_pct", 0.0), "min", "%"),
        (
            "Cost Savings vs Expensive",
            router.get("cost_savings_vs_expensive_pct", 0.0),
            targets.get("cost_savings_min_pct", 0.0),
            "min",
            "%",
        ),
        (
            "Tier 1 P95 Latency",
            router.get("tier_1_p95_latency_ms", 0.0),
            targets.get("tier1_p95_latency_max_ms", 0.0),
            "max",
            "ms",
        ),
        (
            "Escalation Rate",
            router.get("escalation_rate", 0.0) * 100.0,
            targets.get("escalation_rate_max_pct", 100.0),
            "max",
            "%",
        ),
        (
            "Cache Hit Rate",
            router.get("cache_hit_rate", 0.0) * 100.0,
            targets.get("cache_hit_rate_min_pct", 0.0),
            "min",
            "%",
        ),
    ]

    if "multi_source_rate_min_pct" in targets:
        checks.append(
            (
                "Multi-Source Rate",
                router.get("multi_source_rate", 0.0) * 100.0,
                targets.get("multi_source_rate_min_pct", 0.0),
                "min",
                "%",
            )
        )

    rows = []
    for metric, value, target, direction, unit in checks:
        passed = value >= target if direction == "min" else value <= target
        rows.append(
            {
                "metric": metric,
                "value": f"{value:.2f}{unit}",
                "target": (f">= {target:.2f}{unit}" if direction == "min" else f"<= {target:.2f}{unit}"),
                "status": "pass" if passed else "watch",
            }
        )
    return rows


st.set_page_config(page_title="RAG Router Dashboard", layout="wide")
st.title("Multi-Model Cost and Quality Router")
st.caption("Track cost, latency, routing tiers, and accuracy tradeoffs.")

summary = _load_summary()
eval_df = _load_eval_results()
query_df = _load_query_log()
sweep_df = _load_threshold_sweep()

selected_domain = str(summary.get("domain", "support")) if isinstance(summary, dict) else "support"
eval_view = eval_df
query_view = query_df

if not eval_df.empty and "domain" in eval_df.columns:
    domains = sorted(str(item) for item in eval_df["domain"].dropna().unique())
    if domains:
        default_idx = domains.index(selected_domain) if selected_domain in domains else 0
        selected_domain = st.selectbox("Domain Preset", domains, index=default_idx)
        eval_view = eval_df[eval_df["domain"] == selected_domain]

if not query_df.empty and "domain" in query_df.columns:
    query_view = query_df[query_df["domain"] == selected_domain]

summary_view = _scenario_summary(eval_view) if not eval_view.empty else summary
if isinstance(summary_view, dict):
    summary_view["domain"] = selected_domain

if not summary_view or "router" not in summary_view:
    st.warning("No evaluation artifacts found yet. Run: python -m src.main eval")
else:
    col1, col2, col3 = st.columns(3)
    router = summary_view.get("router", {})
    col1.metric("Router Accuracy", f"{router.get('accuracy', 0) * 100:.1f}%")
    col2.metric("Router Total Cost", f"${router.get('total_cost_usd', 0):.4f}")
    col3.metric("Cost Savings vs Expensive", f"{router.get('cost_savings_vs_expensive_pct', 0):.1f}%")

    st.subheader("Domain KPI Targets")
    st.caption(f"Preset: {selected_domain}")
    kpi_df = pd.DataFrame(_kpi_rows(router, selected_domain))
    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

if not query_view.empty and "cache_hit" in query_view.columns:
    st.subheader("Semantic Cache Performance")
    hit_rate = float(query_view["cache_hit"].mean()) * 100.0
    st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")

if not eval_view.empty:
    st.subheader("Cost vs Accuracy by Scenario")
    agg = (
        eval_view.groupby("scenario", as_index=False)
        .agg(accuracy=("correct", "mean"), total_cost_usd=("cost_usd", "sum"), avg_latency_ms=("latency_ms", "mean"))
        .sort_values("total_cost_usd")
    )

    fig = px.scatter(
        agg,
        x="total_cost_usd",
        y="accuracy",
        size="avg_latency_ms",
        color="scenario",
        text="scenario",
        title="Scenario Tradeoff Frontier",
        labels={"total_cost_usd": "Total Cost (USD)", "accuracy": "Accuracy"},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Query Routing Outcomes")
    view_cols = [
        "id",
        "domain",
        "scenario",
        "difficulty",
        "tier",
        "model",
        "cost_usd",
        "latency_ms",
        "similarity",
        "correct",
        "escalated",
        "final_confidence",
        "source_count",
        "route_reason",
        "cache_hit",
        "cache_similarity",
    ]
    safe_view_cols = [col for col in view_cols if col in eval_view.columns]
    st.dataframe(eval_view[safe_view_cols], use_container_width=True, hide_index=True)

if not sweep_df.empty:
    sweep_view = sweep_df
    if "domain" in sweep_df.columns:
        sweep_view = sweep_df[sweep_df["domain"] == selected_domain]
    st.subheader("Threshold Sweep Frontier")
    sweep_fig = px.line(
        sweep_view,
        x="threshold",
        y=["router_accuracy", "router_cost_savings_vs_expensive_pct"],
        markers=True,
        title="Accuracy and Cost Savings Across Confidence Thresholds",
    )
    st.plotly_chart(sweep_fig, use_container_width=True)
    st.dataframe(sweep_view, use_container_width=True, hide_index=True)

if not query_view.empty:
    st.subheader("Live Query Log")
    if "timestamp_utc" in query_view.columns:
        query_view = query_view.sort_values("timestamp_utc", ascending=False)
    live_cols = [
        "timestamp_utc",
        "domain",
        "scenario",
        "query",
        "classification",
        "tier",
        "model",
        "cost_usd",
        "latency_ms",
        "final_confidence",
        "escalated",
        "cache_hit",
        "cache_similarity",
    ]
    safe_live_cols = [col for col in live_cols if col in query_view.columns]
    st.dataframe(
        query_view[safe_live_cols],
        use_container_width=True,
        hide_index=True,
    )
