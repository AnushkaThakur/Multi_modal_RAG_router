from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _csv_to_list(value: str, default: list[str]) -> list[str]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    corpus_dir: Path
    vectorstore_dir: Path
    logs_dir: Path
    eval_set_path: Path
    eval_results_path: Path
    eval_summary_path: Path
    threshold_sweep_path: Path
    threshold_sweep_summary_path: Path
    sweep_runs_dir: Path
    query_log_path: Path
    semantic_cache_path: Path
    embedding_model: str
    classifier_model: str
    default_domain: str
    tier_1_models: list[str]
    tier_2_models: list[str]
    tier_3_models: list[str]
    confidence_threshold: float
    enable_semantic_cache: bool
    semantic_cache_similarity_threshold: float
    semantic_cache_max_entries: int
    semantic_cache_min_confidence: float
    retrieval_top_k: int
    chunk_size: int
    chunk_overlap: int


def load_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    logs_dir = project_root / "logs"
    eval_dir = data_dir / "eval"

    return AppConfig(
        project_root=project_root,
        corpus_dir=data_dir / "corpus",
        vectorstore_dir=project_root / "vectorstore",
        logs_dir=logs_dir,
        eval_set_path=eval_dir / "eval_set.csv",
        eval_results_path=logs_dir / "eval_results.csv",
        eval_summary_path=logs_dir / "eval_summary.json",
        threshold_sweep_path=logs_dir / "threshold_sweep.csv",
        threshold_sweep_summary_path=logs_dir / "threshold_sweep_summary.json",
        sweep_runs_dir=logs_dir / "sweeps",
        query_log_path=logs_dir / "query_log.jsonl",
        semantic_cache_path=logs_dir / "semantic_cache.json",
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        classifier_model=os.getenv("CLASSIFIER_MODEL", "gpt-4o-mini"),
        default_domain=os.getenv("ROUTER_DOMAIN", "support").strip().lower(),
        tier_1_models=_csv_to_list(
            os.getenv("TIER_1_MODELS", "gpt-4o-mini,groq/llama-3.1-8b-instant"),
            ["gpt-4o-mini", "groq/llama-3.1-8b-instant"],
        ),
        tier_2_models=_csv_to_list(
            os.getenv("TIER_2_MODELS", "gpt-4o,claude-3-5-sonnet-latest"),
            ["gpt-4o", "claude-3-5-sonnet-latest"],
        ),
        tier_3_models=_csv_to_list(
            os.getenv("TIER_3_MODELS", "o3-mini,gpt-4o"),
            ["o3-mini", "gpt-4o"],
        ),
        confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.72")),
        enable_semantic_cache=_env_bool("ENABLE_SEMANTIC_CACHE", True),
        semantic_cache_similarity_threshold=float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.9")),
        semantic_cache_max_entries=int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", "1000")),
        semantic_cache_min_confidence=float(os.getenv("SEMANTIC_CACHE_MIN_CONFIDENCE", "0.65")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "4")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
    )
