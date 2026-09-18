from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    AMBIGUOUS = "ambiguous"


@dataclass
class QueryClassification:
    label: QueryComplexity
    confidence: float
    rationale: str


@dataclass
class RAGResponse:
    query: str
    answer: str
    model: str
    tier: str
    cost_usd: float
    latency_ms: float
    model_confidence: float
    heuristic_confidence: float
    final_confidence: float
    escalated: bool
    retrieved_sources: list[str]
    domain: str = "support"
    route_reason: str = ""
    cache_hit: bool = False
    cache_similarity: float = 0.0
