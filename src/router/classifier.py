from __future__ import annotations

from src.config import AppConfig
from src.rag.llm_client import call_json_model
from src.rag.schemas import QueryClassification, QueryComplexity


CLASSIFIER_SYSTEM_PROMPT = """
You classify user queries for RAG routing.
Return strict JSON with keys: label, confidence, rationale.
Allowed labels: simple, complex, ambiguous.

Rules:
- simple: direct lookup/fact retrieval from one chunk or one source.
- complex: multi-hop reasoning, comparison, synthesis across multiple facts.
- ambiguous: missing detail, broad intent, or unclear objective.
""".strip()


def _heuristic_label(query: str) -> QueryClassification:
    lowered = query.lower()
    complex_markers = ["why", "compare", "tradeoff", "analyze", "explain how", "best way", "strategy"]
    ambiguous_markers = ["tell me about", "help", "what should i do", "anything"]

    if any(marker in lowered for marker in complex_markers):
        return QueryClassification(QueryComplexity.COMPLEX, 0.62, "Heuristic: contains reasoning marker keywords.")
    if any(marker in lowered for marker in ambiguous_markers):
        return QueryClassification(QueryComplexity.AMBIGUOUS, 0.58, "Heuristic: contains broad or underspecified language.")
    return QueryClassification(QueryComplexity.SIMPLE, 0.61, "Heuristic: defaults to factual lookup.")


class QueryClassifier:
    def __init__(self, config: AppConfig):
        self.config = config

    def classify(self, query: str) -> QueryClassification:
        messages = [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Query: {query}\nReturn JSON only.",
            },
        ]

        try:
            result = call_json_model(
                model=self.config.classifier_model,
                messages=messages,
                fallback_models=self.config.tier_1_models,
                temperature=0.0,
                max_tokens=120,
            )
            payload = result.parsed_json or {}
            label = str(payload.get("label", "")).strip().lower()
            confidence = float(payload.get("confidence", 0.55))
            rationale = str(payload.get("rationale", "No rationale provided."))

            if label not in {"simple", "complex", "ambiguous"}:
                return _heuristic_label(query)

            return QueryClassification(
                label=QueryComplexity(label),
                confidence=max(0.0, min(1.0, confidence)),
                rationale=rationale,
            )
        except Exception:
            return _heuristic_label(query)
