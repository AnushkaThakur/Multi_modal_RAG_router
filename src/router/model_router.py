from __future__ import annotations

from dataclasses import dataclass

from src.config import AppConfig
from src.rag.llm_client import ModelCallResult, call_json_model
from src.rag.schemas import QueryClassification, QueryComplexity, RAGResponse
from src.router.confidence import blend_confidence, heuristic_confidence
from src.router.domain_policy import DomainPolicy


ANSWER_SYSTEM_PROMPT = """
You answer questions using ONLY the provided context.
If the context is insufficient, say so explicitly.
Return strict JSON with keys:
- answer: string
- confidence: number between 0 and 1
""".strip()


@dataclass
class _GenerationResult:
    answer: str
    model: str
    tier: str
    cost_usd: float
    latency_ms: float
    model_confidence: float
    heuristic_confidence: float
    final_confidence: float


class ModelRouter:
    def __init__(self, config: AppConfig, domain_policy: DomainPolicy):
        self.config = config
        self.domain_policy = domain_policy

    def models_for_tier(self, tier: str) -> list[str]:
        if tier == "tier_1":
            primary = self.config.tier_1_models
            backup = self.config.tier_2_models + self.config.tier_3_models
        elif tier == "tier_2":
            primary = self.config.tier_2_models
            backup = self.config.tier_3_models + self.config.tier_1_models
        elif tier == "tier_3":
            primary = self.config.tier_3_models
            backup = self.config.tier_2_models + self.config.tier_1_models
        else:
            raise ValueError(f"Unknown tier: {tier}")

        seen: set[str] = set()
        ordered: list[str] = []
        for model in [*primary, *backup]:
            cleaned = model.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    def _generate(self, query: str, context: str, tier: str) -> _GenerationResult:
        models = self.models_for_tier(tier)
        primary_model = models[0]
        fallbacks = models[1:] if len(models) > 1 else None

        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Context:\n"
                    f"{context}\n\n"
                    "Question:\n"
                    f"{query}\n\n"
                    "Return JSON only."
                ),
            },
        ]

        result: ModelCallResult = call_json_model(
            model=primary_model,
            messages=messages,
            fallback_models=fallbacks,
            temperature=0.0,
            max_tokens=450,
        )

        payload = result.parsed_json or {}
        answer = str(payload.get("answer", "")).strip() or result.raw_content.strip()
        model_conf = float(payload.get("confidence", 0.55))
        model_conf = max(0.0, min(1.0, model_conf))
        heur_conf = heuristic_confidence(answer)
        final_conf = blend_confidence(model_conf, heur_conf)

        return _GenerationResult(
            answer=answer,
            model=result.model,
            tier=tier,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
            model_confidence=model_conf,
            heuristic_confidence=heur_conf,
            final_confidence=final_conf,
        )

    def answer(
        self,
        query: str,
        context: str,
        classification: QueryClassification,
        forced_tier: str | None = None,
        source_count: int = 1,
    ) -> tuple[RAGResponse, bool]:
        if forced_tier is not None:
            initial_tier = forced_tier
            route_reason = f"Forced tier selection: {forced_tier}"
        else:
            initial_tier, route_reason = self.domain_policy.route_tier(query=query, classification=classification)

        primary = self._generate(query=query, context=context, tier=initial_tier)
        escalated = False

        best = primary
        should_escalate = (
            forced_tier is None
            and initial_tier != "tier_3"
            and self.domain_policy.should_escalate(
                tier=initial_tier,
                final_confidence=primary.final_confidence,
                source_count=source_count,
            )
        )

        if should_escalate:
            escalated = True
            route_reason = f"{route_reason} | escalation triggered"
            escalated_result = self._generate(query=query, context=context, tier="tier_3")
            if escalated_result.final_confidence >= primary.final_confidence:
                best = _GenerationResult(
                    answer=escalated_result.answer,
                    model=escalated_result.model,
                    tier=escalated_result.tier,
                    cost_usd=primary.cost_usd + escalated_result.cost_usd,
                    latency_ms=primary.latency_ms + escalated_result.latency_ms,
                    model_confidence=escalated_result.model_confidence,
                    heuristic_confidence=escalated_result.heuristic_confidence,
                    final_confidence=escalated_result.final_confidence,
                )
                route_reason = f"{route_reason} | escalated answer selected"
            else:
                best = _GenerationResult(
                    answer=primary.answer,
                    model=primary.model,
                    tier=primary.tier,
                    cost_usd=primary.cost_usd + escalated_result.cost_usd,
                    latency_ms=primary.latency_ms + escalated_result.latency_ms,
                    model_confidence=primary.model_confidence,
                    heuristic_confidence=primary.heuristic_confidence,
                    final_confidence=primary.final_confidence,
                )
                route_reason = f"{route_reason} | escalation attempted, primary kept"

        response = RAGResponse(
            query=query,
            answer=best.answer,
            model=best.model,
            tier=best.tier,
            cost_usd=best.cost_usd,
            latency_ms=best.latency_ms,
            model_confidence=best.model_confidence,
            heuristic_confidence=best.heuristic_confidence,
            final_confidence=best.final_confidence,
            escalated=escalated,
            retrieved_sources=[],
            domain=self.domain_policy.name.value,
            route_reason=route_reason,
        )
        return response, escalated
