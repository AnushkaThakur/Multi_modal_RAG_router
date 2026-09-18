from __future__ import annotations

import time

from src.cache.semantic_cache import SemanticCache
from src.config import AppConfig
from src.rag.schemas import QueryClassification, QueryComplexity, RAGResponse
from src.rag.vector_store import build_retriever
from src.router.classifier import QueryClassifier
from src.router.domain_policy import build_domain_policy
from src.router.model_router import ModelRouter
from src.utils.logger import QueryLogger


def _format_context(docs) -> tuple[str, list[str]]:
    parts = []
    sources: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        source = str(doc.metadata.get("source", f"doc_{idx}"))
        sources.append(source)
        parts.append(f"[{idx}] source={source}\n{doc.page_content}")
    return "\n\n".join(parts), sources


class CostAwareRAGPipeline:
    def __init__(self, config: AppConfig, domain: str | None = None):
        self.config = config
        self.domain_policy = build_domain_policy(domain or config.default_domain, config.confidence_threshold)
        self.classifier = QueryClassifier(config)
        self.router = ModelRouter(config, self.domain_policy)
        self.retriever = build_retriever(config)
        self.logger = QueryLogger(config.query_log_path)
        self.cache = SemanticCache(config) if config.enable_semantic_cache else None

    def answer(
        self,
        query: str,
        scenario: str = "router",
        forced_tier: str | None = None,
        use_cache: bool = True,
    ) -> tuple[RAGResponse, QueryClassification]:
        query_embedding: list[float] | None = None
        if use_cache and self.cache and forced_tier is None:
            lookup_started = time.perf_counter()
            hit, query_embedding = self.cache.lookup(query)
            if hit is not None:
                response = RAGResponse(
                    query=query,
                    answer=hit.answer,
                    model=hit.model,
                    tier=hit.tier,
                    cost_usd=0.0,
                    latency_ms=(time.perf_counter() - lookup_started) * 1000,
                    model_confidence=hit.final_confidence,
                    heuristic_confidence=hit.final_confidence,
                    final_confidence=hit.final_confidence,
                    escalated=False,
                    retrieved_sources=hit.retrieved_sources,
                    domain=self.domain_policy.name.value,
                    route_reason="Semantic cache hit",
                    cache_hit=True,
                    cache_similarity=hit.similarity,
                )
                classification = QueryClassification(
                    label=QueryComplexity.SIMPLE,
                    confidence=1.0,
                    rationale="Semantic cache hit",
                )
                self.logger.log(
                    response=response,
                    classification=classification,
                    scenario=scenario,
                    domain=self.domain_policy.name.value,
                )
                return response, classification

        classification = self.classifier.classify(query)
        if forced_tier is not None:
            classification = QueryClassification(
                label=QueryComplexity.AMBIGUOUS,
                confidence=1.0,
                rationale=f"Forced tier run: {forced_tier}",
            )

        docs = self.retriever.invoke(query)
        context, sources = _format_context(docs)
        response, _ = self.router.answer(
            query=query,
            context=context,
            classification=classification,
            forced_tier=forced_tier,
            source_count=len(sources),
        )
        response.retrieved_sources = sources
        response.cache_hit = False

        if use_cache and self.cache and forced_tier is None:
            self.cache.put(query=query, response=response, query_embedding=query_embedding)

        self.logger.log(
            response=response,
            classification=classification,
            scenario=scenario,
            domain=self.domain_policy.name.value,
        )
        return response, classification
