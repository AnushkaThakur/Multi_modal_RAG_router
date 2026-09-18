from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig
from src.rag.schemas import RAGResponse


@dataclass
class SemanticCacheHit:
    answer: str
    model: str
    tier: str
    final_confidence: float
    retrieved_sources: list[str]
    similarity: float


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    def __init__(self, config: AppConfig):
        self.path = config.semantic_cache_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = config.semantic_cache_similarity_threshold
        self.max_entries = config.semantic_cache_max_entries
        self.min_store_confidence = config.semantic_cache_min_confidence
        self.embedder = HuggingFaceEmbeddings(model_name=config.embedding_model)
        self.entries: list[dict[str, Any]] = self._load_entries()

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
        except Exception:
            return []
        return []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.entries, ensure_ascii=True), encoding="utf-8")

    def lookup(self, query: str) -> tuple[SemanticCacheHit | None, list[float]]:
        query_embedding = self.embedder.embed_query(query)
        if not self.entries:
            return None, query_embedding

        best_entry = None
        best_score = -1.0
        for entry in self.entries:
            cached_vector = entry.get("embedding")
            if not isinstance(cached_vector, list):
                continue
            score = _cosine_similarity(query_embedding, [float(v) for v in cached_vector])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self.similarity_threshold:
            return None, query_embedding

        hit = SemanticCacheHit(
            answer=str(best_entry.get("answer", "")),
            model=str(best_entry.get("model", "semantic-cache")),
            tier=str(best_entry.get("tier", "cache")),
            final_confidence=float(best_entry.get("final_confidence", 0.7)),
            retrieved_sources=list(best_entry.get("retrieved_sources", [])),
            similarity=float(best_score),
        )
        return hit, query_embedding

    def put(self, query: str, response: RAGResponse, query_embedding: list[float] | None = None) -> None:
        if response.final_confidence < self.min_store_confidence:
            return

        embedding = query_embedding or self.embedder.embed_query(query)
        new_entry = {
            "query": query,
            "answer": response.answer,
            "model": response.model,
            "tier": response.tier,
            "final_confidence": response.final_confidence,
            "retrieved_sources": response.retrieved_sources,
            "embedding": embedding,
        }

        replace_idx = None
        for idx, entry in enumerate(self.entries):
            cached_vector = entry.get("embedding")
            if not isinstance(cached_vector, list):
                continue
            score = _cosine_similarity(embedding, [float(v) for v in cached_vector])
            if score >= 0.98:
                replace_idx = idx
                break

        if replace_idx is not None:
            self.entries[replace_idx] = new_entry
        else:
            self.entries.append(new_entry)

        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]
        self._save()
