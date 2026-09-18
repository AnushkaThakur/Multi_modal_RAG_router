from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.rag.schemas import QueryClassification, QueryComplexity


class DomainName(str, Enum):
    SUPPORT = "support"
    DEVDOCS = "devdocs"
    COMPLIANCE = "compliance"


DOMAIN_CHOICES = [member.value for member in DomainName]


DOMAIN_KPI_TARGETS: dict[str, dict[str, float]] = {
    "support": {
        "accuracy_min_pct": 88.0,
        "cost_savings_min_pct": 35.0,
        "tier1_p95_latency_max_ms": 4000.0,
        "escalation_rate_max_pct": 45.0,
        "cache_hit_rate_min_pct": 5.0,
    },
    "devdocs": {
        "accuracy_min_pct": 90.0,
        "cost_savings_min_pct": 40.0,
        "tier1_p95_latency_max_ms": 4500.0,
        "escalation_rate_max_pct": 55.0,
        "cache_hit_rate_min_pct": 4.0,
    },
    "compliance": {
        "accuracy_min_pct": 94.0,
        "cost_savings_min_pct": 15.0,
        "tier1_p95_latency_max_ms": 6000.0,
        "escalation_rate_max_pct": 80.0,
        "cache_hit_rate_min_pct": 0.0,
        "multi_source_rate_min_pct": 80.0,
    },
}


def normalize_domain_name(domain: str | None) -> DomainName:
    if not domain:
        return DomainName.SUPPORT
    cleaned = domain.strip().lower()
    for member in DomainName:
        if member.value == cleaned:
            return member
    return DomainName.SUPPORT


def _contains_any(lowered: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in lowered for keyword in keywords)


@dataclass(frozen=True)
class DomainPolicy:
    name: DomainName
    default_confidence_threshold: float
    threshold_by_tier: dict[str, float]
    min_sources_by_tier: dict[str, int]
    support_high_risk_keywords: tuple[str, ...]
    devdocs_force_tier3_keywords: tuple[str, ...]
    devdocs_lookup_keywords: tuple[str, ...]
    devdocs_howto_keywords: tuple[str, ...]
    compliance_force_tier3_keywords: tuple[str, ...]

    def route_tier(self, query: str, classification: QueryClassification) -> tuple[str, str]:
        lowered = query.lower()

        if self.name == DomainName.SUPPORT:
            if _contains_any(lowered, self.support_high_risk_keywords):
                return "tier_3", "Support policy: high-risk intent detected"
            if classification.label == QueryComplexity.SIMPLE:
                return "tier_1", "Support policy: simple retrieval"
            return "tier_2", "Support policy: reasoning/ambiguous query"

        if self.name == DomainName.DEVDOCS:
            if _contains_any(lowered, self.devdocs_force_tier3_keywords):
                return "tier_3", "DevDocs policy: debugging/architecture keyword"
            if _contains_any(lowered, self.devdocs_lookup_keywords) and classification.label != QueryComplexity.COMPLEX:
                return "tier_1", "DevDocs policy: direct API lookup"
            if _contains_any(lowered, self.devdocs_howto_keywords):
                return "tier_2", "DevDocs policy: how-to guidance"
            if classification.label == QueryComplexity.SIMPLE:
                return "tier_1", "DevDocs policy: simple query"
            return "tier_2", "DevDocs policy: medium/complex query"

        if self.name == DomainName.COMPLIANCE:
            if _contains_any(lowered, self.compliance_force_tier3_keywords):
                return "tier_3", "Compliance policy: high-risk regulatory language"
            if classification.label == QueryComplexity.SIMPLE:
                return "tier_2", "Compliance policy: low-risk lookup"
            return "tier_3", "Compliance policy: interpretation requires stronger model"

        if classification.label == QueryComplexity.SIMPLE:
            return "tier_1", "Default policy: simple query"
        return "tier_2", "Default policy: non-simple query"

    def should_escalate(self, tier: str, final_confidence: float, source_count: int) -> bool:
        threshold = self.threshold_by_tier.get(tier, self.default_confidence_threshold)
        min_sources = self.min_sources_by_tier.get(tier, 1)
        return final_confidence < threshold or source_count < min_sources


def build_domain_policy(domain: str | None, base_threshold: float) -> DomainPolicy:
    domain_name = normalize_domain_name(domain)

    support_thresholds = {
        "tier_1": 0.78,
        "tier_2": 0.74,
        "tier_3": 0.68,
    }
    devdocs_thresholds = {
        "tier_1": 0.80,
        "tier_2": 0.76,
        "tier_3": 0.72,
    }
    compliance_thresholds = {
        "tier_1": 0.90,
        "tier_2": 0.84,
        "tier_3": 0.88,
    }

    if domain_name == DomainName.SUPPORT:
        return DomainPolicy(
            name=domain_name,
            default_confidence_threshold=support_thresholds.get("tier_2", base_threshold),
            threshold_by_tier=support_thresholds,
            min_sources_by_tier={"tier_1": 1, "tier_2": 1, "tier_3": 1},
            support_high_risk_keywords=(
                "chargeback",
                "fraud",
                "legal",
                "privacy request",
                "account lock",
                "refund exception",
            ),
            devdocs_force_tier3_keywords=(),
            devdocs_lookup_keywords=(),
            devdocs_howto_keywords=(),
            compliance_force_tier3_keywords=(),
        )

    if domain_name == DomainName.DEVDOCS:
        return DomainPolicy(
            name=domain_name,
            default_confidence_threshold=devdocs_thresholds.get("tier_2", base_threshold),
            threshold_by_tier=devdocs_thresholds,
            min_sources_by_tier={"tier_1": 1, "tier_2": 1, "tier_3": 1},
            support_high_risk_keywords=(),
            devdocs_force_tier3_keywords=(
                "traceback",
                "stack trace",
                "failing test",
                "migration",
                "breaking change",
                "architecture",
                "debug",
            ),
            devdocs_lookup_keywords=(
                "endpoint",
                "parameter",
                "header",
                "status code",
                "default",
                "field",
                "what is",
            ),
            devdocs_howto_keywords=(
                "how to",
                "guide",
                "implement",
                "configure",
                "example",
            ),
            compliance_force_tier3_keywords=(),
        )

    return DomainPolicy(
        name=DomainName.COMPLIANCE,
        default_confidence_threshold=compliance_thresholds.get("tier_3", base_threshold),
        threshold_by_tier=compliance_thresholds,
        min_sources_by_tier={"tier_1": 2, "tier_2": 1, "tier_3": 2},
        support_high_risk_keywords=(),
        devdocs_force_tier3_keywords=(),
        devdocs_lookup_keywords=(),
        devdocs_howto_keywords=(),
        compliance_force_tier3_keywords=(
            "sox",
            "gdpr",
            "hipaa",
            "pci",
            "regulatory",
            "legal",
            "audit",
            "breach",
        ),
    )
