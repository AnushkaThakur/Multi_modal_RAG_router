from __future__ import annotations

import re


HEDGING_PATTERNS = [
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bnot sure\b",
    r"\bunclear\b",
    r"\bcannot determine\b",
    r"\bcould be\b",
]


def heuristic_confidence(answer: str) -> float:
    answer = answer.strip()
    if not answer:
        return 0.0

    length_score = min(len(answer) / 350.0, 1.0)
    hedge_hits = sum(bool(re.search(pattern, answer, flags=re.IGNORECASE)) for pattern in HEDGING_PATTERNS)
    hedge_penalty = min(hedge_hits * 0.15, 0.6)

    return max(0.0, min(1.0, 0.55 + 0.45 * length_score - hedge_penalty))


def blend_confidence(self_reported: float, heuristic: float) -> float:
    self_reported = max(0.0, min(1.0, self_reported))
    heuristic = max(0.0, min(1.0, heuristic))
    return round(0.7 * self_reported + 0.3 * heuristic, 4)
