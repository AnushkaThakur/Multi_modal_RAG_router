from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from src.rag.schemas import QueryClassification, RAGResponse


class QueryLogger:
    def __init__(self, query_log_path):
        self.query_log_path = query_log_path
        self.query_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, response: RAGResponse, classification: QueryClassification, scenario: str, domain: str):
        payload = asdict(response)
        payload["classification"] = classification.label.value
        payload["classification_confidence"] = classification.confidence
        payload["classification_rationale"] = classification.rationale
        payload["scenario"] = scenario
        payload["domain"] = domain
        payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

        with self.query_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
