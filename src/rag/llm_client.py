from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import completion, completion_cost


# Drop provider-unsupported params instead of hard-failing a request.
litellm.drop_params = True


@dataclass
class ModelCallResult:
    model: str
    raw_content: str
    parsed_json: dict[str, Any] | None
    cost_usd: float
    latency_ms: float


def _dedupe_models(models: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for model in models:
        cleaned = model.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _normalized_model_name(model: str) -> str:
    return model.split("/", 1)[-1].strip().lower()


def _effective_temperature(model: str, default_temperature: float) -> float:
    normalized = _normalized_model_name(model)
    if normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4"):
        return 1.0
    return default_temperature


def _extract_json(content: str) -> dict[str, Any] | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def call_json_model(
    model: str,
    messages: list[dict[str, str]],
    fallback_models: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> ModelCallResult:
    models_to_try = _dedupe_models([model] + (fallback_models or []))
    errors: list[str] = []

    for candidate in models_to_try:
        started = time.perf_counter()
        try:
            response = completion(
                model=candidate,
                messages=messages,
                temperature=_effective_temperature(candidate, temperature),
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - started) * 1000

            content = response.choices[0].message.content or ""
            cost_usd = 0.0
            try:
                cost_usd = float(completion_cost(completion_response=response))
            except Exception:
                cost_usd = 0.0

            return ModelCallResult(
                model=response.model,
                raw_content=content,
                parsed_json=_extract_json(content),
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")

    joined_errors = " | ".join(errors) if errors else "No candidate models were provided."
    raise RuntimeError(f"All model attempts failed. {joined_errors}")
