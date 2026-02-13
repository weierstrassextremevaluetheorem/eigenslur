from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from app.schemas import ContextLabel
from app.services.labeler import HeuristicLabeler
from app.services.prompting import LabelerPromptSet
from app.services.text import normalize_term


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class _UsageClassification(BaseModel):
    is_targeted: bool
    target_type: Literal["individual", "group", "none", "unknown"]
    is_quoted: bool
    is_reclaimed: bool
    targetedness_0_1: float = Field(ge=0.0, le=1.0)
    confidence_0_1: float = Field(ge=0.0, le=1.0)


class _SeverityClassification(BaseModel):
    severity_0_1: float = Field(ge=0.0, le=1.0)
    harm_type: Literal["harassment", "hate", "threat", "none", "unknown"]
    violence_signal: bool
    confidence_0_1: float = Field(ge=0.0, le=1.0)


@dataclass
class OpenAIJSONLabeler:
    api_key: str
    model: str
    prompts: LabelerPromptSet
    timeout_seconds: float = 20.0
    fallback: HeuristicLabeler = field(default_factory=HeuristicLabeler)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except (
            ImportError
        ) as error:  # pragma: no cover - exercised only with optional dep
            raise RuntimeError(
                "openai package not installed; install with `pip install -e .[llm]`"
            ) from error
        self._client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    def _run_json_prompt(
        self, system_prompt: str, payload: dict[str, object]
    ) -> dict[str, object]:
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        text = completion.choices[0].message.content
        if not text:
            raise RuntimeError("LLM returned an empty response.")
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response is not a JSON object.")
        return parsed

    def label_context(
        self, term: str, context: str, locale: str = "en-US"
    ) -> ContextLabel:
        normalized = normalize_term(term)
        shared_payload = {
            "term": normalized,
            "sentence": context,
            "surrounding_context": context,
            "locale": locale,
        }

        try:
            usage = _UsageClassification.model_validate(
                self._run_json_prompt(self.prompts.usage_prompt, shared_payload)
            )
            severity = _SeverityClassification.model_validate(
                self._run_json_prompt(
                    self.prompts.severity_prompt,
                    {
                        **shared_payload,
                        "policy_rubric": (
                            "0.0-0.2 neutral, 0.2-0.5 ambiguous insult,"
                            " 0.5-0.8 targeted abuse, 0.8-1.0 violent hate."
                        ),
                    },
                )
            )
        except (RuntimeError, ValidationError, json.JSONDecodeError):
            fallback = self.fallback.label_context(normalized, context, locale=locale)
            return ContextLabel(
                targetedness=fallback.targetedness,
                severity=fallback.severity,
                reclaimed=fallback.reclaimed,
                is_quoted=fallback.is_quoted,
                confidence=fallback.confidence,
                rationale_code="llm_fallback_heuristic_v1",
            )

        confidence = _clamp((usage.confidence_0_1 + severity.confidence_0_1) / 2.0)
        return ContextLabel(
            targetedness=usage.targetedness_0_1,
            severity=severity.severity_0_1,
            reclaimed=usage.is_reclaimed,
            is_quoted=usage.is_quoted,
            confidence=confidence,
            rationale_code="openai_json_v1",
        )

    def label_batch(
        self, term: str, contexts: list[str], locale: str = "en-US"
    ) -> list[ContextLabel]:
        return [
            self.label_context(term=term, context=text, locale=locale)
            for text in contexts
        ]
