from __future__ import annotations

from dataclasses import dataclass

from app.schemas import ContextLabel
from app.services.text import normalize_term


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


@dataclass
class HeuristicLabeler:
    second_person_cues: tuple[str, ...] = (
        " you ",
        " your ",
        " yourself ",
        "@",
    )
    aggression_cues: tuple[str, ...] = (
        "hate",
        "kill",
        "attack",
        "destroy",
        "worthless",
        "disgusting",
        "stupid",
        "filthy",
    )
    reclaim_cues: tuple[str, ...] = (
        "we",
        "our",
        "ours",
        "us",
        "reclaim",
        "reclaimed",
    )

    def label_context(
        self, term: str, context: str, locale: str = "en-US"
    ) -> ContextLabel:
        _ = locale  # Reserved for locale-specific rules.
        term_norm = normalize_term(term)
        text = f" {context.lower()} "

        is_quoted = f'"{term_norm}"' in text or f"'{term_norm}'" in text
        targeted_hits = sum(1 for cue in self.second_person_cues if cue in text)
        aggression_hits = sum(1 for cue in self.aggression_cues if cue in text)
        reclaim_hits = sum(1 for cue in self.reclaim_cues if f" {cue} " in text)

        targetedness = 0.2 + (0.25 * targeted_hits)
        if "they" in text or "those people" in text:
            targetedness += 0.15
        if is_quoted:
            targetedness *= 0.75

        severity = 0.15 + (0.17 * aggression_hits)
        if "!" in context:
            severity += 0.05
        if is_quoted:
            severity *= 0.65

        reclaimed = reclaim_hits >= 2 and term_norm in text
        if reclaimed:
            severity *= 0.55
            targetedness *= 0.8

        confidence = 0.58 + min(0.25, 0.04 * (targeted_hits + aggression_hits))
        if is_quoted and targeted_hits == 0:
            confidence -= 0.08
        confidence = _clamp(confidence)

        return ContextLabel(
            targetedness=_clamp(targetedness),
            severity=_clamp(severity),
            reclaimed=reclaimed,
            is_quoted=is_quoted,
            confidence=confidence,
            rationale_code="heuristic_v1",
        )

    def label_batch(
        self, term: str, contexts: list[str], locale: str = "en-US"
    ) -> list[ContextLabel]:
        return [
            self.label_context(term=term, context=text, locale=locale)
            for text in contexts
        ]
