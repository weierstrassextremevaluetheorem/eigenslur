from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    TermHistoryResponse,
    TermScoreResponse,
    TextScoreResponse,
    TextTermScore,
)
from app.services.fusion import FeatureVector, FusionEngine
from app.services.labeler_base import Labeler
from app.services.spectral import (
    build_cooccurrence_graph,
    context_covariance_largest_eigenvalue,
    term_graph_spectral_radius,
)
from app.services.storage import SQLiteStore
from app.services.text import (
    normalize_term,
    split_sentences,
    token_sequence_contains,
    tokenize,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def tuned_band_thresholds(
    default_review: float,
    default_block: float,
    quantiles: Mapping[str, float] | None,
    min_samples: int = 80,
) -> tuple[float, float]:
    if quantiles is None:
        return default_review, default_block

    sample_count = int(quantiles.get("sample_count", 0.0))
    score_p70 = quantiles.get("score_p70")
    score_p90 = quantiles.get("score_p90")
    if sample_count < min_samples or score_p70 is None or score_p90 is None:
        return default_review, default_block

    review = _clamp((0.7 * default_review) + (0.3 * score_p70), 0.2, 0.75)
    block_candidate = (0.7 * default_block) + (0.3 * score_p90)
    min_block = review + 0.08
    block = _clamp(max(block_candidate, min_block), min_block, 0.95)
    return review, block


def score_band(
    score: float,
    review_threshold: float,
    block_threshold: float,
) -> Literal["monitor", "review", "block"]:
    if score >= block_threshold:
        return "block"
    if score >= review_threshold:
        return "review"
    return "monitor"


@dataclass
class ScoreService:
    embedding_dim: int
    labeler: Labeler
    fusion_engine: FusionEngine
    storage: SQLiteStore | None = None

    def score_term(
        self,
        term: str,
        contexts: list[str],
        locale: str,
        trend_velocity: float = 0.0,
        persist: bool = True,
    ) -> TermScoreResponse:
        target = normalize_term(term)
        target_tokens = tokenize(target)
        term_found_in_context = any(
            token_sequence_contains(tokenize(context), target_tokens)
            for context in contexts
        )
        warnings: list[str] = []
        if not term_found_in_context:
            warnings.append(
                "The scored term was not found in any provided context. Add contexts that include the exact term for reliable scoring."
            )

        labels = self.labeler.label_batch(target, contexts, locale=locale)
        sample_size = len(contexts)

        severity_mean = sum(label.severity for label in labels) / sample_size
        targetedness_mean = sum(label.targetedness for label in labels) / sample_size
        reclaimed_rate = sum(1 for label in labels if label.reclaimed) / sample_size

        eigen_ctx = context_covariance_largest_eigenvalue(
            contexts, dim=self.embedding_dim
        )
        graph = build_cooccurrence_graph(contexts)
        eigen_graph = term_graph_spectral_radius(target, graph, hops=2)
        if term_found_in_context and eigen_graph <= 0.0:
            warnings.append(
                "No graph signal was found for this term in the provided contexts. Add more varied contexts where the term co-occurs with descriptive language."
            )

        feature_quantiles: dict[str, float] | None = None
        if self.storage is not None:
            feature_quantiles = self.storage.get_feature_quantiles()

        review_threshold, block_threshold = tuned_band_thresholds(
            default_review=self.fusion_engine.review_threshold,
            default_block=self.fusion_engine.block_threshold,
            quantiles=feature_quantiles,
        )

        fusion = self.fusion_engine.fuse(
            FeatureVector(
                lambda_graph=eigen_graph,
                lambda_ctx=eigen_ctx,
                severity_mean=severity_mean,
                targetedness_mean=targetedness_mean,
                reclaimed_rate=reclaimed_rate,
                trend_velocity=trend_velocity,
                sample_size=sample_size,
            ),
            feature_quantiles=feature_quantiles,
        )
        band = score_band(
            score=fusion.score,
            review_threshold=review_threshold,
            block_threshold=block_threshold,
        )

        response = TermScoreResponse(
            term=target,
            locale=locale,
            sample_size=sample_size,
            eigen_ctx=eigen_ctx,
            eigen_graph=eigen_graph,
            severity_mean=severity_mean,
            targetedness_mean=targetedness_mean,
            reclaimed_rate=reclaimed_rate,
            trend_velocity=trend_velocity,
            score=fusion.score,
            confidence=fusion.confidence,
            band=band,
            model_version=fusion.model_version,
            warnings=warnings,
        )
        if persist and self.storage is not None:
            self.storage.save_term_score(response)
        return response

    def score_text(
        self, text: str, candidate_terms: list[str], locale: str, persist: bool = False
    ) -> TextScoreResponse:
        text_tokens = tokenize(text)
        sentences = split_sentences(text) or [text]
        results: list[TextTermScore] = []

        for candidate in sorted(
            {normalize_term(term) for term in candidate_terms if term.strip()}
        ):
            candidate_tokens = tokenize(candidate)
            if not token_sequence_contains(text_tokens, candidate_tokens):
                continue
            term_contexts = [
                sentence
                for sentence in sentences
                if token_sequence_contains(tokenize(sentence), candidate_tokens)
            ]
            if not term_contexts:
                term_contexts = [text]

            scored = self.score_term(
                term=candidate,
                contexts=term_contexts,
                locale=locale,
                trend_velocity=0.0,
                persist=persist,
            )
            results.append(
                TextTermScore(
                    term=scored.term,
                    score=scored.score,
                    confidence=scored.confidence,
                    band=scored.band,
                )
            )

        return TextScoreResponse(
            locale=locale, terms_found=len(results), results=results
        )

    def get_term_history(self, term: str, limit: int = 50) -> TermHistoryResponse:
        normalized = normalize_term(term)
        if self.storage is None:
            return TermHistoryResponse(term=normalized, count=0, history=[])
        history = self.storage.get_term_history(normalized, limit=limit)
        return TermHistoryResponse(term=normalized, count=len(history), history=history)

    def submit_feedback(self, payload: FeedbackRequest) -> FeedbackResponse:
        if self.storage is None:
            raise RuntimeError("Persistence is disabled; feedback cannot be recorded.")
        feedback_id = self.storage.save_feedback(payload)
        return FeedbackResponse(status="accepted", feedback_id=feedback_id)
