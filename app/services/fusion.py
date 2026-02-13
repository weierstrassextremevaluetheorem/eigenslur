from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from collections.abc import Mapping


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _compress_nonnegative(value: float) -> float:
    return math.tanh(math.log1p(max(0.0, value)))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _safe_quantile(
    feature_quantiles: Mapping[str, float] | None, key: str
) -> float | None:
    if feature_quantiles is None:
        return None
    value = feature_quantiles.get(key)
    if value is None:
        return None
    return float(value)


def _calibrate_nonnegative_feature(
    value: float,
    p50: float | None,
    p90: float | None,
) -> float:
    baseline = _compress_nonnegative(value)
    if p50 is None or p90 is None or p90 <= p50:
        return baseline

    spread = max(1e-6, p90 - p50)
    z_score = (value - p50) / spread
    calibrated = _sigmoid(1.2 * z_score)
    return _clamp((0.45 * baseline) + (0.55 * calibrated))


@dataclass(frozen=True)
class FeatureVector:
    lambda_graph: float
    lambda_ctx: float
    severity_mean: float
    targetedness_mean: float
    reclaimed_rate: float
    trend_velocity: float
    sample_size: int


@dataclass(frozen=True)
class FusionOutput:
    score: float
    confidence: float
    band: Literal["monitor", "review", "block"]
    linear_value: float
    model_version: str


@dataclass
class FusionEngine:
    review_threshold: float = 0.35
    block_threshold: float = 0.65
    model_version: str = "fusion_v1"

    # Coefficients from initial rule-based prior. Replace with calibrated model later.
    b0: float = -0.8
    b1: float = 0.9
    b2: float = 0.7
    b3: float = 1.1
    b4: float = 1.0
    b5: float = 0.9
    b6: float = 0.4

    def fuse(
        self,
        features: FeatureVector,
        feature_quantiles: Mapping[str, float] | None = None,
    ) -> FusionOutput:
        graph_signal = _calibrate_nonnegative_feature(
            features.lambda_graph,
            _safe_quantile(feature_quantiles, "eigen_graph_p50"),
            _safe_quantile(feature_quantiles, "eigen_graph_p90"),
        )
        ctx_signal = _calibrate_nonnegative_feature(
            features.lambda_ctx,
            _safe_quantile(feature_quantiles, "eigen_ctx_p50"),
            _safe_quantile(feature_quantiles, "eigen_ctx_p90"),
        )

        linear = (
            self.b0
            + (self.b1 * graph_signal)
            + (self.b2 * ctx_signal)
            + (self.b3 * features.severity_mean)
            + (self.b4 * features.targetedness_mean)
            - (self.b5 * features.reclaimed_rate)
            + (self.b6 * features.trend_velocity)
        )
        score = _sigmoid(linear)

        sample_strength = min(1.0, features.sample_size / 20.0)
        confidence = 0.45 + (0.25 * abs((score - 0.5) * 2.0)) + (0.2 * sample_strength)
        confidence -= 0.1 * features.reclaimed_rate
        confidence = _clamp(confidence)

        if score >= self.block_threshold:
            band: Literal["monitor", "review", "block"] = "block"
        elif score >= self.review_threshold:
            band = "review"
        else:
            band = "monitor"

        return FusionOutput(
            score=_clamp(score),
            confidence=confidence,
            band=band,
            linear_value=linear,
            model_version=self.model_version,
        )
