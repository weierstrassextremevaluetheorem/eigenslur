from app.services.fusion import FeatureVector, FusionEngine


def test_fusion_quantile_calibration_changes_signal() -> None:
    engine = FusionEngine()
    quantiles = {
        "eigen_graph_p50": 0.08,
        "eigen_graph_p90": 0.32,
        "eigen_ctx_p50": 0.04,
        "eigen_ctx_p90": 0.22,
    }

    lower = engine.fuse(
        FeatureVector(
            lambda_graph=0.05,
            lambda_ctx=0.04,
            severity_mean=0.5,
            targetedness_mean=0.5,
            reclaimed_rate=0.0,
            trend_velocity=0.0,
            sample_size=10,
        ),
        feature_quantiles=quantiles,
    )
    higher = engine.fuse(
        FeatureVector(
            lambda_graph=0.5,
            lambda_ctx=0.2,
            severity_mean=0.5,
            targetedness_mean=0.5,
            reclaimed_rate=0.0,
            trend_velocity=0.0,
            sample_size=10,
        ),
        feature_quantiles=quantiles,
    )

    assert higher.score > lower.score


def test_fusion_without_quantiles_still_produces_bounded_values() -> None:
    engine = FusionEngine()
    result = engine.fuse(
        FeatureVector(
            lambda_graph=1.0,
            lambda_ctx=0.2,
            severity_mean=0.6,
            targetedness_mean=0.7,
            reclaimed_rate=0.1,
            trend_velocity=0.2,
            sample_size=8,
        )
    )

    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.confidence <= 1.0
