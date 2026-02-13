from app.services.scoring import score_band, tuned_band_thresholds


def test_tuned_band_thresholds_without_quantiles_uses_defaults() -> None:
    review, block = tuned_band_thresholds(0.35, 0.65, quantiles=None)
    assert review == 0.35
    assert block == 0.65


def test_tuned_band_thresholds_with_quantiles_blends_history() -> None:
    review, block = tuned_band_thresholds(
        0.35,
        0.65,
        quantiles={
            "sample_count": 160.0,
            "score_p70": 0.52,
            "score_p90": 0.82,
        },
    )
    assert 0.39 < review < 0.43
    assert 0.69 < block < 0.73
    assert block > review + 0.08


def test_score_band_applies_threshold_order() -> None:
    assert score_band(0.2, 0.35, 0.65) == "monitor"
    assert score_band(0.5, 0.35, 0.65) == "review"
    assert score_band(0.8, 0.35, 0.65) == "block"
