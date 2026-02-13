from app.services.spectral import (
    build_cooccurrence_graph,
    context_covariance_largest_eigenvalue,
    term_graph_spectral_radius,
)


def test_context_covariance_largest_eigenvalue_nonnegative() -> None:
    contexts = [
        "this is a neutral sentence",
        "this sentence is more hostile toward you",
        "community members reclaimed the term for themselves",
    ]
    value = context_covariance_largest_eigenvalue(contexts, dim=128)
    assert value >= 0.0


def test_context_covariance_single_context_uses_internal_views() -> None:
    varied = context_covariance_largest_eigenvalue(
        ["example alpha beta gamma delta epsilon"], dim=128
    )
    repetitive = context_covariance_largest_eigenvalue(
        ["example example example example"], dim=128
    )

    assert varied > 0.0
    assert abs(varied - repetitive) > 1e-6


def test_term_graph_spectral_radius_nonnegative() -> None:
    contexts = [
        "alpha beta gamma",
        "beta gamma delta",
        "alpha delta epsilon",
    ]
    graph = build_cooccurrence_graph(contexts)
    radius = term_graph_spectral_radius("beta", graph)
    assert radius >= 0.0


def test_term_graph_spectral_radius_single_context_varies_by_structure() -> None:
    minimal_graph = build_cooccurrence_graph(["you are awful example"])
    richer_graph = build_cooccurrence_graph(["example alpha beta gamma delta epsilon"])

    minimal = term_graph_spectral_radius("example", minimal_graph)
    richer = term_graph_spectral_radius("example", richer_graph)

    assert minimal > 0.0
    assert richer > 0.0
    assert abs(minimal - richer) > 1e-6


def test_term_graph_spectral_radius_hyphenated_term() -> None:
    contexts = [
        "alpha history-term beta",
        "history-term gamma delta",
    ]
    graph = build_cooccurrence_graph(contexts)
    radius = term_graph_spectral_radius("history-term", graph)
    assert radius > 0.0


def test_windowed_cooccurrence_reduces_long_range_edges() -> None:
    near_graph = build_cooccurrence_graph(
        ["alpha beta"], window_size=2, stopwords=frozenset()
    )
    far_graph = build_cooccurrence_graph(
        ["alpha x1 x2 x3 beta"], window_size=2, stopwords=frozenset()
    )

    assert "beta" in near_graph.adjacency.get("alpha", {})
    assert "beta" not in far_graph.adjacency.get("alpha", {})


def test_stopwords_do_not_create_graph_signal() -> None:
    graph = build_cooccurrence_graph(["alpha the and of to"], window_size=3)
    radius = term_graph_spectral_radius("alpha", graph)
    assert radius == 0.0
