from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from app.services.text import normalize_term, split_sentences, tokenize

_DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True)
class CooccurrenceGraph:
    adjacency: dict[str, dict[str, float]]
    token_doc_frequency: dict[str, int]
    context_count: int


def _token_hash(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "big", signed=False)
    index = raw % dim
    sign = -1.0 if ((raw >> 63) & 1) else 1.0
    return index, sign


def embed_text(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        idx, sign = _token_hash(token, dim)
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _covariance_views(contexts: list[str]) -> list[str]:
    cleaned = [text.strip() for text in contexts if text and text.strip()]
    if len(cleaned) >= 2:
        return cleaned
    if not cleaned:
        return []

    source = cleaned[0]
    sentence_views = split_sentences(source)
    if len(sentence_views) >= 2:
        return sentence_views

    tokens = tokenize(source)
    if len(tokens) < 2:
        return cleaned

    window = min(8, max(3, int(math.sqrt(len(tokens))) + 1))
    stride = max(1, window // 2)
    spans: list[str] = []
    for start in range(0, len(tokens), stride):
        chunk = tokens[start : start + window]
        if len(chunk) < 2:
            continue
        spans.append(" ".join(chunk))

    deduped = list(dict.fromkeys(spans))
    if len(deduped) >= 2:
        return deduped

    midpoint = max(1, len(tokens) // 2)
    halves = [" ".join(tokens[:midpoint]), " ".join(tokens[midpoint:])]
    return [segment for segment in halves if segment]


def context_covariance_largest_eigenvalue(contexts: list[str], dim: int = 256) -> float:
    views = _covariance_views(contexts)
    if len(views) < 2:
        return 0.0

    matrix = np.vstack([embed_text(text, dim=dim) for text in views])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(matrix.shape[0] - 1)

    shrinkage = min(0.35, 4.0 / float(matrix.shape[0] + 3))
    diagonal_target = np.diag(np.diag(covariance))
    covariance = ((1.0 - shrinkage) * covariance) + (shrinkage * diagonal_target)

    eigvals = np.linalg.eigvalsh(covariance)
    return float(max(0.0, eigvals[-1]))


def _filter_tokens(
    text: str, stopwords: frozenset[str], min_token_length: int
) -> list[str]:
    tokens = tokenize(text)
    return [
        token
        for token in tokens
        if len(token) >= min_token_length and token not in stopwords
    ]


def build_cooccurrence_graph(
    contexts: list[str],
    window_size: int = 6,
    min_token_length: int = 2,
    stopwords: frozenset[str] = _DEFAULT_STOPWORDS,
) -> CooccurrenceGraph:
    if window_size < 2:
        raise ValueError("window_size must be >= 2")

    pair_document_frequency: Counter[tuple[str, str]] = Counter()
    pair_proximity_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
    token_document_frequency: Counter[str] = Counter()
    context_count = 0

    for text in contexts:
        tokens = _filter_tokens(
            text=text,
            stopwords=stopwords,
            min_token_length=min_token_length,
        )
        if not tokens:
            continue

        context_count += 1
        token_document_frequency.update(set(tokens))

        context_pair_proximity: dict[tuple[str, str], float] = {}
        for left, token_a in enumerate(tokens):
            right_edge = min(len(tokens), left + window_size)
            for right in range(left + 1, right_edge):
                token_b = tokens[right]
                if token_a == token_b:
                    continue
                pair = (token_a, token_b) if token_a < token_b else (token_b, token_a)
                proximity = 1.0 / float(right - left)
                existing = context_pair_proximity.get(pair)
                if existing is None or proximity > existing:
                    context_pair_proximity[pair] = proximity

        pair_document_frequency.update(context_pair_proximity.keys())
        for pair, proximity in context_pair_proximity.items():
            pair_proximity_sum[pair] += proximity

    if context_count == 0 or not pair_document_frequency:
        return CooccurrenceGraph(
            adjacency={},
            token_doc_frequency=dict(token_document_frequency),
            context_count=context_count,
        )

    graph: dict[str, dict[str, float]] = defaultdict(dict)
    for (token_a, token_b), pair_count in pair_document_frequency.items():
        df_a = token_document_frequency[token_a]
        df_b = token_document_frequency[token_b]
        if df_a == 0 or df_b == 0:
            continue

        pmi = math.log((pair_count * context_count) / (df_a * df_b))
        support = pair_count / context_count
        mean_proximity = pair_proximity_sum[(token_a, token_b)] / pair_count
        ppmi = max(0.0, pmi) + (support * mean_proximity)
        if ppmi <= 0.0:
            continue

        graph[token_a][token_b] = ppmi
        graph[token_b][token_a] = ppmi

    return CooccurrenceGraph(
        adjacency={node: dict(neighbors) for node, neighbors in graph.items()},
        token_doc_frequency=dict(token_document_frequency),
        context_count=context_count,
    )


def _ego_nodes(
    graph: dict[str, dict[str, float]],
    center: str,
    hops: int = 2,
    max_nodes: int = 128,
) -> list[str]:
    if center not in graph:
        return []

    visited = {center}
    frontier = {center}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in graph.get(node, {}):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_frontier.add(neighbor)
                if len(visited) >= max_nodes:
                    return sorted(visited)
        if not next_frontier:
            break
        frontier = next_frontier
    return sorted(visited)


def _unpack_graph(
    graph: CooccurrenceGraph | dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, int], int]:
    if isinstance(graph, CooccurrenceGraph):
        return graph.adjacency, graph.token_doc_frequency, graph.context_count
    return graph, {}, 0


def _idf_weight(
    token: str,
    token_document_frequency: dict[str, int],
    context_count: int,
) -> float:
    if context_count <= 0:
        return 1.0
    df = token_document_frequency.get(token, 0)
    return 1.0 + math.log((context_count + 1.0) / (df + 1.0))


def _non_trivial_normalized_spectral_signal(
    nodes: list[str],
    graph: dict[str, dict[str, float]],
) -> float:
    index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for node_a in nodes:
        row = index[node_a]
        for node_b, weight in graph.get(node_a, {}).items():
            if node_b not in index:
                continue
            col = index[node_b]
            adjacency[row, col] = weight

    symmetric = (adjacency + adjacency.T) / 2.0
    degrees = symmetric.sum(axis=1)
    if not np.any(degrees > 0):
        return 0.0

    inv_sqrt = np.zeros_like(degrees)
    mask = degrees > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(degrees[mask])
    normalized = symmetric * inv_sqrt[:, np.newaxis] * inv_sqrt[np.newaxis, :]

    eigvals = np.linalg.eigvalsh(normalized)
    if eigvals.size <= 1:
        return 0.0

    non_trivial = float(np.max(np.abs(eigvals[:-1])))
    coverage = 1.0 - math.exp(-(len(nodes) - 1) / 3.0)
    return max(0.0, non_trivial * coverage)


def term_graph_spectral_radius(
    term: str,
    graph: CooccurrenceGraph | dict[str, dict[str, float]],
    hops: int = 2,
) -> float:
    adjacency, token_document_frequency, context_count = _unpack_graph(graph)

    targets = tokenize(normalize_term(term))
    if not targets:
        return 0.0

    weighted_radius = 0.0
    weight_total = 0.0

    for target in targets:
        nodes = _ego_nodes(adjacency, target, hops=hops)
        if len(nodes) < 2:
            continue

        signal = _non_trivial_normalized_spectral_signal(nodes, adjacency)
        if signal <= 0.0:
            continue

        token_weight = _idf_weight(target, token_document_frequency, context_count)
        weighted_radius += signal * token_weight
        weight_total += token_weight

    if weight_total == 0.0:
        return 0.0
    return weighted_radius / weight_total
