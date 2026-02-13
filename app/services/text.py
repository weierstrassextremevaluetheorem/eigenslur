import re
from typing import Sequence

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[_'-][a-z0-9]+)*")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_term(value: str) -> str:
    normalized = value.strip().lower()
    tokens = tokenize(normalized)
    if not tokens:
        return normalized
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def token_sequence_contains(tokens: Sequence[str], term_tokens: Sequence[str]) -> bool:
    if not term_tokens:
        return False
    window = len(term_tokens)
    if window > len(tokens):
        return False
    return any(
        tokens[idx : idx + window] == list(term_tokens)
        for idx in range(len(tokens) - window + 1)
    )


def split_sentences(text: str) -> list[str]:
    chunks = [segment.strip() for segment in SENTENCE_SPLIT_PATTERN.split(text)]
    return [chunk for chunk in chunks if chunk]
