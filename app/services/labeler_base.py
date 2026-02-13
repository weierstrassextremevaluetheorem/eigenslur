from __future__ import annotations

from typing import Protocol

from app.schemas import ContextLabel


class Labeler(Protocol):
    def label_batch(
        self, term: str, contexts: list[str], locale: str = "en-US"
    ) -> list[ContextLabel]: ...
