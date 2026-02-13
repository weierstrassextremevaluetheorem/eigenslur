from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np

from app.schemas import FeedbackRequest, TermScoreHistoryItem, TermScoreResponse
from app.services.text import normalize_term


@dataclass
class SQLiteStore:
    database_path: str
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _connection: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self) -> None:
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS term_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    locale TEXT NOT NULL,
                    sample_size INTEGER NOT NULL,
                    eigen_ctx REAL NOT NULL,
                    eigen_graph REAL NOT NULL,
                    severity_mean REAL NOT NULL,
                    targetedness_mean REAL NOT NULL,
                    reclaimed_rate REAL NOT NULL,
                    trend_velocity REAL NOT NULL,
                    score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    band TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_term_scores_term_created
                    ON term_scores(term, created_at DESC);

                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    locale TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    proposed_band TEXT,
                    proposed_score REAL,
                    notes TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_term_created
                    ON feedback(term, created_at DESC);
                """
            )
            self._connection.commit()

    def save_term_score(self, score: TermScoreResponse) -> int:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO term_scores (
                    term, locale, sample_size, eigen_ctx, eigen_graph,
                    severity_mean, targetedness_mean, reclaimed_rate, trend_velocity,
                    score, confidence, band, model_version
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalize_term(score.term),
                    score.locale,
                    score.sample_size,
                    score.eigen_ctx,
                    score.eigen_graph,
                    score.severity_mean,
                    score.targetedness_mean,
                    score.reclaimed_rate,
                    score.trend_velocity,
                    score.score,
                    score.confidence,
                    score.band,
                    score.model_version,
                ),
            )
            self._connection.commit()
            last_row_id = cursor.lastrowid
            if last_row_id is None:
                raise RuntimeError("Failed to persist term score.")
            return last_row_id

    def get_feature_quantiles(
        self,
        sample_limit: int = 1000,
        min_samples: int = 40,
    ) -> dict[str, float] | None:
        safe_limit = max(min_samples, min(sample_limit, 5000))
        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT eigen_ctx, eigen_graph, score
                FROM term_scores
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = cursor.fetchall()

        if len(rows) < min_samples:
            return None

        eigen_ctx = np.array(
            [float(row["eigen_ctx"]) for row in rows], dtype=np.float64
        )
        eigen_graph = np.array(
            [float(row["eigen_graph"]) for row in rows], dtype=np.float64
        )
        scores = np.array([float(row["score"]) for row in rows], dtype=np.float64)
        if eigen_ctx.size == 0 or eigen_graph.size == 0:
            return None

        return {
            "sample_count": float(scores.size),
            "eigen_ctx_p50": float(np.quantile(eigen_ctx, 0.5)),
            "eigen_ctx_p90": float(np.quantile(eigen_ctx, 0.9)),
            "eigen_graph_p50": float(np.quantile(eigen_graph, 0.5)),
            "eigen_graph_p90": float(np.quantile(eigen_graph, 0.9)),
            "score_p70": float(np.quantile(scores, 0.7)),
            "score_p90": float(np.quantile(scores, 0.9)),
        }

    def get_term_history(
        self, term: str, limit: int = 50
    ) -> list[TermScoreHistoryItem]:
        safe_limit = max(1, min(limit, 200))
        with self._lock:
            cursor = self._connection.execute(
                """
                SELECT id, term, locale, score, confidence, band, model_version, created_at
                FROM term_scores
                WHERE term = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (normalize_term(term), safe_limit),
            )
            rows = cursor.fetchall()

        history: list[TermScoreHistoryItem] = []
        for row in rows:
            band_value = str(row["band"])
            band: Literal["monitor", "review", "block"]
            if band_value == "block":
                band = "block"
            elif band_value == "review":
                band = "review"
            else:
                band = "monitor"

            history.append(
                TermScoreHistoryItem(
                    id=int(row["id"]),
                    term=str(row["term"]),
                    locale=str(row["locale"]),
                    score=float(row["score"]),
                    confidence=float(row["confidence"]),
                    band=band,
                    model_version=str(row["model_version"]),
                    created_at=str(row["created_at"]),
                )
            )
        return history

    def save_feedback(self, feedback: FeedbackRequest) -> int:
        with self._lock:
            cursor = self._connection.execute(
                """
                INSERT INTO feedback (
                    term, locale, feedback_type, proposed_band, proposed_score, notes
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    normalize_term(feedback.term),
                    feedback.locale,
                    feedback.feedback_type,
                    feedback.proposed_band,
                    feedback.proposed_score,
                    feedback.notes,
                ),
            )
            self._connection.commit()
            last_row_id = cursor.lastrowid
            if last_row_id is None:
                raise RuntimeError("Failed to persist feedback.")
            return last_row_id
