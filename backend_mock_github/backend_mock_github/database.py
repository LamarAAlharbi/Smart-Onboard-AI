"""SQLite persistence for finance intern assessment decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3
from pathlib import Path

from rule_engine import AssessmentDecision


@dataclass(frozen=True)
class AssessmentResult:
    """A persisted assessment row."""

    id: int
    intern_id: str
    score: float
    status: str
    priority: str
    weak_topics: str
    created_at: str


class AssessmentRepository:
    """Owns the SQLite schema and writes rule-engine decisions."""

    def __init__(self, database_path: str | Path = "intern_assessments.db") -> None:
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        """Create the database table if it does not yet exist."""
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS assessment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intern_id TEXT NOT NULL,
                    score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    weak_topics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(self, decision: AssessmentDecision) -> AssessmentResult:
        """Persist a decision and return its database representation."""
        created_at = datetime.now(timezone.utc).isoformat()
        weak_topics = ", ".join(decision.weak_topics)
        try:
            with self._connection() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO assessment_results
                        (intern_id, score, status, priority, weak_topics, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        decision.intern_id,
                        decision.score,
                        decision.status.value,
                        decision.priority.value,
                        weak_topics,
                        created_at,
                    ),
                )
                row_id = cursor.lastrowid
        except sqlite3.Error as error:
            raise RuntimeError(f"Unable to save assessment result: {error}") from error

        if row_id is None:  # Defensive; SQLite normally always returns an ID.
            raise RuntimeError("Database did not return an assessment result ID.")
        return AssessmentResult(
            id=row_id,
            intern_id=decision.intern_id,
            score=decision.score,
            status=decision.status.value,
            priority=decision.priority.value,
            weak_topics=weak_topics,
            created_at=created_at,
        )

    def _connection(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.database_path)
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as error:
            raise RuntimeError(f"Unable to connect to database: {error}") from error
