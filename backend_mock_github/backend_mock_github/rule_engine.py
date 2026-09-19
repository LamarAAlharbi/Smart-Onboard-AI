"""Business rules for finance intern assessment evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Priority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class QualificationStatus(str, Enum):
    CRITICAL_WEAKNESS = "CRITICAL_WEAKNESS"
    MODERATE_RISK = "MODERATE_RISK"
    QUALIFIED = "QUALIFIED"


@dataclass(frozen=True)
class InternAssessment:
    """Validated input received by the assessment pipeline."""

    intern_id: str
    name: str
    score: float
    weak_topics: tuple[str, ...]


@dataclass(frozen=True)
class AssessmentDecision:
    """Rule-engine outcome passed through the rest of the pipeline."""

    intern_id: str
    name: str
    score: float
    weak_topics: tuple[str, ...]
    status: QualificationStatus
    priority: Priority


def evaluate_assessment(assessment: InternAssessment) -> AssessmentDecision:
    """Categorize an assessment score using the specified inclusive thresholds."""
    score = _validate_score(assessment.score)
    if score < 0.50:
        status, priority = QualificationStatus.CRITICAL_WEAKNESS, Priority.HIGH
    elif score <= 0.70:
        status, priority = QualificationStatus.MODERATE_RISK, Priority.MEDIUM
    else:
        status, priority = QualificationStatus.QUALIFIED, Priority.LOW

    return AssessmentDecision(
        intern_id=_required_text(assessment.intern_id, "intern_id"),
        name=_required_text(assessment.name, "name"),
        score=score,
        weak_topics=_normalize_topics(assessment.weak_topics),
        status=status,
        priority=priority,
    )


def _validate_score(score: float) -> float:
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score must be a numeric value between 0 and 1.")
    normalized = float(score)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("score must be between 0 and 1.")
    return normalized


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required and must be non-empty.")
    return value.strip()


def _normalize_topics(topics: Sequence[str]) -> tuple[str, ...]:
    if isinstance(topics, (str, bytes)):
        raise ValueError("weak_topics must be a sequence of topic strings.")
    cleaned = tuple(topic.strip() for topic in topics if isinstance(topic, str) and topic.strip())
    if not cleaned:
        raise ValueError("At least one non-empty weak topic is required.")
    return cleaned
