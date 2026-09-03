"""
Phase 30 — Native Feedback Processor for EVORA.

Processes and integrates user feedback.

Supports:
  - Feedback collection
  - Feedback classification
  - Feedback integration
  - Feedback analytics
  - Integration with TrainingPipeline
  - Integration with SelfReflection
  - Integration with KnowledgeConsolidation

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class FeedbackType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    RATING = "rating"


class FeedbackStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    INTEGRATED = "integrated"
    REJECTED = "rejected"


@dataclass
class Feedback:
    """A feedback entry."""
    feedback_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    feedback_type: FeedbackType = FeedbackType.SUGGESTION
    status: FeedbackStatus = FeedbackStatus.PENDING
    content: str = ""
    source: str = ""
    rating: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "feedback_type": self.feedback_type.value,
            "status": self.status.value,
            "content": self.content,
            "source": self.source,
            "rating": self.rating,
            "context": self.context,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "processed_at": self.processed_at,
        }


@dataclass
class FeedbackAnalytics:
    """Feedback analytics."""
    total_feedback: int = 0
    positive_count: int = 0
    negative_count: int = 0
    average_rating: float = 0.0
    by_type: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_feedback": self.total_feedback,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "average_rating": self.average_rating,
            "by_type": self.by_type,
            "by_source": self.by_source,
        }


# ---------------------------------------------------------------------------
# Native Feedback Processor
# ---------------------------------------------------------------------------

class NativeFeedbackProcessor:
    """Native feedback processor for EVORA.

    Processes and integrates user feedback.
    """

    def __init__(
        self,
        training_pipeline: Any = None,
        self_reflection: Any = None,
        knowledge_consolidation: Any = None,
        logger: Optional[Any] = None,
    ):
        self.training_pipeline = training_pipeline
        self.self_reflection = self_reflection
        self.knowledge_consolidation = knowledge_consolidation
        self.logger = logger
        self._feedback: dict[str, Feedback] = {}
        self._analytics = FeedbackAnalytics()

    def submit_feedback(self, content: str, feedback_type: FeedbackType = FeedbackType.SUGGESTION, source: str = "", rating: float = 0.0, context: dict[str, Any] = None) -> Feedback:
        """Submit feedback for processing."""
        context = context or {}
        feedback = Feedback(
            feedback_type=feedback_type,
            content=content,
            source=source,
            rating=rating,
            context=context,
        )
        self._feedback[feedback.feedback_id] = feedback
        self._update_analytics(feedback)
        return feedback

    def process_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Process a feedback entry."""
        feedback = self._feedback.get(feedback_id)
        if feedback is None:
            return None
        feedback.status = FeedbackStatus.PROCESSED
        feedback.processed_at = datetime.now().isoformat()
        if self.self_reflection is not None:
            try:
                if hasattr(self.self_reflection, "add_improvement_area"):
                    self.self_reflection.add_improvement_area(feedback.content[:100])
            except Exception:
                pass
        return feedback

    def integrate_feedback(self, feedback_id: str) -> bool:
        """Integrate processed feedback into learning."""
        feedback = self._feedback.get(feedback_id)
        if feedback is None or feedback.status != FeedbackStatus.PROCESSED:
            return False
        if self.training_pipeline is not None:
            try:
                from evora.brain.intelligence.training import OutcomeType
                outcome = OutcomeType.SUCCESS if feedback.feedback_type == FeedbackType.POSITIVE else OutcomeType.FAILURE
                self.training_pipeline.record_training_example(
                    session_id=context.get("session_id", ""),
                    task_id=context.get("task_id", ""),
                    project=context.get("project", ""),
                    component="feedback",
                    input_data={"content": feedback.content, "type": feedback.feedback_type.value},
                    output_data={"rating": feedback.rating, "processed": True},
                    outcome=outcome,
                    confidence=0.7,
                )
            except Exception:
                pass
        feedback.status = FeedbackStatus.INTEGRATED
        return True

    def get_feedback(self, feedback_id: str) -> Optional[Feedback]:
        """Get feedback by ID."""
        return self._feedback.get(feedback_id)

    def get_all_feedback(self, status: FeedbackStatus = None) -> list[Feedback]:
        """Get all feedback, optionally filtered by status."""
        feedbacks = list(self._feedback.values())
        if status is not None:
            feedbacks = [f for f in feedbacks if f.status == status]
        return feedbacks

    def get_analytics(self) -> dict[str, Any]:
        """Get feedback analytics."""
        return self._analytics.to_dict()

    def _update_analytics(self, feedback: Feedback) -> None:
        """Update analytics with new feedback."""
        self._analytics.total_feedback += 1
        if feedback.feedback_type == FeedbackType.POSITIVE:
            self._analytics.positive_count += 1
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            self._analytics.negative_count += 1
        self._analytics.by_type[feedback.feedback_type.value] = self._analytics.by_type.get(feedback.feedback_type.value, 0) + 1
        if feedback.source:
            self._analytics.by_source[feedback.source] = self._analytics.by_source.get(feedback.source, 0) + 1
        if feedback.rating > 0:
            total = self._analytics.total_feedback
            self._analytics.average_rating = ((self._analytics.average_rating * (total - 1)) + feedback.rating) / total
