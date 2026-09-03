"""
Phase 30 — Native Feedback Processor tests.

Verifies:
1. Feedback has correct structure
2. FeedbackAnalytics has correct structure
3. FeedbackType enum exists
4. FeedbackStatus enum exists
5. NativeFeedbackProcessor initializes
6. NativeFeedbackProcessor submits feedback
7. NativeFeedbackProcessor processes feedback
8. NativeFeedbackProcessor integrates feedback
9. NativeFeedbackProcessor gets feedback by ID
10. NativeFeedbackProcessor gets all feedback
11. NativeFeedbackProcessor returns analytics
12. Feedback integrates with TrainingPipeline
13. No ModelManager dependency
14. No external dependencies
15. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.feedback_processor import (
    Feedback,
    FeedbackAnalytics,
    FeedbackStatus,
    FeedbackType,
    NativeFeedbackProcessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def feedback_processor():
    return NativeFeedbackProcessor(logger=MagicMock())


@pytest.fixture
def feedback_with_training():
    training = MagicMock()
    return NativeFeedbackProcessor(training_pipeline=training, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestFeedback
# ---------------------------------------------------------------------------

class TestFeedback:
    """Test Feedback."""

    def test_default_feedback(self):
        feedback = Feedback()
        assert feedback.feedback_id != ""
        assert feedback.feedback_type == FeedbackType.SUGGESTION

    def test_feedback_to_dict(self):
        feedback = Feedback(content="Great job", feedback_type=FeedbackType.POSITIVE, rating=5.0)
        data = feedback.to_dict()
        assert data["content"] == "Great job"
        assert data["rating"] == 5.0


# ---------------------------------------------------------------------------
# TestFeedbackAnalytics
# ---------------------------------------------------------------------------

class TestFeedbackAnalytics:
    """Test FeedbackAnalytics."""

    def test_default_analytics(self):
        analytics = FeedbackAnalytics()
        assert analytics.total_feedback == 0
        assert analytics.average_rating == 0.0

    def test_analytics_to_dict(self):
        analytics = FeedbackAnalytics(total_feedback=10, positive_count=7, average_rating=4.0)
        data = analytics.to_dict()
        assert data["total_feedback"] == 10
        assert data["average_rating"] == 4.0


# ---------------------------------------------------------------------------
# TestNativeFeedbackProcessor
# ---------------------------------------------------------------------------

class TestNativeFeedbackProcessor:
    """Test NativeFeedbackProcessor."""

    def test_feedback_processor_initializes(self, feedback_processor):
        assert feedback_processor is not None

    def test_submit_positive_feedback(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Great job", feedback_type=FeedbackType.POSITIVE)
        assert feedback.feedback_id != ""
        assert feedback.feedback_type == FeedbackType.POSITIVE

    def test_submit_negative_feedback(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Not good", feedback_type=FeedbackType.NEGATIVE)
        assert feedback.feedback_type == FeedbackType.NEGATIVE

    def test_submit_feedback_with_rating(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Good", rating=4.5)
        assert feedback.rating == 4.5

    def test_submit_feedback_with_source(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Test", source="user")
        assert feedback.source == "user"

    def test_process_feedback(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Test")
        processed = feedback_processor.process_feedback(feedback.feedback_id)
        assert processed is not None
        assert processed.status == FeedbackStatus.PROCESSED

    def test_process_feedback_missing(self, feedback_processor):
        processed = feedback_processor.process_feedback("nonexistent")
        assert processed is None

    def test_integrate_feedback(self, feedback_with_training):
        feedback = feedback_with_training.submit_feedback("Test")
        feedback_with_training.process_feedback(feedback.feedback_id)
        result = feedback_with_training.integrate_feedback(feedback.feedback_id)
        assert result is True
        assert feedback.status == FeedbackStatus.INTEGRATED

    def test_integrate_feedback_unprocessed(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Test")
        result = feedback_processor.integrate_feedback(feedback.feedback_id)
        assert result is False

    def test_integrate_feedback_missing(self, feedback_processor):
        result = feedback_processor.integrate_feedback("nonexistent")
        assert result is False

    def test_get_feedback(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("Test")
        retrieved = feedback_processor.get_feedback(feedback.feedback_id)
        assert retrieved is not None
        assert retrieved.content == "Test"

    def test_get_feedback_missing(self, feedback_processor):
        retrieved = feedback_processor.get_feedback("nonexistent")
        assert retrieved is None

    def test_get_all_feedback(self, feedback_processor):
        feedback_processor.submit_feedback("Feedback 1")
        feedback_processor.submit_feedback("Feedback 2")
        all_feedback = feedback_processor.get_all_feedback()
        assert len(all_feedback) == 2

    def test_get_all_feedback_filtered(self, feedback_processor):
        feedback_processor.submit_feedback("Feedback 1")
        processed_feedback = feedback_processor.get_all_feedback(FeedbackStatus.PROCESSED)
        assert len(processed_feedback) == 0

    def test_get_analytics(self, feedback_processor):
        feedback_processor.submit_feedback("Positive", feedback_type=FeedbackType.POSITIVE)
        feedback_processor.submit_feedback("Negative", feedback_type=FeedbackType.NEGATIVE)
        analytics = feedback_processor.get_analytics()
        assert analytics["total_feedback"] == 2
        assert analytics["positive_count"] == 1
        assert analytics["negative_count"] == 1


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 30 security boundaries."""

    def test_no_model_manager_in_feedback(self):
        import evora.brain.intelligence.feedback_processor as fb_mod
        source = Path(fb_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.feedback_processor as fb_mod
        source = Path(fb_mod.__file__).read_text(encoding="utf-8")
        import_section = False
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_section = True
            elif import_section and stripped and not stripped.startswith("#"):
                break
            if import_section:
                for forbidden in ["openai", "anthropic", "ollama", "requests", "aiohttp", "httpx", "urllib", "socket"]:
                    assert forbidden not in stripped.lower(), f"Found forbidden dependency: {forbidden}"


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 30 works offline."""

    def test_feedback_processor_works_offline(self, feedback_processor):
        feedback = feedback_processor.submit_feedback("offline test")
        assert feedback is not None

    def test_analytics_offline(self, feedback_processor):
        feedback_processor.submit_feedback("test")
        analytics = feedback_processor.get_analytics()
        assert isinstance(analytics, dict)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 30 architecture readiness."""

    def test_native_feedback_processor_exists(self):
        from evora.brain.intelligence.feedback_processor import NativeFeedbackProcessor
        assert NativeFeedbackProcessor is not None

    def test_feedback_exists(self):
        from evora.brain.intelligence.feedback_processor import Feedback
        assert Feedback is not None

    def test_feedback_analytics_exists(self):
        from evora.brain.intelligence.feedback_processor import FeedbackAnalytics
        assert FeedbackAnalytics is not None

    def test_feedback_type_enum_exists(self):
        from evora.brain.intelligence.feedback_processor import FeedbackType
        assert FeedbackType.POSITIVE is not None
        assert FeedbackType.NEGATIVE is not None

    def test_feedback_status_enum_exists(self):
        from evora.brain.intelligence.feedback_processor import FeedbackStatus
        assert FeedbackStatus.PENDING is not None
        assert FeedbackStatus.INTEGRATED is not None

    def test_feedback_reuses_training_pipeline(self, feedback_with_training):
        assert feedback_with_training.training_pipeline is not None
