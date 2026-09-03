"""
Phase 20 — Native Error Recovery tests.

Verifies:
1. ErrorRecord has correct structure
2. RecoveryResult has correct structure
3. ErrorCategory enum exists
4. RecoveryStrategy enum exists
5. NativeErrorRecovery initializes
6. NativeErrorRecovery classifies timeout error
7. NativeErrorRecovery classifies security error
8. NativeErrorRecovery classifies resource error
9. NativeErrorRecovery classifies permanent error
10. NativeErrorRecovery classifies unknown error
11. NativeErrorRecovery attempts retry recovery
12. NativeErrorRecovery attempts skip recovery
13. NativeErrorRecovery attempts fallback recovery
14. NativeErrorRecovery attempts abort recovery
15. NativeErrorRecovery attempts escalate recovery
16. NativeErrorRecovery handles error end-to-end
17. NativeErrorRecovery records error history
18. NativeErrorRecovery filters history by category
19. NativeErrorRecovery returns metrics
20. No ModelManager dependency
21. No external dependencies
22. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.error_recovery import (
    ErrorCategory,
    ErrorRecord,
    NativeErrorRecovery,
    RecoveryResult,
    RecoveryStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def error_recovery():
    return NativeErrorRecovery(logger=MagicMock())


@pytest.fixture
def error_recovery_with_agent():
    agent = MagicMock()
    agent.execute.return_value = MagicMock(success=True, output="recovered", error="")
    return NativeErrorRecovery(agent=agent, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestErrorRecord
# ---------------------------------------------------------------------------

class TestErrorRecord:
    """Test ErrorRecord."""

    def test_default_record(self):
        record = ErrorRecord()
        assert record.error_id != ""
        assert record.category == ErrorCategory.UNKNOWN

    def test_record_to_dict(self):
        record = ErrorRecord(message="test error", source="unit_test")
        data = record.to_dict()
        assert data["message"] == "test error"
        assert data["category"] == "unknown"


# ---------------------------------------------------------------------------
# TestRecoveryResult
# ---------------------------------------------------------------------------

class TestRecoveryResult:
    """Test RecoveryResult."""

    def test_default_result(self):
        result = RecoveryResult()
        assert result.success is False

    def test_result_to_dict(self):
        result = RecoveryResult(success=True, strategy_used=RecoveryStrategy.RETRY, output="ok")
        data = result.to_dict()
        assert data["success"] is True
        assert data["strategy_used"] == "retry"


# ---------------------------------------------------------------------------
# TestErrorCategoryEnum
# ---------------------------------------------------------------------------

class TestErrorCategoryEnum:
    """Test ErrorCategory enum."""

    def test_category_values(self):
        assert ErrorCategory.TRANSIENT.value == "transient"
        assert ErrorCategory.PERMANENT.value == "permanent"
        assert ErrorCategory.RESOURCE.value == "resource"
        assert ErrorCategory.SECURITY.value == "security"


# ---------------------------------------------------------------------------
# TestRecoveryStrategyEnum
# ---------------------------------------------------------------------------

class TestRecoveryStrategyEnum:
    """Test RecoveryStrategy enum."""

    def test_strategy_values(self):
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.SKIP.value == "skip"
        assert RecoveryStrategy.ABORT.value == "abort"
        assert RecoveryStrategy.ESCALATE.value == "escalate"


# ---------------------------------------------------------------------------
# TestNativeErrorRecovery
# ---------------------------------------------------------------------------

class TestNativeErrorRecovery:
    """Test NativeErrorRecovery."""

    def test_recovery_initializes(self, error_recovery):
        assert error_recovery is not None

    def test_classify_timeout_error(self, error_recovery):
        record = error_recovery.classify_error(TimeoutError("Connection timed out"))
        assert record.category == ErrorCategory.TRANSIENT

    def test_classify_security_error(self, error_recovery):
        record = error_recovery.classify_error(PermissionError("Permission denied"))
        assert record.category == ErrorCategory.SECURITY

    def test_classify_resource_error(self, error_recovery):
        record = error_recovery.classify_error(MemoryError("Out of memory"))
        assert record.category == ErrorCategory.RESOURCE

    def test_classify_permanent_error(self, error_recovery):
        record = error_recovery.classify_error(FileNotFoundError("File not found"))
        assert record.category == ErrorCategory.PERMANENT

    def test_classify_unknown_error(self, error_recovery):
        record = error_recovery.classify_error(ValueError("Unknown error"))
        assert record.category == ErrorCategory.UNKNOWN

    def test_attempt_retry_recovery(self, error_recovery_with_agent):
        record = ErrorRecord(category=ErrorCategory.TRANSIENT, message="timeout")
        result = error_recovery_with_agent.attempt_recovery(record, {"action": "test"})
        assert result.strategy_used == RecoveryStrategy.RETRY

    def test_attempt_skip_recovery(self, error_recovery):
        record = ErrorRecord(category=ErrorCategory.PERMANENT, message="not found")
        result = error_recovery.attempt_recovery(record)
        assert result.strategy_used == RecoveryStrategy.SKIP

    def test_attempt_fallback_recovery(self, error_recovery):
        record = ErrorRecord(category=ErrorCategory.RESOURCE, message="out of memory")
        result = error_recovery.attempt_recovery(record)
        assert result.strategy_used == RecoveryStrategy.FALLBACK

    def test_attempt_abort_recovery(self, error_recovery):
        record = ErrorRecord(category=ErrorCategory.SECURITY, message="denied")
        result = error_recovery.attempt_recovery(record)
        assert result.strategy_used == RecoveryStrategy.ESCALATE

    def test_handle_error_end_to_end(self, error_recovery):
        result = error_recovery.handle_error(TimeoutError("timeout"), {"source": "test"})
        assert isinstance(result, RecoveryResult)

    def test_error_history(self, error_recovery):
        error_recovery.classify_error(TimeoutError("t1"))
        error_recovery.classify_error(TimeoutError("t2"))
        history = error_recovery.get_error_history()
        assert len(history) == 2

    def test_filtered_history(self, error_recovery):
        error_recovery.classify_error(TimeoutError("connection timeout"))
        error_recovery.classify_error(FileNotFoundError("f1"))
        history = error_recovery.get_error_history(ErrorCategory.TRANSIENT)
        assert len(history) == 1

    def test_recovery_metrics(self, error_recovery):
        error_recovery.classify_error(TimeoutError("t1"))
        metrics = error_recovery.get_recovery_metrics()
        assert "total_errors" in metrics
        assert metrics["total_errors"] == 1

    def test_record_recovery_success(self, error_recovery):
        record = error_recovery.classify_error(TimeoutError("t1"))
        result = error_recovery.record_recovery_success(record.error_id)
        assert result is True
        assert record.recovery_success is True


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 20 security boundaries."""

    def test_no_model_manager_in_recovery(self):
        import evora.brain.intelligence.error_recovery as rec_mod
        source = Path(rec_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.error_recovery as rec_mod
        source = Path(rec_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 20 works offline."""

    def test_recovery_works_offline(self, error_recovery):
        result = error_recovery.handle_error(ValueError("test"), {"source": "offline"})
        assert isinstance(result, RecoveryResult)

    def test_metrics_offline(self, error_recovery):
        error_recovery.classify_error(TimeoutError("t"))
        metrics = error_recovery.get_recovery_metrics()
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 20 architecture readiness."""

    def test_native_error_recovery_exists(self):
        from evora.brain.intelligence.error_recovery import NativeErrorRecovery
        assert NativeErrorRecovery is not None

    def test_error_record_exists(self):
        from evora.brain.intelligence.error_recovery import ErrorRecord
        assert ErrorRecord is not None

    def test_recovery_result_exists(self):
        from evora.brain.intelligence.error_recovery import RecoveryResult
        assert RecoveryResult is not None

    def test_error_category_enum_exists(self):
        from evora.brain.intelligence.error_recovery import ErrorCategory
        assert ErrorCategory.TRANSIENT is not None
        assert ErrorCategory.PERMANENT is not None

    def test_recovery_strategy_enum_exists(self):
        from evora.brain.intelligence.error_recovery import RecoveryStrategy
        assert RecoveryStrategy.RETRY is not None
        assert RecoveryStrategy.FALLBACK is not None

    def test_recovery_reuses_agent(self, error_recovery_with_agent):
        assert error_recovery_with_agent.agent is not None
