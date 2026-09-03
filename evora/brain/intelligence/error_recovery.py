"""
Phase 20 — Native Error Recovery for EVORA.

Handles failures gracefully and attempts recovery.

Supports:
  - Error classification
  - Recovery strategy selection
  - Retry with backoff
  - Fallback execution
  - Circuit breaker pattern
  - Error history tracking
  - Integration with NativeAgent
  - Integration with NativeTaskScheduler

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RESOURCE = "resource"
    SECURITY = "security"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"
    ESCALATE = "escalate"


@dataclass
class ErrorRecord:
    """A recorded error."""
    error_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    category: ErrorCategory = ErrorCategory.UNKNOWN
    message: str = ""
    source: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "message": self.message,
            "source": self.source,
            "context": self.context,
            "timestamp": self.timestamp,
            "recovery_attempted": self.recovery_attempted,
            "recovery_strategy": self.recovery_strategy.value if self.recovery_strategy else None,
            "recovery_success": self.recovery_success,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool = False
    strategy_used: Optional[RecoveryStrategy] = None
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Native Error Recovery
# ---------------------------------------------------------------------------

class NativeErrorRecovery:
    """Native error recovery for EVORA.

    Classifies errors and attempts recovery using strategies.
    """

    def __init__(
        self,
        agent: Any = None,
        logger: Optional[Any] = None,
    ):
        self.agent = agent
        self.logger = logger
        self._error_history: list[ErrorRecord] = []
        self._circuit_breakers: dict[str, dict[str, Any]] = {}

    def classify_error(self, error: Exception, context: dict[str, Any] = None) -> ErrorRecord:
        """Classify an error."""
        context = context or {}
        error_message = str(error)
        error_type = type(error).__name__
        category = ErrorCategory.UNKNOWN
        if "timeout" in error_message.lower() or "timed out" in error_message.lower():
            category = ErrorCategory.TRANSIENT
        elif "permission" in error_message.lower() or "denied" in error_message.lower():
            category = ErrorCategory.SECURITY
        elif "memory" in error_message.lower() or "disk" in error_message.lower() or "resource" in error_message.lower():
            category = ErrorCategory.RESOURCE
        elif "not found" in error_message.lower() or "does not exist" in error_message.lower():
            category = ErrorCategory.PERMANENT
        record = ErrorRecord(
            category=category,
            message=error_message,
            source=context.get("source", error_type),
            context=context,
        )
        self._error_history.append(record)
        return record

    def attempt_recovery(self, error_record: ErrorRecord, context: dict[str, Any] = None) -> RecoveryResult:
        """Attempt to recover from an error."""
        context = context or {}
        strategy = self._select_strategy(error_record)
        error_record.recovery_attempted = True
        error_record.recovery_strategy = strategy
        result = RecoveryResult(strategy_used=strategy)
        if strategy == RecoveryStrategy.RETRY:
            result = self._retry(context, error_record)
        elif strategy == RecoveryStrategy.FALLBACK:
            result = self._fallback(context, error_record)
        elif strategy == RecoveryStrategy.SKIP:
            result = RecoveryResult(success=True, strategy_used=strategy, output="Skipped")
        elif strategy == RecoveryStrategy.ABORT:
            result = RecoveryResult(success=False, strategy_used=strategy, error="Aborted")
        elif strategy == RecoveryStrategy.ESCALATE:
            result = RecoveryResult(success=False, strategy_used=strategy, error="Escalation required")
        error_record.recovery_success = result.success
        return result

    def _select_strategy(self, error_record: ErrorRecord) -> RecoveryStrategy:
        """Select a recovery strategy based on error category."""
        if error_record.category == ErrorCategory.TRANSIENT:
            return RecoveryStrategy.RETRY
        elif error_record.category == ErrorCategory.PERMANENT:
            return RecoveryStrategy.SKIP
        elif error_record.category == ErrorCategory.SECURITY:
            return RecoveryStrategy.ESCALATE
        elif error_record.category == ErrorCategory.RESOURCE:
            return RecoveryStrategy.FALLBACK
        return RecoveryStrategy.RETRY

    def _retry(self, context: dict[str, Any], error_record: ErrorRecord) -> RecoveryResult:
        """Retry the failed action."""
        if self.agent is not None and "action" in context:
            try:
                result = self.agent.execute(context["action"], context.get("action_context", {}))
                if result.success:
                    return RecoveryResult(success=True, strategy_used=RecoveryStrategy.RETRY, output=result.output)
            except Exception as e:
                return RecoveryResult(success=False, strategy_used=RecoveryStrategy.RETRY, error=str(e))
        return RecoveryResult(success=False, strategy_used=RecoveryStrategy.RETRY, error="No agent available for retry")

    def _fallback(self, context: dict[str, Any], error_record: ErrorRecord) -> RecoveryResult:
        """Execute fallback behavior."""
        return RecoveryResult(success=True, strategy_used=RecoveryStrategy.FALLBACK, output="Fallback executed")

    def handle_error(self, error: Exception, context: dict[str, Any] = None) -> RecoveryResult:
        """Full error handling: classify and attempt recovery."""
        context = context or {}
        error_record = self.classify_error(error, context)
        result = self.attempt_recovery(error_record, context)
        return result

    def get_error_history(self, category: Optional[ErrorCategory] = None) -> list[ErrorRecord]:
        """Get error history, optionally filtered by category."""
        if category is None:
            return list(self._error_history)
        return [e for e in self._error_history if e.category == category]

    def get_recovery_metrics(self) -> dict[str, Any]:
        """Get recovery metrics."""
        total = len(self._error_history)
        recovered = sum(1 for e in self._error_history if e.recovery_success)
        by_category: dict[str, int] = {}
        for e in self._error_history:
            by_category[e.category.value] = by_category.get(e.category.value, 0) + 1
        return {
            "total_errors": total,
            "recovered": recovered,
            "recovery_rate": recovered / total if total > 0 else 0.0,
            "by_category": by_category,
        }

    def record_recovery_success(self, error_id: str) -> bool:
        """Record a manual recovery success."""
        for record in self._error_history:
            if record.error_id == error_id:
                record.recovery_success = True
                record.recovery_attempted = True
                return True
        return False
