"""
Phase 28 — Native Execution Monitor for EVORA.

Monitors task execution and detects anomalies.

Supports:
  - Execution tracking
  - Anomaly detection
  - Performance monitoring
  - Resource usage tracking
  - Integration with NativeAgent
  - Integration with NativeTaskScheduler
  - Integration with NativeErrorRecovery

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

class ExecutionStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AnomalyType(str, Enum):
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    BEHAVIOR = "behavior"
    TIMING = "timing"


@dataclass
class ExecutionRecord:
    """An execution record."""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = ""
    status: ExecutionStatus = ExecutionStatus.RUNNING
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: str = ""
    duration: float = 0.0
    resource_usage: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "resource_usage": self.resource_usage,
            "output": self.output,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class Anomaly:
    """A detected anomaly."""
    anomaly_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    anomaly_type: AnomalyType = AnomalyType.PERFORMANCE
    severity: str = "medium"
    description: str = ""
    record_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "description": self.description,
            "record_id": self.record_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Native Execution Monitor
# ---------------------------------------------------------------------------

class NativeExecutionMonitor:
    """Native execution monitor for EVORA.

    Monitors task execution and detects anomalies.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._records: dict[str, ExecutionRecord] = {}
        self._anomalies: list[Anomaly] = []
        self._baselines: dict[str, dict[str, float]] = {}

    def start_execution(self, task_id: str, metadata: dict[str, Any] = None) -> ExecutionRecord:
        """Start monitoring an execution."""
        record = ExecutionRecord(task_id=task_id, metadata=metadata or {})
        self._records[record.record_id] = record
        return record

    def end_execution(self, record_id: str, status: ExecutionStatus, output: str = "", error: str = "") -> Optional[ExecutionRecord]:
        """End monitoring an execution."""
        record = self._records.get(record_id)
        if record is None:
            return None
        record.status = status
        record.end_time = datetime.now().isoformat()
        start = datetime.fromisoformat(record.start_time)
        end = datetime.fromisoformat(record.end_time)
        record.duration = (end - start).total_seconds()
        record.output = output
        record.error = error
        self._detect_anomalies(record)
        return record

    def update_resource_usage(self, record_id: str, usage: dict[str, Any]) -> bool:
        """Update resource usage for an execution."""
        record = self._records.get(record_id)
        if record is None:
            return False
        record.resource_usage.update(usage)
        return True

    def _detect_anomalies(self, record: ExecutionRecord) -> None:
        """Detect anomalies in an execution record."""
        task_id = record.task_id
        if task_id not in self._baselines:
            self._baselines[task_id] = {"avg_duration": 0.0, "avg_output_length": 0.0}
        baseline = self._baselines[task_id]
        if baseline["avg_duration"] > 0 and record.duration > baseline["avg_duration"] * 3:
            self._anomalies.append(Anomaly(
                anomaly_type=AnomalyType.TIMING,
                severity="high",
                description=f"Execution took {record.duration:.1f}s, expected ~{baseline['avg_duration']:.1f}s",
                record_id=record.record_id,
            ))
        if record.status == ExecutionStatus.FAILED:
            self._anomalies.append(Anomaly(
                anomaly_type=AnomalyType.BEHAVIOR,
                severity="high",
                description=f"Execution failed: {record.error}",
                record_id=record.record_id,
            ))
        n = len([r for r in self._records.values() if r.task_id == task_id])
        baseline["avg_duration"] = (baseline["avg_duration"] * (n - 1) + record.duration) / n

    def get_anomalies(self, anomaly_type: AnomalyType = None) -> list[Anomaly]:
        """Get detected anomalies."""
        if anomaly_type is None:
            return list(self._anomalies)
        return [a for a in self._anomalies if a.anomaly_type == anomaly_type]

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        total = len(self._records)
        by_status: dict[str, int] = {}
        for record in self._records.values():
            by_status[record.status.value] = by_status.get(record.status.value, 0) + 1
        return {
            "total_executions": total,
            "by_status": by_status,
            "total_anomalies": len(self._anomalies),
        }

    def get_record(self, record_id: str) -> Optional[ExecutionRecord]:
        """Get an execution record by ID."""
        return self._records.get(record_id)

    def clear_history(self) -> None:
        """Clear execution history."""
        self._records = {}
        self._anomalies = []
        self._baselines = {}
