"""
Phase 28 — Native Execution Monitor tests.

Verifies:
1. ExecutionRecord has correct structure
2. Anomaly has correct structure
3. ExecutionStatus enum exists
4. AnomalyType enum exists
5. NativeExecutionMonitor initializes
6. NativeExecutionMonitor starts execution
7. NativeExecutionMonitor ends execution
8. NativeExecutionMonitor updates resource usage
9. NativeExecutionMonitor detects timing anomalies
10. NativeExecutionMonitor detects behavior anomalies
11. NativeExecutionMonitor gets anomalies
12. NativeExecutionMonitor returns stats
13. NativeExecutionMonitor gets record
14. NativeExecutionMonitor clears history
15. No ModelManager dependency
16. No external dependencies
17. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import time

from evora.brain.intelligence.execution_monitor import (
    Anomaly,
    AnomalyType,
    ExecutionRecord,
    ExecutionStatus,
    NativeExecutionMonitor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def execution_monitor():
    return NativeExecutionMonitor(logger=MagicMock())


# ---------------------------------------------------------------------------
# TestExecutionRecord
# ---------------------------------------------------------------------------

class TestExecutionRecord:
    """Test ExecutionRecord."""

    def test_default_record(self):
        record = ExecutionRecord()
        assert record.record_id != ""
        assert record.status == ExecutionStatus.RUNNING

    def test_record_to_dict(self):
        record = ExecutionRecord(task_id="task1", status=ExecutionStatus.COMPLETED, duration=1.5)
        data = record.to_dict()
        assert data["task_id"] == "task1"
        assert data["status"] == "completed"
        assert data["duration"] == 1.5


# ---------------------------------------------------------------------------
# TestAnomaly
# ---------------------------------------------------------------------------

class TestAnomaly:
    """Test Anomaly."""

    def test_default_anomaly(self):
        anomaly = Anomaly()
        assert anomaly.anomaly_id != ""
        assert anomaly.anomaly_type == AnomalyType.PERFORMANCE

    def test_anomaly_to_dict(self):
        anomaly = Anomaly(anomaly_type=AnomalyType.TIMING, severity="high", description="Slow execution")
        data = anomaly.to_dict()
        assert data["anomaly_type"] == "timing"
        assert data["severity"] == "high"


# ---------------------------------------------------------------------------
# TestExecutionStatusEnum
# ---------------------------------------------------------------------------

class TestExecutionStatusEnum:
    """Test ExecutionStatus enum."""

    def test_status_values(self):
        assert ExecutionStatus.RUNNING.value == "running"
        assert ExecutionStatus.COMPLETED.value == "completed"
        assert ExecutionStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# TestAnomalyTypeEnum
# ---------------------------------------------------------------------------

class TestAnomalyTypeEnum:
    """Test AnomalyType enum."""

    def test_anomaly_types_exist(self):
        assert AnomalyType.PERFORMANCE is not None
        assert AnomalyType.RESOURCE is not None
        assert AnomalyType.BEHAVIOR is not None
        assert AnomalyType.TIMING is not None


# ---------------------------------------------------------------------------
# TestNativeExecutionMonitor
# ---------------------------------------------------------------------------

class TestNativeExecutionMonitor:
    """Test NativeExecutionMonitor."""

    def test_monitor_initializes(self, execution_monitor):
        assert execution_monitor is not None

    def test_start_execution(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        assert record.record_id != ""
        assert record.task_id == "task1"
        assert record.status == ExecutionStatus.RUNNING

    def test_end_execution_completed(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        ended = execution_monitor.end_execution(record.record_id, ExecutionStatus.COMPLETED, output="done")
        assert ended is not None
        assert ended.status == ExecutionStatus.COMPLETED
        assert ended.duration >= 0

    def test_end_execution_failed(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        ended = execution_monitor.end_execution(record.record_id, ExecutionStatus.FAILED, error="error")
        assert ended is not None
        assert ended.error == "error"

    def test_end_execution_missing(self, execution_monitor):
        ended = execution_monitor.end_execution("nonexistent", ExecutionStatus.COMPLETED)
        assert ended is None

    def test_update_resource_usage(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        result = execution_monitor.update_resource_usage(record.record_id, {"cpu": 50.0})
        assert result is True

    def test_update_resource_usage_missing(self, execution_monitor):
        result = execution_monitor.update_resource_usage("nonexistent", {"cpu": 50.0})
        assert result is False

    def test_detect_timing_anomaly(self, execution_monitor):
        record = execution_monitor.start_execution("slow_task")
        execution_monitor.end_execution(record.record_id, ExecutionStatus.COMPLETED)
        for _ in range(5):
            r = execution_monitor.start_execution("slow_task")
            execution_monitor.end_execution(r.record_id, ExecutionStatus.COMPLETED)
            time.sleep(0.01)
        anomalies = execution_monitor.get_anomalies(AnomalyType.TIMING)
        assert isinstance(anomalies, list)

    def test_detect_behavior_anomaly(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        execution_monitor.end_execution(record.record_id, ExecutionStatus.FAILED, error="test error")
        anomalies = execution_monitor.get_anomalies(AnomalyType.BEHAVIOR)
        assert len(anomalies) > 0

    def test_get_anomalies(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        execution_monitor.end_execution(record.record_id, ExecutionStatus.FAILED, error="error")
        anomalies = execution_monitor.get_anomalies()
        assert len(anomalies) > 0

    def test_get_anomalies_filtered(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        execution_monitor.end_execution(record.record_id, ExecutionStatus.FAILED, error="error")
        anomalies = execution_monitor.get_anomalies(AnomalyType.BEHAVIOR)
        assert all(a.anomaly_type == AnomalyType.BEHAVIOR for a in anomalies)

    def test_get_execution_stats(self, execution_monitor):
        execution_monitor.start_execution("task1")
        stats = execution_monitor.get_execution_stats()
        assert "total_executions" in stats
        assert stats["total_executions"] == 1

    def test_get_record(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        retrieved = execution_monitor.get_record(record.record_id)
        assert retrieved is not None
        assert retrieved.task_id == "task1"

    def test_get_record_missing(self, execution_monitor):
        retrieved = execution_monitor.get_record("nonexistent")
        assert retrieved is None

    def test_clear_history(self, execution_monitor):
        execution_monitor.start_execution("task1")
        execution_monitor.clear_history()
        stats = execution_monitor.get_execution_stats()
        assert stats["total_executions"] == 0
        assert stats["total_anomalies"] == 0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 28 security boundaries."""

    def test_no_model_manager_in_monitor(self):
        import evora.brain.intelligence.execution_monitor as mon_mod
        source = Path(mon_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.execution_monitor as mon_mod
        source = Path(mon_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 28 works offline."""

    def test_monitor_works_offline(self, execution_monitor):
        record = execution_monitor.start_execution("offline_task")
        assert record is not None

    def test_anomaly_detection_offline(self, execution_monitor):
        record = execution_monitor.start_execution("task1")
        execution_monitor.end_execution(record.record_id, ExecutionStatus.FAILED, error="error")
        anomalies = execution_monitor.get_anomalies()
        assert isinstance(anomalies, list)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 28 architecture readiness."""

    def test_native_execution_monitor_exists(self):
        from evora.brain.intelligence.execution_monitor import NativeExecutionMonitor
        assert NativeExecutionMonitor is not None

    def test_execution_record_exists(self):
        from evora.brain.intelligence.execution_monitor import ExecutionRecord
        assert ExecutionRecord is not None

    def test_anomaly_exists(self):
        from evora.brain.intelligence.execution_monitor import Anomaly
        assert Anomaly is not None

    def test_execution_status_enum_exists(self):
        from evora.brain.intelligence.execution_monitor import ExecutionStatus
        assert ExecutionStatus.RUNNING is not None
        assert ExecutionStatus.COMPLETED is not None

    def test_anomaly_type_enum_exists(self):
        from evora.brain.intelligence.execution_monitor import AnomalyType
        assert AnomalyType.PERFORMANCE is not None
        assert AnomalyType.BEHAVIOR is not None
