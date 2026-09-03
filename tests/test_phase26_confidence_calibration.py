"""
Phase 26 — Native Confidence Calibration tests.

Verifies:
1. ConfidenceRecord has correct structure
2. CalibrationMetrics has correct structure
3. NativeConfidenceCalibration initializes
4. NativeConfidenceCalibration records predictions
5. NativeConfidenceCalibration calibrates confidence
6. NativeConfidenceCalibration returns calibration report
7. NativeConfidenceCalibration gets records
8. NativeConfidenceCalibration clears records
9. Calibration detects overconfidence
10. Calibration detects underconfidence
11. Calibration computes calibration error
12. Calibration tracks by task type
13. Calibration handles empty records
14. No ModelManager dependency
15. No external dependencies
16. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.confidence_calibration import (
    CalibrationMetrics,
    ConfidenceRecord,
    NativeConfidenceCalibration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calibration():
    return NativeConfidenceCalibration(logger=MagicMock())


@pytest.fixture
def calibration_with_history():
    cal = NativeConfidenceCalibration(logger=MagicMock())
    for i in range(10):
        cal.record_prediction(0.5, i % 2 == 0, task_type="test")
    return cal


# ---------------------------------------------------------------------------
# TestConfidenceRecord
# ---------------------------------------------------------------------------

class TestConfidenceRecord:
    """Test ConfidenceRecord."""

    def test_default_record(self):
        record = ConfidenceRecord()
        assert record.record_id != ""
        assert record.predicted_confidence == 0.5

    def test_record_to_dict(self):
        record = ConfidenceRecord(predicted_confidence=0.8, actual_outcome=True)
        data = record.to_dict()
        assert data["predicted_confidence"] == 0.8
        assert data["actual_outcome"] is True


# ---------------------------------------------------------------------------
# TestCalibrationMetrics
# ---------------------------------------------------------------------------

class TestCalibrationMetrics:
    """Test CalibrationMetrics."""

    def test_default_metrics(self):
        metrics = CalibrationMetrics()
        assert metrics.total_predictions == 0
        data = metrics.to_dict()
        assert data["accuracy"] == 0.0

    def test_metrics_to_dict(self):
        metrics = CalibrationMetrics(total_predictions=10, correct_predictions=7)
        data = metrics.to_dict()
        assert data["accuracy"] == 0.7


# ---------------------------------------------------------------------------
# TestNativeConfidenceCalibration
# ---------------------------------------------------------------------------

class TestNativeConfidenceCalibration:
    """Test NativeConfidenceCalibration."""

    def test_calibration_initializes(self, calibration):
        assert calibration is not None

    def test_record_prediction(self, calibration):
        record = calibration.record_prediction(0.8, True)
        assert record.record_id != ""
        assert record.predicted_confidence == 0.8

    def test_record_with_task_type(self, calibration):
        record = calibration.record_prediction(0.7, False, task_type="coding")
        assert record.task_type == "coding"

    def test_calibrate_empty(self, calibration):
        result = calibration.calibrate(0.5)
        assert result == 0.5

    def test_calibrate_with_history(self, calibration_with_history):
        result = calibration_with_history.calibrate(0.5, task_type="test")
        assert 0.0 <= result <= 1.0

    def test_get_calibration_report(self, calibration):
        calibration.record_prediction(0.8, True)
        report = calibration.get_calibration_report()
        assert "total_predictions" in report
        assert report["total_predictions"] == 1

    def test_get_records(self, calibration):
        calibration.record_prediction(0.8, True, task_type="test")
        records = calibration.get_records()
        assert len(records) == 1

    def test_get_records_filtered(self, calibration):
        calibration.record_prediction(0.8, True, task_type="test")
        calibration.record_prediction(0.9, False, task_type="coding")
        records = calibration.get_records("test")
        assert len(records) == 1
        assert records[0].task_type == "test"

    def test_clear_records(self, calibration):
        calibration.record_prediction(0.8, True)
        calibration.clear_records()
        records = calibration.get_records()
        assert len(records) == 0


# ---------------------------------------------------------------------------
# TestCalibrationBehavior
# ---------------------------------------------------------------------------

class TestCalibrationBehavior:
    """Test calibration behavior."""

    def test_overconfidence_detection(self, calibration):
        for _ in range(10):
            calibration.record_prediction(0.9, False)
        report = calibration.get_calibration_report()
        assert report["overconfidence_rate"] > 0.0

    def test_underconfidence_detection(self, calibration):
        for _ in range(10):
            calibration.record_prediction(0.1, True)
        report = calibration.get_calibration_report()
        assert report["underconfidence_rate"] > 0.0

    def test_calibration_error_computed(self, calibration):
        for _ in range(10):
            calibration.record_prediction(0.5, True)
        report = calibration.get_calibration_report()
        assert "calibration_error" in report

    def test_accuracy_tracking(self, calibration):
        for i in range(10):
            calibration.record_prediction(0.5, i % 2 == 0)
        report = calibration.get_calibration_report()
        assert 0.0 <= report["accuracy"] <= 1.0

    def test_bins_tracking(self, calibration):
        calibration.record_prediction(0.1, True)
        calibration.record_prediction(0.5, True)
        calibration.record_prediction(0.9, True)
        report = calibration.get_calibration_report()
        assert "bins" in report
        assert len(report["bins"]) > 0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 26 security boundaries."""

    def test_no_model_manager_in_calibration(self):
        import evora.brain.intelligence.confidence_calibration as cal_mod
        source = Path(cal_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.confidence_calibration as cal_mod
        source = Path(cal_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 26 works offline."""

    def test_calibration_works_offline(self, calibration):
        record = calibration.record_prediction(0.8, True)
        assert record is not None

    def test_calibrate_offline(self, calibration):
        calibration.record_prediction(0.5, True)
        result = calibration.calibrate(0.5)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 26 architecture readiness."""

    def test_native_confidence_calibration_exists(self):
        from evora.brain.intelligence.confidence_calibration import NativeConfidenceCalibration
        assert NativeConfidenceCalibration is not None

    def test_confidence_record_exists(self):
        from evora.brain.intelligence.confidence_calibration import ConfidenceRecord
        assert ConfidenceRecord is not None

    def test_calibration_metrics_exists(self):
        from evora.brain.intelligence.confidence_calibration import CalibrationMetrics
        assert CalibrationMetrics is not None
