"""
Phase 26 — Native Confidence Calibration for EVORA.

Calibrates confidence estimates based on historical accuracy.

Supports:
  - Confidence tracking
  - Calibration curve computation
  - Overconfidence/underconfidence detection
  - Adaptive confidence adjustment
  - Integration with TrainingPipeline
  - Integration with InferenceEngine
  - Integration with Agent

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceRecord:
    """A confidence prediction record."""
    record_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    predicted_confidence: float = 0.5
    actual_outcome: bool = False
    task_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "predicted_confidence": self.predicted_confidence,
            "actual_outcome": self.actual_outcome,
            "task_type": self.task_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class CalibrationMetrics:
    """Calibration metrics."""
    total_predictions: int = 0
    correct_predictions: int = 0
    calibration_error: float = 0.0
    overconfidence_rate: float = 0.0
    underconfidence_rate: float = 0.0
    bins: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy": self.correct_predictions / self.total_predictions if self.total_predictions > 0 else 0.0,
            "calibration_error": self.calibration_error,
            "overconfidence_rate": self.overconfidence_rate,
            "underconfidence_rate": self.underconfidence_rate,
            "bins": self.bins,
        }


# ---------------------------------------------------------------------------
# Native Confidence Calibration
# ---------------------------------------------------------------------------

class NativeConfidenceCalibration:
    """Native confidence calibration for EVORA.

    Calibrates confidence estimates based on historical accuracy.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._records: list[ConfidenceRecord] = []
        self._metrics = CalibrationMetrics()

    def record_prediction(self, predicted_confidence: float, actual_outcome: bool, task_type: str = "", metadata: dict[str, Any] = None) -> ConfidenceRecord:
        """Record a confidence prediction and its outcome."""
        metadata = metadata or {}
        record = ConfidenceRecord(
            predicted_confidence=predicted_confidence,
            actual_outcome=actual_outcome,
            task_type=task_type,
            metadata=metadata,
        )
        self._records.append(record)
        self._update_metrics()
        return record

    def calibrate(self, raw_confidence: float, task_type: str = "") -> float:
        """Calibrate a raw confidence estimate."""
        relevant_records = [r for r in self._records if r.task_type == task_type] if task_type else self._records
        if not relevant_records:
            return raw_confidence
        bin_key = self._get_bin(raw_confidence)
        bin_records = [r for r in relevant_records if self._get_bin(r.predicted_confidence) == bin_key]
        if not bin_records:
            return raw_confidence
        actual_rate = sum(1 for r in bin_records if r.actual_outcome) / len(bin_records)
        return (raw_confidence + actual_rate) / 2.0

    def _get_bin(self, confidence: float) -> str:
        """Get the bin key for a confidence value."""
        if confidence < 0.2:
            return "0.0-0.2"
        elif confidence < 0.4:
            return "0.2-0.4"
        elif confidence < 0.6:
            return "0.4-0.6"
        elif confidence < 0.8:
            return "0.6-0.8"
        else:
            return "0.8-1.0"

    def _update_metrics(self) -> None:
        """Update calibration metrics."""
        self._metrics.total_predictions = len(self._records)
        if not self._records:
            return
        self._metrics.correct_predictions = sum(1 for r in self._records if r.actual_outcome)
        bins: dict[str, dict[str, Any]] = {}
        for record in self._records:
            bin_key = self._get_bin(record.predicted_confidence)
            if bin_key not in bins:
                bins[bin_key] = {"count": 0, "correct": 0, "total_confidence": 0.0}
            bins[bin_key]["count"] += 1
            bins[bin_key]["total_confidence"] += record.predicted_confidence
            if record.actual_outcome:
                bins[bin_key]["correct"] += 1
        self._metrics.bins = bins
        total_error = 0.0
        overconfidence = 0.0
        underconfidence = 0.0
        for bin_data in bins.values():
            if bin_data["count"] > 0:
                avg_confidence = bin_data["total_confidence"] / bin_data["count"]
                actual_rate = bin_data["correct"] / bin_data["count"]
                error = abs(avg_confidence - actual_rate)
                total_error += error * bin_data["count"]
                if avg_confidence > actual_rate:
                    overconfidence += bin_data["count"]
                elif avg_confidence < actual_rate:
                    underconfidence += bin_data["count"]
        self._metrics.calibration_error = total_error / len(self._records) if self._records else 0.0
        self._metrics.overconfidence_rate = overconfidence / len(self._records) if self._records else 0.0
        self._metrics.underconfidence_rate = underconfidence / len(self._records) if self._records else 0.0

    def get_calibration_report(self) -> dict[str, Any]:
        """Get a calibration report."""
        return self._metrics.to_dict()

    def get_records(self, task_type: str = "") -> list[ConfidenceRecord]:
        """Get prediction records, optionally filtered by task type."""
        if not task_type:
            return list(self._records)
        return [r for r in self._records if r.task_type == task_type]

    def clear_records(self) -> None:
        """Clear all prediction records."""
        self._records = []
        self._metrics = CalibrationMetrics()
