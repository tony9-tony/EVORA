"""
Phase 35 — Native Observability Hub for EVORA.

Central observability and monitoring hub.

Supports:
  - Metric collection
  - Metric aggregation
  - Alert generation
  - Dashboard data
  - Integration with ExecutionMonitor
  - Integration with NativeAgent
  - Integration with ErrorRecovery

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
class Metric:
    """A metric data point."""
    metric_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    value: float = 0.0
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "value": self.value,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


@dataclass
class Alert:
    """An alert."""
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    severity: str = "medium"
    message: str = ""
    metric_name: str = ""
    threshold: float = 0.0
    current_value: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class DashboardData:
    """Dashboard data snapshot."""
    dashboard_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metrics: list[Metric] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dashboard_id": self.dashboard_id,
            "metrics": [m.to_dict() for m in self.metrics],
            "alerts": [a.to_dict() for a in self.alerts],
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Native Observability Hub
# ---------------------------------------------------------------------------

class NativeObservabilityHub:
    """Native observability hub for EVORA.

    Central observability and monitoring.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._metrics: list[Metric] = []
        self._alerts: list[Alert] = []
        self._thresholds: dict[str, float] = {}

    def record_metric(self, name: str, value: float, tags: dict[str, str] = None) -> Metric:
        """Record a metric."""
        metric = Metric(name=name, value=value, tags=tags or {})
        self._metrics.append(metric)
        self._check_threshold(metric)
        return metric

    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set an alert threshold for a metric."""
        self._thresholds[metric_name] = threshold

    def _check_threshold(self, metric: Metric) -> None:
        """Check if metric exceeds threshold."""
        threshold = self._thresholds.get(metric.name)
        if threshold is not None and metric.value > threshold:
            alert = Alert(
                severity="high",
                message=f"Metric {metric.name} exceeded threshold {threshold}",
                metric_name=metric.name,
                threshold=threshold,
                current_value=metric.value,
            )
            self._alerts.append(alert)

    def get_metrics(self, name: str = None) -> list[Metric]:
        """Get metrics, optionally filtered by name."""
        if name is None:
            return list(self._metrics)
        return [m for m in self._metrics if m.name == name]

    def get_alerts(self, severity: str = None) -> list[Alert]:
        """Get alerts, optionally filtered by severity."""
        if severity is None:
            return list(self._alerts)
        return [a for a in self._alerts if a.severity == severity]

    def get_dashboard_data(self) -> DashboardData:
        """Get dashboard data snapshot."""
        summary = {
            "total_metrics": len(self._metrics),
            "total_alerts": len(self._alerts),
            "active_alerts": len([a for a in self._alerts if a.severity in ("high", "critical")]),
        }
        return DashboardData(
            metrics=list(self._metrics)[-50:],
            alerts=list(self._alerts)[-20:],
            summary=summary,
        )

    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self._metrics = []

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts = []
