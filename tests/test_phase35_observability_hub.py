"""
Phase 35 — Native Observability Hub tests.

Verifies:
1. Metric has correct structure
2. Alert has correct structure
3. DashboardData has correct structure
4. NativeObservabilityHub initializes
5. NativeObservabilityHub records metrics
6. NativeObservabilityHub sets thresholds
7. NativeObservabilityHub generates alerts
8. NativeObservabilityHub gets metrics
9. NativeObservabilityHub gets alerts
10. NativeObservabilityHub gets dashboard data
11. NativeObservabilityHub clears metrics
12. NativeObservabilityHub clears alerts
13. No ModelManager dependency
14. No external dependencies
15. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.observability_hub import (
    Alert,
    DashboardData,
    Metric,
    NativeObservabilityHub,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def observability_hub():
    return NativeObservabilityHub(logger=MagicMock())


# ---------------------------------------------------------------------------
# TestMetric
# ---------------------------------------------------------------------------

class TestMetric:
    """Test Metric."""

    def test_default_metric(self):
        metric = Metric()
        assert metric.metric_id != ""
        assert metric.value == 0.0

    def test_metric_to_dict(self):
        metric = Metric(name="cpu_usage", value=75.5, tags={"host": "server1"})
        data = metric.to_dict()
        assert data["name"] == "cpu_usage"
        assert data["value"] == 75.5


# ---------------------------------------------------------------------------
# TestAlert
# ---------------------------------------------------------------------------

class TestAlert:
    """Test Alert."""

    def test_default_alert(self):
        alert = Alert()
        assert alert.alert_id != ""
        assert alert.severity == "medium"

    def test_alert_to_dict(self):
        alert = Alert(message="High CPU", metric_name="cpu", threshold=90.0, current_value=95.0)
        data = alert.to_dict()
        assert data["message"] == "High CPU"
        assert data["current_value"] == 95.0


# ---------------------------------------------------------------------------
# TestNativeObservabilityHub
# ---------------------------------------------------------------------------

class TestNativeObservabilityHub:
    """Test NativeObservabilityHub."""

    def test_observability_hub_initializes(self, observability_hub):
        assert observability_hub is not None

    def test_record_metric(self, observability_hub):
        metric = observability_hub.record_metric("cpu_usage", 75.0)
        assert metric.metric_id != ""
        assert metric.value == 75.0

    def test_record_metric_with_tags(self, observability_hub):
        metric = observability_hub.record_metric("cpu_usage", 75.0, tags={"host": "server1"})
        assert metric.tags["host"] == "server1"

    def test_set_threshold(self, observability_hub):
        observability_hub.set_threshold("cpu_usage", 90.0)
        assert "cpu_usage" in observability_hub._thresholds

    def test_threshold_generates_alert(self, observability_hub):
        observability_hub.set_threshold("cpu_usage", 90.0)
        observability_hub.record_metric("cpu_usage", 95.0)
        alerts = observability_hub.get_alerts()
        assert len(alerts) > 0

    def test_no_alert_below_threshold(self, observability_hub):
        observability_hub.set_threshold("cpu_usage", 90.0)
        observability_hub.record_metric("cpu_usage", 85.0)
        alerts = observability_hub.get_alerts()
        assert len(alerts) == 0

    def test_get_metrics(self, observability_hub):
        observability_hub.record_metric("cpu", 75.0)
        observability_hub.record_metric("memory", 60.0)
        metrics = observability_hub.get_metrics()
        assert len(metrics) == 2

    def test_get_metrics_filtered(self, observability_hub):
        observability_hub.record_metric("cpu", 75.0)
        observability_hub.record_metric("memory", 60.0)
        cpu_metrics = observability_hub.get_metrics("cpu")
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].name == "cpu"

    def test_get_alerts(self, observability_hub):
        observability_hub.set_threshold("cpu", 90.0)
        observability_hub.record_metric("cpu", 95.0)
        alerts = observability_hub.get_alerts()
        assert len(alerts) > 0

    def test_get_alerts_filtered(self, observability_hub):
        observability_hub.set_threshold("cpu", 90.0)
        observability_hub.record_metric("cpu", 95.0)
        high_alerts = observability_hub.get_alerts("high")
        assert all(a.severity == "high" for a in high_alerts)

    def test_get_dashboard_data(self, observability_hub):
        observability_hub.record_metric("cpu", 75.0)
        observability_hub.set_threshold("cpu", 90.0)
        observability_hub.record_metric("cpu", 95.0)
        dashboard = observability_hub.get_dashboard_data()
        assert isinstance(dashboard, DashboardData)
        assert "total_metrics" in dashboard.summary

    def test_clear_metrics(self, observability_hub):
        observability_hub.record_metric("cpu", 75.0)
        observability_hub.clear_metrics()
        metrics = observability_hub.get_metrics()
        assert len(metrics) == 0

    def test_clear_alerts(self, observability_hub):
        observability_hub.set_threshold("cpu", 90.0)
        observability_hub.record_metric("cpu", 95.0)
        observability_hub.clear_alerts()
        alerts = observability_hub.get_alerts()
        assert len(alerts) == 0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 35 security boundaries."""

    def test_no_model_manager_in_observability(self):
        import evora.brain.intelligence.observability_hub as obs_mod
        source = Path(obs_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.observability_hub as obs_mod
        source = Path(obs_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 35 works offline."""

    def test_observability_hub_works_offline(self, observability_hub):
        metric = observability_hub.record_metric("offline_metric", 42.0)
        assert metric is not None

    def test_dashboard_offline(self, observability_hub):
        observability_hub.record_metric("test", 10.0)
        dashboard = observability_hub.get_dashboard_data()
        assert isinstance(dashboard, DashboardData)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 35 architecture readiness."""

    def test_native_observability_hub_exists(self):
        from evora.brain.intelligence.observability_hub import NativeObservabilityHub
        assert NativeObservabilityHub is not None

    def test_metric_exists(self):
        from evora.brain.intelligence.observability_hub import Metric
        assert Metric is not None

    def test_alert_exists(self):
        from evora.brain.intelligence.observability_hub import Alert
        assert Alert is not None

    def test_dashboard_data_exists(self):
        from evora.brain.intelligence.observability_hub import DashboardData
        assert DashboardData is not None
