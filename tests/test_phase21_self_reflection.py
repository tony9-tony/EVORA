"""
Phase 21 — Native Self-Reflection tests.

Verifies:
1. Reflection has correct structure
2. PerformanceMetrics has correct structure
3. ReflectionType enum exists
4. ReflectionSeverity enum exists
5. NativeSelfReflection initializes
6. NativeSelfReflection records task results
7. NativeSelfReflection generates performance reflections
8. NativeSelfReflection identifies weaknesses
9. NativeSelfReflection identifies strengths
10. NativeSelfReflection suggests improvements
11. NativeSelfReflection returns reflections
12. NativeSelfReflection filters reflections by type
13. NativeSelfReflection gets latest reflection
14. NativeSelfReflection returns metrics
15. NativeSelfReflection clears reflections
16. NativeSelfReflection adds improvement areas
17. Reflection integrates with IntelligenceRuntime
18. No ModelManager dependency
19. No external dependencies
20. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.self_reflection import (
    NativeSelfReflection,
    PerformanceMetrics,
    Reflection,
    ReflectionSeverity,
    ReflectionType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def self_reflection():
    return NativeSelfReflection(logger=MagicMock())


@pytest.fixture
def self_reflection_with_runtime():
    runtime = MagicMock()
    return NativeSelfReflection(
        intelligence_runtime=runtime,
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# TestReflection
# ---------------------------------------------------------------------------

class TestReflection:
    """Test Reflection."""

    def test_default_reflection(self):
        reflection = Reflection()
        assert reflection.reflection_id != ""
        assert reflection.reflection_type == ReflectionType.GENERAL

    def test_reflection_to_dict(self):
        reflection = Reflection(
            title="Test reflection",
            description="A test reflection",
            confidence=0.8,
        )
        data = reflection.to_dict()
        assert data["title"] == "Test reflection"
        assert data["confidence"] == 0.8


# ---------------------------------------------------------------------------
# TestPerformanceMetrics
# ---------------------------------------------------------------------------

class TestPerformanceMetrics:
    """Test PerformanceMetrics."""

    def test_default_metrics(self):
        metrics = PerformanceMetrics()
        assert metrics.total_tasks == 0
        data = metrics.to_dict()
        assert data["success_rate"] == 0.0

    def test_metrics_to_dict(self):
        metrics = PerformanceMetrics(total_tasks=10, successful_tasks=8)
        data = metrics.to_dict()
        assert data["total_tasks"] == 10
        assert data["success_rate"] == 0.8


# ---------------------------------------------------------------------------
# TestReflectionTypeEnum
# ---------------------------------------------------------------------------

class TestReflectionTypeEnum:
    """Test ReflectionType enum."""

    def test_reflection_types_exist(self):
        assert ReflectionType.PERFORMANCE is not None
        assert ReflectionType.WEAKNESS is not None
        assert ReflectionType.STRENGTH is not None
        assert ReflectionType.IMPROVEMENT is not None


# ---------------------------------------------------------------------------
# TestReflectionSeverityEnum
# ---------------------------------------------------------------------------

class TestReflectionSeverityEnum:
    """Test ReflectionSeverity enum."""

    def test_severity_values(self):
        assert ReflectionSeverity.LOW.value == "low"
        assert ReflectionSeverity.MEDIUM.value == "medium"
        assert ReflectionSeverity.HIGH.value == "high"
        assert ReflectionSeverity.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# TestNativeSelfReflection
# ---------------------------------------------------------------------------

class TestNativeSelfReflection:
    """Test NativeSelfReflection."""

    def test_self_reflection_initializes(self, self_reflection):
        assert self_reflection is not None

    def test_record_successful_task(self, self_reflection):
        self_reflection.record_task_result(success=True, duration=1.0)
        metrics = self_reflection.get_metrics()
        assert metrics["total_tasks"] == 1
        assert metrics["successful_tasks"] == 1

    def test_record_failed_task(self, self_reflection):
        self_reflection.record_task_result(success=False, error="timeout")
        metrics = self_reflection.get_metrics()
        assert metrics["failed_tasks"] == 1

    def test_record_multiple_tasks(self, self_reflection):
        self_reflection.record_task_result(success=True, duration=1.0)
        self_reflection.record_task_result(success=True, duration=2.0)
        self_reflection.record_task_result(success=False, error="error")
        metrics = self_reflection.get_metrics()
        assert metrics["total_tasks"] == 3
        assert metrics["success_rate"] == 2 / 3

    def test_reflect_generates_insights(self, self_reflection):
        self_reflection.record_task_result(success=False, error="timeout")
        reflections = self_reflection.reflect()
        assert isinstance(reflections, list)

    def test_get_reflections(self, self_reflection):
        self_reflection.reflect()
        reflections = self_reflection.get_reflections()
        assert isinstance(reflections, list)

    def test_get_reflections_filtered(self, self_reflection):
        self_reflection.reflect()
        perf_reflections = self_reflection.get_reflections(ReflectionType.PERFORMANCE)
        assert isinstance(perf_reflections, list)

    def test_get_latest_reflection(self, self_reflection):
        self_reflection.reflect()
        latest = self_reflection.get_latest_reflection()
        assert latest is None or isinstance(latest, Reflection)

    def test_add_improvement_area(self, self_reflection):
        self_reflection.add_improvement_area("testing")
        metrics = self_reflection.get_metrics()
        assert "testing" in metrics["improvement_areas"]

    def test_clear_reflections(self, self_reflection):
        self_reflection.reflect()
        self_reflection.clear_reflections()
        reflections = self_reflection.get_reflections()
        assert len(reflections) == 0

    def test_metrics_success_rate(self, self_reflection):
        for _ in range(10):
            self_reflection.record_task_result(success=True)
        metrics = self_reflection.get_metrics()
        assert metrics["success_rate"] == 1.0


# ---------------------------------------------------------------------------
# TestReflectionGeneration
# ---------------------------------------------------------------------------

class TestReflectionGeneration:
    """Test reflection generation."""

    def test_low_success_rate_generates_performance_reflection(self, self_reflection):
        for _ in range(10):
            self_reflection.record_task_result(success=False, error="fail")
        reflections = self_reflection.reflect()
        perf_reflections = [r for r in reflections if r.reflection_type == ReflectionType.PERFORMANCE]
        assert len(perf_reflections) > 0

    def test_high_success_rate_generates_strength_reflection(self, self_reflection):
        for _ in range(10):
            self_reflection.record_task_result(success=True)
        reflections = self_reflection.reflect()
        strength_reflections = [r for r in reflections if r.reflection_type == ReflectionType.STRENGTH]
        assert len(strength_reflections) > 0

    def test_common_failures_generate_weakness_reflections(self, self_reflection):
        for _ in range(5):
            self_reflection.record_task_result(success=False, error="timeout error")
        reflections = self_reflection.reflect()
        weakness_reflections = [r for r in reflections if r.reflection_type == ReflectionType.WEAKNESS]
        assert len(weakness_reflections) > 0

    def test_improvement_areas_generate_suggestions(self, self_reflection):
        self_reflection.add_improvement_area("code quality")
        reflections = self_reflection.reflect()
        improvement_reflections = [r for r in reflections if r.reflection_type == ReflectionType.IMPROVEMENT]
        assert len(improvement_reflections) > 0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 21 security boundaries."""

    def test_no_model_manager_in_reflection(self):
        import evora.brain.intelligence.self_reflection as ref_mod
        source = Path(ref_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.self_reflection as ref_mod
        source = Path(ref_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 21 works offline."""

    def test_reflection_works_offline(self, self_reflection):
        self_reflection.record_task_result(success=True)
        reflections = self_reflection.reflect()
        assert isinstance(reflections, list)

    def test_metrics_offline(self, self_reflection):
        self_reflection.record_task_result(success=True)
        metrics = self_reflection.get_metrics()
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 21 architecture readiness."""

    def test_native_self_reflection_exists(self):
        from evora.brain.intelligence.self_reflection import NativeSelfReflection
        assert NativeSelfReflection is not None

    def test_reflection_exists(self):
        from evora.brain.intelligence.self_reflection import Reflection
        assert Reflection is not None

    def test_performance_metrics_exists(self):
        from evora.brain.intelligence.self_reflection import PerformanceMetrics
        assert PerformanceMetrics is not None

    def test_reflection_type_enum_exists(self):
        from evora.brain.intelligence.self_reflection import ReflectionType
        assert ReflectionType.PERFORMANCE is not None
        assert ReflectionType.WEAKNESS is not None

    def test_reflection_severity_enum_exists(self):
        from evora.brain.intelligence.self_reflection import ReflectionSeverity
        assert ReflectionSeverity.LOW is not None
        assert ReflectionSeverity.HIGH is not None

    def test_reflection_reuses_runtime(self, self_reflection_with_runtime):
        assert self_reflection_with_runtime.intelligence_runtime is not None
