"""
Phase 27 — Native Tool Optimizer tests.

Verifies:
1. ToolPerformance has correct structure
2. ToolRecommendation has correct structure
3. NativeToolOptimizer initializes
4. NativeToolOptimizer recommends tool
5. NativeToolOptimizer records tool use
6. NativeToolOptimizer gets tool performance
7. NativeToolOptimizer gets all performance
8. NativeToolOptimizer returns optimization metrics
9. NativeToolOptimizer clears history
10. Tool optimizer integrates with ToolRegistry
11. No ModelManager dependency
12. No external dependencies
13. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.tool_optimizer import (
    NativeToolOptimizer,
    ToolPerformance,
    ToolRecommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool_optimizer():
    return NativeToolOptimizer(logger=MagicMock())


@pytest.fixture
def optimizer_with_registry():
    registry = MagicMock()
    return NativeToolOptimizer(tool_registry=registry, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestToolPerformance
# ---------------------------------------------------------------------------

class TestToolPerformance:
    """Test ToolPerformance."""

    def test_default_performance(self):
        perf = ToolPerformance()
        assert perf.tool_name == ""
        assert perf.total_uses == 0

    def test_performance_to_dict(self):
        perf = ToolPerformance(tool_name="read_file", success_rate=0.9)
        data = perf.to_dict()
        assert data["tool_name"] == "read_file"
        assert data["success_rate"] == 0.9


# ---------------------------------------------------------------------------
# TestToolRecommendation
# ---------------------------------------------------------------------------

class TestToolRecommendation:
    """Test ToolRecommendation."""

    def test_default_recommendation(self):
        rec = ToolRecommendation()
        assert rec.recommendation_id != ""
        assert rec.confidence == 0.0

    def test_recommendation_to_dict(self):
        rec = ToolRecommendation(tool_name="read_file", confidence=0.8, reasoning="High success rate")
        data = rec.to_dict()
        assert data["tool_name"] == "read_file"
        assert data["confidence"] == 0.8


# ---------------------------------------------------------------------------
# TestNativeToolOptimizer
# ---------------------------------------------------------------------------

class TestNativeToolOptimizer:
    """Test NativeToolOptimizer."""

    def test_optimizer_initializes(self, tool_optimizer):
        assert tool_optimizer is not None

    def test_recommend_tool_no_history(self, tool_optimizer):
        rec = tool_optimizer.recommend_tool("analyze code")
        assert isinstance(rec, ToolRecommendation)
        assert rec.tool_name != ""

    def test_recommend_tool_with_history(self, tool_optimizer):
        tool_optimizer.record_tool_use("read_file", True, 1.0)
        tool_optimizer.record_tool_use("edit_file", True, 0.5)
        rec = tool_optimizer.recommend_tool("read a file")
        assert isinstance(rec, ToolRecommendation)

    def test_record_tool_use(self, tool_optimizer):
        tool_optimizer.record_tool_use("read_file", True, 1.0)
        perf = tool_optimizer.get_tool_performance("read_file")
        assert perf is not None
        assert perf.total_uses == 1
        assert perf.success_rate == 1.0

    def test_record_multiple_uses(self, tool_optimizer):
        tool_optimizer.record_tool_use("read_file", True, 1.0)
        tool_optimizer.record_tool_use("read_file", True, 2.0)
        tool_optimizer.record_tool_use("read_file", False, 0.5)
        perf = tool_optimizer.get_tool_performance("read_file")
        assert perf.total_uses == 3
        assert perf.success_rate == 2 / 3

    def test_get_tool_performance_missing(self, tool_optimizer):
        perf = tool_optimizer.get_tool_performance("nonexistent")
        assert perf is None

    def test_get_all_performance(self, tool_optimizer):
        tool_optimizer.record_tool_use("tool1", True)
        tool_optimizer.record_tool_use("tool2", True)
        all_perf = tool_optimizer.get_all_performance()
        assert len(all_perf) == 2

    def test_get_optimization_metrics(self, tool_optimizer):
        tool_optimizer.record_tool_use("tool1", True)
        metrics = tool_optimizer.get_optimization_metrics()
        assert "total_tools_tracked" in metrics
        assert metrics["total_tools_tracked"] == 1

    def test_clear_history(self, tool_optimizer):
        tool_optimizer.record_tool_use("tool1", True)
        tool_optimizer.clear_history()
        all_perf = tool_optimizer.get_all_performance()
        assert len(all_perf) == 0

    def test_recommendation_has_alternatives(self, tool_optimizer):
        tool_optimizer.record_tool_use("tool1", True)
        tool_optimizer.record_tool_use("tool2", True)
        tool_optimizer.record_tool_use("tool3", True)
        rec = tool_optimizer.recommend_tool("test")
        assert isinstance(rec.alternatives, list)


# ---------------------------------------------------------------------------
# TestToolOptimization
# ---------------------------------------------------------------------------

class TestToolOptimization:
    """Test tool optimization behavior."""

    def test_success_rate_affects_recommendation(self, tool_optimizer):
        tool_optimizer.record_tool_use("good_tool", True, 1.0)
        tool_optimizer.record_tool_use("good_tool", True, 1.0)
        tool_optimizer.record_tool_use("bad_tool", False, 1.0)
        tool_optimizer.record_tool_use("bad_tool", False, 1.0)
        rec = tool_optimizer.recommend_tool("test", available_tools=["good_tool", "bad_tool"])
        assert rec.tool_name == "good_tool"

    def test_average_duration_affects_recommendation(self, tool_optimizer):
        tool_optimizer.record_tool_use("fast_tool", True, 0.1)
        tool_optimizer.record_tool_use("slow_tool", True, 10.0)
        rec = tool_optimizer.recommend_tool("test", available_tools=["fast_tool", "slow_tool"])
        assert rec.tool_name == "fast_tool"


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 27 security boundaries."""

    def test_no_model_manager_in_optimizer(self):
        import evora.brain.intelligence.tool_optimizer as opt_mod
        source = Path(opt_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.tool_optimizer as opt_mod
        source = Path(opt_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 27 works offline."""

    def test_optimizer_works_offline(self, tool_optimizer):
        rec = tool_optimizer.recommend_tool("test")
        assert isinstance(rec, ToolRecommendation)

    def test_record_use_offline(self, tool_optimizer):
        tool_optimizer.record_tool_use("tool", True)
        perf = tool_optimizer.get_tool_performance("tool")
        assert perf is not None


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 27 architecture readiness."""

    def test_native_tool_optimizer_exists(self):
        from evora.brain.intelligence.tool_optimizer import NativeToolOptimizer
        assert NativeToolOptimizer is not None

    def test_tool_performance_exists(self):
        from evora.brain.intelligence.tool_optimizer import ToolPerformance
        assert ToolPerformance is not None

    def test_tool_recommendation_exists(self):
        from evora.brain.intelligence.tool_optimizer import ToolRecommendation
        assert ToolRecommendation is not None

    def test_optimizer_reuses_tool_registry(self, optimizer_with_registry):
        assert optimizer_with_registry.tool_registry is not None
