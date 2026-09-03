"""
Phase 18 — Native Goal Decomposition tests.

Verifies:
1. Subtask has correct structure
2. DecomposedGoal has correct structure
3. DependencyType enum exists
4. SubtaskStatus enum exists
5. NativeGoalDecomposer initializes
6. NativeGoalDecomposer decomposes with runtime
7. NativeGoalDecomposer decomposes without runtime (heuristic)
8. NativeGoalDecomposer returns correct execution order
9. NativeGoalDecomposer marks subtask completed
10. NativeGoalDecomposer marks subtask failed
11. NativeGoalDecomposer gets ready subtasks
12. NativeGoalDecomposer gets next subtask
13. NativeGoalDecomposer stores decompositions
14. NativeGoalDecomposer returns metrics
15. Heuristic test decomposition works
16. Heuristic refactor decomposition works
17. Heuristic default decomposition works
18. Decomposition integrates with IntelligenceRuntime
19. No ModelManager dependency
20. No external dependencies
21. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.goal_decomposition import (
    DecomposedGoal,
    DependencyType,
    NativeGoalDecomposer,
    Subtask,
    SubtaskStatus,
)
from evora.logger import Logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def decomposer():
    return NativeGoalDecomposer(logger=Logger("evora-test-p18", "info", None))


@pytest.fixture
def decomposer_with_runtime():
    runtime = MagicMock()
    plan = MagicMock()
    step = MagicMock()
    step.to_dict.return_value = {"name": "Step 1", "action_type": "analyze_code", "action_args": {}}
    plan.steps = [step]
    plan.confidence = 0.8
    plan.requires_approval = False
    runtime.plan = MagicMock(return_value=plan)
    return NativeGoalDecomposer(
        intelligence_runtime=runtime,
        logger=Logger("evora-test-p18-rt", "info", None),
    )


# ---------------------------------------------------------------------------
# TestSubtask
# ---------------------------------------------------------------------------

class TestSubtask:
    """Test Subtask."""

    def test_default_subtask(self):
        subtask = Subtask()
        assert subtask.subtask_id != ""
        assert subtask.status == SubtaskStatus.PENDING

    def test_subtask_to_dict(self):
        subtask = Subtask(
            description="Run tests",
            action_type="run_tests",
            priority=1,
        )
        data = subtask.to_dict()
        assert data["description"] == "Run tests"
        assert data["action_type"] == "run_tests"
        assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# TestDecomposedGoal
# ---------------------------------------------------------------------------

class TestDecomposedGoal:
    """Test DecomposedGoal."""

    def test_default_goal(self):
        goal = DecomposedGoal()
        assert goal.goal_id != ""
        assert goal.overall_status == SubtaskStatus.PENDING

    def test_goal_to_dict(self):
        goal = DecomposedGoal(original_goal="Run tests")
        data = goal.to_dict()
        assert data["original_goal"] == "Run tests"
        assert "subtasks" in data


# ---------------------------------------------------------------------------
# TestDependencyTypeEnum
# ---------------------------------------------------------------------------

class TestDependencyTypeEnum:
    """Test DependencyType enum."""

    def test_dependency_types_exist(self):
        assert DependencyType.SEQUENTIAL is not None
        assert DependencyType.PARALLEL is not None
        assert DependencyType.CONDITIONAL is not None

    def test_dependency_type_values(self):
        assert DependencyType.SEQUENTIAL.value == "sequential"
        assert DependencyType.PARALLEL.value == "parallel"
        assert DependencyType.CONDITIONAL.value == "conditional"


# ---------------------------------------------------------------------------
# TestSubtaskStatusEnum
# ---------------------------------------------------------------------------

class TestSubtaskStatusEnum:
    """Test SubtaskStatus enum."""

    def test_status_values(self):
        assert SubtaskStatus.PENDING.value == "pending"
        assert SubtaskStatus.READY.value == "ready"
        assert SubtaskStatus.COMPLETED.value == "completed"
        assert SubtaskStatus.FAILED.value == "failed"


# ---------------------------------------------------------------------------
# TestNativeGoalDecomposer
# ---------------------------------------------------------------------------

class TestNativeGoalDecomposer:
    """Test NativeGoalDecomposer."""

    def test_decomposer_initializes(self, decomposer):
        assert decomposer is not None

    def test_decompose_returns_goal(self, decomposer):
        result = decomposer.decompose("Run tests")
        assert isinstance(result, DecomposedGoal)
        assert result.original_goal == "Run tests"

    def test_decompose_creates_subtasks(self, decomposer):
        result = decomposer.decompose("Run tests")
        assert len(result.subtasks) > 0

    def test_decompose_stores_result(self, decomposer):
        result = decomposer.decompose("Run tests")
        stored = decomposer.get_decomposition(result.goal_id)
        assert stored is not None
        assert stored.goal_id == result.goal_id

    def test_decompose_with_runtime(self, decomposer_with_runtime):
        result = decomposer_with_runtime.decompose("Analyze code")
        assert isinstance(result, DecomposedGoal)
        assert len(result.subtasks) > 0


# ---------------------------------------------------------------------------
# TestExecutionOrder
# ---------------------------------------------------------------------------

class TestExecutionOrder:
    """Test execution order computation."""

    def test_execution_order_sequential(self, decomposer):
        goal = decomposer.decompose("Refactor code")
        batches = decomposer.get_execution_order(goal)
        assert len(batches) > 0

    def test_execution_order_respects_dependencies(self, decomposer):
        goal = decomposer.decompose("Refactor code")
        batches = decomposer.get_execution_order(goal)
        completed_ids: set[str] = set()
        for batch in batches:
            for subtask in batch:
                assert all(d in completed_ids for d in subtask.dependencies)
            for subtask in batch:
                completed_ids.add(subtask.subtask_id)

    def test_get_next_subtask(self, decomposer):
        goal = decomposer.decompose("Run tests")
        next_task = decomposer.get_next_subtask(goal)
        assert next_task is not None
        assert next_task.status == SubtaskStatus.PENDING

    def test_get_ready_subtasks(self, decomposer):
        goal = decomposer.decompose("Run tests")
        ready = decomposer.get_ready_subtasks(goal)
        assert len(ready) > 0


# ---------------------------------------------------------------------------
# TestSubtaskStateTransitions
# ---------------------------------------------------------------------------

class TestSubtaskStateTransitions:
    """Test subtask state transitions."""

    def test_mark_completed(self, decomposer):
        goal = decomposer.decompose("Run tests")
        subtask = goal.subtasks[0]
        result = decomposer.mark_subtask_completed(goal, subtask.subtask_id, {"output": "ok"})
        assert result is True
        assert subtask.status == SubtaskStatus.COMPLETED
        assert subtask.result == {"output": "ok"}

    def test_mark_failed(self, decomposer):
        goal = decomposer.decompose("Run tests")
        subtask = goal.subtasks[0]
        result = decomposer.mark_subtask_failed(goal, subtask.subtask_id, "Error")
        assert result is True
        assert subtask.status == SubtaskStatus.FAILED
        assert subtask.error == "Error"

    def test_overall_status_completed(self, decomposer):
        goal = decomposer.decompose("Run tests")
        for subtask in goal.subtasks:
            decomposer.mark_subtask_completed(goal, subtask.subtask_id)
        assert goal.overall_status == SubtaskStatus.COMPLETED

    def test_overall_status_failed(self, decomposer):
        goal = decomposer.decompose("Run tests")
        if goal.subtasks:
            decomposer.mark_subtask_failed(goal, goal.subtasks[0].subtask_id)
        assert goal.overall_status == SubtaskStatus.FAILED


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test decomposition metrics."""

    def test_get_metrics(self, decomposer):
        goal = decomposer.decompose("Run tests")
        metrics = decomposer.get_metrics(goal)
        assert "total_subtasks" in metrics
        assert "completed" in metrics
        assert "progress" in metrics
        assert metrics["total_subtasks"] == len(goal.subtasks)

    def test_metrics_progress(self, decomposer):
        goal = decomposer.decompose("Run tests")
        if len(goal.subtasks) >= 2:
            decomposer.mark_subtask_completed(goal, goal.subtasks[0].subtask_id)
            metrics = decomposer.get_metrics(goal)
            assert metrics["completed"] == 1
            assert metrics["progress"] > 0.0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 18 security boundaries."""

    def test_no_model_manager_in_decomposition(self):
        import evora.brain.intelligence.goal_decomposition as dec_mod
        source = Path(dec_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.goal_decomposition as dec_mod
        source = Path(dec_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 18 works offline."""

    def test_decompose_offline(self, decomposer):
        result = decomposer.decompose("offline goal")
        assert isinstance(result, DecomposedGoal)
        assert len(result.subtasks) > 0

    def test_execution_order_offline(self, decomposer):
        goal = decomposer.decompose("offline goal")
        batches = decomposer.get_execution_order(goal)
        assert isinstance(batches, list)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 18 architecture readiness."""

    def test_native_goal_decomposer_exists(self):
        from evora.brain.intelligence.goal_decomposition import NativeGoalDecomposer
        assert NativeGoalDecomposer is not None

    def test_subtask_exists(self):
        from evora.brain.intelligence.goal_decomposition import Subtask
        assert Subtask is not None

    def test_decomposed_goal_exists(self):
        from evora.brain.intelligence.goal_decomposition import DecomposedGoal
        assert DecomposedGoal is not None

    def test_dependency_type_enum_exists(self):
        from evora.brain.intelligence.goal_decomposition import DependencyType
        assert DependencyType.SEQUENTIAL is not None
        assert DependencyType.PARALLEL is not None

    def test_subtask_status_enum_exists(self):
        from evora.brain.intelligence.goal_decomposition import SubtaskStatus
        assert SubtaskStatus.PENDING is not None
        assert SubtaskStatus.COMPLETED is not None

    def test_decomposer_reuses_intelligence_runtime(self, decomposer_with_runtime):
        assert decomposer_with_runtime.intelligence_runtime is not None

    def test_decomposer_produces_executable_plan(self, decomposer):
        goal = decomposer.decompose("Build a feature")
        batches = decomposer.get_execution_order(goal)
        assert len(batches) > 0
        all_subtasks = [s for batch in batches for s in batch]
        assert len(all_subtasks) == len(goal.subtasks)
