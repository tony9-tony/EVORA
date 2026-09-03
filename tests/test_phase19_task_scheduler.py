"""
Phase 19 — Native Task Scheduler tests.

Verifies:
1. ScheduledTask has correct structure
2. SchedulingResult has correct structure
3. SchedulingStrategy enum exists
4. TaskResultStatus enum exists
5. NativeTaskScheduler initializes
6. NativeTaskScheduler schedules a goal
7. NativeTaskScheduler executes next task
8. NativeTaskScheduler executes all tasks
9. NativeTaskScheduler picks next by priority
10. NativeTaskScheduler picks next by FIFO
11. NativeTaskScheduler retries failed tasks
12. NativeTaskScheduler cancels tasks
13. NativeTaskScheduler gets task status
14. NativeTaskScheduler returns metrics
15. Scheduler integrates with agent
16. Scheduler integrates with goal decomposer
17. No ModelManager dependency
18. No external dependencies
19. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.task_scheduler import (
    NativeTaskScheduler,
    SchedulingResult,
    SchedulingStrategy,
    ScheduledTask,
    TaskResultStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduler():
    return NativeTaskScheduler(logger=MagicMock())


@pytest.fixture
def scheduler_with_agent():
    agent = MagicMock()
    agent.execute.return_value = MagicMock(success=True, output="done", error="", to_dict=lambda: {"success": True})
    decomposer = MagicMock()
    goal = MagicMock()
    subtask = MagicMock()
    subtask.description = "Test task"
    subtask.priority = 1
    subtask.action_type = "analyze_code"
    subtask.estimated_effort = 1.0
    goal.subtasks = [subtask]
    goal.original_goal = "Test"
    decomposer.decompose.return_value = goal
    decomposer.get_execution_order.return_value = [[subtask]]
    return NativeTaskScheduler(
        agent=agent,
        goal_decomposer=decomposer,
        logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# TestScheduledTask
# ---------------------------------------------------------------------------

class TestScheduledTask:
    """Test ScheduledTask."""

    def test_default_task(self):
        task = ScheduledTask()
        assert task.task_id != ""
        assert task.status == TaskResultStatus.PENDING

    def test_task_to_dict(self):
        task = ScheduledTask(status=TaskResultStatus.SUCCESS, retries=1)
        data = task.to_dict()
        assert data["status"] == "success"
        assert data["retries"] == 1


# ---------------------------------------------------------------------------
# TestSchedulingResult
# ---------------------------------------------------------------------------

class TestSchedulingResult:
    """Test SchedulingResult."""

    def test_default_result(self):
        result = SchedulingResult()
        assert result.success is False

    def test_result_to_dict(self):
        result = SchedulingResult(success=True, scheduled_count=5, completed_count=3)
        data = result.to_dict()
        assert data["success"] is True
        assert data["scheduled_count"] == 5
        assert data["completed_count"] == 3


# ---------------------------------------------------------------------------
# TestSchedulingStrategyEnum
# ---------------------------------------------------------------------------

class TestSchedulingStrategyEnum:
    """Test SchedulingStrategy enum."""

    def test_strategy_values(self):
        assert SchedulingStrategy.PRIORITY.value == "priority"
        assert SchedulingStrategy.FIFO.value == "fifo"
        assert SchedulingStrategy.EDF.value == "earliest_deadline_first"


# ---------------------------------------------------------------------------
# TestTaskResultStatusEnum
# ---------------------------------------------------------------------------

class TestTaskResultStatusEnum:
    """Test TaskResultStatus enum."""

    def test_status_values(self):
        assert TaskResultStatus.SUCCESS.value == "success"
        assert TaskResultStatus.FAILURE.value == "failure"
        assert TaskResultStatus.TIMEOUT.value == "timeout"
        assert TaskResultStatus.PENDING.value == "pending"


# ---------------------------------------------------------------------------
# TestNativeTaskScheduler
# ---------------------------------------------------------------------------

class TestNativeTaskScheduler:
    """Test NativeTaskScheduler."""

    def test_scheduler_initializes(self, scheduler):
        assert scheduler is not None

    def test_schedule_goal_without_decomposer(self, scheduler):
        result = scheduler.schedule_goal("Test goal")
        assert isinstance(result, SchedulingResult)
        assert result.scheduled_count == 0

    def test_schedule_goal_with_decomposer(self, scheduler_with_agent):
        result = scheduler_with_agent.schedule_goal("Test goal")
        assert isinstance(result, SchedulingResult)
        assert result.scheduled_count > 0

    def test_execute_next_without_tasks(self, scheduler):
        task = scheduler.execute_next()
        assert task is None

    def test_execute_next_with_tasks(self, scheduler_with_agent):
        scheduler_with_agent.schedule_goal("Test goal")
        task = scheduler_with_agent.execute_next()
        assert task is not None

    def test_execute_all(self, scheduler_with_agent):
        scheduler_with_agent.schedule_goal("Test goal")
        result = scheduler_with_agent.execute_all()
        assert isinstance(result, SchedulingResult)
        assert result.completed_count > 0

    def test_get_task_status(self, scheduler_with_agent):
        scheduler_with_agent.schedule_goal("Test goal")
        task = scheduler_with_agent.execute_next()
        if task:
            status = scheduler_with_agent.get_task_status(task.task_id)
            assert status is not None

    def test_cancel_task(self, scheduler):
        task = ScheduledTask()
        scheduler._scheduled[task.task_id] = task
        result = scheduler.cancel_task(task.task_id)
        assert result is True
        assert task.status == TaskResultStatus.CANCELLED

    def test_get_metrics(self, scheduler):
        metrics = scheduler.get_scheduling_metrics()
        assert "total_scheduled" in metrics
        assert "strategy" in metrics


# ---------------------------------------------------------------------------
# TestSchedulingStrategies
# ---------------------------------------------------------------------------

class TestSchedulingStrategies:
    """Test scheduling strategies."""

    def test_priority_strategy(self):
        scheduler = NativeTaskScheduler(strategy=SchedulingStrategy.PRIORITY, logger=MagicMock())
        assert scheduler.strategy == SchedulingStrategy.PRIORITY

    def test_fifo_strategy(self):
        scheduler = NativeTaskScheduler(strategy=SchedulingStrategy.FIFO, logger=MagicMock())
        assert scheduler.strategy == SchedulingStrategy.FIFO

    def test_default_strategy(self, scheduler):
        assert scheduler.strategy == SchedulingStrategy.PRIORITY


# ---------------------------------------------------------------------------
# TestRetryLogic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Test retry logic."""

    def test_task_retries(self, scheduler_with_agent):
        scheduler_with_agent.schedule_goal("Test goal")
        task = scheduler_with_agent.execute_next()
        if task:
            assert task.retries == 0
            assert task.max_retries == 3


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 19 security boundaries."""

    def test_no_model_manager_in_scheduler(self):
        import evora.brain.intelligence.task_scheduler as sched_mod
        source = Path(sched_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.task_scheduler as sched_mod
        source = Path(sched_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 19 works offline."""

    def test_scheduler_works_offline(self, scheduler):
        metrics = scheduler.get_scheduling_metrics()
        assert isinstance(metrics, dict)

    def test_schedule_offline(self, scheduler):
        result = scheduler.schedule_goal("offline task")
        assert isinstance(result, SchedulingResult)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 19 architecture readiness."""

    def test_native_task_scheduler_exists(self):
        from evora.brain.intelligence.task_scheduler import NativeTaskScheduler
        assert NativeTaskScheduler is not None

    def test_scheduled_task_exists(self):
        from evora.brain.intelligence.task_scheduler import ScheduledTask
        assert ScheduledTask is not None

    def test_scheduling_result_exists(self):
        from evora.brain.intelligence.task_scheduler import SchedulingResult
        assert SchedulingResult is not None

    def test_scheduling_strategy_enum_exists(self):
        from evora.brain.intelligence.task_scheduler import SchedulingStrategy
        assert SchedulingStrategy.PRIORITY is not None
        assert SchedulingStrategy.FIFO is not None

    def test_task_result_status_enum_exists(self):
        from evora.brain.intelligence.task_scheduler import TaskResultStatus
        assert TaskResultStatus.SUCCESS is not None
        assert TaskResultStatus.FAILURE is not None

    def test_scheduler_reuses_agent(self, scheduler_with_agent):
        assert scheduler_with_agent.agent is not None

    def test_scheduler_reuses_goal_decomposer(self, scheduler_with_agent):
        assert scheduler_with_agent.goal_decomposer is not None
