"""
Tests for EVORA Phase 2 task state and decision abstractions.
"""

import json
import pytest

from evora.task import (
    TaskState,
    Observation,
    Decision,
    TestResult,
    ActionResult,
)
from evora.evaluation import EvaluationOutcome


class TestObservation:

    def test_create_observation(self):
        obs = Observation(
            type="file_created",
            source="write_file",
            data={"path": "/test/file.py"},
            success=True,
        )
        assert obs.type == "file_created"
        assert obs.success is True
        assert obs.data["path"] == "/test/file.py"

    def test_to_dict(self):
        obs = Observation(type="command_success", source="test", success=True)
        d = obs.to_dict()
        assert d["type"] == "command_success"
        assert d["success"] is True

    def test_from_dict(self):
        data = {
            "type": "test_passed",
            "source": "run_tests",
            "data": {"output": "all good"},
            "timestamp": 1234567890.0,
            "success": True,
        }
        obs = Observation.from_dict(data)
        assert obs.type == "test_passed"
        assert obs.data["output"] == "all good"


class TestDecision:

    def test_create_decision(self):
        dec = Decision(
            action="execute_tool",
            reason="Need to create a file",
            tool="write_file",
            arguments={"path": "test.py", "content": "print('hi')"},
            confidence=0.9,
        )
        assert dec.action == "execute_tool"
        assert dec.tool == "write_file"
        assert dec.requires_approval is False

    def test_to_dict(self):
        dec = Decision(action="plan", reason="test")
        d = dec.to_dict()
        assert d["action"] == "plan"
        assert d["reason"] == "test"

    def test_from_dict(self):
        data = {
            "action": "report",
            "reason": "done",
            "tool": None,
            "arguments": {},
            "expected_outcome": "",
            "risk_level": "safe",
            "requires_approval": False,
            "confidence": 1.0,
        }
        dec = Decision.from_dict(data)
        assert dec.action == "report"
        assert dec.confidence == 1.0


class TestActionResult:

    def test_success_result(self):
        result = ActionResult(
            success=True,
            tool="write_file",
            output="File written",
            data={"bytes": 42},
        )
        assert result.success is True
        assert result.data["bytes"] == 42

    def test_failure_result(self):
        result = ActionResult(
            success=False,
            tool="read_file",
            error="File not found",
        )
        assert result.success is False
        assert "File not found" in result.error


class TestTestResult:

    def test_passed(self):
        tr = TestResult(command="pytest", passed=True, output="2 passed")
        assert tr.passed is True
        assert "2 passed" in tr.output

    def test_failed(self):
        tr = TestResult(command="pytest", passed=False, error="1 failed")
        assert tr.passed is False

    def test_to_dict(self):
        tr = TestResult(command="pytest", passed=True)
        d = tr.to_dict()
        assert d["command"] == "pytest"


class TestTaskState:

    def test_create_with_defaults(self):
        state = TaskState(request="Create a calculator")
        assert state.request == "Create a calculator"
        assert state.goal == "Create a calculator"
        assert state.status == "idle"
        assert state.attempts == 0
        assert state.max_attempts == 10
        assert state.observations == []
        assert state.actions == []
        assert state.errors == []
        assert state.is_complete is False
        assert state.is_failed is False
        assert state.is_cancelled is False

    def test_task_id_is_unique(self):
        s1 = TaskState(request="test1")
        s2 = TaskState(request="test2")
        assert s1.task_id != s2.task_id

    def test_add_observation(self):
        state = TaskState(request="test")
        obs = Observation(type="file_created", source="write", success=True)
        state.add_observation(obs)
        assert len(state.observations) == 1

    def test_add_decision(self):
        state = TaskState(request="test")
        dec = Decision(action="plan", reason="initial plan")
        state.add_decision(dec)
        assert len(state.decisions) == 1

    def test_add_error(self):
        state = TaskState(request="test")
        state.add_error("Something went wrong")
        assert len(state.errors) == 1
        assert "Something went wrong" in state.errors[0]

    def test_mark_complete(self):
        state = TaskState(request="test")
        state.mark_complete("All done")
        assert state.is_complete is True
        assert state.status == "completed"
        assert state.final_result == "All done"

    def test_mark_failed(self):
        state = TaskState(request="test")
        state.mark_failed("Failed to complete")
        assert state.is_failed is True
        assert state.status == "failed"
        assert state.final_result == "Failed to complete"

    def test_mark_cancelled(self):
        state = TaskState(request="test")
        state.mark_cancelled("User cancelled")
        assert state.is_cancelled is True
        assert state.status == "cancelled"

    def test_exceeded_retry_limit(self):
        state = TaskState(request="test", max_attempts=3)
        state.attempts = 3
        assert state.exceeded_retry_limit() is True

        state.attempts = 2
        assert state.exceeded_retry_limit() is False

    def test_increment_attempt(self):
        state = TaskState(request="test")
        count = state.increment_attempt()
        assert count == 1
        assert state.attempts == 1

    def test_to_dict_and_from_dict(self):
        state = TaskState(
            request="Create a calculator",
            goal="Build calculator",
            workspace="/tmp",
            project_context={"framework": "Python"},
        )
        state.add_observation(Observation(type="file_created", source="write", success=True))
        state.add_decision(Decision(action="plan", reason="initial"))
        state.add_error("test error")
        state.mark_complete("Done")

        d = state.to_dict()
        restored = TaskState.from_dict(d)

        assert restored.request == "Create a calculator"
        assert restored.goal == "Build calculator"
        assert restored.workspace == "/tmp"
        assert len(restored.observations) == 1
        assert len(restored.decisions) == 1
        assert len(restored.errors) == 1
        assert restored.is_complete is True

    def test_snapshot(self):
        state = TaskState(request="test task")
        state.attempts = 3
        snap = state.snapshot()
        assert snap["request" if "request" in snap else "task_id"] in snap or "task_id" in snap
        assert snap["attempts"] == 3
        assert snap["status"] == "idle"

    def test_add_action(self):
        state = TaskState(request="test")
        result = ActionResult(success=True, tool="write_file", output="written")
        state.add_action("write_file", {"path": "test.py"}, result)
        assert len(state.actions) == 1
        assert state.actions[0]["tool"] == "write_file"

    def test_completed_steps_set_serialized(self):
        state = TaskState(request="test")
        state.completed_steps = {"step-1", "step-2"}
        d = state.to_dict()
        assert isinstance(d["completed_steps"], list)

    def test_remaining_steps(self):
        state = TaskState(request="test")
        state.remaining_steps = [{"id": "s1", "name": "step 1"}]
        assert len(state.remaining_steps) == 1
        assert state.remaining_steps[0]["name"] == "step 1"
