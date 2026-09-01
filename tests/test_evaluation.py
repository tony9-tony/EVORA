"""
Tests for EVORA Phase 2 evaluator.
"""

import pytest
from unittest.mock import MagicMock

from evora.evaluation import Evaluator, EvaluationOutcome, EvaluationResult
from evora.task import TaskState, Observation, TestResult


def make_state(**kwargs) -> TaskState:
    defaults = {
        "request": "Create a calculator",
        "goal": "Create a calculator",
        "workspace": "/tmp/test",
        "project_context": {"languages": {"Python": 100}},
    }
    defaults.update(kwargs)
    return TaskState(**defaults)


class TestEvaluator:

    def test_creation(self):
        ev = Evaluator()
        assert ev is not None

    def test_evaluate_terminal_complete(self):
        ev = Evaluator()
        state = make_state()
        state.mark_complete("Done")
        obs = Observation(type="noop", source="agent", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.SUCCESS

    def test_evaluate_terminal_failed(self):
        ev = Evaluator()
        state = make_state()
        state.mark_failed("Broke")
        obs = Observation(type="noop", source="agent", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.FAILURE

    def test_evaluate_terminal_cancelled(self):
        ev = Evaluator()
        state = make_state()
        state.mark_cancelled("User cancelled")
        obs = Observation(type="noop", source="agent", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.BLOCKED

    def test_evaluate_error(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="error", source="test", success=False, data={"error": "boom"})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.FAILURE
        assert result.confidence > 0.5

    def test_evaluate_command_failed(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="command_failed", source="test", success=False,
                          data={"error": "not found", "return_code": 1})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.FAILURE

    def test_evaluate_test_failed(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="test_failed", source="run_tests", success=False,
                          data={"error": "1 failed", "output": "FAIL"})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.FAILURE
        assert "test" in result.reason.lower() or "fail" in result.reason.lower()

    def test_evaluate_test_passed(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="test_passed", source="run_tests", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.SUCCESS or result.outcome == EvaluationOutcome.PROGRESS

    def test_evaluate_file_created(self):
        ev = Evaluator()
        state = make_state()
        state.remaining_steps = []
        obs = Observation(type="file_created", source="write_file", success=True, data={"path": "test.py"})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.PROGRESS
        assert result.confidence >= 0.7

    def test_evaluate_command_success(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="command_success", source="execute_command", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.PROGRESS

    def test_evaluate_file_missing(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="file_missing", source="read_file", success=False, data={"path": "x.py"})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.BLOCKED

    def test_evaluate_approval_granted(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="approval_granted", source="approval", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.PROGRESS

    def test_evaluate_approval_denied(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="approval_denied", source="approval", success=False)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.BLOCKED

    def test_evaluate_plan_created(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="plan_created", source="planner", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.PROGRESS

    def test_evaluate_unknown_type(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="unknown_thing", source="test", success=True)
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.PROGRESS
        assert result.confidence < 1.0

    def test_evaluate_completion_success(self):
        ev = Evaluator()
        state = make_state()
        state.remaining_steps = []
        state.add_test_result(TestResult(command="pytest", passed=True))
        result = ev.evaluate_completion(state)
        assert result.outcome == EvaluationOutcome.SUCCESS
        assert result.confidence >= 0.9

    def test_evaluate_completion_with_remaining_steps(self):
        ev = Evaluator()
        state = make_state()
        state.remaining_steps = [{"id": "s1"}]
        result = ev.evaluate_completion(state)
        assert result.outcome == EvaluationOutcome.PROGRESS
        assert result.confidence < 0.5

    def test_evaluate_completion_with_failing_tests(self):
        ev = Evaluator()
        state = make_state()
        state.remaining_steps = []
        state.add_test_result(TestResult(command="pytest", passed=False, error="1 failed"))
        result = ev.evaluate_completion(state)
        assert result.outcome == EvaluationOutcome.FAILURE

    def test_evaluate_completion_with_no_tests(self):
        ev = Evaluator()
        state = make_state()
        state.remaining_steps = []
        result = ev.evaluate_completion(state)
        assert result.outcome == EvaluationOutcome.SUCCESS

    def test_evaluation_result_to_dict(self):
        result = EvaluationResult(
            outcome=EvaluationOutcome.PROGRESS,
            confidence=0.8,
            reason="All good",
            recommendations=["continue"],
        )
        d = result.to_dict()
        assert d["outcome"] == "progress"
        assert d["confidence"] == 0.8
        assert d["recommendations"] == ["continue"]

    def test_evaluate_build_failed(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="build_failed", source="execute_command", success=False,
                          data={"error": "syntax error"})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.FAILURE

    def test_evaluate_action_failed(self):
        ev = Evaluator()
        state = make_state()
        obs = Observation(type="action_failed", source="write_file", success=False,
                          data={"error": "permission denied"})
        result = ev.evaluate(state, obs)
        assert result.outcome == EvaluationOutcome.FAILURE
