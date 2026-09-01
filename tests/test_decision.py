"""
Tests for EVORA Phase 2 decision engine.
"""

import pytest
from unittest.mock import MagicMock

from evora.task import TaskState, Observation, Decision
from evora.decision import DecisionEngine


def make_state(**kwargs) -> TaskState:
    defaults = {
        "request": "Create a calculator",
        "goal": "Create a calculator",
        "workspace": "/tmp/test",
        "project_context": {"languages": {"Python": 100}},
    }
    defaults.update(kwargs)
    return TaskState(**defaults)


class TestDecisionEngine:

    def test_engine_creation(self):
        engine = DecisionEngine()
        assert engine.max_retries == 3
        assert engine.auto_approve is False

    def test_decide_understand_from_idle(self):
        engine = DecisionEngine()
        state = TaskState(request="Build a calculator")
        state.status = "idle"

        decision = engine.decide_next(state)
        assert decision.action == "understand"
        assert decision.confidence == 1.0

    def test_decide_analyze_when_no_context(self):
        engine = DecisionEngine()
        state = make_state()
        state.status = "understood"
        state.goal = "Create a calculator"
        state.project_context = {}

        decision = engine.decide_next(state)
        assert decision.action == "analyze"

    def test_decide_plan_when_no_plan(self):
        engine = DecisionEngine()
        state = make_state(project_context={"languages": {"Python": 100}})
        state.status = "analyzed"

        decision = engine.decide_next(state)
        assert decision.action == "plan"

    def test_decide_ask_approval(self):
        engine = DecisionEngine()
        state = make_state(project_context={"languages": {"Python": 100}})
        state.status = "awaiting_approval"
        state.plan = {"title": "Test Plan"}
        state.remaining_steps = [{"id": "s1", "name": "step1", "action_type": "create_file"}]

        decision = engine.decide_next(state)
        assert decision.action == "ask_approval"
        assert decision.requires_approval is True

    def test_decide_execute_next_step(self):
        engine = DecisionEngine()
        state = make_state(project_context={"test": True})
        state.status = "executing"
        state.plan = {"title": "Test"}
        state.remaining_steps = [
            {"id": "s1", "name": "Create file", "action_type": "create_file", "action_args": {"path": "test.py"}, "estimated_effort": "low"},
        ]
        state.completed_steps = set()

        decision = engine.decide_next(state)
        assert decision.action == "execute_tool"
        assert decision.tool == "write_file"
        assert decision.arguments["path"] == "test.py"

    def test_decide_run_tests_after_steps_complete(self):
        engine = DecisionEngine()
        state = make_state(project_context={"test": True})
        state.status = "executing"
        state.plan = {"title": "Test"}
        state.remaining_steps = []
        state.completed_steps = {"s1", "s2"}
        state.test_results = []

        decision = engine.decide_next(state)
        assert decision.action == "run_tests"

    def test_decide_fix_on_test_failure(self):
        engine = DecisionEngine()
        state = make_state()
        state.status = "testing"
        state.remaining_steps = []
        state.completed_steps = {"s1"}
        state.plan = {"title": "Test Plan"}

        from evora.task import TestResult
        state.add_test_result(TestResult(command="pytest", passed=False, error="1 failed"))

        decision = engine.decide_next(state)
        assert decision.action == "fix_error"

    def test_decide_report_on_complete(self):
        engine = DecisionEngine()
        state = make_state()
        state.mark_complete("All done")

        decision = engine.decide_next(state)
        assert decision.action == "report"

    def test_decide_report_on_failed(self):
        engine = DecisionEngine()
        state = make_state()
        state.mark_failed("Something broke")

        decision = engine.decide_next(state)
        assert decision.action == "report"

    def test_decide_report_on_cancelled(self):
        engine = DecisionEngine()
        state = make_state()
        state.mark_cancelled("User cancelled")

        decision = engine.decide_next(state)
        assert decision.action == "report"

    def test_max_retries_in_fix(self):
        engine = DecisionEngine(max_retries=2)
        state = make_state()
        state.status = "testing"
        state.remaining_steps = []
        state.plan = {"title": "Test Plan"}

        from evora.task import TestResult
        state.add_test_result(TestResult(command="pytest", passed=False, error="1 failed"))

        decision1 = engine.decide_next(state)
        assert decision1.action == "fix_error"

        decision2 = engine.decide_next(state)
        assert decision2.action == "fix_error"

        decision3 = engine.decide_next(state)
        assert decision3.action == "report"
        assert "Max retries" in decision3.reason or "max retry" in decision3.reason

    def test_auto_approve_changes_risk(self):
        engine = DecisionEngine(auto_approve=True)
        state = make_state()
        state.status = "executing"
        state.plan = {"title": "Test"}
        state.remaining_steps = [
            {"id": "s1", "name": "Run command", "action_type": "run_command", "action_args": {"command": "ls"}, "estimated_effort": "low"},
        ]
        state.completed_steps = set()

        decision = engine.decide_next(state)
        assert decision.action == "execute_tool"
        assert decision.requires_approval is False

    def test_action_type_to_tool_mapping(self):
        engine = DecisionEngine()
        assert engine._action_type_to_tool("create_file") == "write_file"
        assert engine._action_type_to_tool("edit_file") == "edit_file"
        assert engine._action_type_to_tool("read_file") == "read_file"
        assert engine._action_type_to_tool("run_command") == "execute_command"
        assert engine._action_type_to_tool("run_tests") == "execute_command"
        assert engine._action_type_to_tool("analyze") == "list_directory"
        assert engine._action_type_to_tool("create_directory") == "create_directory"

    def test_decide_understand_sets_goal(self):
        engine = DecisionEngine()
        state = TaskState(request="Write a test file", goal="")

        decision = engine.decide_next(state)
        assert decision.action == "understand"
        assert state.goal == "Write a test file"

    def test_reset_retry_count(self):
        engine = DecisionEngine()
        engine._retry_counts["fix_attempt"] = 5
        engine.reset_retry_count()
        assert "fix_attempt" not in engine._retry_counts

    def test_check_completion_with_no_remaining(self):
        engine = DecisionEngine()
        state = make_state()
        state.remaining_steps = []

        from evora.task import TestResult
        state.add_test_result(TestResult(command="pytest", passed=True))

        assert engine._check_completion(state) is True

    def test_check_completion_with_remaining(self):
        engine = DecisionEngine()
        state = make_state()
        state.remaining_steps = [{"id": "s1"}]

        assert engine._check_completion(state) is False
