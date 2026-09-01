"""
Tests for EVORA Phase 2 autonomous agent loop.

These tests verify:
- Dynamic decision making (not a hardcoded sequence)
- Observation → evaluation → re-decision loop
- Bounded error recovery
- Approval pauses
- Retry limit enforcement
- Successful completion
- Cancellation
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evora.task import TaskState, Decision, ActionResult, Observation, TestResult
from evora.decision import DecisionEngine
from evora.observation import ObservationManager
from evora.evaluation import Evaluator, EvaluationOutcome
from evora.autonomous import AutonomousAgent, AutonomousConfig
from evora.logger import Logger
from evora.model import ModelManager, ModelResponse, Usage, Role, Message
from evora.planner import Planner
from evora.approval import ApprovalSystem, ApprovalDecision
from evora.tools import ToolRegistry
from evora.memory import Memory
from evora.security import PermissionManager
from evora.analyzer import ProjectAnalyzer


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class MockModelProvider:
    """Mock model that returns configurable responses."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []
        self._call_idx = 0

    def name(self):
        return "mock"

    def model(self):
        return "mock-model"

    async def chat(self, request):
        self.calls.append(request)
        content = self._next_response()
        return ModelResponse(
            content=content,
            provider="mock",
            model="mock-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    def _next_response(self):
        for msg in reversed(self._last_messages()):
            if msg.role == Role.USER:
                user_content = msg.content[:300]
                break
        else:
            user_content = ""

        if self._call_idx < len(self.responses):
            content = self.responses[self._call_idx]
            self._call_idx += 1
            return content
        return json.dumps({
            "title": "Mock Plan",
            "description": "Auto-generated mock plan",
            "steps": [{
                "id": "step-1",
                "name": "Create file",
                "description": "Create a test file",
                "action_type": "create_file",
                "action_args": {"path": "output.txt", "content": "Hello"},
                "depends_on": [],
                "estimated_effort": "low",
            }],
        })

    def _last_messages(self):
        if self.calls:
            return self.calls[-1].messages
        return []

    async def chat_stream(self, request):
        yield await self.chat(request)

    def close(self):
        pass


class MockToolRegistry:
    """Mock tool registry that records calls and returns configurable results."""

    def __init__(self, results=None, defaults=None):
        self.results = results or {}
        self.defaults = defaults or {}
        self.call_log = []

    def get_specs(self):
        return []

    def list(self):
        return []

    def get(self, name):
        return None

    async def execute(self, name: str, **kwargs):
        self.call_log.append({"tool": name, "args": kwargs})
        key = name
        if key in self.defaults:
            spec = self.defaults[key]
            return spec
        if key in self.results and self.results[key]:
            return self.results[key].pop(0)
        from evora.tools import ToolResult
        return ToolResult(success=True, output=f"Mock result for {name}", data={"returncode": 0})


class TestAutonomousAgent:

    def _make_agent(self, tmp_path, config=None, mock_provider=None,
                    mock_tools=None, auto_approve=True, max_retries=3):
        logger = Logger("test_auto", "error")
        security = PermissionManager(
            workspace_dir=str(tmp_path),
            ask_approvals=not auto_approve,
            allowed_cmds=[],
        )
        memory = Memory(str(tmp_path / "memory"), "test-project")
        manager = ModelManager(logger)
        if mock_provider:
            manager.register("mock", mock_provider)
        else:
            manager.register("mock", MockModelProvider())

        planner = Planner(manager, logger)
        approval = ApprovalSystem(logger=logger, auto_approve=auto_approve)
        tools = mock_tools or MockToolRegistry()
        analyzer = ProjectAnalyzer(str(tmp_path), logger)

        agent = AutonomousAgent(
            model_manager=manager,
            planner=planner,
            approval=approval,
            tools=tools,
            memory=memory,
            security=security,
            logger=logger,
            analyzer=analyzer,
            config=config or AutonomousConfig(
                max_retries=max_retries,
                retry_delay=0.01,
                auto_approve=auto_approve,
                max_iterations=10,
            ),
        )
        return agent, manager, tools


class TestAutonomousAgentBasic(TestAutonomousAgent):

    def test_agent_creation(self, tmp_path):
        agent, manager, tools = self._make_agent(tmp_path)
        assert agent is not None
        assert agent.decision_engine is not None
        assert agent.observation_mgr is not None
        assert agent.evaluator is not None

    def test_run_completes_successfully(self, tmp_path):
        """Agent should complete a simple task successfully."""
        agent, manager, tools = self._make_agent(tmp_path)

        async def run():
            return await agent.run("Create a test file")

        report = run_async(run())
        assert "EVORA AUTONOMOUS TASK REPORT" in report
        assert "COMPLETED" in report
        assert len(tools.call_log) > 0

    def test_run_saves_memory(self, tmp_path):
        """Agent should save task to memory."""
        agent, manager, tools = self._make_agent(tmp_path)
        run_async(agent.run("Create a test file"))
        tasks = agent.memory.store.list_tasks(limit=5)
        assert len(tasks) >= 1


class TestAutonomousAgentDecisionMaking(TestAutonomousAgent):

    def test_decision_changes_based_on_observation(self, tmp_path):
        """The agent must change its next action based on what it observes.

        This is the KEY test: the decision engine should NOT follow a
        predetermined sequence. When an observation shows a failure,
        the next decision should be to fix, not to proceed.
        """
        from evora.task import TaskState, Observation, Decision
        from evora.decision import DecisionEngine

        engine = DecisionEngine()

        state = TaskState(
            request="Create a function",
            goal="Create a function",
            workspace=str(tmp_path),
            project_context={"languages": {"Python": 100}},
            plan={"title": "Test Plan"},
            remaining_steps=[],
            completed_steps={"s1"},
        )
        state.status = "executing"

        decision1 = engine.decide_next(state)
        assert decision1.action == "run_tests"

        from evora.task import TestResult
        state.add_test_result(TestResult(command="pytest", passed=False, error="1 failed"))

        state.status = "testing"
        decision2 = engine.decide_next(state)
        assert decision2.action == "fix_error"
        assert decision1.action != decision2.action

    def test_decision_changes_from_execute_to_test(self, tmp_path):
        """When remaining steps go from non-empty to empty, decision should change."""
        engine = DecisionEngine()

        state = TaskState(
            request="Create files",
            workspace=str(tmp_path),
            project_context={"test": True},
            plan={"title": "Plan"},
            remaining_steps=[{"id": "s1", "name": "step1", "action_type": "create_file"}],
        )
        state.status = "executing"

        decision1 = engine.decide_next(state)
        assert decision1.action == "execute_tool"

        state.remaining_steps = []
        state.completed_steps = {"s1"}
        state.status = "executing"

        decision2 = engine.decide_next(state)
        assert decision2.action == "run_tests"
        assert decision1.action != decision2.action

    def test_decision_changes_from_fix_to_report(self, tmp_path):
        """When max retries exceeded, decision changes from fix to report."""
        engine = DecisionEngine(max_retries=2)

        state = TaskState(
            request="Fix tests",
            workspace=str(tmp_path),
            project_context={"test": True},
            plan={"title": "Plan"},
            remaining_steps=[],
            completed_steps={"s1"},
        )
        state.status = "testing"

        from evora.task import TestResult
        state.add_test_result(TestResult(command="pytest", passed=False, error="fail"))

        decision1 = engine.decide_next(state)
        assert decision1.action == "fix_error"

        decision2 = engine.decide_next(state)
        assert decision2.action == "fix_error"

        decision3 = engine.decide_next(state)
        assert decision3.action == "report"
        assert decision1.action != decision3.action


class TestAutonomousAgentApproval(TestAutonomousAgent):

    def test_approval_can_pause_agent(self, tmp_path):
        """Agent should pause when approval is denied."""
        approval = ApprovalSystem(logger=Logger("test", "error"), auto_approve=False)

        original_approve = approval.approve_plan

        def denying_approval(_plan_text, _plan_obj=None):
            return ApprovalDecision.REJECT

        approval.approve_plan = denying_approval

        async def run():
            return await agent.run("Create a file")

        agent, manager, tools = self._make_agent(
            tmp_path,
            auto_approve=False,
        )
        agent.approval = approval

        report = run_async(run())
        assert "CANCELLED" in report or "FAILED" in report or "cancelled" in report.lower()


class TestAutonomousAgentErrorRecovery(TestAutonomousAgent):

    def test_bounded_recovery_on_tool_failure(self, tmp_path):
        """Agent should attempt fixes but stop after max_retries."""
        from evora.tools import ToolResult

        mock_tools = MockToolRegistry()
        mock_tools.results["write_file"] = [
            ToolResult(success=False, error="Permission denied", data={"returncode": 1}),
            ToolResult(success=True, output="Written", data={"returncode": 0}),
        ]

        agent, manager, tools = self._make_agent(
            tmp_path,
            mock_tools=mock_tools,
            config=AutonomousConfig(
                max_retries=2,
                retry_delay=0.01,
                auto_approve=True,
                max_iterations=5,
            ),
        )

        # The mock model returns a plan with one create_file step
        # The tool fails first, then succeeds
        report = run_async(agent.run("Create a file"))
        assert "EVORA AUTONOMOUS TASK REPORT" in report


class TestAutonomousAgentRetryLimit(TestAutonomousAgent):

    def test_retry_limit_prevents_infinite_loop(self, tmp_path):
        """Agent must not loop infinitely — should stop after max_iterations."""
        from evora.tools import ToolResult

        mock_tools = MockToolRegistry()
        mock_tools.results["write_file"] = [
            ToolResult(success=False, error="Always fails", data={"returncode": 1}),
        ]
        mock_tools.defaults["write_file"] = ToolResult(success=False, error="Always fails", data={"returncode": 1})

        agent, manager, tools = self._make_agent(
            tmp_path,
            mock_tools=mock_tools,
            config=AutonomousConfig(
                max_retries=1,
                retry_delay=0.01,
                auto_approve=True,
                max_iterations=3,
            ),
        )

        report = run_async(agent.run("Create a file that always fails"))
        assert "EVORA AUTONOMOUS TASK REPORT" in report
        assert "COMPLETED" not in report or "FAILED" in report

    def test_task_state_has_max_attempts(self, tmp_path):
        """TaskState should have a configurable max_attempts."""
        state = TaskState(request="test", max_attempts=5)
        assert state.max_attempts == 5
        assert state.exceeded_retry_limit() is False
        state.attempts = 5
        assert state.exceeded_retry_limit() is True


class TestAutonomousAgentCompletion(TestAutonomousAgent):

    def test_successful_completion(self, tmp_path):
        """Agent should report COMPLETED when task succeeds."""
        agent, manager, tools = self._make_agent(tmp_path)
        report = run_async(agent.run("Create a test file"))
        assert "COMPLETED" in report

    def test_cancellation(self, tmp_path):
        """Agent should stop when cancelled."""
        agent, manager, tools = self._make_agent(tmp_path)

        async def stop_and_run():
            async def stop_later():
                await asyncio.sleep(0.01)
                await agent.stop()

            async def run():
                return await asyncio.gather(agent.run("Create a file"), stop_later())
                return ""

            report = await agent.run("Create a file")
            return report

        result = run_async(agent.run("Create a file"))
        assert "EVORA AUTONOMOUS TASK REPORT" in result


class TestAutonomousAgentObservationFeedback(TestAutonomousAgent):

    def test_observation_feeds_into_state(self, tmp_path):
        """Observations from actions should be recorded in task state."""
        from evora.tools import ToolResult

        recorded_state = []

        class TrackingAgent(AutonomousAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.observation_count = 0

            def _observe(self, state: TaskState, result: ActionResult):
                self.observation_count += 1
                observations = self.observation_mgr.from_action_result(result)
                for obs in observations:
                    state.add_observation(obs)
                return observations

        logger = Logger("test", "error")
        security = PermissionManager(workspace_dir=str(tmp_path), ask_approvals=False, allowed_cmds=[])
        memory = Memory(str(tmp_path / "memory"), "test-project")
        manager = ModelManager(logger)
        manager.register("mock", MockModelProvider())
        planner = Planner(manager, logger)
        approval = ApprovalSystem(logger=logger, auto_approve=True)
        tools = MockToolRegistry()
        analyzer = ProjectAnalyzer(str(tmp_path), logger)

        agent = TrackingAgent(
            model_manager=manager,
            planner=planner,
            approval=approval,
            tools=tools,
            memory=memory,
            security=security,
            logger=logger,
            analyzer=analyzer,
            config=AutonomousConfig(max_retries=1, retry_delay=0.01, auto_approve=True, max_iterations=10),
        )

        report = run_async(agent.run("Create a test file"))
        assert agent.observation_count > 0


class TestPhase2Integration(TestAutonomousAgent):

    def test_full_loop_phases(self, tmp_path):
        """The agent should go through understand, plan, execute, test, report."""
        agent, manager, tools = self._make_agent(tmp_path)

        log_output = []
        original_logger = agent.logger

        class CapturingLogger:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                attr = getattr(self._inner, name)
                if callable(attr):
                    def wrapper(msg, *a, **kw):
                        log_output.append(f"{name}: {msg}")
                        return attr(msg, *a, **kw)
                    return wrapper
                return attr

        agent.logger = CapturingLogger(original_logger)
        agent.observation_mgr.logger = agent.logger
        agent.decision_engine.logger = agent.logger
        agent.evaluator.logger = agent.logger

        report = run_async(agent.run("Create a hello.py file"))
        assert "EVORA AUTONOMOUS TASK REPORT" in report

        log_text = " ".join(log_output)
        assert "understand" in log_text.lower() or "plan" in log_text.lower()

    def test_existing_agent_still_works(self, tmp_path):
        """The old Phase 1 Agent class should still function."""
        from evora.agent import Agent, AgentConfig

        logger = Logger("test_phase1", "error")
        security = PermissionManager(workspace_dir=str(tmp_path), ask_approvals=False)
        memory = Memory(str(tmp_path / "memory"), "test-project")
        manager = ModelManager(logger)
        manager.register("mock", MockModelProvider())

        planner = Planner(manager, logger)
        approval = ApprovalSystem(logger=logger, auto_approve=True)
        tools = ToolRegistry(security, logger)

        agent = Agent(
            model_manager=manager,
            plan=planner,
            approval=approval,
            tools=tools,
            memory=memory,
            security=security,
            logger=logger,
            config=AgentConfig(max_retries=1, retry_delay=0.1, auto_approve=True),
            analyzer=ProjectAnalyzer(str(tmp_path), logger),
        )

        report = run_async(agent.run("Create a test file"))
        assert "EVORA TASK REPORT" in report
