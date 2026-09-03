"""
Phase 17 — Native Agent tests.

Verifies:
1. AgentState enum has correct states
2. AgentActionType enum has correct types
3. AgentObservation has correct structure
4. AgentAction has correct structure
5. AgentResult has correct structure
6. NativeAgent initializes without crashing
7. NativeAgent execute runs full cycle
8. NativeAgent observe generates observations
9. NativeAgent understand parses goal
10. NativeAgent reason generates reasoning
11. NativeAgent plan creates plan
12. NativeAgent request_authorization handles approval
13. NativeAgent decide_action chooses first step
14. NativeAgent act executes action
15. NativeAgent test validates result
16. NativeAgent evaluate assesses outcome
17. NativeAgent learn records lesson
18. NativeAgent get_state returns state
19. Agent respects security boundaries (no ModelManager)
20. Agent works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.agent_intelligence import (
    AgentAction,
    AgentActionType,
    AgentObservation,
    AgentResult,
    AgentState,
    NativeAgent,
)
from evora.logger import Logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def native_agent():
    return NativeAgent(logger=Logger("evora-test-p17", "info", None))


@pytest.fixture
def agent_with_deps():
    from unittest.mock import AsyncMock
    from evora.approval import ApprovalDecision

    runtime = MagicMock()
    native_plan = MagicMock()
    native_plan.to_dict.return_value = {
        "steps": [{"id": "step-1", "action_type": "analyze_project", "name": "Analyze", "action_args": {"path": "."}}],
        "confidence": 0.6,
        "requires_approval": True,
        "limitations": [],
    }
    runtime.plan = AsyncMock(return_value=native_plan)

    reasoning_result = MagicMock()
    reasoning_result.to_dict.return_value = {
        "decision": "proceed", "action": "analyze", "confidence": 0.5,
        "reasoning_summary": "Test reasoning", "limitations": [],
    }
    runtime.reason = AsyncMock(return_value=reasoning_result)

    # Mock tool that returns a proper ToolResult
    mock_tool = MagicMock()
    tool_result = MagicMock()
    tool_result.success = True
    tool_result.output = "Tool executed"
    tool_result.error = ""
    mock_tool.execute = AsyncMock(return_value=tool_result)

    tool_registry = MagicMock()
    tool_registry.get.return_value = mock_tool
    tool_registry.list.return_value = []

    approval_system = MagicMock()
    approval_system.approve_plan.return_value = ApprovalDecision.APPROVE

    permission_manager = MagicMock()
    permission_manager.check_command_safety.return_value = MagicMock()
    permission_manager.request_approval.return_value = True

    identity_service = MagicMock()
    identity_service.require_authority.return_value = MagicMock()

    return NativeAgent(
        intelligence_runtime=runtime,
        comprehension_intelligence=MagicMock(),
        conversation_manager=MagicMock(),
        tool_registry=tool_registry,
        permission_manager=permission_manager,
        approval_system=approval_system,
        identity_service=identity_service,
        training_pipeline=MagicMock(),
        logger=Logger("evora-test-p17-deps", "info", None),
    )


# ---------------------------------------------------------------------------
# TestAgentStateEnum
# ---------------------------------------------------------------------------

class TestAgentStateEnum:
    """Test AgentState enum."""

    def test_idle_state_exists(self):
        assert AgentState.IDLE is not None
        assert AgentState.IDLE.value == "idle"

    def test_all_states_exist(self):
        expected = ["idle", "observing", "understanding", "reasoning", "planning",
                    "requesting_authorization", "acting", "testing", "evaluating", "learning", "error"]
        for state in expected:
            assert AgentState(state) is not None


# ---------------------------------------------------------------------------
# TestAgentActionTypeEnum
# ---------------------------------------------------------------------------

class TestAgentActionTypeEnum:
    """Test AgentActionType enum."""

    def test_action_types_exist(self):
        assert AgentActionType.READ_FILE is not None
        assert AgentActionType.WRITE_FILE is not None
        assert AgentActionType.EDIT_FILE is not None
        assert AgentActionType.EXECUTE_COMMAND is not None
        assert AgentActionType.RUN_TESTS is not None
        assert AgentActionType.COMPLETE is not None
        assert AgentActionType.ABORT is not None


# ---------------------------------------------------------------------------
# TestAgentObservation
# ---------------------------------------------------------------------------

class TestAgentObservation:
    """Test AgentObservation."""

    def test_default_observation(self):
        obs = AgentObservation()
        assert obs.observation_id != ""
        assert obs.timestamp != ""

    def test_observation_to_dict(self):
        obs = AgentObservation(
            observation_type="test",
            source="unit_test",
            data={"key": "value"},
        )
        data = obs.to_dict()
        assert data["observation_type"] == "test"
        assert data["source"] == "unit_test"
        assert data["data"]["key"] == "value"


# ---------------------------------------------------------------------------
# TestAgentAction
# ---------------------------------------------------------------------------

class TestAgentAction:
    """Test AgentAction."""

    def test_default_action(self):
        action = AgentAction()
        assert action.action_id != ""
        assert action.confidence == 0.0

    def test_action_to_dict(self):
        action = AgentAction(
            action_type="analyze_code",
            description="Analyze the codebase",
            parameters={"target": "src/"},
            reasoning="Need to understand structure",
        )
        data = action.to_dict()
        assert data["action_type"] == "analyze_code"
        assert data["reasoning"] == "Need to understand structure"


# ---------------------------------------------------------------------------
# TestAgentResult
# ---------------------------------------------------------------------------

class TestAgentResult:
    """Test AgentResult."""

    def test_default_result(self):
        result = AgentResult()
        assert result.success is False

    def test_result_to_dict(self):
        result = AgentResult(
            action_id="act-123",
            success=True,
            output="Done",
            lesson_learned="Use faster algorithm",
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["lesson_learned"] == "Use faster algorithm"


# ---------------------------------------------------------------------------
# TestNativeAgent
# ---------------------------------------------------------------------------

class TestNativeAgent:
    """Test NativeAgent."""

    def test_agent_initializes(self, native_agent):
        assert native_agent is not None
        state = native_agent.get_state()
        assert state["state"] == "idle"

    def test_execute_returns_result(self, agent_with_deps):
        result = agent_with_deps.execute("analyze code", {})
        assert isinstance(result, AgentResult)
        assert result.action_id != ""

    def test_agent_history(self, agent_with_deps):
        agent_with_deps.execute("task 1", {})
        agent_with_deps.execute("task 2", {})
        state = agent_with_deps.get_state()
        assert state["history_count"] == 2

    def test_agent_tracks_observations(self, agent_with_deps):
        result = agent_with_deps.execute("test goal", {"project": "evora"})
        assert isinstance(result.observations, list)

    def test_agent_evaluates_success(self, agent_with_deps):
        result = agent_with_deps.execute("test goal", {})
        assert "evaluation" in result.to_dict()
        assert result.to_dict()["evaluation"] is not None

    def test_agent_records_lesson(self, agent_with_deps):
        result = agent_with_deps.execute("test goal", {})
        assert isinstance(result.lesson_learned, str)


# ---------------------------------------------------------------------------
# TestAgentStateTransitions
# ---------------------------------------------------------------------------

class TestAgentStateTransitions:
    """Test agent state transitions."""

    def test_agent_observes(self, native_agent):
        obs = native_agent.observe({})
        assert isinstance(obs, list)

    def test_agent_understands(self, native_agent):
        understanding = native_agent.understand("analyze Python", {})
        assert isinstance(understanding, dict)
        assert "goal" in understanding

    def test_agent_reasons(self, native_agent):
        reasoning = native_agent.reason("test goal", {"goal": "test goal"}, {})
        assert isinstance(reasoning, dict)

    def test_agent_plans(self, native_agent):
        plan = native_agent.plan("test goal", {}, {})
        assert isinstance(plan, dict)

    def test_agent_decides_action(self, native_agent):
        plan = {"steps": [{"action_type": "analyze_code", "name": "Analyze"}], "requires_approval": False}
        reasoning = {"confidence": 0.8, "reasoning_summary": "Good plan"}
        action = native_agent.decide_action(plan, reasoning)
        assert isinstance(action, AgentAction)
        assert action.action_type == "analyze_code"

    def test_agent_acts(self, native_agent):
        action = AgentAction(action_type="analyze_project", description="Analyze", requires_approval=False)
        result = native_agent.act(action, {})
        assert isinstance(result, AgentResult)

    def test_agent_tests(self, native_agent):
        test_result = native_agent.test(AgentResult(success=True), {})
        assert isinstance(test_result, dict)

    def test_agent_evaluates(self, native_agent):
        evaluation = native_agent.evaluate("goal", AgentResult(success=True), {"passed": True}, {})
        assert isinstance(evaluation, dict)
        assert "goal_achieved" in evaluation

    def test_agent_learns(self, native_agent):
        lesson = native_agent.learn("goal", AgentResult(success=True), {"confidence": 0.8}, {})
        assert isinstance(lesson, str)


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 17 security boundaries."""

    def test_no_model_manager_in_agent(self):
        import evora.brain.intelligence.agent_intelligence as agent_mod
        source = Path(agent_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.agent_intelligence as agent_mod
        source = Path(agent_mod.__file__).read_text(encoding="utf-8")
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

    def test_agent_uses_existing_security(self, agent_with_deps):
        result = agent_with_deps.execute("test", {})
        assert isinstance(result, AgentResult)

    def test_agent_cannot_bypass_security(self):
        agent = NativeAgent()
        assert not hasattr(agent, "grant_authority")
        assert not hasattr(agent, "approve_self")
        assert not hasattr(agent, "bypass_security")


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 17 works offline."""

    def test_agent_works_offline(self, native_agent):
        result = native_agent.execute("offline task", {})
        assert isinstance(result, AgentResult)

    def test_agent_state_offline(self, agent_with_deps):
        agent_with_deps.execute("offline goal", {})
        state = agent_with_deps.get_state()
        assert state["state"] in ["idle", "error"]


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 17 architecture readiness."""

    def test_native_agent_exists(self):
        from evora.brain.intelligence.agent_intelligence import NativeAgent
        assert NativeAgent is not None

    def test_agent_state_enum_exists(self):
        from evora.brain.intelligence.agent_intelligence import AgentState
        assert AgentState.IDLE is not None

    def test_agent_action_type_enum_exists(self):
        from evora.brain.intelligence.agent_intelligence import AgentActionType
        assert AgentActionType.ANALYZE_CODE is not None

    def test_agent_observation_exists(self):
        from evora.brain.intelligence.agent_intelligence import AgentObservation
        assert AgentObservation is not None

    def test_agent_action_exists(self):
        from evora.brain.intelligence.agent_intelligence import AgentAction
        assert AgentAction is not None

    def test_agent_result_exists(self):
        from evora.brain.intelligence.agent_intelligence import AgentResult
        assert AgentResult is not None

    def test_agent_reuses_intelligence_runtime(self, agent_with_deps):
        assert agent_with_deps.intelligence_runtime is not None

    def test_agent_reuses_security(self, agent_with_deps):
        assert agent_with_deps.permission_manager is not None
        assert agent_with_deps.approval_system is not None
        assert agent_with_deps.identity_service is not None
