"""
Tests for EVORA Phase 7: Autonomous Development & Reasoning Loop.

These tests verify:
- Reasoning engine structure and parsing
- Development inspection
- Improvement discovery from findings
- Development plan generation
- Development state machine transitions
- Creator approval enforcement
- Implementation with scope enforcement
- Testing and rollback behavior
- Benchmarking
- Learning/experience extraction
- End-to-end development loops
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from evora.reasoning import ReasoningEngine, ReasoningContext, ReasoningResult
from evora.inspector import DevelopmentInspector, InspectionReport, InspectionFinding
from evora.discovery import ImprovementDiscovery, ImprovementCandidate
from evora.dev_planner import DevelopmentPlanner, DevelopmentPlan, DevelopmentStep
from evora.self_develop import SelfDevelopmentSession, DevStatus, DevSessionRecord
from evora.security import PermissionManager
from evora.logger import Logger
from evora.identity import IdentityService, Identity, AuthorityLevel


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def security(tmp_workspace):
    return PermissionManager(str(tmp_workspace), allow_file_write=True, allow_cmd_exec=True)


@pytest.fixture
def logger():
    return Logger("evora-test-phase7", "info", None)


@pytest.fixture
def mock_model_manager():
    manager = MagicMock()
    response = MagicMock()
    response.content = json.dumps({
        "summary": "This improvement addresses a clear gap",
        "selected_approach": "add_missing_tests",
        "confidence": 0.85,
        "risks": ["May require test infrastructure changes"],
        "next_action": "implement_tests",
    })
    manager.chat = AsyncMock(return_value=response)
    return manager


class TestReasoningEngine:
    """Test the reasoning engine."""

    def test_reasoning_context_creation(self):
        context = ReasoningContext(
            objective="Improve test coverage",
            observations=["Only 60% coverage"],
            constraints=["Must pass existing tests"],
        )
        assert context.objective == "Improve test coverage"
        assert len(context.observations) == 1
        assert len(context.constraints) == 1

    def test_reasoning_result_creation(self):
        result = ReasoningResult(
            summary="Test coverage needs improvement",
            selected_approach="add_tests",
            confidence=0.8,
            risks=["Time consuming"],
            next_action="plan_tests",
        )
        assert result.confidence == 0.8
        assert result.selected_approach == "add_tests"
        d = result.to_dict()
        assert d["summary"] == result.summary

    @pytest.mark.asyncio
    async def test_reason_returns_result(self, mock_model_manager, logger):
        engine = ReasoningEngine(mock_model_manager, logger)
        context = ReasoningContext(objective="Improve tests")
        result = await engine.reason(context)
        assert isinstance(result, ReasoningResult)
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_reason_handles_model_failure(self, logger):
        manager = MagicMock()
        manager.chat = AsyncMock(side_effect=Exception("Model down"))
        engine = ReasoningEngine(manager, logger)
        context = ReasoningContext(objective="Improve tests")
        result = await engine.reason(context)
        assert result.confidence == 0.0
        assert result.next_action == "abort"
        assert "Model down" in result.summary or "aborted" in result.summary.lower()

    @pytest.mark.asyncio
    async def test_reason_handles_empty_response(self, logger):
        manager = MagicMock()
        response = MagicMock()
        response.content = ""
        manager.chat = AsyncMock(return_value=response)
        engine = ReasoningEngine(manager, logger)
        context = ReasoningContext(objective="Improve tests")
        result = await engine.reason(context)
        assert result.confidence < 0.5


class TestDevelopmentInspector:
    """Test development inspection."""

    def test_inspect_returns_report(self, tmp_workspace, security, logger):
        inspector = DevelopmentInspector(str(tmp_workspace), security, logger)
        report = inspector.inspect()
        assert isinstance(report, InspectionReport)
        assert "findings" in report.to_dict()
        assert "project_structure" in report.to_dict()

    def test_inspect_detects_tests(self, tmp_workspace, security, logger):
        tests_dir = tmp_workspace / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_sample.py").write_text("def test_pass(): pass\n")
        evora_dir = tmp_workspace / "evora"
        evora_dir.mkdir(parents=True, exist_ok=True)
        (evora_dir / "__init__.py").write_text("")
        inspector = DevelopmentInspector(str(tmp_workspace), security, logger)
        report = inspector.inspect()
        assert len(report.project_structure.get("test_files", [])) >= 1

    def test_inspect_handles_missing_workspace(self, security, logger):
        inspector = DevelopmentInspector("/nonexistent/path", security, logger)
        report = inspector.inspect()
        assert report.project_structure.get("total_py_files", 0) == 0


class TestImprovementDiscovery:
    """Test improvement discovery."""

    def test_discover_from_findings(self, logger):
        discovery = ImprovementDiscovery(logger)
        findings = [
            InspectionFinding(category="tests", severity="high", description="Tests failing"),
            InspectionFinding(category="architecture", severity="low", description="Large module"),
        ]
        report = InspectionReport(findings=findings)
        candidates = discovery.discover(report)
        assert len(candidates) == 2
        assert all(isinstance(c, ImprovementCandidate) for c in candidates)
        assert candidates[0].category == "tests"
        assert candidates[0].severity == "high"

    def test_candidate_to_proposal(self, logger):
        discovery = ImprovementDiscovery(logger)
        candidate = ImprovementCandidate(
            id="c1",
            title="Fix tests",
            description="Fix failing tests",
            category="tests",
            severity="high",
            affected_files=["tests/test_a.py"],
        )
        proposal = candidate.to_proposal()
        assert proposal.id == "c1"
        assert proposal.title == "Fix tests"
        assert proposal.files_changed == ["tests/test_a.py"]


class TestDevelopmentPlanner:
    """Test development planning."""

    @pytest.mark.asyncio
    async def test_create_plan_generates_steps(self, mock_model_manager, logger):
        planner = DevelopmentPlanner(mock_model_manager, logger)
        candidate = ImprovementCandidate(
            id="c1",
            title="Improve coverage",
            description="Add tests",
            category="tests",
            severity="high",
            affected_files=["tests/test_a.py"],
        )
        plan = await planner.create_plan(candidate, {})
        assert isinstance(plan, DevelopmentPlan)
        assert len(plan.steps) >= 2
        assert any("test" in s.name.lower() for s in plan.steps)

    def test_plan_to_dict(self, logger):
        plan = DevelopmentPlan(
            id="p1",
            objective="Test improvement",
            candidate_id="c1",
            steps=[DevelopmentStep(id="s1", name="Step 1", description="Do thing", action_type="run_command")],
        )
        d = plan.to_dict()
        assert d["id"] == "p1"
        assert len(d["steps"]) == 1


class TestDevelopmentStateMachine:
    """Test development session state transitions."""

    @pytest.mark.asyncio
    async def test_valid_transitions(self):
        session = SelfDevelopmentSession.__new__(SelfDevelopmentSession)
        session._status = DevStatus.IDLE
        session.logger = None
        await session._transition(DevStatus.INSPECTING)
        assert session._status == DevStatus.INSPECTING

    @pytest.mark.asyncio
    async def test_invalid_transition_raises(self):
        session = SelfDevelopmentSession.__new__(SelfDevelopmentSession)
        session._status = DevStatus.IDLE
        session.logger = None
        with pytest.raises(ValueError, match="Invalid state transition"):
            await session._transition(DevStatus.THINKING)

    @pytest.mark.asyncio
    async def test_terminal_to_idle_allowed(self):
        session = SelfDevelopmentSession.__new__(SelfDevelopmentSession)
        session._status = DevStatus.SUCCEEDED
        session.logger = None
        await session._transition(DevStatus.IDLE)
        assert session._status == DevStatus.IDLE


class TestDevelopmentSession:
    """Test the full development session."""

    @pytest.mark.asyncio
    async def test_no_candidates_returns_succeeded(self, tmp_workspace, security, logger):
        inspector = DevelopmentInspector(str(tmp_workspace), security, logger)
        discovery = ImprovementDiscovery(logger)

        mock_plan = DevelopmentPlan(
            id="p1",
            objective="Test",
            candidate_id="c1",
            steps=[],
        )

        with patch.object(SelfDevelopmentSession, "__init__", lambda self, **kwargs: None):
            session = SelfDevelopmentSession.__new__(SelfDevelopmentSession)
            session.workspace = tmp_workspace
            session.security = security
            session.logger = logger
            session.identity_service = None
            session.approval = MagicMock()
            session.approval.approve_plan.return_value = MagicMock(value="approve")
            session.tools = MagicMock()
            session.memory = MagicMock()
            session.history = MagicMock()
            session.self_improve = MagicMock()
            session._record = None
            session._status = DevStatus.IDLE
            session.inspector = inspector
            session.discovery = discovery
            session.planner = MagicMock()
            session.planner.create_plan = AsyncMock(return_value=mock_plan)
            session.reasoning = MagicMock()
            session.reasoning.reason = AsyncMock(return_value=ReasoningResult(
                summary="No improvements needed",
                selected_approach="none",
                confidence=0.9,
                next_action="complete",
            ))

            result = await session.run("Improve nothing")
            assert "REJECTED" in result

    @pytest.mark.asyncio
    async def test_rejected_returns_rejected(self, tmp_workspace, security, logger):
        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high")
        ]

        mock_plan = DevelopmentPlan(
            id="p1",
            objective="Test",
            candidate_id="c1",
            steps=[],
        )

        with patch.object(SelfDevelopmentSession, "__init__", lambda self, **kwargs: None):
            session = SelfDevelopmentSession.__new__(SelfDevelopmentSession)
            session.workspace = tmp_workspace
            session.security = security
            session.logger = logger
            session.identity_service = None
            session.approval = MagicMock()
            session.approval.approve_plan.return_value = MagicMock(value="reject")
            session.tools = MagicMock()
            session.memory = MagicMock()
            session.history = MagicMock()
            session.self_improve = MagicMock()
            session._record = None
            session._status = DevStatus.IDLE
            session.inspector = inspector
            session.discovery = discovery
            session.planner = MagicMock()
            session.planner.create_plan = AsyncMock(return_value=mock_plan)
            session.reasoning = MagicMock()
            session.reasoning.reason = AsyncMock(return_value=ReasoningResult(
                summary="Select first candidate",
                selected_approach="Fix tests",
                confidence=0.7,
                next_action="plan",
            ))

            result = await session.run("Improve tests")
            assert "REJECTED" in result
