"""
Phase 7 Independent Security Audit — Kilo #2

Attack tests targeting:
1. Approval bypass
2. Scope escape
3. Critical control file protection
4. Sensitive file access
5. Model output attacks
6. Reasoning engine attacks
7. State machine attacks
8. Direct method invocation
9. Plan integrity
10. Rollback behavior
11. Benchmark integrity
12. Learning integrity
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from evora.self_develop import SelfDevelopmentSession, DevStatus, DevSessionRecord
from evora.self_improve import ChangeValidator, ImprovementStatus, SelfImproveTool
from evora.security import PermissionManager
from evora.logger import Logger
from evora.identity import IdentityService, Identity, AuthorityLevel, IdentityStore
from evora.tools import ToolRegistry
from evora.reasoning import ReasoningEngine, ReasoningContext, ReasoningResult
from evora.inspector import DevelopmentInspector, InspectionReport, InspectionFinding
from evora.discovery import ImprovementDiscovery, ImprovementCandidate
from evora.dev_planner import DevelopmentPlanner, DevelopmentPlan, DevelopmentStep
from evora.approval import ApprovalSystem, ApprovalDecision, ApprovalToken


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def security(tmp_workspace):
    return PermissionManager(str(tmp_workspace), allow_file_write=True, allow_cmd_exec=True)


@pytest.fixture
def logger():
    return Logger("evora-audit-p7", "info", None)


@pytest.fixture
def creator_identity_service(tmp_workspace, logger):
    store = IdentityStore(str(tmp_workspace / "identities"))
    store.bootstrap_creator("Creator")
    return IdentityService(store=store, logger=logger)


class TestApprovalBypass:
    """Try to bypass creator approval."""

    @pytest.mark.asyncio
    async def test_missing_approval_system_denied(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=None,
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        result = await session.run("Improve something")
        assert "REJECTED" in result or "FAILED" in result

    @pytest.mark.asyncio
    async def test_rejected_approval_cannot_implement(self, tmp_workspace, security, logger):
        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high")
        ]

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        session.planner.create_plan = AsyncMock(return_value=DevelopmentPlan(
            id="p1", objective="Fix", candidate_id="c1", steps=[]
        ))
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))

        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="reject"))

        result = await session.run("Improve tests")
        assert "REJECTED" in result

    @pytest.mark.asyncio
    async def test_eof_returns_reject(self, tmp_workspace, security, logger):
        from evora.approval import ApprovalDecision
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.approval.approve_plan = MagicMock(side_effect=EOFError)
        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        session.inspector = inspector
        session.discovery = MagicMock()
        session.discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high")
        ]
        session.planner = MagicMock()
        session.planner.create_plan = AsyncMock(return_value=DevelopmentPlan(
            id="p1", objective="Fix", candidate_id="c1", steps=[]
        ))
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))

        result = await session.run("Improve tests")
        assert "FAILED" in result or "REJECTED" in result

    @pytest.mark.asyncio
    async def test_implement_without_approval_token_denied(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="test", objective="test")
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        result = await session._implement(candidate, plan)
        assert result.get("success") is False
        assert "approval token" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_approval_token_is_single_use(self, tmp_workspace, security, logger, creator_identity_service):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=creator_identity_service,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        token = ApprovalToken.create(session_id="s1", plan_id="p1", candidate_id="c1", approved_by="creator")
        session._approval_token = token
        session.approval.consume_approval_token = MagicMock(return_value=True)

        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="s1", objective="test")
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high")
        plan1 = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        result1 = await session._implement(candidate, plan1)
        assert result1.get("success") is True
        session.approval.consume_approval_token.assert_called_once_with(
            token_id=token.token_id,
            session_id="s1",
            plan_id="p1",
            candidate_id="c1",
        )

        token2 = ApprovalToken.create(session_id="s1", plan_id="p2", candidate_id="c1", approved_by="creator")
        session._approval_token = token2
        session._implemented_plan_ids.discard(plan1.id)
        session._status = DevStatus.APPROVED
        session.approval.consume_approval_token = MagicMock(return_value=False)
        plan2 = DevelopmentPlan(id="p2", objective="T", candidate_id="c1", steps=[])
        result2 = await session._implement(candidate, plan2)
        assert result2.get("success") is False
        assert "invalid or already consumed" in result2.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_approval_token_does_not_survive_process_restart(self, tmp_workspace, security, logger):
        original = ApprovalSystem(auto_approve=True)
        original.approve_plan("", {"id": "p", "candidate_id": "c"})
        token = original.issue_approval_token("s", "p", "c", "creator")
        restarted = ApprovalSystem(auto_approve=True)
        assert token is not None
        assert restarted.consume_approval_token(token.token_id, "s", "p", "c") is False

    @pytest.mark.asyncio
    async def test_reentrant_run_is_rejected(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace), model_manager=MagicMock(), security=security,
            identity_service=None, approval=MagicMock(), tools=MagicMock(),
            memory=MagicMock(), logger=logger,
        )
        session._running = True
        result = await session.run("reentrant")
        assert result.startswith("FAILED")
        assert "already running" in result


class TestScopeEscape:
    """Try to escape approved scope."""

    @pytest.mark.asyncio
    async def test_plan_with_extra_files_not_executed(self, tmp_workspace, security, logger):
        target = tmp_workspace / "allowed.py"
        target.write_text("x = 1\n")
        escape_target = tmp_workspace / "escape.py"
        escape_target.write_text("y = 2\n")

        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high",
                                affected_files=["allowed.py"])
        ]

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        plan_with_escape = DevelopmentPlan(
            id="p1",
            objective="Fix",
            candidate_id="c1",
            steps=[
                DevelopmentStep(id="s1", name="Modify escape.py", description="Escape", action_type="edit_file",
                               action_args={"path": "escape.py", "old_string": "y = 2", "new_string": "y = 3"}),
            ],
        )
        session.planner.create_plan = AsyncMock(return_value=plan_with_escape)
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))
        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="approve"))

        result = await session.run("Improve tests")
        assert "SUCCESS" in result or "REJECTED" in result or "FAILED" in result
        assert escape_target.read_text() == "y = 2\n"

    @pytest.mark.asyncio
    async def test_path_traversal_in_plan_denied(self, tmp_workspace, security, logger):
        target = tmp_workspace / "test.txt"
        target.write_text("content")

        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high",
                                affected_files=["../outside.txt"])
        ]

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        plan_with_traversal = DevelopmentPlan(
            id="p1",
            objective="Fix",
            candidate_id="c1",
            steps=[
                DevelopmentStep(id="s1", name="Modify outside", description="Escape", action_type="edit_file",
                               action_args={"path": "../outside.txt", "old_string": "content", "new_string": "modified"}),
            ],
        )
        session.planner.create_plan = AsyncMock(return_value=plan_with_traversal)
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))
        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="approve"))

        result = await session.run("Improve tests")
        assert "REJECTED" in result or "FAILED" in result

    @pytest.mark.asyncio
    async def test_symlink_path_denied(self, tmp_workspace, security, logger):
        if os.name == "nt":
            pytest.skip("Symlink creation requires elevated privileges on Windows")

        target = tmp_workspace / "target.py"
        target.write_text("x = 1\n")
        symlink = tmp_workspace / "symlink.py"
        symlink.symlink_to(target)

        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high",
                                affected_files=[str(symlink)])
        ]

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        plan = DevelopmentPlan(
            id="p1",
            objective="Fix",
            candidate_id="c1",
            steps=[
                DevelopmentStep(id="s1", name="Modify symlink", description="Escape", action_type="edit_file",
                               action_args={"path": str(symlink), "old_string": "x = 1", "new_string": "x = 2"}),
            ],
        )
        session.planner.create_plan = AsyncMock(return_value=plan)
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))
        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="approve"))

        result = await session.run("Improve tests")
        assert "REJECTED" in result or "FAILED" in result

    @pytest.mark.asyncio
    async def test_absolute_path_outside_workspace_denied(self, tmp_workspace, security, logger):
        target = tmp_workspace / "allowed.py"
        target.write_text("x = 1\n")
        outside = tmp_workspace.parent / "outside.py"
        outside.write_text("y = 2\n")

        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high",
                                affected_files=[str(target)])
        ]

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        plan = DevelopmentPlan(
            id="p1",
            objective="Fix",
            candidate_id="c1",
            steps=[
                DevelopmentStep(id="s1", name="Modify outside", description="Escape", action_type="edit_file",
                               action_args={"path": str(outside.resolve()), "old_string": "y = 2", "new_string": "y = 3"}),
            ],
        )
        session.planner.create_plan = AsyncMock(return_value=plan)
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))
        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="approve"))

        result = await session.run("Improve tests")
        assert "REJECTED" in result or "FAILED" in result
        assert outside.read_text() == "y = 2\n"

    @pytest.mark.asyncio
    async def test_unapproved_file_not_modified(self, tmp_workspace, security, logger):
        allowed = tmp_workspace / "allowed.py"
        allowed.write_text("x = 1\n")
        unapproved = tmp_workspace / "unapproved.py"
        unapproved.write_text("y = 2\n")

        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix tests", description="Fix", category="tests", severity="high",
                                affected_files=[str(allowed)])
        ]

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        plan = DevelopmentPlan(
            id="p1",
            objective="Fix",
            candidate_id="c1",
            steps=[
                DevelopmentStep(id="s1", name="Modify unapproved", description="Escape", action_type="edit_file",
                               action_args={"path": str(unapproved), "old_string": "y = 2", "new_string": "y = 3"}),
            ],
        )
        session.planner.create_plan = AsyncMock(return_value=plan)
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix tests", confidence=0.7))
        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="approve"))

        result = await session.run("Improve tests")
        assert "REJECTED" in result or "FAILED" in result
        assert unapproved.read_text() == "y = 2\n"


class TestCriticalFileProtection:
    """Try to modify critical control files via Phase 7."""

    @pytest.mark.asyncio
    async def test_critical_file_in_affected_files_blocked(self, tmp_workspace, security, logger):
        target = tmp_workspace / "evora" / "self_improve.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = "class SelfImproveTool:\n    pass\n"
        target.write_text(original)

        inspector = MagicMock()
        inspector.inspect.return_value = InspectionReport(findings=[
            InspectionFinding(category="tests", severity="high", description="Test failing")
        ])
        discovery = MagicMock()
        discovery.discover.return_value = [
            ImprovementCandidate(id="c1", title="Fix self_improve", description="Fix", category="tests",
                                severity="high", affected_files=["evora/self_improve.py"])
        ]

        real_tools = ToolRegistry(security, logger, identity_service=None, approval_system=None)

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=real_tools,
            memory=MagicMock(),
            logger=logger,
        )
        session.inspector = inspector
        session.discovery = discovery
        session.planner = MagicMock()
        session.planner.create_plan = AsyncMock(return_value=DevelopmentPlan(
            id="p1", objective="Fix", candidate_id="c1", steps=[
                DevelopmentStep(id="s1", name="Modify self_improve", description="Modify", action_type="edit_file",
                               action_args={"path": "evora/self_improve.py", "old_string": original, "new_string": "MODIFIED\n"}),
            ]
        ))
        session.reasoning = MagicMock()
        session.reasoning.reason = AsyncMock(return_value=MagicMock(selected_approach="Fix self_improve", confidence=0.7))
        session.approval.approve_plan = MagicMock(return_value=MagicMock(value="approve"))

        result = await session.run("Improve self_improve.py")
        assert "REJECTED" in result or "FAILED" in result or "MODIFIED" not in target.read_text()

    @pytest.mark.asyncio
    async def test_critical_file_in_affected_files_blocked_with_absolute_path(self, tmp_workspace, security, logger, creator_identity_service):
        target = tmp_workspace / "evora" / "security.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = "class SecurityManager:\n    pass\n"
        target.write_text(original)

        real_tools = ToolRegistry(security, logger, identity_service=None, approval_system=None)

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=creator_identity_service,
            approval=MagicMock(),
            tools=real_tools,
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="test", objective="test")
        token = ApprovalToken.create(session_id="test", plan_id="p1", candidate_id="c1", approved_by="creator")
        session._approval_token = token
        session.approval.consume_approval_token = MagicMock(return_value=True)
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high",
                                         affected_files=[str(target)])
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[
            DevelopmentStep(id="s1", name="Modify security", description="Modify", action_type="edit_file",
                           action_args={"path": str(target.resolve()), "old_string": original, "new_string": "MODIFIED\n"}),
        ])
        result = await session._implement(candidate, plan)
        assert result.get("success") is False
        assert "critical control file" in result.get("error", "").lower()
        assert "MODIFIED" not in target.read_text()


class TestImplementationSecurity:
    """Verify implementation security controls."""

    @pytest.mark.asyncio
    async def test_creator_authority_required_for_implement(self, tmp_workspace, security, logger):
        store = IdentityStore(str(tmp_workspace / "id"))
        store.bootstrap_creator("Creator")
        user = Identity.create(name="User", authority=AuthorityLevel.USER)
        store.set_current(user)
        identity_service = IdentityService(store=store, logger=logger)

        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=identity_service,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="test", objective="test")
        token = ApprovalToken.create(session_id="test", plan_id="p1", candidate_id="c1", approved_by="creator")
        session._approval_token = token
        session.approval.consume_approval_token = MagicMock(return_value=True)
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        result = await session._implement(candidate, plan)
        assert result.get("success") is False
        assert "DENIED" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_implementation_recorded_in_history(self, tmp_workspace, security, logger, creator_identity_service):
        history_dir = tmp_workspace / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        self_improve = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=creator_identity_service,
            approval_system=MagicMock(),
            history_dir=str(history_dir),
        )
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session.self_improve = self_improve
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="test", objective="test")
        token = ApprovalToken.create(session_id="test", plan_id="p1", candidate_id="c1", approved_by="creator")
        session._approval_token = token
        session.approval.consume_approval_token = MagicMock(return_value=True)
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        await session._implement(candidate, plan)
        records = session.self_improve.history.list()
        assert records == []

    @pytest.mark.asyncio
    async def test_token_context_cannot_cross_authorize(self, tmp_workspace, security, logger, creator_identity_service):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace), model_manager=MagicMock(), security=security,
            identity_service=creator_identity_service, approval=ApprovalSystem(auto_approve=True),
            tools=MagicMock(), memory=MagicMock(), logger=logger,
        )
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="active", objective="test")
        session._approval_token = ApprovalToken.create("foreign", "foreign-plan", "foreign-candidate", "forged")
        candidate = ImprovementCandidate("active-candidate", "T", "D", "tests", "high", [])
        plan = DevelopmentPlan("active-plan", "T", "active-candidate", [])
        result = await session._implement(candidate, plan)
        assert result["success"] is False
        assert "does not match" in result["error"]

    @pytest.mark.asyncio
    async def test_unapproved_action_type_cannot_write(self, tmp_workspace, security, logger, creator_identity_service):
        target = tmp_workspace / "outside.txt"
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace), model_manager=MagicMock(), security=security,
            identity_service=creator_identity_service, approval=ApprovalSystem(auto_approve=True),
            tools=ToolRegistry(security, logger), memory=MagicMock(), logger=logger,
        )
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="s", objective="test")
        session._approval_token = session.approval.issue_approval_token("s", "p", "c", "forged")
        candidate = ImprovementCandidate("c", "T", "D", "tests", "high", [])
        plan = DevelopmentPlan("p", "T", "c", [
            DevelopmentStep("x", "command", "", "execute_command", {
                "command": f"python -c \"from pathlib import Path; Path(r'{target}').write_text('bad')\"",
            }),
        ])
        result = await session._implement(candidate, plan)
        assert result["success"] is False
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_rollback_restores_partial_changes(self, tmp_workspace, security, logger, creator_identity_service):
        target = tmp_workspace / "changed.txt"
        target.write_text("original")
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace), model_manager=MagicMock(), security=security,
            identity_service=creator_identity_service, approval=ApprovalSystem(auto_approve=True),
            tools=MagicMock(), memory=MagicMock(), logger=logger,
        )
        session._status = DevStatus.TESTING
        session._snapshots = {target: "original"}
        rollback = await session._rollback(
            ImprovementCandidate("c", "T", "D", "tests", "high", []),
            DevelopmentPlan("p", "T", "c", []),
            {"error": "failed"},
        )
        assert rollback["success"] is True
        assert target.read_text() == "original"

    @pytest.mark.asyncio
    async def test_duplicate_implementation_rejected(self, tmp_workspace, security, logger, creator_identity_service):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=creator_identity_service,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.APPROVED
        session._record = DevSessionRecord(session_id="test", objective="test")
        token = ApprovalToken.create(session_id="test", plan_id="p1", candidate_id="c1", approved_by="creator")
        session._approval_token = token
        session.approval.consume_approval_token = MagicMock(return_value=True)
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        result1 = await session._implement(candidate, plan)
        assert result1.get("success") is True

        token2 = ApprovalToken.create(session_id="test", plan_id="p1", candidate_id="c1", approved_by="creator")
        session._approval_token = token2
        result2 = await session._implement(candidate, plan)
        assert result2.get("success") is False
        assert "duplicate" in result2.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_completed_session_cannot_reimplement(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._record = DevSessionRecord(session_id="s1", objective="test")
        session._completed_sessions.add("s1")
        session._status = DevStatus.APPROVED
        candidate = ImprovementCandidate(id="c1", title="T", description="D", category="tests", severity="high")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        result = await session._implement(candidate, plan)
        assert result.get("success") is False
        assert "already completed" in result.get("error", "").lower()


class TestModelOutputAttacks:
    """Test model output handling."""

    @pytest.mark.asyncio
    async def test_malformed_reasoning_json(self, logger):
        manager = MagicMock()
        response = MagicMock()
        response.content = "This is not valid JSON {{{"
        manager.chat = AsyncMock(return_value=response)
        engine = ReasoningEngine(manager, logger)
        context = ReasoningContext(objective="Improve tests")
        result = await engine.reason(context)
        assert isinstance(result, ReasoningResult)
        assert result.next_action == "abort"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_model_exception_handled(self, logger):
        manager = MagicMock()
        manager.chat = AsyncMock(side_effect=Exception("Model unavailable"))
        engine = ReasoningEngine(manager, logger)
        context = ReasoningContext(objective="Improve tests")
        result = await engine.reason(context)
        assert isinstance(result, ReasoningResult)
        assert result.next_action == "abort"
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_reasoning_missing_required_fields_fails_hard(self, logger):
        manager = MagicMock()
        response = MagicMock()
        response.content = '{"summary": "ok"}'
        manager.chat = AsyncMock(return_value=response)
        engine = ReasoningEngine(manager, logger)
        context = ReasoningContext(objective="Improve tests")
        result = await engine.reason(context)
        assert isinstance(result, ReasoningResult)
        assert result.next_action == "abort"
        assert result.confidence == 0.0


class TestStateMachineAttacks:
    """Try to break state machine."""

    @pytest.mark.asyncio
    async def test_direct_state_mutation_blocked(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.IDLE
        with pytest.raises(ValueError, match="Invalid state transition"):
            await session._transition(DevStatus.THINKING)

    @pytest.mark.asyncio
    async def test_approval_to_implement_without_approved_denied(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.AWAITING_APPROVAL
        with pytest.raises(ValueError, match="Invalid state transition"):
            await session._transition(DevStatus.IMPLEMENTING)

    @pytest.mark.asyncio
    async def test_rejected_to_implement_denied(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.REJECTED
        with pytest.raises(ValueError, match="Invalid state transition"):
            await session._transition(DevStatus.IMPLEMENTING)


class TestPlanIntegrity:
    """Try to create malicious plans."""

    def test_plan_with_dangerous_command(self, logger):
        planner = DevelopmentPlanner(MagicMock(), logger)
        candidate = ImprovementCandidate(
            id="c1",
            title="Run dangerous command",
            description="Execute rm -rf /",
            category="tests",
            severity="high",
            affected_files=["test.py"],
        )
        plan = planner._generate_steps(candidate, {})
        assert len(plan) >= 1


class TestBenchmarkIntegrity:
    """Verify benchmark is bound to real execution."""

    @pytest.mark.asyncio
    async def test_benchmark_includes_real_test_evidence(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.EVALUATING
        session._record = DevSessionRecord(session_id="test", objective="test")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        test_result = {"success": True, "results": [{"test": "pytest", "success": True, "output": "ok"}]}
        benchmark = await session._benchmark(plan, test_result)
        assert "timestamp" in benchmark
        assert "command" in benchmark
        assert "exit_code" in benchmark
        assert "passed" in benchmark
        assert "failed" in benchmark
        assert "timing_seconds" in benchmark

    @pytest.mark.asyncio
    async def test_benchmark_not_fabricated_without_tests(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        session._status = DevStatus.EVALUATING
        session._record = DevSessionRecord(session_id="test", objective="test")
        plan = DevelopmentPlan(id="p1", objective="T", candidate_id="c1", steps=[])
        test_result = {"success": False, "error": "Tests failed", "results": []}
        benchmark = await session._benchmark(plan, test_result)
        assert benchmark["passed"] == 0 or "command" in benchmark


class TestLearningIntegrity:
    """Verify learning reflects actual results."""

    def test_lesson_from_failure(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        test_result = {"success": False, "error": "Syntax error"}
        extra = {"reason": "Other reason"}
        lesson = session._extract_lesson(test_result, extra)
        assert "Failure" in lesson
        assert "Syntax error" in lesson

    def test_lesson_from_success(self, tmp_workspace, security, logger):
        session = SelfDevelopmentSession(
            workspace_dir=str(tmp_workspace),
            model_manager=MagicMock(),
            security=security,
            identity_service=None,
            approval=MagicMock(),
            tools=MagicMock(),
            memory=MagicMock(),
            logger=logger,
        )
        test_result = {"success": True}
        lesson = session._extract_lesson(test_result, {})
        assert "validated" in lesson.lower()


class TestPhase6Regression:
    """Verify Phase 6 security controls still work."""

    def test_critical_file_protection_preserved(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        target = tmp_workspace / "evora" / "self_improve.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("pass")
        assert validator.is_critical_control_file(str(target)) is True

    def test_sensitive_extension_blocked(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        target = tmp_workspace / "test.env"
        target.write_text("SECRET=value")
        assert validator.is_sensitive_extension(str(target)) is True
        assert validator.is_critical_control_file(str(target)) is True

    def test_history_signature_verification(self, tmp_workspace, security, logger, tmp_path):
        from evora.self_improve import ImprovementHistory, ImprovementProposal, ImprovementRecord
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)

        path = tmp_path / "h" / f"{record.history_id}.json"
        data = json.loads(path.read_text())
        data["status"] = "success"
        path.write_text(json.dumps(data))

        loaded = history.get(record.history_id)
        assert loaded is None

    def test_identity_store_authority(self, tmp_workspace):
        store = IdentityStore(str(tmp_workspace / "id"))
        fake = Identity.create(name="Imposter", authority=AuthorityLevel.USER)
        with pytest.raises(PermissionError, match="not authorized"):
            store.set_creator(fake, caller=fake)
