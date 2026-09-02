"""
Phase 6 Security Attack Tests — Kilo #2 Independent Auditor

These tests attempt to break the Phase 6 self-improvement system.
Each test documents the attack vector and expected defense.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evora.self_improve import (
    ImprovementStatus,
    ImprovementProposal,
    ImprovementRecord,
    ImprovementHistory,
    ChangeValidator,
    ImprovementPlanner,
    SelfImproveTool,
)
from evora.security import PermissionManager
from evora.logger import Logger
from evora.identity import IdentityService, Identity, AuthorityLevel, IdentityStore


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def security(tmp_workspace):
    return PermissionManager(str(tmp_workspace), allow_file_write=True, allow_cmd_exec=True)


@pytest.fixture
def logger():
    return Logger("evora-test-audit", "info", None)


class TestApprovalBypass:
    """Try to apply changes without legitimate approval."""

    @pytest.mark.asyncio
    async def test_apply_without_approval_system_denied(self, tmp_workspace, security, logger, tmp_path):
        """No approval_system → apply must fail."""
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=None,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        target.write_text("x = 1\n")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="x = 1", new_string="x = 2")
        assert result.success is False
        assert "No approval system" in result.error

    @pytest.mark.asyncio
    async def test_apply_with_rejected_decision_denied(self, tmp_workspace, security, logger, tmp_path):
        """Explicit reject → apply must fail."""
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="reject")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        target.write_text("x = 1\n")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="x = 1", new_string="x = 2")
        assert result.success is False
        assert "rejected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_apply_with_modify_decision_denied(self, tmp_workspace, security, logger, tmp_path):
        """MODIFY is not APPROVE → apply must fail."""
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="modify")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        target.write_text("x = 1\n")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="x = 1", new_string="x = 2")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_apply_with_cancel_decision_denied(self, tmp_workspace, security, logger, tmp_path):
        """CANCEL is not APPROVE → apply must fail."""
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="cancel")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        target.write_text("x = 1\n")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="x = 1", new_string="x = 2")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_apply_with_explain_decision_denied(self, tmp_workspace, security, logger, tmp_path):
        """EXPLAIN is not APPROVE → apply must fail."""
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="explain")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        target.write_text("x = 1\n")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="x = 1", new_string="x = 2")
        assert result.success is False


class TestWorkspaceBoundary:
    """Try to escape the workspace boundary."""

    @pytest.mark.asyncio
    async def test_path_traversal_parent_dir_denied(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        result = await tool.execute(
            action="apply",
            file_path="../escape.txt",
            old_string="x",
            new_string="y",
        )
        assert result.success is False
        assert "Pre-validation failed" in result.error or "outside the workspace" in result.error

    @pytest.mark.asyncio
    async def test_absolute_path_outside_workspace_denied(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        outside = Path(tempfile.gettempdir()) / "evil_escape.txt"
        result = await tool.execute(
            action="apply",
            file_path=str(outside),
            old_string="x",
            new_string="y",
        )
        assert result.success is False
        assert "Pre-validation failed" in result.error or "outside the workspace" in result.error

    @pytest.mark.asyncio
    async def test_windows_drive_letter_escape(self, tmp_workspace, security, logger, tmp_path):
        if os.name != "nt":
            pytest.skip("Windows-specific test")
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        result = await tool.execute(
            action="apply",
            file_path="C:\\Windows\\system.ini",
            old_string="x",
            new_string="y",
        )
        assert result.success is False
        assert "Pre-validation failed" in result.error or "outside the workspace" in result.error


class TestStateManipulation:
    """Try to manipulate improvement state to bypass checks."""

    def test_direct_success_record_rejected(self, tmp_workspace, security, logger, tmp_path):
        """Directly recording SUCCESS should be rejected."""
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.SUCCESS)
        with pytest.raises(ValueError, match="Cannot directly record terminal state"):
            history.record(record)

    def test_rejected_record_persists_correctly(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.REJECTED)
        history.record(record)
        loaded = history.get(record.history_id)
        assert loaded.status == ImprovementStatus.REJECTED

    def test_manual_status_override_detected(self, tmp_workspace, security, logger, tmp_path):
        """Manually modifying JSON should break signature and be rejected."""
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)

        path = tmp_path / "h" / f"{record.history_id}.json"
        data = json.loads(path.read_text())
        data["status"] = "success"
        data["test_result"] = "passed=999, failed=0"
        path.write_text(json.dumps(data))

        loaded = history.get(record.history_id)
        assert loaded is None


class TestCreatorImpersonation:
    """Try to impersonate the creator identity."""

    def test_bootstrap_creator_twice_denied(self, tmp_workspace):
        store = IdentityStore(str(tmp_workspace / "id"))
        id1 = store.bootstrap_creator("Anthony")
        assert id1.is_creator
        with pytest.raises(PermissionError):
            store.bootstrap_creator("FakeCreator")

    def test_set_creator_without_is_creator_flag(self, tmp_workspace):
        store = IdentityStore(str(tmp_workspace / "id"))
        fake = Identity.create(name="Imposter", authority=AuthorityLevel.USER)
        store.set_creator(fake)
        loaded = store.get_creator()
        assert loaded is not None
        assert loaded.is_creator
        assert loaded.authority == AuthorityLevel.CREATOR

    def test_identity_by_username_not_hardcoded(self, tmp_workspace):
        store = IdentityStore(str(tmp_workspace / "id"))
        store.bootstrap_creator("NotTony")
        creator = store.get_creator()
        assert creator.is_creator
        assert creator.name == "NotTony"


class TestHistoryTampering:
    """History files have integrity protection."""

    def test_history_files_include_signature(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)

        path = tmp_path / "h" / f"{record.history_id}.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "signature" in data
        assert data["signature"] is not None

    def test_terminal_record_cannot_be_overwritten(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)

        record.status = ImprovementStatus.APPROVED
        history.update(record)
        record.status = ImprovementStatus.RUNNING
        history.update(record)
        record.status = ImprovementStatus.SUCCESS
        history.update(record)

        loaded = history.get(record.history_id)
        assert loaded.status == ImprovementStatus.SUCCESS

        with pytest.raises(ValueError, match="Refusing to modify terminal record"):
            loaded.status = ImprovementStatus.FAILED
            history.update(loaded)


class TestSecretScanning:
    """Test that secrets are caught in proposed content."""

    def test_sk_key_detected(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        found = validator.contains_secrets('key = "sk-abc123def456ghi789jkl012mno345pqr678"')
        assert len(found) >= 1

    def test_api_key_assignment_detected(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        found = validator.contains_secrets('API_KEY = "my-secret-key-12345"')
        assert len(found) >= 1

    def test_password_detected(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        found = validator.contains_secrets('password = "SuperSecret123!"')
        assert len(found) >= 1

    def test_token_detected(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        found = validator.contains_secrets('token = "ghp_abc123def456ghi789jkl012mno345"')
        assert len(found) >= 1

    def test_clean_code_not_flagged(self, tmp_workspace, security, logger):
        validator = ChangeValidator(str(tmp_workspace), security, logger)
        found = validator.contains_secrets("def foo():\n    return 1\n")
        assert len(found) == 0


class TestRollbackBehavior:
    """Verify rollback on validation/test failure."""

    @pytest.mark.asyncio
    async def test_rollback_on_post_validation_failure(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        original = "x = 1\n"
        target.write_text(original)

        with patch.object(tool.validator, "validate_after",
                          return_value={"valid": False, "errors": ["Syntax error"], "warnings": []}):
            result = await tool.execute(
                action="apply",
                file_path=str(target),
                old_string="x = 1",
                new_string="bad syntax(",
            )
            assert result.success is False
            assert "rolled back" in result.error
            assert target.read_text() == original

    @pytest.mark.asyncio
    async def test_rollback_on_test_failure(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        original = "x = 1\n"
        target.write_text(original)

        with patch.object(tool.validator, "validate_after",
                          return_value={"valid": True, "errors": [], "warnings": []}):
            with patch.object(tool.validator, "validate_tests",
                              return_value={"valid": False, "passed": 0, "failed": 1, "output": "FAIL"}):
                result = await tool.execute(
                    action="apply",
                    file_path=str(target),
                    old_string="x = 1",
                    new_string="x = 2",
                )
                assert result.success is False
                assert "rolled back" in result.error
                assert target.read_text() == original
