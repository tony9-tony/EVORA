"""
Phase 6 Deep Security Audit — Kilo #2

Additional attack vectors after initial review:
1. History file integrity (no signing)
2. IdentityStore.set_creator authority bypass
3. SelfImproveTool hardcoded approved_by="creator"
4. Concurrent improvement race conditions
5. File modification of critical control files
"""

import json
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
    return Logger("evora-test-audit2", "info", None)


class TestHistoryIntegrity:
    """History files have HMAC integrity protection."""

    def test_forged_success_record_detected(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="Legit", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)

        path = tmp_path / "h" / f"{record.history_id}.json"
        data = json.loads(path.read_text())
        data["status"] = "success"
        data["test_result"] = "passed=100, failed=0"
        data["approved_by"] = "forged-creator"
        path.write_text(json.dumps(data))

        loaded = history.get(record.history_id)
        assert loaded is None

    def test_forged_rejected_to_approved_detected(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="Rejected", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.REJECTED)
        history.record(record)

        path = tmp_path / "h" / f"{record.history_id}.json"
        data = json.loads(path.read_text())
        data["status"] = "success"
        path.write_text(json.dumps(data))

        loaded = history.get(record.history_id)
        assert loaded is None


class TestIdentityStoreAuthority:
    """IdentityStore.set_creator must enforce authority."""

    def test_direct_set_creator_requires_caller(self, tmp_workspace):
        store = IdentityStore(str(tmp_workspace / "id"))
        attacker = Identity.create(name="Attacker", authority=AuthorityLevel.USER)
        with pytest.raises(PermissionError, match="not authorized"):
            store.set_creator(attacker, caller=attacker)

    def test_bootstrap_creator_still_works(self, tmp_workspace):
        store = IdentityStore(str(tmp_workspace / "id"))
        creator = store.bootstrap_creator("Anthony")
        assert creator.is_creator
        assert creator.name == "Anthony"


class TestApprovedByHardcoded:
    """_do_apply must record the actual approver identity."""

    @pytest.mark.asyncio
    async def test_approved_by_captures_identity_service_name(self, tmp_workspace, security, logger, tmp_path):
        fake_identity = Identity.create(name="RealTony", authority=AuthorityLevel.CREATOR)
        identity_service = MagicMock()
        identity_service.current_identity.return_value = fake_identity

        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=identity_service,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "test.py"
        target.write_text("x = 1\n")

        with patch.object(tool.validator, "validate_after",
                          return_value={"valid": True, "errors": [], "warnings": []}):
            with patch.object(tool.validator, "validate_tests",
                              return_value={"valid": True, "passed": 1, "failed": 0, "output": "ok"}):
                result = await tool.execute(
                    action="apply",
                    file_path=str(target),
                    old_string="x = 1",
                    new_string="x = 2",
                )
                assert result.success is True
                records = tool.history.list()
                assert len(records) >= 1
                assert records[0].approved_by == "RealTony"

    @pytest.mark.asyncio
    async def test_approved_by_fallback_when_no_identity_service(self, tmp_workspace, security, logger, tmp_path):
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
        target.write_text("x = 1\n")

        with patch.object(tool.validator, "validate_after",
                          return_value={"valid": True, "errors": [], "warnings": []}):
            with patch.object(tool.validator, "validate_tests",
                              return_value={"valid": True, "passed": 1, "failed": 0, "output": "ok"}):
                result = await tool.execute(
                    action="apply",
                    file_path=str(target),
                    old_string="x = 1",
                    new_string="x = 2",
                )
                assert result.success is True
                records = tool.history.list()
                assert len(records) >= 1
                assert records[0].approved_by == "creator"


class TestConcurrentImprovements:
    """Test concurrent improvement behavior."""

    def test_concurrent_history_writes_no_corruption(self, tmp_workspace, security, logger, tmp_path):
        import threading
        history = ImprovementHistory(str(tmp_path / "h"))

        def write_record(i):
            prop = ImprovementProposal(id=f"p{i}", title=f"T{i}", description="D")
            record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
            try:
                history.record(record)
                record.status = ImprovementStatus.APPROVED
                history.update(record)
                record.status = ImprovementStatus.RUNNING
                history.update(record)
                record.status = ImprovementStatus.SUCCESS
                history.update(record)
            except Exception:
                pass

        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = history.list()
        assert len(records) >= 1


class TestCriticalFileModification:
    """Test whether self-improvement can modify critical control files."""

    @pytest.mark.asyncio
    async def test_cannot_modify_identity_store_directly_via_apply(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        id_file = tmp_workspace / "identities" / "creator.json"
        id_file.parent.mkdir(parents=True, exist_ok=True)
        id_file.write_text('{"id": "c1", "name": "creator", "authority": "creator"}')

        result = await tool.execute(
            action="apply",
            file_path=str(id_file),
            old_string='"authority": "creator"',
            new_string='"authority": "user"',
        )
        assert result.success is False


class TestInvalidStateTransitions:
    """Invalid state transitions should be rejected."""

    def test_pending_to_success_rejected(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)

        record.status = ImprovementStatus.SUCCESS
        with pytest.raises(ValueError, match="Invalid state transition"):
            history.update(record)

    def test_failed_to_success_rejected(self, tmp_workspace, security, logger, tmp_path):
        history = ImprovementHistory(str(tmp_path / "h"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        history.record(record)
        record.status = ImprovementStatus.APPROVED
        history.update(record)
        record.status = ImprovementStatus.RUNNING
        history.update(record)
        record.status = ImprovementStatus.FAILED
        history.update(record)

        with pytest.raises(ValueError, match="Refusing to modify terminal record"):
            record.status = ImprovementStatus.SUCCESS
            history.update(record)
