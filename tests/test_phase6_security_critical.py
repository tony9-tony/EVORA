"""
Deep critical file tests — can EVORA modify its own control files?
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evora.self_improve import SelfImproveTool, ImprovementStatus
from evora.security import PermissionManager
from evora.logger import Logger


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def security(tmp_workspace):
    return PermissionManager(str(tmp_workspace), allow_file_write=True, allow_cmd_exec=True)


@pytest.fixture
def logger():
    return Logger("evora-test-critical", "info", None)


class TestCriticalFileModification:
    """Critical control files must be rejected by self-improvement."""

    @pytest.mark.asyncio
    async def test_modify_self_improve_py_blocked(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "evora" / "self_improve.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("class SelfImproveTool:\n    pass\n")

        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="class SelfImproveTool:\n    pass\n",
            new_string="class SelfImproveTool:\n    # MODIFIED\n    pass\n",
        )
        assert result.success is False
        assert "critical control file" in result.error.lower()
        assert "MODIFIED" not in target.read_text()

    @pytest.mark.asyncio
    async def test_modify_identity_py_blocked(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "evora" / "identity.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("class IdentityStore:\n    pass\n")

        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="class IdentityStore:\n    pass\n",
            new_string="class IdentityStore:\n    # COMPROMISED\n    pass\n",
        )
        assert result.success is False
        assert "critical control file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_modify_approval_py_blocked(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "evora" / "approval.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("class ApprovalSystem:\n    pass\n")

        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="class ApprovalSystem:\n    pass\n",
            new_string="class ApprovalSystem:\n    # BYPASSED\n    pass\n",
        )
        assert result.success is False
        assert "critical control file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_modify_security_py_blocked(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "evora" / "security.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("class PermissionManager:\n    pass\n")

        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="class PermissionManager:\n    pass\n",
            new_string="class PermissionManager:\n    # DISABLED\n    pass\n",
        )
        assert result.success is False
        assert "critical control file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_modify_creator_json_blocked(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=None,
            approval_system=approval,
            history_dir=str(tmp_path / "h"),
        )
        target = tmp_workspace / "identities" / "creator.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = '{"id": "c1", "name": "creator", "authority": "creator"}'
        target.write_text(original)

        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string='"authority": "creator"',
            new_string='"authority": "user"',
        )
        assert result.success is False
        assert "critical control file" in result.error.lower()
        assert '"authority": "creator"' in target.read_text()
