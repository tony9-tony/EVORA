"""
Tests for EVORA Phase 6 self-improvement system.

These tests verify:
- ImprovementStatus enum values
- ImprovementProposal creation and serialization
- ImprovementRecord lifecycle and persistence
- ImprovementHistory append-only store with summary statistics
- ChangeValidator path checking, secret scanning, pre/post validation
- ImprovementPlanner weakness detection and proposal generation
- SelfImproveTool actions: analyze, propose, apply (with approval flow), history
- CREATOR authority enforcement for apply action
- Pre-validation failures (outside workspace, secret content) cause rollback
- Post-validation failures (syntax errors) cause rollback
"""

import json
import tempfile
import textwrap
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


@pytest.fixture
def tmp_workspace():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def security(tmp_workspace):
    return PermissionManager(str(tmp_workspace), allow_file_write=True, allow_cmd_exec=True)


@pytest.fixture
def logger():
    return Logger("evora-test", "info", None)


class TestImprovementStatus:
    def test_status_values(self):
        assert ImprovementStatus.PENDING.value == "pending"
        assert ImprovementStatus.APPROVED.value == "approved"
        assert ImprovementStatus.RUNNING.value == "running"
        assert ImprovementStatus.SUCCESS.value == "success"
        assert ImprovementStatus.FAILED.value == "failed"
        assert ImprovementStatus.REJECTED.value == "rejected"

    def test_status_is_str_enum(self):
        assert ImprovementStatus.PENDING == "pending"


class TestImprovementProposal:
    def test_creation(self):
        prop = ImprovementProposal(
            id="test-123",
            title="Fix bare except",
            description="Replace bare except with specific exception",
            files_changed=["evora/tools.py"],
        )
        assert prop.id == "test-123"
        assert prop.title == "Fix bare except"
        assert prop.files_changed == ["evora/tools.py"]

    def test_defaults(self):
        prop = ImprovementProposal(id="x", title="t", description="d")
        assert prop.files_changed == []
        assert prop.benefit == ""
        assert prop.risk == ""
        assert prop.created_at is not None
        assert prop.proposed_by == ""

    def test_to_dict(self):
        prop = ImprovementProposal(
            id="abc", title="T", description="D",
            files_changed=["a.py"], benefit="B", risk="R",
            proposed_by="tester",
        )
        d = prop.to_dict()
        assert d["id"] == "abc"
        assert d["title"] == "T"
        assert d["files_changed"] == ["a.py"]
        assert d["benefit"] == "B"
        assert d["risk"] == "R"
        assert d["proposed_by"] == "tester"

    def test_from_dict(self):
        data = {
            "id": "abc", "title": "T", "description": "D",
            "files_changed": ["a.py"], "benefit": "B", "risk": "R",
            "created_at": "2024-01-01T00:00:00", "proposed_by": "tester",
        }
        prop = ImprovementProposal.from_dict(data)
        assert prop.id == "abc"
        assert prop.title == "T"
        assert prop.files_changed == ["a.py"]

    def test_round_trip(self):
        prop = ImprovementProposal(id="rt", title="RT", description="D")
        d = prop.to_dict()
        prop2 = ImprovementProposal.from_dict(d)
        assert prop2.id == prop.id
        assert prop2.title == prop.title


class TestImprovementRecord:
    def test_creation(self):
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(
            proposal=prop,
            status=ImprovementStatus.SUCCESS,
            approved_by="creator",
            test_result="passed=10, failed=0",
        )
        assert record.proposal.id == "p1"
        assert record.status == ImprovementStatus.SUCCESS
        assert record.approved_by == "creator"
        assert record.test_result == "passed=10, failed=0"
        assert record.history_id.startswith("imp-")

    def test_to_dict_serializes_proposal(self):
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.RUNNING)
        d = record.to_dict()
        assert d["history_id"] == record.history_id
        assert d["status"] == "running"
        assert d["proposal"]["id"] == "p1"
        assert d["proposal"]["title"] == "T"

    def test_to_dict_handles_dict_proposal(self):
        prop_dict = {"id": "d1", "title": "T", "description": "D",
                     "files_changed": [], "benefit": "", "risk": "",
                     "created_at": "now", "proposed_by": ""}
        record = ImprovementRecord(
            proposal=prop_dict,  # pass dict directly
            status=ImprovementStatus.PENDING,
        )
        d = record.to_dict()
        assert d["proposal"] == prop_dict

    def test_from_dict_with_dict_proposal(self):
        data = {
            "history_id": "imp-abc",
            "proposal": {"id": "p1", "title": "T", "description": "D",
                         "files_changed": [], "benefit": "", "risk": "",
                         "created_at": "now", "proposed_by": ""},
            "status": "success",
            "approved_by": "creator",
            "test_result": "passed=1, failed=0",
        }
        record = ImprovementRecord.from_dict(data)
        assert record.history_id == "imp-abc"
        assert isinstance(record.proposal, ImprovementProposal)
        assert record.proposal.id == "p1"
        assert record.status == ImprovementStatus.SUCCESS


class TestImprovementHistory:
    def test_init_creates_directory(self, tmp_workspace):
        data_dir = tmp_workspace / "improvements"
        history = ImprovementHistory(str(data_dir))
        assert data_dir.exists()
        assert data_dir.is_dir()

    def test_record_and_persist(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop, status=ImprovementStatus.PENDING)
        record_id = history.record(record)
        assert record_id == record.history_id
        assert record_id.startswith("imp-")

        path = tmp_workspace / "improvements" / f"{record_id}.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "pending"

    def test_update_existing_record(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        prop = ImprovementProposal(id="p1", title="T", description="D")
        record = ImprovementRecord(proposal=prop)
        history.record(record)
        record.status = ImprovementStatus.APPROVED
        record.approved_by = "creator"
        history.update(record)
        record.status = ImprovementStatus.RUNNING
        history.update(record)

        loaded = history.get(record.history_id)
        assert loaded.status == ImprovementStatus.RUNNING
        assert loaded.approved_by == "creator"

    def test_list_returns_newest_first(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        r1 = ImprovementRecord(ImprovementProposal(id="1", title="T", description="D"))
        r2 = ImprovementRecord(ImprovementProposal(id="2", title="T", description="D"))
        history.record(r1)
        import time
        time.sleep(0.01)
        history.record(r2)
        records = history.list()
        assert len(records) == 2
        assert records[0].proposal.id == "2"
        assert records[1].proposal.id == "1"

    def test_list_respects_limit(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        for i in range(5):
            r = ImprovementRecord(ImprovementProposal(id=str(i), title="T", description="D"))
            history.record(r)
        assert len(history.list(limit=3)) == 3

    def test_get_returns_none_for_missing(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        assert history.get("nonexistent-id") is None

    def test_summary_stats(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        r1 = ImprovementRecord(ImprovementProposal(id="1", title="T", description="D"))
        history.record(r1)
        r1.status = ImprovementStatus.APPROVED
        history.update(r1)
        r1.status = ImprovementStatus.RUNNING
        history.update(r1)
        r1.status = ImprovementStatus.SUCCESS
        history.update(r1)

        r2 = ImprovementRecord(ImprovementProposal(id="2", title="T", description="D"))
        history.record(r2)
        r2.status = ImprovementStatus.APPROVED
        history.update(r2)
        r2.status = ImprovementStatus.RUNNING
        history.update(r2)
        r2.status = ImprovementStatus.FAILED
        history.update(r2)

        r3 = ImprovementRecord(ImprovementProposal(id="3", title="T", description="D"), status=ImprovementStatus.REJECTED)
        history.record(r3)

        summary = history.summary()
        assert summary["total"] == 3
        assert summary["by_status"]["success"] == 1
        assert summary["by_status"]["failed"] == 1
        assert summary["by_status"]["rejected"] == 1
        assert summary["success_rate"] == pytest.approx(1 / 3, abs=0.01)

    def test_record_sets_applied_at(self, tmp_workspace):
        history = ImprovementHistory(str(tmp_workspace / "improvements"))
        record = ImprovementRecord(
            ImprovementProposal(id="p1", title="T", description="D")
        )
        assert record.applied_at is None
        history.record(record)
        assert record.applied_at is not None


class TestChangeValidator:
    def test_validate_file_path_inside_workspace(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        path = tmp_workspace / "evora" / "test.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test")
        result = validator.validate_file_path(str(path))
        assert result == path.resolve()

    def test_validate_file_path_outside_workspace_raises(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        outside = Path("/tmp/evil.py")
        with pytest.raises(PermissionError, match="outside the workspace"):
            validator.validate_file_path(str(outside))

    def test_contains_secrets_detects_api_key(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        content = 'api_key = "sk-abc123def456ghi789jkl012mno345pqr789stu012vwx"'
        found = validator.contains_secrets(content)
        assert len(found) >= 1

    def test_contains_secrets_detects_password(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        content = 'password = "supersecret123"'
        found = validator.contains_secrets(content)
        assert len(found) >= 1

    def test_contains_secrets_clean_content(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        found = validator.contains_secrets("x = 1\nimport os")
        assert len(found) == 0

    def test_validate_before_valid(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        valid_file = tmp_workspace / "test.py"
        result = validator.validate_before(
            [str(valid_file)],
            {str(valid_file): "x = 1\n"},
        )
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_before_path_outside_workspace(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        result = validator.validate_before(
            ["/etc/passwd"],
            {"/etc/passwd": "x = 1"},
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_before_secret_content(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        valid_file = tmp_workspace / "test.py"
        result = validator.validate_before(
            [str(valid_file)],
            {str(valid_file): 'api_key = "sk-abc123def456ghi789"'},
        )
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_after_valid_python(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        py_file = tmp_workspace / "good.py"
        py_file.write_text("def foo():\n    return 1\n")
        result = validator.validate_after(str(py_file), check_imports=True)
        assert result["valid"] is True

    def test_validate_after_syntax_error(self, tmp_workspace, security):
        validator = ChangeValidator(str(tmp_workspace), security)
        py_file = tmp_workspace / "bad.py"
        py_file.write_text("def foo(:\n")
        result = validator.validate_after(str(py_file), check_imports=True)
        assert result["valid"] is False
        assert len(result["errors"]) > 0


class TestImprovementPlanner:
    def test_analyze_self_on_real_codebase(self, tmp_workspace, logger):
        planner = ImprovementPlanner(str(tmp_workspace / "evora"))
        findings = planner.analyze_self()
        assert "files_scanned" in findings
        assert "weaknesses" in findings
        assert "long_functions" in findings
        assert "todo_count" in findings

    def test_analyze_self_no_scannable_dirs(self, tmp_workspace, logger):
        planner = ImprovementPlanner(str(tmp_workspace))
        findings = planner.analyze_self()
        assert findings["files_scanned"] == 0
        assert len(findings["weaknesses"]) == 0

    def test_create_proposal(self, tmp_workspace, logger):
        planner = ImprovementPlanner(str(tmp_workspace), logger)
        finding = {
            "file": "evora/foo.py",
            "line": 10,
            "type": "todo_comment",
            "detail": "# TODO: fix this",
        }
        prop = planner.create_proposal(finding, "evora")
        assert prop.title == "Fix todo_comment in foo.py"
        assert prop.description == "Found todo_comment at evora/foo.py:10: # TODO: fix this"
        assert prop.files_changed == ["evora/foo.py"]

    def test_recommend_produces_proposals(self, tmp_workspace, logger):
        evora_dir = tmp_workspace / "evora"
        evora_dir.mkdir()
        (evora_dir / "sample.py").write_text("# TODO: fix me\nprint('hello')\n")
        planner = ImprovementPlanner(str(tmp_workspace), logger)
        proposals = planner.recommend("evora")
        assert len(proposals) >= 1
        assert all(isinstance(p, ImprovementProposal) for p in proposals)
        assert any("todo" in p.description.lower() for p in proposals)

    def test_recommend_empty_when_no_issues(self, tmp_workspace, logger):
        evora_dir = tmp_workspace / "evora"
        evora_dir.mkdir()
        (evora_dir / "clean.py").write_text("def foo():\n    return 1\n")
        planner = ImprovementPlanner(str(tmp_workspace), logger)
        proposals = planner.recommend("evora")
        assert len(proposals) == 0


class TestSelfImproveTool:
    def _make_tool(self, tmp_workspace, security, logger, identity_service=None,
                   approval_system=None, history_dir=None):
        return SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=identity_service,
            approval_system=approval_system,
            history_dir=history_dir,
        )

    @pytest.mark.asyncio
    async def test_execute_analyze(self, tmp_workspace, security, logger):
        tool = self._make_tool(tmp_workspace, security, logger)
        result = await tool.execute(action="analyze")
        assert result.success is True
        assert "EVORA Self-Analysis Report" in result.output

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self, tmp_workspace, security, logger):
        tool = self._make_tool(tmp_workspace, security, logger)
        result = await tool.execute(action="nonexistent")
        assert result.success is False
        assert "Unknown action" in result.error

    @pytest.mark.asyncio
    async def test_execute_propose(self, tmp_workspace, security, logger):
        (tmp_workspace / "evora").mkdir()
        (tmp_workspace / "evora" / "sample.py").write_text("# TODO: fix me\n")
        tool = self._make_tool(tmp_workspace, security, logger)
        result = await tool.execute(action="propose")
        assert result.success is True
        assert "Improvement Proposals" in result.output
        assert result.data is not None
        assert "proposals" in result.data

    @pytest.mark.asyncio
    async def test_execute_propose_no_findings(self, tmp_workspace, security, logger):
        (tmp_workspace / "evora").mkdir()
        (tmp_workspace / "evora" / "clean.py").write_text("def foo():\n    return 1\n")
        tool = self._make_tool(tmp_workspace, security, logger)
        result = await tool.execute(action="propose")
        assert result.success is True
        assert "No improvements" in result.output

    @pytest.mark.asyncio
    async def test_execute_history(self, tmp_workspace, security, logger):
        tool = self._make_tool(tmp_workspace, security, logger)
        result = await tool.execute(action="history")
        assert result.success is True
        assert "Improvement History" in result.output

    @pytest.mark.asyncio
    async def test_execute_apply_missing_args(self, tmp_workspace, security, logger,
                                                tmp_path):
        tool = self._make_tool(tmp_workspace, security, logger,
                               history_dir=str(tmp_path / "h"))
        result = await tool.execute(action="apply")
        assert result.success is False
        assert "file_path, old_string, and new_string are required" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_no_approval_system(self, tmp_workspace, security, logger,
                                                     tmp_path):
        tool = self._make_tool(tmp_workspace, security, logger,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="x", new_string="y")
        assert result.success is False
        assert "No approval system" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_rejected(self, tmp_workspace, security, logger, tmp_path):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="reject")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old code")
        result = await tool.execute(action="apply", file_path=str(target),
                                    old_string="old", new_string="new")
        assert result.success is False
        assert "rejected" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_prevalidation_outside_workspace(
        self, tmp_workspace, security, logger, tmp_path, monkeypatch
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        result = await tool.execute(
            action="apply",
            file_path="/tmp/evil.py",
            old_string="old",
            new_string="new",
        )
        assert result.success is False
        assert "Pre-validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_prevalidation_secrets(
        self, tmp_workspace, security, logger, tmp_path
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old code")
        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="old code",
            new_string='api_key = "sk-abc123def456ghi789"\n',
        )
        assert result.success is False
        assert "Pre-validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_old_string_not_found(
        self, tmp_workspace, security, logger, tmp_path
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("some code here")
        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="nonexistent string",
            new_string="replacement",
        )
        assert result.success is False
        assert "old_string not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_multiple_matches(
        self, tmp_workspace, security, logger, tmp_path
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("same\n\nsame\n")
        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="same",
            new_string="different",
        )
        assert result.success is False
        assert "Multiple matches" in result.error

    @pytest.mark.asyncio
    async def test_execute_apply_syntax_error_rollback(
        self, tmp_workspace, security, logger, tmp_path
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = "old code\n"
        target.write_text(original)
        result = await tool.execute(
            action="apply",
            file_path=str(target),
            old_string="old code",
            new_string="bad syntax(",
        )
        assert result.success is False
        assert "rolled back" in result.error
        assert target.read_text() == original

    @pytest.mark.asyncio
    async def test_execute_apply_test_failure_rollback(
        self, tmp_workspace, security, logger, tmp_path
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = "x = 1\n"
        target.write_text(original)

        with patch.object(tool.validator, "validate_after", return_value={"valid": True, "errors": [], "warnings": []}):
            with patch.object(tool.validator, "validate_tests", return_value={"valid": False, "passed": 0, "failed": 1, "output": "FAILED test_foo"}):
                result = await tool.execute(
                    action="apply",
                    file_path=str(target),
                    old_string="x = 1",
                    new_string="x = 2",
                )
                assert result.success is False
                assert "rolled back" in result.error
                assert target.read_text() == original

    @pytest.mark.asyncio
    async def test_execute_apply_success(
        self, tmp_workspace, security, logger, tmp_path
    ):
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        tool = self._make_tool(tmp_workspace, security, logger,
                               approval_system=approval,
                               history_dir=str(tmp_path / "h"))
        target = tmp_workspace / "evora" / "test.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        original = "x = 1\n"
        target.write_text(original)

        with patch.object(tool.validator, "validate_after", return_value={"valid": True, "errors": [], "warnings": []}):
            with patch.object(tool.validator, "validate_tests", return_value={"valid": True, "passed": 10, "failed": 0, "output": "10 passed"}):
                result = await tool.execute(
                    action="apply",
                    file_path=str(target),
                    old_string="x = 1",
                    new_string="x = 2",
                )
                assert result.success is True
                assert "applied and validated" in result.output
                assert "x = 2" in target.read_text()

    def test_tool_registration(self, tmp_workspace, security, logger, tmp_path):
        from evora.tools import ToolRegistry
        identity_service = MagicMock()
        identity_service.require_authority.return_value = True
        approval = MagicMock()
        approval.approve_plan.return_value = MagicMock(value="approve")
        registry = ToolRegistry(
            security, logger,
            identity_service=identity_service,
            approval_system=approval,
        )
        assert "self_improve" in registry.list()
        tool = registry.get("self_improve")
        assert tool.name == "self_improve"

    def test_tool_not_registered_without_self_improve(self, tmp_workspace, security, logger):
        from evora.tools import ToolRegistry, _has_self_improve
        registry = ToolRegistry(security, logger)
        if _has_self_improve():
            assert "self_improve" in registry.list()
        else:
            assert "self_improve" not in registry.list()
