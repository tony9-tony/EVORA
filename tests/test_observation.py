"""
Tests for EVORA Phase 2 observation manager.
"""

import pytest
from unittest.mock import MagicMock

from evora.observation import ObservationManager
from evora.task import ActionResult, Observation


class TestObservationManager:

    def test_creation(self):
        mgr = ObservationManager()
        assert mgr is not None

    def test_observe_file_created(self):
        mgr = ObservationManager()
        obs = mgr.observe_file_created("/path/to/file.py")
        assert obs.type == "file_created"
        assert obs.success is True
        assert obs.data["path"] == "/path/to/file.py"
        assert obs.source == "write_file"

    def test_observe_file_modified(self):
        mgr = ObservationManager()
        obs = mgr.observe_file_modified("/path/to/file.py")
        assert obs.type == "file_modified"
        assert obs.success is True

    def test_observe_file_deleted(self):
        mgr = ObservationManager()
        obs = mgr.observe_file_deleted("/path/to/file.py")
        assert obs.type == "file_deleted"
        assert obs.success is True

    def test_observe_directory_created(self):
        mgr = ObservationManager()
        obs = mgr.observe_directory_created("/path/to/dir")
        assert obs.type == "directory_created"
        assert obs.success is True

    def test_observe_command_success(self):
        mgr = ObservationManager()
        obs = mgr.observe_command_success("ls -la", output="total 10", return_code=0)
        assert obs.type == "command_success"
        assert obs.success is True
        assert obs.data["command"] == "ls -la"
        assert obs.data["return_code"] == 0

    def test_observe_command_failed(self):
        mgr = ObservationManager()
        obs = mgr.observe_command_failed("rm -rf /", error="permission denied", output="", return_code=1)
        assert obs.type == "command_failed"
        assert obs.success is False
        assert obs.data["command"] == "rm -rf /"
        assert obs.data["return_code"] == 1

    def test_observe_test_passed(self):
        mgr = ObservationManager()
        obs = mgr.observe_test_passed(output="5 passed", command="pytest")
        assert obs.type == "test_passed"
        assert obs.success is True
        assert obs.data["command"] == "pytest"

    def test_observe_test_failed(self):
        mgr = ObservationManager()
        obs = mgr.observe_test_failed(output="1 failed", error="assertion error")
        assert obs.type == "test_failed"
        assert obs.success is False
        assert obs.data["error"] == "assertion error"

    def test_observe_build_failed(self):
        mgr = ObservationManager()
        obs = mgr.observe_build_failed(output="build error", error="syntax error")
        assert obs.type == "build_failed"
        assert obs.success is False

    def test_observe_file_missing(self):
        mgr = ObservationManager()
        obs = mgr.observe_file_missing("/missing/file.py")
        assert obs.type == "file_missing"
        assert obs.success is False
        assert obs.data["path"] == "/missing/file.py"

    def test_observe_plan_created(self):
        mgr = ObservationManager()
        obs = mgr.observe_plan_created("Test Plan", 3)
        assert obs.type == "plan_created"
        assert obs.success is True
        assert obs.data["title"] == "Test Plan"
        assert obs.data["steps"] == 3

    def test_observe_approval_granted(self):
        mgr = ObservationManager()
        obs = mgr.observe_approval(granted=True, reason="User approved")
        assert obs.type == "approval_granted"
        assert obs.success is True

    def test_observe_approval_denied(self):
        mgr = ObservationManager()
        obs = mgr.observe_approval(granted=False, reason="User denied")
        assert obs.type == "approval_denied"
        assert obs.success is False

    def test_observe_error(self):
        mgr = ObservationManager()
        obs = mgr.observe_error("Connection refused", context="api_call")
        assert obs.type == "error"
        assert obs.success is False
        assert obs.data["error"] == "Connection refused"
        assert obs.source == "api_call"

    def test_from_action_result_write_file(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=True,
            tool="write_file",
            arguments={"path": "test.py", "content": "print('hi')"},
            output="File written successfully",
        )
        observations = mgr.from_action_result(result)
        assert len(observations) >= 1
        assert observations[0].type == "file_created"
        assert observations[0].data["path"] == "test.py"

    def test_from_action_result_read_file(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=True,
            tool="read_file",
            arguments={"path": "test.py"},
            output="print('hi')\n",
            data={"path": "test.py", "lines": 1},
        )
        observations = mgr.from_action_result(result)
        assert len(observations) >= 1
        assert observations[0].type == "file_read"

    def test_from_action_result_command_success(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=True,
            tool="execute_command",
            arguments={"command": "ls"},
            output="file1.py\nfile2.py",
            data={"returncode": 0},
        )
        observations = mgr.from_action_result(result)
        assert len(observations) >= 1
        assert observations[0].type == "command_success"

    def test_from_action_result_command_failed(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=False,
            tool="execute_command",
            arguments={"command": "python test.py"},
            output="Error output",
            error="Syntax error",
            data={"returncode": 1},
        )
        observations = mgr.from_action_result(result)
        assert len(observations) >= 1
        assert observations[0].type == "command_failed"
        assert observations[0].success is False

    def test_from_action_result_test_passed(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=True,
            tool="run_tests",
            arguments={"command": "pytest"},
            output="5 passed",
        )
        observations = mgr.from_action_result(result)
        assert len(observations) >= 1
        assert observations[0].type == "test_passed"

    def test_from_action_result_test_failed(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=False,
            tool="run_tests",
            arguments={"command": "pytest"},
            output="1 failed",
            error="AssertionError",
        )
        observations = mgr.from_action_result(result)
        assert len(observations) >= 1
        assert observations[0].type == "test_failed"

    def test_from_action_result_action_failed(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=False,
            tool="write_file",
            error="Permission denied",
        )
        observations = mgr.from_action_result(result)
        assert len(observations) == 1
        assert observations[0].type == "action_failed"
        assert observations[0].success is False

    def test_from_action_result_unknown_tool(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=True,
            tool="unknown_tool",
            output="something",
        )
        observations = mgr.from_action_result(result)
        assert len(observations) == 1
        assert observations[0].type == "action_success"

    def test_from_action_result_empty_observations(self):
        mgr = ObservationManager()
        result = ActionResult(
            success=True,
            tool="planner",
            output="plan created",
        )
        observations = mgr.from_action_result(result)
        assert len(observations) == 1
        assert observations[0].type == "action_success"
