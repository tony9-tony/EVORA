"""
Tests for the EVORA tools system.
"""

import os
import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.tools import (
    ToolRegistry,
    ToolResult,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    CreateDirTool,
    ListDirTool,
    SearchFilesTool,
    ExecuteCommandTool,
    RunTestsTool,
    GitStatusTool,
    GitDiffTool,
    GitBranchTool,
    GitLogTool,
    AnalyzeProjectTool,
    AnalyzeCodeTool,
    WebSearchTool,
    WebFetchTool,
)
from evora.security import PermissionManager, PermissionLevel
from evora.logger import Logger


def run_async(coro):
    """Helper to run async tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def security(tmp_path):
    return PermissionManager(workspace_dir=str(tmp_path))


@pytest.fixture
def logger():
    return Logger("test_tools", "error")


@pytest.fixture
def registry(security, logger):
    return ToolRegistry(security, logger)


class TestToolResult:

    def test_success(self):
        tr = ToolResult(success=True, output="hello")
        assert tr.success is True

    def test_to_dict(self):
        tr = ToolResult(success=True, output="hello", error="", data={"key": "val"})
        d = tr.to_dict()
        assert d["success"] is True
        assert d["output"] == "hello"
        assert d["data"] == {"key": "val"}


class TestReadFileTool:

    def test_read_file(self, tmp_path, security, logger):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        tool = ReadFileTool(security, logger)
        result = run_async(tool.execute(path=str(test_file)))
        assert result.success is True
        assert "Hello World" in result.output


class TestWriteFileTool:

    def test_write_file(self, tmp_path, security, logger):
        test_file = tmp_path / "output.txt"

        tool = WriteFileTool(security, logger)
        result = run_async(tool.execute(path=str(test_file), content="New content"))
        assert result.success is True
        assert test_file.exists()
        assert test_file.read_text() == "New content"

    def test_read_after_write(self, tmp_path, security, logger):
        test_file = tmp_path / "rw_test.txt"

        write_tool = WriteFileTool(security, logger)
        read_tool = ReadFileTool(security, logger)

        run_async(write_tool.execute(path=str(test_file), content="Read me back"))
        result = run_async(read_tool.execute(path=str(test_file)))
        assert result.success
        assert "Read me back" in result.output


class TestEditFileTool:

    def test_edit_file(self, tmp_path, security, logger):
        test_file = tmp_path / "edit.txt"
        test_file.write_text("old content")

        tool = EditFileTool(security, logger)
        result = run_async(tool.execute(
            path=str(test_file),
            old_string="old",
            new_string="new"
        ))
        assert result.success is True
        assert "new content" in test_file.read_text()

    def test_edit_file_not_found(self, tmp_path, security, logger):
        tool = EditFileTool(security, logger)
        result = run_async(tool.execute(
            path=str(tmp_path / "nonexistent.txt"),
            old_string="old",
            new_string="new"
        ))
        assert result.success is False


class TestCreateDirTool:

    def test_create_dir(self, tmp_path, security, logger):
        path = tmp_path / "newdir"

        tool = CreateDirTool(security, logger)
        result = run_async(tool.execute(path=str(path)))
        assert result.success is True
        assert path.exists()


class TestListDirTool:

    def test_list_dir(self, tmp_path, security, logger):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "file.txt").write_text("hello")

        tool = ListDirTool(security, logger)
        result = run_async(tool.execute(path=str(tmp_path)))
        assert result.success is True
        assert "file.txt" in result.output
        assert "subdir" in result.output


class TestSearchFilesTool:

    def test_search_files(self, tmp_path, security, logger):
        (tmp_path / "test_file.py").write_text("content")
        (tmp_path / "other.py").write_text("content")

        tool = SearchFilesTool(security, logger)
        result = run_async(tool.execute(pattern="*.py", path=str(tmp_path)))
        assert result.success is True
        assert "test_file.py" in result.output


class TestExecuteCommandTool:

    def test_execute_safe_command(self, tmp_path, security, logger):
        tool = ExecuteCommandTool(security, logger)
        result = run_async(tool.execute(command="echo hello"))
        assert result.success is True
        assert "hello" in result.output


class TestToolRegistry:

    def test_registry_has_tools(self, security, logger):
        registry = ToolRegistry(security, logger)
        tools = registry.list()
        assert "read_file" in tools
        assert "write_file" in tools
        assert "edit_file" in tools
        assert "execute_command" in tools

    def test_registry_execute(self, tmp_path, security, logger):
        registry = ToolRegistry(security, logger)
        result = run_async(registry.execute("list_directory", path=str(tmp_path)))
        assert result.success is True

    def test_registry_get_spec(self, security, logger):
        registry = ToolRegistry(security, logger)
        specs = registry.get_specs()
        assert len(specs) > 0
        for spec in specs:
            assert "name" in spec.get("function", {})


class TestRunTestsTool:

    def test_run_tests_no_framework(self, tmp_path, security, logger):
        """RunTestsTool should return error when no test framework is detected."""
        tool = RunTestsTool(security, logger)
        result = run_async(tool.execute())
        assert result.success is False
        assert "No test framework" in result.error or "not detected" in result.error.lower()

    def test_run_tests_with_pytest(self, tmp_path, security, logger):
        """RunTestsTool should detect pytest and run it."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_path / "test_dummy.py").write_text("def test_pass():\n    assert True\n")

        tool = RunTestsTool(security, logger)
        result = run_async(tool.execute())
        assert result.success is True
        assert "1 passed" in result.output

    def test_run_tests_with_framework_override(self, tmp_path, security, logger):
        """RunTestsTool should use the provided framework."""
        tool = RunTestsTool(security, logger)
        result = run_async(tool.execute(framework="python", path="-c 'print(\"tests\")'"))
        assert result.success is True


class TestGitTools:

    def test_git_status_outside_workspace(self, tmp_path, security, logger):
        """GitStatusTool should return error for paths outside workspace."""
        tool = GitStatusTool(security, logger)
        result = run_async(tool.execute(path="/tmp/nonexistent"))
        assert result.success is False
        assert "outside" in result.error or "Permission" in result.error

    def test_git_status_no_git_repo(self, tmp_path, security, logger):
        """GitStatusTool should handle repos without git."""
        tool = GitStatusTool(security, logger)
        result = run_async(tool.execute(path=str(tmp_path)))
        assert not result.success or "fatal" in result.output.lower() or result.error


class TestAnalyzeProjectTool:

    def test_analyze_empty_project(self, tmp_path, security, logger):
        """AnalyzeProjectTool should handle empty directories."""
        tool = AnalyzeProjectTool(security, logger)
        result = run_async(tool.execute())
        assert result.success is True
        import json
        data = json.loads(result.output)
        assert "files" in data
        assert "languages" in data
        assert data["file_count"] == 0

    def test_analyze_project_with_files(self, tmp_path, security, logger):
        """AnalyzeProjectTool should detect Python files."""
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")

        # Override workspace
        security.workspace_dir = tmp_path.resolve()

        tool = AnalyzeProjectTool(security, logger)
        result = run_async(tool.execute(path=str(tmp_path)))
        assert result.success is True
        import json
        data = json.loads(result.output)
        assert "main.py" in data["files"]
        assert "Python" in data["languages"]
        assert data["build_system"] is not None


class TestAnalyzeCodeTool:

    def test_analyze_code_python(self, tmp_path, security, logger):
        """AnalyzeCodeTool should parse Python files."""
        test_file = tmp_path / "test.py"
        test_file.write_text("class Foo:\n    def bar(self):\n        pass\n")

        tool = AnalyzeCodeTool(security, logger)
        result = run_async(tool.execute(path=str(test_file)))
        assert result.success is True
        assert "test.py" in result.output
        assert "Python" in result.output
        assert "classes" in result.data

    def test_analyze_code_file_not_found(self, tmp_path, security, logger):
        tool = AnalyzeCodeTool(security, logger)
        result = run_async(tool.execute(path=str(tmp_path / "nonexistent.py")))
        assert result.success is False
        assert "not found" in result.error.lower()


class TestWebSearchTool:

    def test_web_search_not_approved(self, tmp_path, security, logger):
        """WebSearchTool should reject approval when no callbacks return True."""
        security.ask_approvals = True  # We need approval
        # No approval callbacks registered, so approval will fail
        tool = WebSearchTool(security, logger)
        result = run_async(tool.execute(query="test"))
        assert result.success is False
        assert "not approved" in result.error.lower()

    def test_web_search_approved(self, tmp_path, security, logger):
        """WebSearchTool should work when auto-approved."""
        security.ask_approvals = False  # Auto-approve ASK-level
        tool = WebSearchTool(security, logger)
        # This test requires internet access; may fail in offline environments
        import httpx
        try:
            result = run_async(tool.execute(query="EVORA AI"))
            # We can't guarantee results, but it shouldn't error on approval
            assert "not approved" not in result.error.lower()
        except Exception:
            pytest.skip("Network unavailable for web search test")
