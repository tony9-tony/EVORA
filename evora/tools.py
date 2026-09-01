"""
Tool system for EVORA.

Provides controlled, permission-checked tools that the AI agent can use.
Each tool has a permission level and returns structured results.

Tools:
    read_file       - Read file contents
    write_file      - Create/overwrite a file
    edit_file       - Edit a file with search/replace
    create_dir      - Create a directory
    list_dir        - List directory contents
    search_files    - Search for files by name pattern
    search_content  - Search file contents with regex
    execute_command - Run a command (permission-checked)
    run_tests       - Run test suite
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from evora.security import PermissionLevel, PermissionManager
from evora.logger import Logger


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "data": self.data,
        }


class Tool:
    """Base class for EVORA tools."""

    name: str
    description: str
    permission: PermissionLevel = PermissionLevel.SAFE
    parameters: dict[str, Any] = {}

    def __init__(self, security: PermissionManager, logger: Optional[Logger] = None):
        self.security = security
        self.logger = logger

    async def execute(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def to_tool_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": [k for k, v in self.parameters.items() if v.get("required")],
                }
            }
        }


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file from the workspace."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the file to read.", "required": True},
        "start_line": {"type": "integer", "description": "Starting line number (1-indexed).", "required": False},
        "end_line": {"type": "integer", "description": "Ending line number.", "required": False},
    }

    async def execute(self, path: str, start_line: int = None, end_line: int = None) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        if not full_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        if not full_path.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")

        try:
            content = full_path.read_text(encoding="utf-8")
            lines = content.splitlines(keepends=True)

            if start_line or end_line:
                start = (start_line or 1) - 1
                end = end_line or len(lines)
                lines = lines[start:end]
                content = "".join(lines)

            return ToolResult(
                success=True,
                output=content,
                data={"path": str(full_path), "lines": len(lines)}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a file with content."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the file to write.", "required": True},
        "content": {"type": "string", "description": "Content to write to the file.", "required": True},
        "mode": {"type": "string", "description": "Write mode: 'w' for overwrite, 'a' for append.", "required": False},
    }

    async def execute(self, path: str, content: str, mode: str = "w") -> ToolResult:
        level = self.security.check_file_write(path)
        if level == PermissionLevel.DANGEROUS:
            return ToolResult(success=False, error=f"Write blocked: path outside workspace")

        if level == PermissionLevel.ASK:
            if not self.security.request_approval(
                f"write_file {path}", PermissionLevel.ASK,
                "File write outside default scope"
            ):
                return ToolResult(success=False, error="Write operation not approved")

        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            write_mode = "a" if mode == "a" else "w"
            encoding = "utf-8"

            if write_mode == "w":
                full_path.write_text(content, encoding=encoding)
            else:
                with open(full_path, "a", encoding=encoding) as f:
                    f.write(content)

            if self.logger:
                self.logger.code(f"Wrote {len(content)} bytes to {path}")

            return ToolResult(
                success=True,
                output=f"File written successfully: {path}",
                data={"path": str(full_path), "bytes": len(content)}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}")


class EditFileTool(Tool):
    name = "edit_file"
    description = "Edit a file by replacing old_string with new_string."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the file to edit.", "required": True},
        "old_string": {"type": "string", "description": "String to search for.", "required": True},
        "new_string": {"type": "string", "description": "Replacement string.", "required": True},
        "replace_all": {"type": "boolean", "description": "Replace all occurrences.", "required": False},
    }

    async def execute(self, path: str, old_string: str, new_string: str, replace_all: bool = False) -> ToolResult:
        level = self.security.check_file_write(path)
        if level == PermissionLevel.DANGEROUS:
            return ToolResult(success=False, error=f"Edit blocked: path outside workspace")

        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        if not full_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        try:
            content = full_path.read_text(encoding="utf-8")

            if old_string not in content:
                return ToolResult(success=False, error=f"String not found in file: {old_string[:50]}...")

            count = content.count(old_string) if replace_all else 1
            if not replace_all and content.count(old_string) > 1:
                return ToolResult(success=False, error=f"Multiple matches found. Use replace_all=true or provide more context.")

            if replace_all:
                content = content.replace(old_string, new_string)
            else:
                content = content.replace(old_string, new_string, 1)

            full_path.write_text(content, encoding="utf-8")

            if self.logger:
                self.logger.code(f"Edited {path}: replaced {count} occurrence(s)")

            return ToolResult(
                success=True,
                output=f"File edited successfully: {path}",
                data={"path": str(full_path), "replacements": count}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to edit file: {e}")


class CreateDirTool(Tool):
    name = "create_directory"
    description = "Create a directory (and parent directories if needed)."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the directory to create.", "required": True},
    }

    async def execute(self, path: str) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            full_path.mkdir(parents=True, exist_ok=True)
            return ToolResult(
                success=True,
                output=f"Directory created: {path}",
                data={"path": str(full_path)}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to create directory: {e}")


class ListDirTool(Tool):
    name = "list_directory"
    description = "List contents of a directory."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the directory to list.", "required": True},
    }

    async def execute(self, path: str = ".") -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        if not full_path.exists():
            return ToolResult(success=False, error=f"Directory not found: {path}")

        try:
            items = []
            for entry in sorted(full_path.iterdir()):
                items.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else 0,
                    "modified": entry.stat().st_mtime,
                })
            return ToolResult(
                success=True,
                output="\n".join(f"{'📁' if i['type'] == 'dir' else '📄'} {i['name']}" for i in items),
                data={"items": items}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to list directory: {e}")


class SearchFilesTool(Tool):
    name = "search_files"
    description = "Search for files matching a pattern (glob)."
    permission = PermissionLevel.SAFE
    parameters = {
        "pattern": {"type": "string", "description": "Glob pattern to match files.", "required": True},
        "path": {"type": "string", "description": "Directory to search in.", "required": False},
    }

    async def execute(self, pattern: str, path: str = ".") -> ToolResult:
        try:
            search_dir = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            matches = []
            for match in search_dir.rglob(pattern):
                try:
                    match.relative_to(search_dir)
                except ValueError:
                    pass
                try:
                    matches.append(str(match.relative_to(search_dir)))
                except ValueError:
                    matches.append(str(match))
            return ToolResult(
                success=True,
                output=f"Found {len(matches)} matching files:\n" + "\n".join(matches[:50]),
                data={"matches": matches, "total": len(matches)}
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Search failed: {e}")


class SearchContentTool(Tool):
    name = "search_content"
    description = "Search file contents for a regex pattern."
    permission = PermissionLevel.SAFE
    parameters = {
        "query": {"type": "string", "description": "Regex pattern to search for.", "required": True},
        "path": {"type": "string", "description": "Directory or file to search.", "required": False},
        "file_pattern": {"type": "string", "description": "File glob pattern to filter.", "required": False},
    }

    async def execute(self, query: str, path: str = ".", file_pattern: str = "*") -> ToolResult:
        try:
            search_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error as e:
            return ToolResult(success=False, error=f"Invalid regex: {e}")

        results = []
        root = search_path
        if search_path.is_file():
            files = [search_path]
        else:
            files = list(search_path.rglob(file_pattern))

        for f in files:
            if not f.is_file():
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        results.append({
                            "file": str(f.relative_to(root)) if f.is_relative_to(root) else str(f),
                            "line": i,
                            "content": line.strip(),
                        })
                        if len(results) >= 100:
                            break
            except Exception:
                continue
            if len(results) >= 100:
                break

        output_lines = [f"{r['file']}:{r['line']}: {r['content']}" for r in results]
        return ToolResult(
            success=True,
            output="\n".join(output_lines) if output_lines else "No matches found.",
            data={"matches": results, "total": len(results)}
        )


class ExecuteCommandTool(Tool):
    name = "execute_command"
    description = "Execute a shell command within the workspace. Commands are checked for safety."
    permission = PermissionLevel.ASK
    parameters = {
        "command": {"type": "string", "description": "The command to execute.", "required": True},
        "timeout": {"type": "integer", "description": "Timeout in seconds.", "required": False},
        "cwd": {"type": "string", "description": "Working directory for the command.", "required": False},
    }

    async def execute(self, command: str, timeout: int = None, cwd: str = None) -> ToolResult:
        level = self.security.check_command_safety(command)
        timeout_val = timeout or self.security.check_command_timeout(command)

        level_str = level.value.upper() if isinstance(level, PermissionLevel) else str(level)
        if level_str == "DANGEROUS":
            if self.logger:
                self.logger.error(f"Blocked dangerous command: {command}")
            return ToolResult(success=False, error=f"Command blocked (dangerous): {command}")

        if level_str == "ASK":
            approved = self.security.request_approval(
                command, PermissionLevel.ASK,
                f"Executing command: {command}"
            )
            if not approved:
                return ToolResult(success=False, error="Command execution not approved")

        if not self.security.check_command_allowed(command):
            return ToolResult(success=False, error=f"Command not in allowed list: {command}")

        work_dir = cwd or self.security.workspace_dir
        work_dir = str(Path(work_dir).resolve())

        try:
            if self.logger:
                self.logger.code(f"Executing: {command} (timeout={timeout_val}s)")

            process = subprocess.Popen(
                command,
                shell=True,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout_val)
                success = process.returncode == 0

                output = stdout
                if stderr:
                    output += "\n" + stderr

                if self.logger:
                    status = "SUCCESS" if success else "FAILED"
                    self.logger.code(f"Command completed ({status}): {command}")

                return ToolResult(
                    success=success,
                    output=output,
                    error=stderr if not success else "",
                    data={
                        "returncode": process.returncode,
                        "command": command,
                    }
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return ToolResult(success=False, error=f"Command timed out after {timeout_val}s: {command}")

        except Exception as e:
            return ToolResult(success=False, error=f"Command execution failed: {e}")


class RunTestsTool(Tool):
    name = "run_tests"
    description = "Run tests for the project. Auto-detects test framework."
    permission = PermissionLevel.ASK
    parameters = {
        "framework": {"type": "string", "description": "Test framework (pytest, go, npm, etc.)", "required": False},
        "path": {"type": "string", "description": "Test path or pattern.", "required": False},
    }

    async def execute(self, framework: str = None, path: str = None) -> ToolResult:
        work_dir = str(Path(self.security.workspace_dir).resolve())
        test_path = path or ""

        if framework:
            cmd = f"{framework} {test_path}".strip()
        else:
            if Path(work_dir, "go.mod").exists():
                cmd = f"go test -v ./... " + test_path
            elif Path(work_dir, "pytest.ini").exists() or Path(work_dir, "pyproject.toml").exists() or any(Path(work_dir).rglob("test_*.py")):
                cmd = "python -m pytest"
            elif Path(work_dir, "package.json").exists():
                cmd = f"npm test {test_path}"
            else:
                return ToolResult(success=False, error="No test framework detected")

        return await self.execute_command(command=cmd)


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self, security: PermissionManager, logger: Optional[Logger] = None):
        self.security = security
        self.logger = logger
        self._tools: dict[str, Tool] = {}
        self._register_all()

    def _register_all(self):
        self.register(ReadFileTool(self.security, self.logger))
        self.register(WriteFileTool(self.security, self.logger))
        self.register(EditFileTool(self.security, self.logger))
        self.register(CreateDirTool(self.security, self.logger))
        self.register(ListDirTool(self.security, self.logger))
        self.register(SearchFilesTool(self.security, self.logger))
        self.register(SearchContentTool(self.security, self.logger))
        self.register(ExecuteCommandTool(self.security, self.logger))
        self.register(RunTestsTool(self.security, self.logger))

    def register(self, tool: Tool):
        self._tools[tool.name] = tool
        if self.logger:
            self.logger.debug(f"registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list(self) -> list[str]:
        return list(self._tools.keys())

    def get_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": [k for k, v in tool.parameters.items() if v.get("required")],
                    }
                }
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return await tool.execute(**kwargs)
