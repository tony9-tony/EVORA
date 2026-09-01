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
    git_status      - Check git status
    git_diff        - View uncommitted changes
    git_commit      - Commit changes
    git_branch      - Manage branches
    git_log         - View commit history
    analyze_project - Analyze project structure and detect frameworks
    analyze_code    - Analyze a source file for structure and quality
    web_search      - Search the web for information
    web_fetch       - Fetch content from a URL
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


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show the working tree status of a Git repository."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the git repository.", "required": False},
    }

    async def execute(self, path: str = ".") -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            result = subprocess.run(
                ["git", "-C", str(full_path), "status"],
                capture_output=True, text=True, timeout=30,
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
                data={"path": str(full_path), "returncode": result.returncode},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Git status failed: {e}")


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show uncommitted changes in a Git repository."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the git repository.", "required": False},
        "staged": {"type": "boolean", "description": "Show staged changes only.", "required": False},
        "file": {"type": "string", "description": "Show diff for a specific file.", "required": False},
    }

    async def execute(self, path: str = ".", staged: bool = False, file: str = None) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            cmd = ["git", "-C", str(full_path), "diff"]
            if staged:
                cmd.append("--staged")
            if file:
                cmd.append("--")
                cmd.append(file)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Git diff failed: {e}")


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Commit staged changes in a Git repository (requires approval)."
    permission = PermissionLevel.ASK
    parameters = {
        "path": {"type": "string", "description": "Path to the git repository.", "required": False},
        "message": {"type": "string", "description": "Commit message.", "required": True},
    }

    async def execute(self, path: str = ".", message: str = "") -> ToolResult:
        if not message:
            return ToolResult(success=False, error="Commit message is required")

        approved = self.security.request_approval(
            f"git commit -m '{message}'", PermissionLevel.ASK,
            "Committing changes to git"
        )
        if not approved:
            return ToolResult(success=False, error="Commit not approved")

        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            result = subprocess.run(
                ["git", "-C", str(full_path), "commit", "-m", message],
                capture_output=True, text=True, timeout=30,
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Git commit failed: {e}")


class GitBranchTool(Tool):
    name = "git_branch"
    description = "List, create, or switch branches in a Git repository (requires approval for modifications)."
    permission = PermissionLevel.ASK
    parameters = {
        "path": {"type": "string", "description": "Path to the git repository.", "required": False},
        "branch": {"type": "string", "description": "Branch name to create/switch to.", "required": False},
        "list_only": {"type": "boolean", "description": "Only list branches, do not create.", "required": False},
    }

    async def execute(self, path: str = ".", branch: str = None, list_only: bool = True) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            if list_only or not branch:
                result = subprocess.run(
                    ["git", "-C", str(full_path), "branch", "--show-current"],
                    capture_output=True, text=True, timeout=30,
                )
                return ToolResult(
                    success=result.returncode == 0,
                    output=result.stdout.strip(),
                    error=result.stderr if result.returncode != 0 else "",
                )

            approved = self.security.request_approval(
                f"git checkout -b {branch}", PermissionLevel.ASK,
                f"Creating/switching to branch '{branch}'"
            )
            if not approved:
                return ToolResult(success=False, error="Branch operation not approved")

            result = subprocess.run(
                ["git", "-C", str(full_path), "checkout", "-b", branch],
                capture_output=True, text=True, timeout=30,
            )
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Git branch failed: {e}")


class GitLogTool(Tool):
    name = "git_log"
    description = "Show commit history of a Git repository."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the git repository.", "required": False},
        "limit": {"type": "integer", "description": "Number of commits to show.", "required": False},
    }

    async def execute(self, path: str = ".", limit: int = 10) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            cmd = ["git", "-C", str(full_path), "log", f"-{limit}", "--oneline", "--decorate"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
                data={"commits": result.stdout.strip().split("\n") if result.stdout.strip() else []},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Git log failed: {e}")


class AnalyzeProjectTool(Tool):
    name = "analyze_project"
    description = "Analyze the project structure, detect languages, frameworks, and test commands."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the project root.", "required": False},
    }

    async def execute(self, path: str = None) -> ToolResult:
        import json as json_mod
        search_path = path or self.security.workspace_dir
        try:
            full_path = self.security.check_workspace_path(search_path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            result = {}
            result["files"] = []
            result["languages"] = {}
            result["frameworks"] = []
            result["build_system"] = None
            result["test_command"] = None
            result["has_git"] = (full_path / ".git").exists()

            lang_extensions = {
                ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
                ".jsx": "JavaScript", ".tsx": "TypeScript", ".go": "Go",
                ".rs": "Rust", ".java": "Java", ".c": "C", ".cpp": "C++",
                ".cs": "C#", ".rb": "Ruby", ".php": "PHP", ".sh": "Shell",
                ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".json": "JSON",
                ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".xml": "XML",
                ".md": "Markdown", ".sql": "SQL", ".kt": "Kotlin", ".swift": "Swift",
            }

            build_files = {
                "pyproject.toml": "Python (poetry/pip)",
                "setup.py": "Python (setuptools)",
                "setup.cfg": "Python (setuptools)",
                "go.mod": "Go",
                "Cargo.toml": "Rust",
                "pom.xml": "Java (Maven)",
                "build.gradle": "Java (Gradle)",
                "package.json": "Node.js",
                "CMakeLists.txt": "C/C++ (CMake)",
                "Makefile": "Make",
            }

            frameworks = {
                "fastapi": "FastAPI", "flask": "Flask", "django": "Django",
                "react": "React", "vue": "Vue", "angular": "Angular",
                "next.config.js": "Next.js", "gatsby": "Gatsby",
                "pytest.ini": "pytest", "tox.ini": "tox",
            }

            test_commands = {
                "pytest.ini": "pytest", "pyproject.toml": "pytest",
                "go.mod": "go test ./...", "Cargo.toml": "cargo test",
                "pom.xml": "mvn test", "build.gradle": "gradle test",
                "package.json": "npm test",
            }

            for entry in full_path.iterdir():
                if entry.is_file():
                    result["files"].append(entry.name)
                    ext = entry.suffix.lower()
                    if ext in lang_extensions:
                        lang = lang_extensions[ext]
                        result["languages"][lang] = result["languages"].get(lang, 0) + 1
                    if entry.name in build_files:
                        result["build_system"] = build_files[entry.name]
                    if entry.name in frameworks:
                        result["frameworks"].append(frameworks[entry.name])
                    if entry.name in test_commands:
                        result["test_command"] = test_commands[entry.name]

            result["file_count"] = len(result["files"])
            return ToolResult(
                success=True,
                output=json_mod.dumps(result, indent=2),
                data=result,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Project analysis failed: {e}")


class AnalyzeCodeTool(Tool):
    name = "analyze_code"
    description = "Analyze a source file for structure: functions, classes, imports, complexity."
    permission = PermissionLevel.SAFE
    parameters = {
        "path": {"type": "string", "description": "Path to the source file.", "required": True},
    }

    async def execute(self, path: str) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(path)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        if not full_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")
        if not full_path.is_file():
            return ToolResult(success=False, error=f"Not a file: {path}")

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            ext = full_path.suffix.lower()

            result = {
                "path": str(full_path),
                "language": lang_extensions.get(ext, "Unknown") if (lang_extensions := {
                    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
                    ".go": "Go", ".rs": "Rust", ".java": "Java", ".c": "C",
                    ".cpp": "C++", ".cs": "C#", ".rb": "Ruby", ".php": "PHP",
                }) else "Unknown",
                "lines": len(lines),
                "classes": [],
                "functions": [],
                "imports": [],
            }

            indent_chars = set()
            for line in lines:
                stripped = line.lstrip()
                if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                    continue
                indent_chars.add(line[:len(line) - len(stripped)])

            for line in lines:
                stripped = line.strip()
                if ext == ".py":
                    if re.match(r"^(class|def)\s+\w+", stripped):
                        parts = stripped.split("(")
                        name = parts[0].replace("class ", "").replace("def ", "").strip()
                        if stripped.startswith("class"):
                            result["classes"].append(name)
                        else:
                            result["functions"].append(name)
                        if stripped.startswith("import ") or stripped.startswith("from "):
                            result["imports"].append(stripped)
                elif ext in (".js", ".ts", ".jsx", ".tsx"):
                    if "function " in stripped and "(" in stripped:
                        result["functions"].append(stripped)
                    elif stripped.startswith("class "):
                        result["classes"].append(stripped)
                    elif stripped.startswith("import ") or stripped.startswith("const ") and "require" in stripped:
                        result["imports"].append(stripped)

            result["uses_tabs"] = "\t" in "".join(indent_chars)
            result["uses_spaces"] = " " in "".join(indent_chars)

            output_lines = [
                f"File: {path}",
                f"Language: {result['language']}",
                f"Lines: {result['lines']}",
                f"Classes: {len(result['classes'])}",
                f"Functions: {len(result['functions'])}",
                f"Imports: {len(result['imports'])}",
            ]
            if result["classes"]:
                output_lines.append(f"  Classes: {', '.join(result['classes'][:10])}")
            if result["functions"]:
                output_lines.append(f"  Functions: {', '.join(result['functions'][:10])}")
            if result["imports"]:
                output_lines.append(f"  Imports: {', '.join(result['imports'][:10])}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                data=result,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Code analysis failed: {e}")


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information. Uses DuckDuckGo (no API key required)."
    permission = PermissionLevel.ASK
    parameters = {
        "query": {"type": "string", "description": "Search query.", "required": True},
        "max_results": {"type": "integer", "description": "Maximum number of results.", "required": False},
    }

    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        approved = self.security.request_approval(
            f"web_search: {query[:50]}", PermissionLevel.ASK,
            "Searching the web for information"
        )
        if not approved:
            return ToolResult(success=False, error="Web search not approved")

        try:
            import httpx as _httpx
            url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
            async with _httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers={"User-Agent": "EVORA/1.0"})
                html = resp.text

            results = []
            for match in re.finditer(
                r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL
            ):
                href = match.group(1)
                title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
                results.append({"title": title, "url": href})
                if len(results) >= max_results:
                    break

            if not results:
                return ToolResult(success=False, error="No results found or web search unavailable")

            output = "\n".join(f"- {r['title']}: {r['url']}" for r in results)
            return ToolResult(
                success=True,
                output=output,
                data={"query": query, "results": results},
            )
        except ImportError:
            return ToolResult(success=False, error="httpx package not installed for web search")
        except Exception as e:
            return ToolResult(success=False, error=f"Web search failed: {e}")


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch and extract text content from a URL."
    permission = PermissionLevel.ASK
    parameters = {
        "url": {"type": "string", "description": "URL to fetch.", "required": True},
        "max_length": {"type": "integer", "description": "Maximum characters to return.", "required": False},
    }

    async def execute(self, url: str, max_length: int = 5000) -> ToolResult:
        approved = self.security.request_approval(
            f"web_fetch: {url[:80]}", PermissionLevel.ASK,
            "Fetching web content"
        )
        if not approved:
            return ToolResult(success=False, error="Web fetch not approved")

        try:
            import httpx as _httpx
            async with _httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "EVORA/1.0"})
                html = resp.text

            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > max_length:
                text = text[:max_length] + "..."

            return ToolResult(
                success=True,
                output=text,
                data={"url": url, "length": len(text)},
            )
        except ImportError:
            return ToolResult(success=False, error="httpx package not installed for web fetch")
        except Exception as e:
            return ToolResult(success=False, error=f"Web fetch failed: {e}")


class SelfAnalyzeTool(Tool):
    name = "self_analyze"
    description = "Analyze EVORA's own codebase for quality, test coverage, and potential improvements."
    permission = PermissionLevel.SAFE
    parameters = {
        "workspace": {"type": "string", "description": "Path to analyze (defaults to workspace).", "required": False},
        "check_tests": {"type": "boolean", "description": "Check test coverage and identify untested files.", "required": False},
        "check_complexity": {"type": "boolean", "description": "Check code complexity and identify long functions.", "required": False},
    }

    async def execute(self, workspace: str = None, check_tests: bool = True, check_complexity: bool = True) -> ToolResult:
        try:
            full_path = self.security.check_workspace_path(workspace or self.security.workspace_dir)
        except PermissionError as e:
            return ToolResult(success=False, error=str(e))

        try:
            import json as json_mod

            result = {
                "issues": [],
                "metrics": {},
                "recommendations": [],
            }

            # Check for test files
            test_files = list(full_path.rglob("test_*.py")) + list(full_path.rglob("*_test.py"))
            source_files = [f for f in full_path.rglob("*.py") if "test_" not in f.name and "_test.py" not in f.name]
            result["metrics"]["test_files"] = len(test_files)
            result["metrics"]["source_files"] = len(source_files)
            result["metrics"]["test_coverage_ratio"] = len(test_files) / max(len(source_files), 1)

            if check_tests:
                source_stems = {f.stem for f in source_files}
                test_stems = set()
                for tf in test_files:
                    if tf.stem.startswith("test_"):
                        test_stems.add(tf.stem[5:])  # Remove 'test_' prefix
                    else:
                        test_stems.add(tf.stem.replace("_test", ""))

                untested = []
                for sf in source_files:
                    stem = sf.stem
                    if stem not in test_stems and not stem.startswith("_"):
                        untested.append(str(sf.relative_to(full_path)))

                if untested:
                    result["issues"].append({
                        "type": "untested_code",
                        "severity": "medium",
                        "detail": f"{len(untested)} source files have no corresponding test file",
                        "files": untested[:20],
                    })
                    result["recommendations"].append(f"Write tests for: {', '.join(untested[:5])}")

            if check_complexity:
                long_functions = []
                for sf in source_files:
                    try:
                        content = sf.read_text(encoding="utf-8", errors="ignore")
                        lines = content.splitlines()
                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            if stripped.startswith("def ") or stripped.startswith("async def "):
                                depth = 0
                                def_lines = 0
                                for j in range(i, min(i + 50, len(lines))):
                                    def_lines += 1
                                    indent_change = lines[j].count("    ") - (lines[j-1].count("    ") if j > 0 else 0)
                                    depth += indent_change
                                    if depth <= 0 and j > i:
                                        break
                                    if def_lines > 20:
                                        long_functions.append({
                                            "file": str(sf.relative_to(full_path)),
                                            "line": i + 1,
                                            "function": stripped.split("(")[0].replace("async ", "").replace("def ", ""),
                                            "lines": def_lines,
                                        })
                                        break
                    except Exception:
                        continue

                if long_functions:
                    result["issues"].append({
                        "type": "complexity",
                        "severity": "low",
                        "detail": f"{len(long_functions)} functions exceed 20 lines",
                        "functions": long_functions[:10],
                    })

            # Check for TODO/FIXME/BUG comments
            todos = []
            for sf in source_files:
                try:
                    content = sf.read_text(encoding="utf-8", errors="ignore")
                    for i, line in enumerate(content.splitlines(), 1):
                        stripped = line.strip().upper()
                        if "TODO" in stripped or "FIXME" in stripped or "BUG" in stripped:
                            todos.append(f"{sf.relative_to(full_path)}:{i}: {line.strip()}")
                except Exception:
                    continue

            if todos:
                result["issues"].append({
                    "type": "technical_debt",
                    "severity": "low",
                    "detail": f"{len(todos)} TODO/FIXME/BUG comments found",
                    "items": todos[:20],
                })

            output_lines = [
                "EVORA Self-Analysis Report",
                "=" * 40,
                f"Source files: {result['metrics']['source_files']}",
                f"Test files: {result['metrics']['test_files']}",
                f"Test coverage ratio: {result['metrics']['test_coverage_ratio']:.1%}",
                "",
                f"Issues found: {len(result['issues'])}",
            ]
            for issue in result["issues"]:
                output_lines.append(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['detail']}")
            if result["recommendations"]:
                output_lines.append("")
                output_lines.append("Recommendations:")
                for rec in result["recommendations"][:10]:
                    output_lines.append(f"  - {rec}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                data=result,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Self-analysis failed: {e}")


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

        exec_tool = ExecuteCommandTool(self.security, self.logger)
        return await exec_tool.execute(command=cmd)


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
        self.register(GitStatusTool(self.security, self.logger))
        self.register(GitDiffTool(self.security, self.logger))
        self.register(GitCommitTool(self.security, self.logger))
        self.register(GitBranchTool(self.security, self.logger))
        self.register(GitLogTool(self.security, self.logger))
        self.register(AnalyzeProjectTool(self.security, self.logger))
        self.register(AnalyzeCodeTool(self.security, self.logger))
        self.register(WebSearchTool(self.security, self.logger))
        self.register(WebFetchTool(self.security, self.logger))
        self.register(SelfAnalyzeTool(self.security, self.logger))

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
