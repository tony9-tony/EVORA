"""
Security and permission system for EVORA.

Provides permission escalation levels and command safety checks.

Permission levels:
    SAFE   - Read-only operations (reading files, listing directories, searching)
    ASK    - Destructive but recoverable (installing deps, modifying config, risky commands)
    DANGEROUS - Irreversible operations (deleting files, resetting databases, self-modification)

The default workspace is restricted to the project directory.
"""

import os
import re
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class PermissionLevel(str, Enum):
    SAFE = "safe"
    ASK = "ask"
    DANGEROUS = "dangerous"


@dataclass
class CommandPermit:
    command: str
    level: PermissionLevel
    reason: str = ""
    safe_args: list[str] = None

    def __post_init__(self):
        if self.safe_args is None:
            self.safe_args = []


DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
    re.compile(r"\bdel\s+/[fs]", re.IGNORECASE),
    re.compile(r"\bformat\s+[a-z]:", re.IGNORECASE),
    re.compile(r"\bdd\b.*\bof=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"\bshutdown\b", re.IGNORECASE),
    re.compile(r"\breboot\b", re.IGNORECASE),
    re.compile(r"\bhalt\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+database\b", re.IGNORECASE),
    re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+.*\bset\b.*\bwhere\b", re.IGNORECASE),
    re.compile(r"\bdelete\s+from\b", re.IGNORECASE),
    re.compile(r"\bkill\b\s+.*\b-kill\b", re.IGNORECASE),
]

ASK_PATTERNS = [
    re.compile(r"\bapt\s+install\b", re.IGNORECASE),
    re.compile(r"\byum\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
    re.compile(r"\bpip\s+install\b", re.IGNORECASE),
    re.compile(r"\bgo\s+install\b", re.IGNORECASE),
    re.compile(r"\bcargo\s+install\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+run\b", re.IGNORECASE),
    re.compile(r"\bdocker\s+build\b", re.IGNORECASE),
    re.compile(r"\bchmod\b", re.IGNORECASE),
    re.compile(r"\bchown\b", re.IGNORECASE),
    re.compile(r"\bgpasswd\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bsystemctl\b", re.IGNORECASE),
]


class PermissionManager:
    """Manages permission levels for operations and validates safety."""

    def __init__(self, workspace_dir: str, allow_file_write: bool = True,
                 allow_cmd_exec: bool = True, allowed_cmds: Optional[list[str]] = None,
                 ask_approvals: bool = True):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.allow_file_write = allow_file_write
        self.allow_cmd_exec = allow_cmd_exec
        self.allowed_cmds = allowed_cmds or []
        self.ask_approvals = ask_approvals
        self._approval_callbacks: list = []

    def add_approval_callback(self, callback):
        """Register a callback that will be invoked for ASK-level operations."""
        self._approval_callbacks.append(callback)

    def check_workspace_path(self, path: str) -> Path:
        """Ensure a path is within the workspace directory."""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError:
            pass

        try:
            resolved.relative_to(self.workspace_dir)
            return resolved
        except ValueError:
            try:
                self.workspace_dir.relative_to(resolved)
                return resolved
            except ValueError:
                pass
            pass
        try:
            resolved.relative_to(self.workspace_dir)
            return resolved
        except ValueError:
            pass

        try:
            resolved.relative_to(self.workspace_dir)
            return resolved
        except ValueError:
            try:
                self.workspace_dir.relative_to(resolved)
                return resolved
            except ValueError:
                pass
        try:
            resolved.relative_to(self.workspace_dir)
            return resolved
        except ValueError:
            pass

        try:
            resolved.relative_to(self.workspace_dir)
            return resolved
        except ValueError:
            try:
                self.workspace_dir.relative_to(resolved)
                return resolved
            except ValueError:
                pass

        from evora.logger import Logger, Stage
        raise PermissionError(
            f"Path '{resolved}' is outside the workspace '{self.workspace_dir}'"
        )

    @staticmethod
    def check_command_safety(command: str) -> PermissionLevel:
        """Determine the permission level required for a command."""
        for pattern in DANGEROUS_PATTERNS:
            if pattern.search(command):
                return PermissionLevel.DANGEROUS
        for pattern in ASK_PATTERNS:
            if pattern.search(command):
                return PermissionLevel.ASK
        return PermissionLevel.SAFE

    def check_command_allowed(self, command: str) -> bool:
        """Check if a command is in the allowed list (if restricted)."""
        if not self.allowed_cmds:
            return True
        try:
            parts = shlex.split(command)
        except ValueError:
            parts = command.split()
        base_cmd = os.path.basename(parts[0]) if parts else ""
        return base_cmd in self.allowed_cmds or parts[0] in self.allowed_cmds

    def request_approval(self, command: str, level: PermissionLevel, reason: str = "") -> bool:
        """Request user approval for a command. Returns True if approved."""
        if level == PermissionLevel.SAFE:
            return True

        if not self.ask_approvals:
            return level != PermissionLevel.DANGEROUS

        for callback in self._approval_callbacks:
            if callback(command, level, reason):
                return True
        return False

    def check_file_write(self, path: str) -> PermissionLevel:
        """Determine the permission level for writing to a file."""
        try:
            self.check_workspace_path(path)
        except PermissionError:
            return PermissionLevel.DANGEROUS

        try:
            resolved = Path(path).resolve()
            if str(resolved).startswith(str(self.workspace_dir / ".evora")):
                return PermissionLevel.SAFE
            if str(resolved).endswith(".pyc") or str(resolved).endswith(".log"):
                return PermissionLevel.SAFE
        except Exception:
            pass

        if not self.allow_file_write:
            return PermissionLevel.ASK

        return PermissionLevel.SAFE

    def check_file_delete(self, path: str) -> PermissionLevel:
        """Determine the permission level for deleting a file."""
        try:
            resolved = Path(path).resolve()
            if not str(resolved).startswith(str(self.workspace_dir)):
                return PermissionLevel.DANGEROUS
        except Exception:
            return PermissionLevel.DANGEROUS

        return PermissionLevel.ASK

    @staticmethod
    def check_command_timeout(command: str, default_timeout: int = 60) -> int:
        """Determine the timeout for a command based on its content."""
        if re.search(r"\b(gem install|pip install|cargo build|go build|go test|npm install|yarn install)\b", command, re.IGNORECASE):
            return 300
        if re.search(r"\b(make|cmake|gradle|mvn)\b", command, re.IGNORECASE):
            return 180
        if re.search(r"\b(python|python3|node)\b", command, re.IGNORECASE):
            return 120
        return default_timeout
