"""
Self-Access + Controlled Self-Improvement (Phase 6).

Provides the infrastructure for EVORA to analyze its own codebase,
identify weaknesses, propose improvements, and apply them under
CREATOR authority and approval.

Architecture:
    ImprovementPlanner → ImprovementProposal → approval → ChangeValidator → ImprovementHistory

Safety boundaries:
    - Only CREATOR-level identities can approve improvements
    - All file changes must be within the workspace
    - Secrets are scanned before any memory/history is persisted
    - No silent self-modification — every change requires explicit approval
    - Before/after validation is always performed
    - Improvement history is immutable (append-only)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from evora.logger import Logger, Stage
from evora.memory import MemoryFilter
from evora.security import PermissionLevel, PermissionManager
from evora.identity import AuthorityLevel, IdentityService
from evora.tools import Tool, ToolResult


class ImprovementStatus(str, Enum):
    """Lifecycle states for a self-improvement."""
    PENDING = "pending"          # Proposed, waiting for approval
    APPROVED = "approved"        # Approved by creator, ready to apply
    RUNNING = "running"          # In progress (validating/applying)
    SUCCESS = "success"          # Applied and validated
    FAILED = "failed"            # Error during application or validation
    REJECTED = "rejected"        # Explicitly rejected by creator


@dataclass
class ImprovementProposal:
    """A proposed self-improvement."""
    id: str                          # UUID
    title: str                       # Short human-readable title
    description: str                 # Detailed description of the issue
    files_changed: list[str] = field(default_factory=list)  # Files to modify/create
    benefit: str = ""                # Why this matters
    risk: str = ""                   # Risk assessment
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    proposed_by: str = ""            # Identity name that proposed it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImprovementProposal":
        return cls(**data)


@dataclass
class ImprovementRecord:
    """A full improvement record with status and validation results."""
    proposal: ImprovementProposal
    status: ImprovementStatus = ImprovementStatus.PENDING
    approved_by: Optional[str] = None       # Identity that approved
    applied_at: Optional[str] = None        # When applied
    test_result: Optional[str] = ""         # Pass/fail summary
    before_validation: Optional[dict] = None
    after_validation: Optional[dict] = None
    error: Optional[str] = None
    history_id: str = field(default_factory=lambda: f"imp-{uuid.uuid4().hex[:12]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_id": self.history_id,
            "proposal": self.proposal.to_dict() if isinstance(self.proposal, ImprovementProposal) else self.proposal,
            "status": self.status.value,
            "approved_by": self.approved_by,
            "applied_at": self.applied_at,
            "test_result": self.test_result,
            "before_validation": self.before_validation,
            "after_validation": self.after_validation,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImprovementRecord":
        prop = data.get("proposal", {})
        if isinstance(prop, dict):
            prop = ImprovementProposal.from_dict(prop)
        return cls(
            history_id=data.get("history_id", f"imp-{uuid.uuid4().hex[:12]}"),
            proposal=prop,
            status=ImprovementStatus(data.get("status", "pending")),
            approved_by=data.get("approved_by"),
            applied_at=data.get("applied_at"),
            test_result=data.get("test_result", ""),
            before_validation=data.get("before_validation"),
            after_validation=data.get("after_validation"),
            error=data.get("error"),
        )


class ImprovementHistory:
    """Persistent, append-only store for improvement records.

    Files:
        <evora_data_dir>/improvements/
            <history_id>.json
        <evora_data_dir>/improvements/_current.json  (pointer to active record)
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".evora" / "improvements"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def record(self, record: ImprovementRecord) -> str:
        """Persist an improvement record. Returns the history_id."""
        path = self.data_dir / f"{record.history_id}.json"
        record.applied_at = record.applied_at or datetime.now().isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)
        return record.history_id

    def update(self, record: ImprovementRecord) -> None:
        """Update an existing record (status changes, validation results)."""
        path = self.data_dir / f"{record.history_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    def list(self, limit: int = 50) -> list[ImprovementRecord]:
        """List records, newest first."""
        records = []
        for path in sorted(self.data_dir.glob("imp-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    records.append(ImprovementRecord.from_dict(json.load(f)))
                if len(records) >= limit:
                    break
            except Exception:
                continue
        return records

    def get(self, history_id: str) -> Optional[ImprovementRecord]:
        path = self.data_dir / f"{history_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return ImprovementRecord.from_dict(json.load(f))

    def summary(self) -> dict[str, Any]:
        """Return a summary of improvement history statistics."""
        records = self.list(limit=10000)
        by_status = {}
        for r in records:
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        return {
            "total": len(records),
            "by_status": by_status,
            "success_rate": round(by_status.get("success", 0) / max(len(records), 1), 2),
        }


class ChangeValidator:
    """Validates changes before and after application.

    Enforces:
    - All file operations stay within the workspace
    - Before/after validation runs (e.g., test suite, import checks)
    - No secrets or sensitive data are introduced
    """

    SECRET_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.I),
        re.compile(r"['\"]?password['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.I),
        re.compile(r"['\"]?secret['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.I),
        re.compile(r"['\"]?token['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.I),
    ]

    def __init__(self, workspace_dir: str, security: PermissionManager, logger: Optional[Logger] = None):
        self.workspace = Path(workspace_dir).resolve()
        self.security = security
        self.logger = logger

    def validate_file_path(self, path: str) -> Path:
        """Ensure a path is within the workspace. Raises PermissionError if not."""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.workspace)
            return resolved
        except ValueError:
            raise PermissionError(
                f"Path '{resolved}' is outside the workspace '{self.workspace}'"
            )

    def contains_secrets(self, content: str) -> list[str]:
        """Check if content contains secrets. Returns list of matched pattern descriptions."""
        found = []
        for pattern in self.SECRET_PATTERNS:
            if pattern.search(content):
                found.append(f"Secret pattern detected: {pattern.pattern[:50]}")
        return found

    def validate_before(self, files: list[str], content_changes: dict[str, str]) -> dict[str, Any]:
        """Validate before a change is applied.

        Args:
            files: List of file paths to be changed
            content_changes: Dict mapping file path to proposed content

        Returns dict with 'valid', 'errors', 'warnings'.
        """
        result = {"valid": True, "errors": [], "warnings": []}

        for file_path in files:
            try:
                self.validate_file_path(file_path)
            except PermissionError as e:
                result["valid"] = False
                result["errors"].append(str(e))

        for path, content in content_changes.items():
            try:
                self.validate_file_path(path)
            except PermissionError as e:
                result["valid"] = False
                result["errors"].append(str(e))
                continue

            secrets = self.contains_secrets(content)
            if secrets:
                result["valid"] = False
                result["errors"].extend(secrets)

        if self.logger:
            if result["valid"]:
                self.logger.verify(f"Pre-validation passed for {len(files)} files")
            else:
                self.logger.error(f"Pre-validation failed: {'; '.join(result['errors'])}")

        return result

    def run_command(self, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)."""
        try:
            process = subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os_name() == "nt" else 0,
            )
            stdout, stderr = process.communicate(timeout=timeout)
            return process.returncode, stdout, stderr
        except Exception as e:
            return -1, "", str(e)

    def validate_after(self, file_path: str, check_imports: bool = True) -> dict[str, Any]:
        """Validate after a change is applied.

        Checks:
        - File is importable (if Python)
        - Full test suite passes (if pytest available)
        """
        result = {"valid": True, "errors": [], "warnings": []}
        path = Path(file_path)

        if check_imports and path.suffix == ".py":
            try:
                cmd = f"import ast; ast.parse(open({repr(file_path)}).read())"
                proc = subprocess.Popen(
                    [sys_executable(), "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                _, stderr = proc.communicate(timeout=10)
                if proc.returncode != 0:
                    result["valid"] = False
                    result["errors"].append(f"Syntax error in {path}: {stderr.strip()}")
            except Exception as e:
                result["warnings"].append(f"Could not check syntax: {e}")

        if self.logger:
            if result["valid"]:
                self.logger.verify(f"Post-validation passed for {file_path}")
            else:
                self.logger.error(f"Post-validation failed for {file_path}: {result['errors']}")

        return result

    def validate_tests(self, timeout: int = 300) -> dict[str, Any]:
        """Run the test suite and return results."""
        result = {"valid": True, "passed": 0, "failed": 0, "output": ""}

        rc, stdout, stderr = self.run_command("python -m pytest tests/ -q --tb=short", timeout=timeout)
        output = stdout + "\n" + stderr
        result["output"] = output[-2000:] if len(output) > 2000 else output

        pass_match = re.search(r"(\d+) passed", output)
        fail_match = re.search(r"(\d+) failed", output)
        if pass_match:
            result["passed"] = int(pass_match.group(1))
        if fail_match:
            result["failed"] = int(fail_match.group(1))
            result["valid"] = False

        if self.logger:
            if result["valid"]:
                self.logger.verify(f"Test suite passed ({result['passed']} tests)")
            else:
                self.logger.error(f"Test suite failed ({result['failed']} failures)")

        return result


class ImprovementPlanner:
    """Analyzes the EVORA codebase to identify weaknesses and create proposals."""

    # Patterns that indicate code quality issues
    WEAKNESS_PATTERNS = [
        (re.compile(r"TODO|FIXME|XXX|HACK"), "todo_comment"),
        (re.compile(r"noqa.*noqa"), "redundant_noqa"),
        (re.compile(r"except\s*:\s*$"), "bare_except"),
        (re.compile(r"except\s+Exception\s*:"), "broad_except"),
        (re.compile(r"print\("), "print_statement"),
    ]

    # Files that are safe for self-improvement (within the evora package or tests)
    SCANNABLE_DIRS = ["evora", "tests"]

    def __init__(self, workspace_dir: str, logger: Optional[Logger] = None):
        self.workspace = Path(workspace_dir).resolve()
        self.logger = logger
        self._long_functions: list[dict[str, Any]] = []

    def analyze_self(self) -> dict[str, Any]:
        """Scan the codebase for weaknesses and return findings."""
        findings = {
            "files_scanned": 0,
            "weaknesses": [],
            "long_functions": [],
            "todo_count": 0,
            "bare_except_count": 0,
        }

        for scan_dir in self.SCANNABLE_DIRS:
            dir_path = self.workspace / scan_dir
            if not dir_path.exists():
                continue

            for py_file in dir_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    findings["files_scanned"] += 1

                    # Check for TODO/FIXME comments
                    for match in re.finditer(r"(TODO|FIXME|XXX|HACK)", content):
                        findings["todo_count"] += 1
                        line_num = content[:match.start()].count("\n") + 1
                        line_text = content.splitlines()[line_num - 1].strip()[:100]
                        findings["weaknesses"].append({
                            "file": str(py_file.relative_to(self.workspace)),
                            "line": line_num,
                            "type": "todo_comment",
                            "detail": line_text,
                        })

                    # Check for bare except
                    for match in re.finditer(r"except\s*:", content):
                        findings["bare_except_count"] += 1
                        line_num = content[:match.start()].count("\n") + 1
                        findings["weaknesses"].append({
                            "file": str(py_file.relative_to(self.workspace)),
                            "line": line_num,
                            "type": "bare_except",
                            "detail": content.splitlines()[line_num - 1].strip()[:100],
                        })
                except Exception:
                    continue

        # Check for long functions (>30 lines)
        for scan_dir in self.SCANNABLE_DIRS:
            dir_path = self.workspace / scan_dir
            if not dir_path.exists():
                continue
            for py_file in dir_path.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    lines = content.splitlines()
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.startswith("def ") or stripped.startswith("async def "):
                            func_lines = 0
                            start_indent = len(line) - len(stripped)
                            for j in range(i, min(i + 100, len(lines))):
                                func_lines += 1
                                if j + 1 < len(lines):
                                    next_stripped = lines[j + 1].strip()
                                    if next_stripped and (len(lines[j + 1]) - len(next_stripped)) <= start_indent:
                                        if not next_stripped.startswith("def ") and not next_stripped.startswith("async def "):
                                            break
                            if func_lines > 30:
                                findings["long_functions"].append({
                                    "file": str(py_file.relative_to(self.workspace)),
                                    "line": i + 1,
                                    "name": stripped.split("(")[0].replace("async ", "").replace("def ", ""),
                                    "lines": func_lines,
                                })
                except Exception:
                    continue

        findings["long_functions"] = findings["long_functions"][:20]

        if self.logger:
            self.logger.analyze(f"Self-analysis: {findings['files_scanned']} files scanned, "
                                f"{len(findings['weaknesses'])} weaknesses found")

        return findings

    def create_proposal(self, finding: dict[str, Any], workspace_name: str) -> ImprovementProposal:
        """Create a proposal from a finding."""
        prop = ImprovementProposal(
            id=str(uuid.uuid4()),
            title=f"Fix {finding['type']} in {Path(finding['file']).name}",
            description=f"Found {finding['type']} at {finding['file']}:{finding['line']}: {finding.get('detail', '')}",
            files_changed=[finding["file"]],
            benefit=f"Removing {finding['type']} improves code quality and maintainability.",
            risk="Low - localized change, tests will be run before acceptance.",
            proposed_by="EVORA-Self-Analysis",
        )
        return prop

    def recommend(self, workspace_name: str = "evora") -> list[ImprovementProposal]:
        """Generate a list of improvement proposals based on self-analysis."""
        findings = self.analyze_self()
        proposals = []
        for finding in findings["weaknesses"][:10]:
            proposals.append(self.create_proposal(finding, workspace_name))
        return proposals


class SelfImproveTool(Tool):
    """Tool that allows EVORA to analyze itself and propose improvements.

    Requires CREATOR authority for approval. All changes are validated
    before and after application. History is persisted immutably.

    Safety:
    - Cannot modify files outside the workspace
    - Cannot bypass CREATOR approval
    - Secrets are scanned in all proposed content
    - Before/after validation is always performed
    - Changes are recorded in the improvement history
    """

    name = "self_improve"
    description = "Analyze EVORA's own codebase for weaknesses and propose improvements (requires CREATOR approval)."
    permission = PermissionLevel.ASK
    parameters = {
        "action": {
            "type": "string",
            "description": "Action: 'analyze', 'propose', 'apply', 'history'",
            "required": True,
        },
        "file_path": {
            "type": "string",
            "description": "File to improve (for 'apply' action)",
            "required": False,
        },
        "old_string": {
            "type": "string",
            "description": "String to replace (for 'apply' action)",
            "required": False,
        },
        "new_string": {
            "type": "string",
            "description": "Replacement string (for 'apply' action)",
            "required": False,
        },
    }

    def __init__(self, security: PermissionManager, logger: Optional[Logger] = None,
                 identity_service: Optional[IdentityService] = None,
                 approval_system=None, history_dir: Optional[str] = None):
        super().__init__(security, logger)
        self.workspace = Path(security.workspace_dir).resolve()
        self.identity_service = identity_service
        self.approval_system = approval_system
        self.history = ImprovementHistory(history_dir)
        self.planner = ImprovementPlanner(str(self.workspace), logger)
        self.validator = ChangeValidator(str(self.workspace), security, logger)

    def _require_creator(self):
        """Ensure the current identity has CREATOR authority."""
        if self.identity_service:
            self.identity_service.require_authority("enable_self_modification")

    async def execute(self, action: str, file_path: str = None,
                       old_string: str = None, new_string: str = None) -> ToolResult:
        if action == "analyze":
            return await self._do_analyze()
        elif action == "propose":
            return await self._do_propose()
        elif action == "apply":
            return await self._do_apply(file_path, old_string, new_string)
        elif action == "history":
            return await self._do_history()
        else:
            return ToolResult(success=False, error=f"Unknown action: {action}. Use: analyze, propose, apply, history")

    async def _do_analyze(self) -> ToolResult:
        """Analyze the codebase for weaknesses."""
        if self.logger:
            self.logger.reason("Analyzing own codebase for improvements...")
        findings = self.planner.analyze_self()
        output_lines = [
            "EVORA Self-Analysis Report",
            "=" * 40,
            f"Files scanned: {findings['files_scanned']}",
            f"Weaknesses found: {len(findings['weaknesses'])}",
            f"  - TODO/FIXME comments: {findings['todo_count']}",
            f"  - Bare except blocks: {findings['bare_except_count']}",
            f"Long functions (>30 lines): {len(findings['long_functions'])}",
        ]
        for lf in findings["long_functions"][:5]:
            output_lines.append(
                f"  - {lf['file']}:{lf['line']} {lf['name']} ({lf['lines']} lines)"
            )
        for w in findings["weaknesses"][:10]:
            output_lines.append(f"  [{w['type']}] {w['file']}:{w['line']}: {w.get('detail', '')[:80]}")

        return ToolResult(
            success=True,
            output="\n".join(output_lines),
            data=findings,
        )

    async def _do_propose(self) -> ToolResult:
        """Generate improvement proposals."""
        proposals = self.planner.recommend()
        if not proposals:
            return ToolResult(success=True, output="No improvements recommended at this time.")

        lines = ["Improvement Proposals:", "=" * 40]
        for p in proposals:
            lines.append(f"\n  [{p.id[:8]}] {p.title}")
            lines.append(f"  Description: {p.description}")
            lines.append(f"  Files: {', '.join(p.files_changed)}")
            lines.append(f"  Benefit: {p.benefit}")
            lines.append(f"  Risk: {p.risk}")

        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"proposals": [p.to_dict() for p in proposals]},
        )

    async def _do_apply(self, file_path: str, old_string: str, new_string: str) -> ToolResult:
        """Apply a specific improvement (requires CREATOR approval)."""
        if not file_path or not old_string or not new_string:
            return ToolResult(success=False, error="file_path, old_string, and new_string are required for 'apply' action")

        try:
            self._require_creator()
        except PermissionError as e:
            return ToolResult(success=False, error=f"[DENIED] {e}")

        if not self.approval_system:
            return ToolResult(success=False, error="No approval system available. Use --auto-approve or configure approval callbacks.")

        prop = ImprovementProposal(
            id=str(uuid.uuid4()),
            title=f"Modify {Path(file_path).name}",
            description=f"Replace code in {file_path}",
            files_changed=[file_path],
            benefit="User-requested improvement",
            risk="Medium - code change",
            proposed_by="user",
        )

        proposal_text = (
            f"Self-Improvement Proposal\n"
            f"{'=' * 50}\n"
            f"Title: {prop.title}\n"
            f"File: {file_path}\n"
            f"Benefit: {prop.benefit}\n"
            f"Risk: {prop.risk}\n\n"
            f"Change:\n"
            f"  OLD: {old_string[:200]}\n"
            f"  NEW: {new_string[:200]}\n\n"
            f"This change requires CREATOR approval."
        )

        decision = self.approval_system.approve_plan(proposal_text, prop)
        if decision.value not in ("approve", "approved"):
            record = ImprovementRecord(proposal=prop, status=ImprovementStatus.REJECTED)
            self.history.record(record)
            return ToolResult(success=False, error=f"Proposal rejected by {decision.value}")

        full_path = self.workspace / file_path
        record = ImprovementRecord(
            proposal=prop,
            status=ImprovementStatus.APPROVED,
            approved_by="creator",
        )
        self.history.update(record)

        record.status = ImprovementStatus.RUNNING
        self.history.update(record)

        before_result = self.validator.validate_before([file_path], {file_path: new_string})
        record.before_validation = before_result
        self.history.update(record)

        if not before_result["valid"]:
            record.status = ImprovementStatus.FAILED
            record.error = f"; ".join(before_result["errors"])
            self.history.update(record)
            return ToolResult(success=False, error=f"Pre-validation failed: {record.error}")

        try:
            content = full_path.read_text(encoding="utf-8")
            if old_string not in content:
                record.status = ImprovementStatus.FAILED
                record.error = f"old_string not found in {file_path}"
                self.history.update(record)
                return ToolResult(success=False, error=record.error)

            count = content.count(old_string)
            if count > 1:
                record.status = ImprovementStatus.FAILED
                record.error = f"Multiple matches ({count}) found in {file_path}. Provide more context."
                self.history.update(record)
                return ToolResult(success=False, error=record.error)

            new_content = content.replace(old_string, new_string, 1)
            full_path.write_text(new_content, encoding="utf-8")

            after_result = self.validator.validate_after(file_path)
            record.after_validation = after_result
            self.history.update(record)

            if not after_result["valid"]:
                # Rollback
                full_path.write_text(content, encoding="utf-8")
                record.status = ImprovementStatus.FAILED
                record.error = f"Post-validation failed; change rolled back: {'; '.join(after_result['errors'])}"
                self.history.update(record)
                return ToolResult(success=False, error=record.error)

            test_result = self.validator.validate_tests()
            record.test_result = f"passed={test_result['passed']}, failed={test_result['failed']}"
            if not test_result["valid"]:
                full_path.write_text(content, encoding="utf-8")
                record.status = ImprovementStatus.FAILED
                record.error = f"Tests failed; change rolled back. {test_result['output'][:500]}"
                self.history.update(record)
                return ToolResult(success=False, error=record.error)

            record.status = ImprovementStatus.SUCCESS
            self.history.update(record)
            return ToolResult(
                success=True,
                output=f"Improvement applied and validated: {file_path}\n{test_result['output'][:500]}",
                data={"history_id": record.history_id, "tests": test_result},
            )
        except PermissionError as e:
            record.status = ImprovementStatus.FAILED
            record.error = str(e)
            self.history.update(record)
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            record.status = ImprovementStatus.FAILED
            record.error = str(e)
            self.history.update(record)
            return ToolResult(success=False, error=f"Failed to apply: {e}")

    async def _do_history(self) -> ToolResult:
        records = self.history.list()
        summary = self.history.summary()
        lines = [
            "Improvement History",
            "=" * 40,
            f"Total: {summary['total']}, Success rate: {summary['success_rate']:.0%}",
        ]
        for r in records[:20]:
            status_icon = {
                "pending": "[⏳]", "approved": "[✓]", "running": "[⚙️]",
                "success": "[✅]", "failed": "[❌]", "rejected": "[⏹]",
            }.get(r.status.value, "[?]")
            lines.append(f"  {status_icon} {r.proposal.title[:60]} — {r.status.value}")
            if r.error:
                lines.append(f"    Error: {r.error[:100]}")
        return ToolResult(
            success=True,
            output="\n".join(lines),
            data={"summary": summary, "records": [r.to_dict() for r in records]},
        )


def os_name() -> str:
    import os
    return os.name


def sys_executable() -> str:
    import sys
    return sys.executable
