"""
Development inspector for EVORA Phase 7.

Inspects the EVORA workspace and identifies:
- project structure
- available components
- tests
- Git state
- existing capabilities
- known weaknesses
- TODOs
- failing tests
- missing tests
- potential technical improvements
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from evora.logger import Logger
from evora.analyzer import ProjectAnalyzer
from evora.self_improve import ImprovementPlanner, ChangeValidator
from evora.security import PermissionManager


@dataclass
class InspectionFinding:
    """A single inspection finding."""

    category: str
    severity: str
    description: str
    location: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "location": self.location,
            "recommendation": self.recommendation,
        }


@dataclass
class InspectionReport:
    """Report from a development inspection."""

    findings: list[InspectionFinding] = field(default_factory=list)
    project_structure: dict[str, Any] = field(default_factory=dict)
    test_summary: dict[str, Any] = field(default_factory=dict)
    git_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "project_structure": self.project_structure,
            "test_summary": self.test_summary,
            "git_state": self.git_state,
            "metadata": self.metadata,
        }


class DevelopmentInspector:
    """Inspects the EVORA workspace for development opportunities."""

    def __init__(self, workspace_dir: str, security: PermissionManager, logger: Optional[Logger] = None):
        self.workspace = Path(workspace_dir).resolve()
        self.security = security
        self.logger = logger

    def inspect(self) -> InspectionReport:
        """Run a full inspection of the workspace."""
        report = InspectionReport()

        report.project_structure = self._inspect_structure()
        report.test_summary = self._inspect_tests()
        report.git_state = self._inspect_git()
        report.findings = self._collect_findings(report)

        if self.logger:
            self.logger.analyze(f"Inspection complete: {len(report.findings)} findings")

        return report

    def _inspect_structure(self) -> dict[str, Any]:
        """Inspect project structure."""
        structure = {
            "evora_modules": [],
            "test_files": [],
            "doc_files": [],
            "config_files": [],
            "total_py_files": 0,
        }

        if not self.workspace.exists():
            return structure

        for py_file in self.workspace.rglob("*.py"):
            rel = str(py_file.relative_to(self.workspace)).replace("\\", "/")
            structure["total_py_files"] += 1
            if rel.startswith("evora/"):
                structure["evora_modules"].append(rel)
            elif rel.startswith("tests/"):
                structure["test_files"].append(rel)

        for doc_file in self.workspace.rglob("*.md"):
            rel = str(doc_file.relative_to(self.workspace))
            structure["doc_files"].append(rel)

        for config_file in ["pyproject.toml", "pytest.ini", ".gitignore"]:
            if (self.workspace / config_file).exists():
                structure["config_files"].append(config_file)

        return structure

    def _inspect_tests(self) -> dict[str, Any]:
        """Run tests and return summary."""
        summary = {"total": 0, "passed": 0, "failed": 0, "errors": []}

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(timeout=120)
            output = stdout + "\n" + stderr

            import re
            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)
            if passed_match:
                summary["passed"] = int(passed_match.group(1))
                summary["total"] += summary["passed"]
            if failed_match:
                summary["failed"] = int(failed_match.group(1))
                summary["total"] += summary["failed"]
                summary["errors"].append(f"{summary['failed']} test(s) failing")
        except Exception as e:
            summary["errors"].append(f"Test inspection failed: {e}")

        return summary

    def _inspect_git(self) -> dict[str, Any]:
        """Inspect Git state."""
        git_state = {"branch": "unknown", "dirty": False, "commits": 0}

        try:
            proc = subprocess.Popen(
                ["git", "branch", "--show-current"],
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, _ = proc.communicate(timeout=10)
            git_state["branch"] = stdout.strip() or "unknown"

            proc = subprocess.Popen(
                ["git", "status", "--porcelain"],
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, _ = proc.communicate(timeout=10)
            git_state["dirty"] = bool(stdout.strip())

            proc = subprocess.Popen(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(self.workspace),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, _ = proc.communicate(timeout=10)
            try:
                git_state["commits"] = int(stdout.strip())
            except ValueError:
                pass
        except Exception:
            pass

        return git_state

    def _collect_findings(self, report: InspectionReport) -> list[InspectionFinding]:
        """Collect findings from inspection data."""
        findings = []

        if report.test_summary.get("failed", 0) > 0:
            findings.append(InspectionFinding(
                category="tests",
                severity="high",
                description=f"{report.test_summary['failed']} failing tests detected",
                recommendation="Investigate and fix failing tests before other improvements",
            ))

        if report.git_state.get("dirty"):
            findings.append(InspectionFinding(
                category="git",
                severity="medium",
                description="Working tree has uncommitted changes",
                recommendation="Commit or stash changes before self-development",
            ))

        evora_modules = report.project_structure.get("evora_modules", [])
        if len(evora_modules) > 20:
            findings.append(InspectionFinding(
                category="architecture",
                severity="low",
                description=f"Large codebase: {len(evora_modules)} Python modules",
                recommendation="Consider modularization or documentation improvements",
            ))

        doc_files = report.project_structure.get("doc_files", [])
        if len(doc_files) < 3:
            findings.append(InspectionFinding(
                category="documentation",
                severity="low",
                description="Limited documentation files",
                recommendation="Add or expand documentation for better maintainability",
            ))

        return findings
