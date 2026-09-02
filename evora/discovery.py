"""
Improvement discovery for EVORA Phase 7.

Generates improvement candidates from inspection findings.
Each candidate contains enough information for EVORA and the creator
to understand the proposed change.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from evora.logger import Logger
from evora.inspector import InspectionReport, InspectionFinding
from evora.self_improve import ImprovementProposal


@dataclass
class ImprovementCandidate:
    """A proposed improvement candidate."""

    id: str
    title: str
    description: str
    category: str
    severity: str
    affected_files: list[str] = field(default_factory=list)
    benefit: str = ""
    risks: list[str] = field(default_factory=list)
    validation_strategy: str = ""
    estimated_scope: str = "medium"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "severity": self.severity,
            "affected_files": self.affected_files,
            "benefit": self.benefit,
            "risks": self.risks,
            "validation_strategy": self.validation_strategy,
            "estimated_scope": self.estimated_scope,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImprovementCandidate":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_proposal(self) -> ImprovementProposal:
        """Convert to a Phase 6 ImprovementProposal for approval."""
        return ImprovementProposal(
            id=self.id,
            title=self.title,
            description=self.description,
            files_changed=self.affected_files,
            benefit=self.benefit,
            risk="; ".join(self.risks) if self.risks else "Low",
            proposed_by="EVORA-Self-Development",
        )


class ImprovementDiscovery:
    """Generates improvement candidates from inspection findings."""

    IMPROVEMENT_PATTERNS = {
        "tests": {
            "template": "Fix failing test: {description}",
            "benefit": "Restores test suite health and prevents regressions",
            "validation": "Run pytest to verify tests pass",
            "scope": "medium",
        },
        "git": {
            "template": "Clean working tree: {description}",
            "benefit": "Ensures clean development state for self-improvement",
            "validation": "Verify git status is clean",
            "scope": "low",
        },
        "architecture": {
            "template": "Improve architecture: {description}",
            "benefit": "Better maintainability and reduced complexity",
            "validation": "Run tests and verify imports",
            "scope": "high",
        },
        "documentation": {
            "template": "Improve documentation: {description}",
            "benefit": "Better developer experience and maintainability",
            "validation": "Verify documentation completeness",
            "scope": "low",
        },
        "security": {
            "template": "Address security concern: {description}",
            "benefit": "Reduces attack surface and improves safety",
            "validation": "Run security tests and verify protections",
            "scope": "high",
        },
    }

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def discover(self, report: InspectionReport) -> list[ImprovementCandidate]:
        """Generate improvement candidates from inspection findings."""
        candidates = []

        for finding in report.findings:
            candidate = self._finding_to_candidate(finding)
            if candidate:
                candidates.append(candidate)

        if self.logger:
            self.logger.analyze(f"Discovery complete: {len(candidates)} candidates")

        return candidates

    def _finding_to_candidate(self, finding: InspectionFinding) -> Optional[ImprovementCandidate]:
        """Convert an inspection finding to an improvement candidate."""
        pattern = self.IMPROVEMENT_PATTERNS.get(finding.category)
        if not pattern:
            return None

        title = pattern["template"].format(description=finding.description)
        affected_files = [finding.location] if finding.location else []

        return ImprovementCandidate(
            id=f"imp-{uuid.uuid4().hex[:12]}",
            title=title,
            description=finding.recommendation or finding.description,
            category=finding.category,
            severity=finding.severity,
            affected_files=affected_files,
            benefit=pattern["benefit"],
            risks=["Requires creator approval", "May affect existing functionality"],
            validation_strategy=pattern["validation"],
            estimated_scope=pattern["scope"],
            metadata={"source_finding": finding.to_dict()},
        )
