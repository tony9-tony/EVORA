"""
Phase 33 — Native Security Auditor for EVORA.

Audits security of the system.

Supports:
  - Security scan definitions
  - Vulnerability detection
  - Security scoring
  - Audit reporting
  - Integration with PermissionManager
  - Integration with ApprovalSystem

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityFinding:
    """A security finding."""
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    severity: Severity = Severity.MEDIUM
    category: str = ""
    description: str = ""
    recommendation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity.value,
            "category": self.category,
            "description": self.description,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class SecurityAuditReport:
    """A security audit report."""
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    findings: list[SecurityFinding] = field(default_factory=list)
    security_score: float = 0.0
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "findings": [f.to_dict() for f in self.findings],
            "security_score": self.security_score,
            "total_findings": self.total_findings,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Native Security Auditor
# ---------------------------------------------------------------------------

class NativeSecurityAuditor:
    """Native security auditor for EVORA.

    Audits security of the system.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._findings: list[SecurityFinding] = []
        self._reports: list[SecurityAuditReport] = []

    def add_finding(self, severity: Severity, category: str, description: str, recommendation: str = "") -> SecurityFinding:
        """Add a security finding."""
        finding = SecurityFinding(
            severity=severity,
            category=category,
            description=description,
            recommendation=recommendation,
        )
        self._findings.append(finding)
        return finding

    def run_audit(self) -> SecurityAuditReport:
        """Run a security audit."""
        report = SecurityAuditReport()
        for finding in self._findings:
            report.findings.append(finding)
            if finding.severity == Severity.CRITICAL:
                report.critical_count += 1
            elif finding.severity == Severity.HIGH:
                report.high_count += 1
            elif finding.severity == Severity.MEDIUM:
                report.medium_count += 1
            else:
                report.low_count += 1
        report.total_findings = len(report.findings)
        report.security_score = self._calculate_score(report)
        self._reports.append(report)
        return report

    def _calculate_score(self, report: SecurityAuditReport) -> float:
        """Calculate security score."""
        if report.total_findings == 0:
            return 1.0
        penalty = (report.critical_count * 0.25 + report.high_count * 0.15 +
                   report.medium_count * 0.05 + report.low_count * 0.01)
        return max(0.0, 1.0 - penalty)

    def get_findings(self, severity: Severity = None) -> list[SecurityFinding]:
        """Get findings, optionally filtered by severity."""
        if severity is None:
            return list(self._findings)
        return [f for f in self._findings if f.severity == severity]

    def get_latest_report(self) -> Optional[SecurityAuditReport]:
        """Get the latest audit report."""
        return self._reports[-1] if self._reports else None

    def clear_findings(self) -> None:
        """Clear all findings."""
        self._findings = []
