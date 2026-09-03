"""
Phase 33 — Native Security Auditor tests.

Verifies:
1. SecurityFinding has correct structure
2. SecurityAuditReport has correct structure
3. Severity enum exists
4. NativeSecurityAuditor initializes
5. NativeSecurityAuditor adds findings
6. NativeSecurityAuditor runs audit
7. NativeSecurityAuditor calculates security score
8. NativeSecurityAuditor gets findings
9. NativeSecurityAuditor gets latest report
10. NativeSecurityAuditor clears findings
11. No ModelManager dependency
12. No external dependencies
13. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.security_auditor import (
    NativeSecurityAuditor,
    SecurityAuditReport,
    SecurityFinding,
    Severity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def security_auditor():
    return NativeSecurityAuditor(logger=MagicMock())


@pytest.fixture
def auditor_with_findings():
    auditor = NativeSecurityAuditor(logger=MagicMock())
    auditor.add_finding(Severity.CRITICAL, "injection", "SQL injection risk", "Use parameterized queries")
    auditor.add_finding(Severity.HIGH, "auth", "Weak authentication", "Use strong authentication")
    auditor.add_finding(Severity.LOW, "logging", "Missing logs", "Add logging")
    return auditor


# ---------------------------------------------------------------------------
# TestSecurityFinding
# ---------------------------------------------------------------------------

class TestSecurityFinding:
    """Test SecurityFinding."""

    def test_default_finding(self):
        finding = SecurityFinding()
        assert finding.finding_id != ""
        assert finding.severity == Severity.MEDIUM

    def test_finding_to_dict(self):
        finding = SecurityFinding(severity=Severity.HIGH, category="injection", description="Risk")
        data = finding.to_dict()
        assert data["severity"] == "high"
        assert data["category"] == "injection"


# ---------------------------------------------------------------------------
# TestSecurityAuditReport
# ---------------------------------------------------------------------------

class TestSecurityAuditReport:
    """Test SecurityAuditReport."""

    def test_default_report(self):
        report = SecurityAuditReport()
        assert report.report_id != ""
        assert report.security_score == 0.0

    def test_report_to_dict(self):
        report = SecurityAuditReport(security_score=0.8, total_findings=5, critical_count=1)
        data = report.to_dict()
        assert data["security_score"] == 0.8
        assert data["critical_count"] == 1


# ---------------------------------------------------------------------------
# TestNativeSecurityAuditor
# ---------------------------------------------------------------------------

class TestNativeSecurityAuditor:
    """Test NativeSecurityAuditor."""

    def test_security_auditor_initializes(self, security_auditor):
        assert security_auditor is not None

    def test_add_finding(self, security_auditor):
        finding = security_auditor.add_finding(Severity.HIGH, "injection", "SQL injection")
        assert finding.finding_id != ""
        assert finding.severity == Severity.HIGH

    def test_run_audit_no_findings(self, security_auditor):
        report = security_auditor.run_audit()
        assert report.security_score == 1.0
        assert report.total_findings == 0

    def test_run_audit_with_findings(self, auditor_with_findings):
        report = auditor_with_findings.run_audit()
        assert report.total_findings == 3
        assert report.critical_count == 1
        assert report.high_count == 1
        assert report.low_count == 1
        assert report.security_score < 1.0

    def test_security_score_calculation(self, auditor_with_findings):
        report = auditor_with_findings.run_audit()
        assert 0.0 <= report.security_score <= 1.0

    def test_get_findings(self, auditor_with_findings):
        findings = auditor_with_findings.get_findings()
        assert len(findings) == 3

    def test_get_findings_filtered(self, auditor_with_findings):
        critical = auditor_with_findings.get_findings(Severity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].severity == Severity.CRITICAL

    def test_get_latest_report(self, auditor_with_findings):
        auditor_with_findings.run_audit()
        report = auditor_with_findings.get_latest_report()
        assert report is not None
        assert report.total_findings == 3

    def test_get_latest_report_no_reports(self, security_auditor):
        report = security_auditor.get_latest_report()
        assert report is None

    def test_clear_findings(self, auditor_with_findings):
        auditor_with_findings.clear_findings()
        findings = auditor_with_findings.get_findings()
        assert len(findings) == 0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 33 security boundaries."""

    def test_no_model_manager_in_auditor(self):
        import evora.brain.intelligence.security_auditor as sec_mod
        source = Path(sec_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.security_auditor as sec_mod
        source = Path(sec_mod.__file__).read_text(encoding="utf-8")
        import_section = False
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_section = True
            elif import_section and stripped and not stripped.startswith("#"):
                break
            if import_section:
                for forbidden in ["openai", "anthropic", "ollama", "requests", "aiohttp", "httpx", "urllib", "socket"]:
                    assert forbidden not in stripped.lower(), f"Found forbidden dependency: {forbidden}"


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 33 works offline."""

    def test_security_auditor_works_offline(self, security_auditor):
        finding = security_auditor.add_finding(Severity.LOW, "test", "test finding")
        assert finding is not None

    def test_audit_offline(self, security_auditor):
        security_auditor.add_finding(Severity.HIGH, "test", "risk")
        report = security_auditor.run_audit()
        assert isinstance(report, SecurityAuditReport)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 33 architecture readiness."""

    def test_native_security_auditor_exists(self):
        from evora.brain.intelligence.security_auditor import NativeSecurityAuditor
        assert NativeSecurityAuditor is not None

    def test_security_finding_exists(self):
        from evora.brain.intelligence.security_auditor import SecurityFinding
        assert SecurityFinding is not None

    def test_security_audit_report_exists(self):
        from evora.brain.intelligence.security_auditor import SecurityAuditReport
        assert SecurityAuditReport is not None

    def test_severity_enum_exists(self):
        from evora.brain.intelligence.security_auditor import Severity
        assert Severity.LOW is not None
        assert Severity.CRITICAL is not None
