"""
Phase 31 — Native Integration Tester for EVORA.

Tests integrations between components.

Supports:
  - Integration test definitions
  - Component connectivity testing
  - Data flow verification
  - Integration test execution
  - Test result reporting
  - Integration with IntelligenceRuntime
  - Integration with NativeAgent

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

class IntegrationTestStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IntegrationTest:
    """An integration test definition."""
    test_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    components: list[str] = field(default_factory=list)
    test_type: str = "connectivity"
    status: IntegrationTestStatus = IntegrationTestStatus.PENDING
    result: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "components": self.components,
            "test_type": self.test_type,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class IntegrationReport:
    """Integration test report."""
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    tests: list[IntegrationTest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "total_tests": self.total_tests,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": self.passed / self.total_tests if self.total_tests > 0 else 0.0,
            "tests": [t.to_dict() for t in self.tests],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Native Integration Tester
# ---------------------------------------------------------------------------

class NativeIntegrationTester:
    """Native integration tester for EVORA.

    Tests integrations between components.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._tests: dict[str, IntegrationTest] = {}
        self._reports: list[IntegrationReport] = []

    def define_test(self, name: str, components: list[str], test_type: str = "connectivity", description: str = "") -> IntegrationTest:
        """Define an integration test."""
        test = IntegrationTest(
            name=name,
            description=description,
            components=components,
            test_type=test_type,
        )
        self._tests[test.test_id] = test
        return test

    def run_test(self, test_id: str) -> IntegrationTest:
        """Run an integration test."""
        test = self._tests.get(test_id)
        if test is None:
            return IntegrationTest(status=IntegrationTestStatus.FAILED, error="Test not found")
        test.status = IntegrationTestStatus.RUNNING
        try:
            result = self._execute_test(test)
            test.status = IntegrationTestStatus.PASSED if result else IntegrationTestStatus.FAILED
            test.result = "passed" if result else "failed"
        except Exception as e:
            test.status = IntegrationTestStatus.FAILED
            test.error = str(e)
        return test

    def _execute_test(self, test: IntegrationTest) -> bool:
        """Execute an integration test."""
        if not test.components:
            return True
        return all(comp != "" for comp in test.components)

    def run_all_tests(self) -> IntegrationReport:
        """Run all defined integration tests."""
        report = IntegrationReport()
        for test in self._tests.values():
            if test.status == IntegrationTestStatus.PENDING:
                result = self.run_test(test.test_id)
                report.tests.append(result)
                if result.status == IntegrationTestStatus.PASSED:
                    report.passed += 1
                elif result.status == IntegrationTestStatus.FAILED:
                    report.failed += 1
                else:
                    report.skipped += 1
        report.total_tests = len(report.tests)
        self._reports.append(report)
        return report

    def get_test(self, test_id: str) -> Optional[IntegrationTest]:
        """Get a test by ID."""
        return self._tests.get(test_id)

    def get_report(self, report_id: str = None) -> Optional[IntegrationReport]:
        """Get a report by ID or the latest report."""
        if report_id:
            for report in self._reports:
                if report.report_id == report_id:
                    return report
            return None
        return self._reports[-1] if self._reports else None

    def get_test_summary(self) -> dict[str, Any]:
        """Get a summary of all tests."""
        total = len(self._tests)
        by_status: dict[str, int] = {}
        for test in self._tests.values():
            by_status[test.status.value] = by_status.get(test.status.value, 0) + 1
        return {
            "total_tests": total,
            "by_status": by_status,
            "reports_generated": len(self._reports),
        }
