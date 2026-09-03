"""
Phase 31 — Native Integration Tester tests.

Verifies:
1. IntegrationTest has correct structure
2. IntegrationReport has correct structure
3. IntegrationTestStatus enum exists
4. NativeIntegrationTester initializes
5. NativeIntegrationTester defines tests
6. NativeIntegrationTester runs test
7. NativeIntegrationTester runs all tests
8. NativeIntegrationTester gets test by ID
9. NativeIntegrationTester gets report
10. NativeIntegrationTester returns test summary
11. No ModelManager dependency
12. No external dependencies
13. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.integration_tester import (
    IntegrationReport,
    IntegrationTest,
    IntegrationTestStatus,
    NativeIntegrationTester,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def integration_tester():
    return NativeIntegrationTester(logger=MagicMock())


# ---------------------------------------------------------------------------
# TestIntegrationTest
# ---------------------------------------------------------------------------

class TestIntegrationTest:
    """Test IntegrationTest."""

    def test_default_test(self):
        test = IntegrationTest()
        assert test.test_id != ""
        assert test.status == IntegrationTestStatus.PENDING

    def test_test_to_dict(self):
        test = IntegrationTest(name="Test connectivity", components=["agent", "runtime"])
        data = test.to_dict()
        assert data["name"] == "Test connectivity"
        assert data["status"] == "pending"


# ---------------------------------------------------------------------------
# TestIntegrationReport
# ---------------------------------------------------------------------------

class TestIntegrationReport:
    """Test IntegrationReport."""

    def test_default_report(self):
        report = IntegrationReport()
        assert report.report_id != ""
        assert report.total_tests == 0

    def test_report_to_dict(self):
        report = IntegrationReport(total_tests=10, passed=8, failed=2)
        data = report.to_dict()
        assert data["total_tests"] == 10
        assert data["success_rate"] == 0.8


# ---------------------------------------------------------------------------
# TestNativeIntegrationTester
# ---------------------------------------------------------------------------

class TestNativeIntegrationTester:
    """Test NativeIntegrationTester."""

    def test_integration_tester_initializes(self, integration_tester):
        assert integration_tester is not None

    def test_define_test(self, integration_tester):
        test = integration_tester.define_test(
            "Test agent-runtime integration",
            components=["agent", "runtime"],
            test_type="connectivity",
        )
        assert test.test_id != ""
        assert test.name == "Test agent-runtime integration"

    def test_run_test(self, integration_tester):
        test = integration_tester.define_test("Test", components=["agent", "runtime"])
        result = integration_tester.run_test(test.test_id)
        assert result.status in (IntegrationTestStatus.PASSED, IntegrationTestStatus.FAILED)

    def test_run_test_missing(self, integration_tester):
        result = integration_tester.run_test("nonexistent")
        assert result.status == IntegrationTestStatus.FAILED

    def test_run_all_tests(self, integration_tester):
        integration_tester.define_test("Test 1", components=["agent"])
        integration_tester.define_test("Test 2", components=["runtime"])
        report = integration_tester.run_all_tests()
        assert report.total_tests == 2

    def test_run_all_tests_empty(self, integration_tester):
        report = integration_tester.run_all_tests()
        assert report.total_tests == 0

    def test_get_test(self, integration_tester):
        test = integration_tester.define_test("Test", components=["agent"])
        retrieved = integration_tester.get_test(test.test_id)
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_test_missing(self, integration_tester):
        retrieved = integration_tester.get_test("nonexistent")
        assert retrieved is None

    def test_get_report(self, integration_tester):
        integration_tester.define_test("Test", components=["agent"])
        integration_tester.run_all_tests()
        report = integration_tester.get_report()
        assert report is not None
        assert report.total_tests > 0

    def test_get_report_by_id(self, integration_tester):
        integration_tester.define_test("Test", components=["agent"])
        report1 = integration_tester.run_all_tests()
        report = integration_tester.get_report(report1.report_id)
        assert report is not None
        assert report.report_id == report1.report_id

    def test_get_report_missing(self, integration_tester):
        report = integration_tester.get_report("nonexistent")
        assert report is None

    def test_get_test_summary(self, integration_tester):
        integration_tester.define_test("Test 1", components=["agent"])
        integration_tester.define_test("Test 2", components=["runtime"])
        summary = integration_tester.get_test_summary()
        assert summary["total_tests"] == 2


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 31 security boundaries."""

    def test_no_model_manager_in_integration_tester(self):
        import evora.brain.intelligence.integration_tester as int_mod
        source = Path(int_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.integration_tester as int_mod
        source = Path(int_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 31 works offline."""

    def test_integration_tester_works_offline(self, integration_tester):
        test = integration_tester.define_test("offline test", components=["agent"])
        assert test is not None

    def test_run_test_offline(self, integration_tester):
        test = integration_tester.define_test("offline", components=["agent"])
        result = integration_tester.run_test(test.test_id)
        assert result.status in (IntegrationTestStatus.PASSED, IntegrationTestStatus.FAILED)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 31 architecture readiness."""

    def test_native_integration_tester_exists(self):
        from evora.brain.intelligence.integration_tester import NativeIntegrationTester
        assert NativeIntegrationTester is not None

    def test_integration_test_exists(self):
        from evora.brain.intelligence.integration_tester import IntegrationTest
        assert IntegrationTest is not None

    def test_integration_report_exists(self):
        from evora.brain.intelligence.integration_tester import IntegrationReport
        assert IntegrationReport is not None

    def test_integration_test_status_enum_exists(self):
        from evora.brain.intelligence.integration_tester import IntegrationTestStatus
        assert IntegrationTestStatus.PENDING is not None
        assert IntegrationTestStatus.PASSED is not None
        assert IntegrationTestStatus.FAILED is not None
