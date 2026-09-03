"""
Phase 32 — Native Benchmark Runner tests.

Verifies:
1. BenchmarkResult has correct structure
2. BenchmarkReport has correct structure
3. NativeBenchmarkRunner initializes
4. NativeBenchmarkRunner registers benchmark
5. NativeBenchmarkRunner runs benchmark
6. NativeBenchmarkRunner gets results
7. NativeBenchmarkRunner gets benchmark report
8. NativeBenchmarkRunner compares benchmarks
9. No ModelManager dependency
10. No external dependencies
11. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.benchmark_runner import (
    BenchmarkReport,
    BenchmarkResult,
    NativeBenchmarkRunner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def benchmark_runner():
    return NativeBenchmarkRunner(logger=MagicMock())


@pytest.fixture
def runner_with_benchmark():
    runner = NativeBenchmarkRunner(logger=MagicMock())
    runner.register_benchmark("fast_task", lambda: "done")
    runner.register_benchmark("slow_task", lambda: time.sleep(0.01) or "done")
    return runner


# ---------------------------------------------------------------------------
# TestBenchmarkResult
# ---------------------------------------------------------------------------

class TestBenchmarkResult:
    """Test BenchmarkResult."""

    def test_default_result(self):
        result = BenchmarkResult()
        assert result.result_id != ""
        assert result.duration == 0.0

    def test_result_to_dict(self):
        result = BenchmarkResult(benchmark_name="test", duration=1.5, score=0.8)
        data = result.to_dict()
        assert data["benchmark_name"] == "test"
        assert data["duration"] == 1.5


# ---------------------------------------------------------------------------
# TestBenchmarkReport
# ---------------------------------------------------------------------------

class TestBenchmarkReport:
    """Test BenchmarkReport."""

    def test_default_report(self):
        report = BenchmarkReport()
        assert report.report_id != ""
        assert report.average_duration == 0.0

    def test_report_to_dict(self):
        report = BenchmarkReport(benchmark_name="test", average_duration=1.0, average_score=0.9)
        data = report.to_dict()
        assert data["benchmark_name"] == "test"
        assert data["average_score"] == 0.9


# ---------------------------------------------------------------------------
# TestNativeBenchmarkRunner
# ---------------------------------------------------------------------------

class TestNativeBenchmarkRunner:
    """Test NativeBenchmarkRunner."""

    def test_benchmark_runner_initializes(self, benchmark_runner):
        assert benchmark_runner is not None

    def test_register_benchmark(self, benchmark_runner):
        benchmark_runner.register_benchmark("test", lambda: "done")
        assert "test" in benchmark_runner._benchmarks

    def test_run_benchmark(self, benchmark_runner):
        benchmark_runner.register_benchmark("test", lambda: "done")
        report = benchmark_runner.run_benchmark("test")
        assert isinstance(report, BenchmarkReport)
        assert report.benchmark_name == "test"
        assert len(report.results) > 0

    def test_run_benchmark_multiple_iterations(self, benchmark_runner):
        benchmark_runner.register_benchmark("test", lambda: "done")
        report = benchmark_runner.run_benchmark("test", iterations=3)
        assert len(report.results) == 3

    def test_run_benchmark_missing(self, benchmark_runner):
        report = benchmark_runner.run_benchmark("nonexistent")
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 0

    def test_get_results(self, benchmark_runner):
        benchmark_runner.register_benchmark("test1", lambda: "done")
        benchmark_runner.run_benchmark("test1")
        results = benchmark_runner.get_results()
        assert len(results) > 0

    def test_get_results_filtered(self, benchmark_runner):
        benchmark_runner.register_benchmark("test1", lambda: "done")
        benchmark_runner.register_benchmark("test2", lambda: "done")
        benchmark_runner.run_benchmark("test1")
        results = benchmark_runner.get_results("test1")
        assert all(r.benchmark_name == "test1" for r in results)

    def test_get_benchmark_report(self, benchmark_runner):
        benchmark_runner.register_benchmark("test", lambda: "done")
        benchmark_runner.run_benchmark("test")
        report = benchmark_runner.get_benchmark_report("test")
        assert report is not None
        assert report.benchmark_name == "test"

    def test_get_benchmark_report_missing(self, benchmark_runner):
        report = benchmark_runner.get_benchmark_report("nonexistent")
        assert report is None

    def test_compare_benchmarks(self, benchmark_runner):
        benchmark_runner.register_benchmark("fast", lambda: "done")
        benchmark_runner.register_benchmark("slow", lambda: time.sleep(0.01) or "done")
        benchmark_runner.run_benchmark("fast")
        benchmark_runner.run_benchmark("slow")
        comparison = benchmark_runner.compare_benchmarks("fast", "slow")
        assert "faster" in comparison
        assert comparison["faster"] == "fast"


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 32 security boundaries."""

    def test_no_model_manager_in_benchmark(self):
        import evora.brain.intelligence.benchmark_runner as bench_mod
        source = Path(bench_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.benchmark_runner as bench_mod
        source = Path(bench_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 32 works offline."""

    def test_benchmark_runner_works_offline(self, benchmark_runner):
        benchmark_runner.register_benchmark("offline_test", lambda: "done")
        report = benchmark_runner.run_benchmark("offline_test")
        assert isinstance(report, BenchmarkReport)

    def test_get_results_offline(self, benchmark_runner):
        benchmark_runner.register_benchmark("test", lambda: "done")
        benchmark_runner.run_benchmark("test")
        results = benchmark_runner.get_results()
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 32 architecture readiness."""

    def test_native_benchmark_runner_exists(self):
        from evora.brain.intelligence.benchmark_runner import NativeBenchmarkRunner
        assert NativeBenchmarkRunner is not None

    def test_benchmark_result_exists(self):
        from evora.brain.intelligence.benchmark_runner import BenchmarkResult
        assert BenchmarkResult is not None

    def test_benchmark_report_exists(self):
        from evora.brain.intelligence.benchmark_runner import BenchmarkReport
        assert BenchmarkReport is not None
