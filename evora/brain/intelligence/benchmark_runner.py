"""
Phase 32 — Native Benchmark Runner for EVORA.

Runs benchmarks to measure performance.

Supports:
  - Benchmark definition
  - Benchmark execution
  - Performance measurement
  - Result comparison
  - Benchmark reporting
  - Integration with NativeAgent
  - Integration with ExecutionMonitor

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A benchmark result."""
    result_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    benchmark_name: str = ""
    duration: float = 0.0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "benchmark_name": self.benchmark_name,
            "duration": self.duration,
            "score": self.score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkReport:
    """A benchmark report."""
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    benchmark_name: str = ""
    results: list[BenchmarkResult] = field(default_factory=list)
    average_duration: float = 0.0
    average_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "benchmark_name": self.benchmark_name,
            "results": [r.to_dict() for r in self.results],
            "average_duration": self.average_duration,
            "average_score": self.average_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Native Benchmark Runner
# ---------------------------------------------------------------------------

class NativeBenchmarkRunner:
    """Native benchmark runner for EVORA.

    Runs benchmarks to measure performance.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._benchmarks: dict[str, Any] = {}
        self._results: list[BenchmarkResult] = []

    def register_benchmark(self, name: str, func: Any, metadata: dict[str, Any] = None) -> None:
        """Register a benchmark function."""
        self._benchmarks[name] = {
            "func": func,
            "metadata": metadata or {},
        }

    def run_benchmark(self, name: str, iterations: int = 1) -> BenchmarkReport:
        """Run a benchmark."""
        if name not in self._benchmarks:
            return BenchmarkReport(benchmark_name=name, results=[])
        benchmark = self._benchmarks[name]
        func = benchmark["func"]
        results = []
        for _ in range(iterations):
            start = time.time()
            try:
                output = func()
                duration = time.time() - start
                score = 1.0 / (duration + 0.001)
            except Exception as e:
                duration = time.time() - start
                score = 0.0
                output = str(e)
            result = BenchmarkResult(
                benchmark_name=name,
                duration=duration,
                score=score,
                metadata={"output": str(output)[:100]},
            )
            results.append(result)
            self._results.append(result)
        avg_duration = sum(r.duration for r in results) / len(results) if results else 0.0
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        report = BenchmarkReport(
            benchmark_name=name,
            results=results,
            average_duration=avg_duration,
            average_score=avg_score,
            metadata=benchmark["metadata"],
        )
        return report

    def get_results(self, benchmark_name: str = None) -> list[BenchmarkResult]:
        """Get benchmark results."""
        if benchmark_name:
            return [r for r in self._results if r.benchmark_name == benchmark_name]
        return list(self._results)

    def get_benchmark_report(self, benchmark_name: str) -> Optional[BenchmarkReport]:
        """Get a benchmark report by name."""
        results = self.get_results(benchmark_name)
        if not results:
            return None
        avg_duration = sum(r.duration for r in results) / len(results)
        avg_score = sum(r.score for r in results) / len(results)
        return BenchmarkReport(
            benchmark_name=benchmark_name,
            results=results,
            average_duration=avg_duration,
            average_score=avg_score,
        )

    def compare_benchmarks(self, name1: str, name2: str) -> dict[str, Any]:
        """Compare two benchmarks."""
        report1 = self.get_benchmark_report(name1)
        report2 = self.get_benchmark_report(name2)
        if not report1 or not report2:
            return {"error": "One or both benchmarks not found"}
        return {
            "benchmark1": name1,
            "benchmark2": name2,
            "duration1": report1.average_duration,
            "duration2": report2.average_duration,
            "score1": report1.average_score,
            "score2": report2.average_score,
            "faster": name1 if report1.average_duration < report2.average_duration else name2,
        }
