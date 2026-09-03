"""
Phase 21 — Native Self-Reflection for EVORA.

Enables the system to analyze its own performance and identify improvements.

Supports:
  - Performance analysis
  - Weakness identification
  - Strength identification
  - Improvement suggestions
  - Reflection history tracking
  - Integration with IntelligenceRuntime
  - Integration with NativeAgent
  - Integration with TrainingPipeline

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

class ReflectionType(str, Enum):
    PERFORMANCE = "performance"
    WEAKNESS = "weakness"
    STRENGTH = "strength"
    IMPROVEMENT = "improvement"
    GENERAL = "general"


class ReflectionSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Reflection:
    """A reflection entry."""
    reflection_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    reflection_type: ReflectionType = ReflectionType.GENERAL
    severity: ReflectionSeverity = ReflectionSeverity.LOW
    title: str = ""
    description: str = ""
    evidence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "reflection_type": self.reflection_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "evidence": self.evidence,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for reflection."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_duration: float = 0.0
    common_failures: list[str] = field(default_factory=list)
    improvement_areas: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0,
            "average_duration": self.average_duration,
            "common_failures": self.common_failures,
            "improvement_areas": self.improvement_areas,
        }


# ---------------------------------------------------------------------------
# Native Self-Reflection
# ---------------------------------------------------------------------------

class NativeSelfReflection:
    """Native self-reflection for EVORA.

    Analyzes performance and generates improvement suggestions.
    """

    def __init__(
        self,
        intelligence_runtime: Any = None,
        agent: Any = None,
        training_pipeline: Any = None,
        logger: Optional[Any] = None,
    ):
        self.intelligence_runtime = intelligence_runtime
        self.agent = agent
        self.training_pipeline = training_pipeline
        self.logger = logger
        self._reflections: list[Reflection] = []
        self._metrics = PerformanceMetrics()

    def reflect(self, context: dict[str, Any] = None) -> list[Reflection]:
        """Perform self-reflection and return insights."""
        context = context or {}
        reflections: list[Reflection] = []
        reflections.extend(self._analyze_performance())
        reflections.extend(self._identify_weaknesses())
        reflections.extend(self._identify_strengths())
        reflections.extend(self._suggest_improvements())
        self._reflections.extend(reflections)
        return reflections

    def _analyze_performance(self) -> list[Reflection]:
        """Analyze performance metrics."""
        reflections = []
        metrics_dict = self._metrics.to_dict()
        success_rate = metrics_dict["success_rate"]
        if success_rate < 0.5 and self._metrics.total_tasks > 0:
            reflections.append(Reflection(
                reflection_type=ReflectionType.PERFORMANCE,
                severity=ReflectionSeverity.HIGH,
                title="Low success rate",
                description=f"Success rate is {success_rate:.1%}, below acceptable threshold",
                evidence=[f"Total tasks: {self._metrics.total_tasks}", f"Failed: {self._metrics.failed_tasks}"],
                suggestions=["Review error recovery strategies", "Analyze common failures"],
                confidence=0.8,
            ))
        return reflections

    def _identify_weaknesses(self) -> list[Reflection]:
        """Identify weaknesses from metrics."""
        reflections = []
        for failure in self._metrics.common_failures[:3]:
            reflections.append(Reflection(
                reflection_type=ReflectionType.WEAKNESS,
                severity=ReflectionSeverity.MEDIUM,
                title=f"Weakness: {failure[:50]}",
                description=f"Recurring failure pattern detected: {failure}",
                evidence=[failure],
                suggestions=["Address root cause", "Add retry logic", "Improve error handling"],
                confidence=0.6,
            ))
        return reflections

    def _identify_strengths(self) -> list[Reflection]:
        """Identify strengths from metrics."""
        reflections = []
        if self._metrics.successful_tasks > self._metrics.failed_tasks and self._metrics.total_tasks > 0:
            reflections.append(Reflection(
                reflection_type=ReflectionType.STRENGTH,
                severity=ReflectionSeverity.LOW,
                title="High success rate",
                description=f"Successfully completed {self._metrics.successful_tasks} tasks",
                evidence=[f"Success rate: {self._metrics.successful_tasks / self._metrics.total_tasks:.1%}"],
                suggestions=["Maintain current approach", "Document successful patterns"],
                confidence=0.7,
            ))
        return reflections

    def _suggest_improvements(self) -> list[Reflection]:
        """Suggest improvements based on metrics."""
        reflections = []
        for area in self._metrics.improvement_areas[:3]:
            reflections.append(Reflection(
                reflection_type=ReflectionType.IMPROVEMENT,
                severity=ReflectionSeverity.MEDIUM,
                title=f"Improvement: {area[:50]}",
                description=f"Potential improvement area identified: {area}",
                evidence=[area],
                suggestions=[f"Address {area}"],
                confidence=0.5,
            ))
        return reflections

    def record_task_result(self, success: bool, duration: float = 0.0, error: str = "") -> None:
        """Record a task result for future reflection."""
        self._metrics.total_tasks += 1
        if success:
            self._metrics.successful_tasks += 1
        else:
            self._metrics.failed_tasks += 1
            if error:
                self._metrics.common_failures.append(error)
        total_duration = self._metrics.average_duration * (self._metrics.total_tasks - 1) + duration
        self._metrics.average_duration = total_duration / self._metrics.total_tasks

    def add_improvement_area(self, area: str) -> None:
        """Add an improvement area."""
        if area not in self._metrics.improvement_areas:
            self._metrics.improvement_areas.append(area)

    def get_reflections(self, reflection_type: Optional[ReflectionType] = None) -> list[Reflection]:
        """Get reflections, optionally filtered by type."""
        if reflection_type is None:
            return list(self._reflections)
        return [r for r in self._reflections if r.reflection_type == reflection_type]

    def get_latest_reflection(self) -> Optional[Reflection]:
        """Get the latest reflection."""
        return self._reflections[-1] if self._reflections else None

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return self._metrics.to_dict()

    def clear_reflections(self) -> None:
        """Clear reflection history."""
        self._reflections = []
