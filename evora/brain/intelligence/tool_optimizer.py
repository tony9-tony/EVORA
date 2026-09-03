"""
Phase 27 — Native Tool Optimizer for EVORA.

Optimizes tool selection and usage.

Supports:
  - Tool recommendation
  - Tool performance tracking
  - Tool selection optimization
  - Tool chaining
  - Integration with ToolRegistry
  - Integration with NativeAgent
  - Integration with IntelligenceRuntime

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

@dataclass
class ToolPerformance:
    """Performance metrics for a tool."""
    tool_name: str = ""
    total_uses: int = 0
    successful_uses: int = 0
    failed_uses: int = 0
    average_duration: float = 0.0
    success_rate: float = 0.0
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "total_uses": self.total_uses,
            "successful_uses": self.successful_uses,
            "failed_uses": self.failed_uses,
            "average_duration": self.average_duration,
            "success_rate": self.success_rate,
            "last_used": self.last_used,
        }


@dataclass
class ToolRecommendation:
    """A tool recommendation."""
    recommendation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    tool_name: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "tool_name": self.tool_name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Native Tool Optimizer
# ---------------------------------------------------------------------------

class NativeToolOptimizer:
    """Native tool optimizer for EVORA.

    Optimizes tool selection and usage.
    """

    def __init__(
        self,
        tool_registry: Any = None,
        intelligence_runtime: Any = None,
        logger: Optional[Any] = None,
    ):
        self.tool_registry = tool_registry
        self.intelligence_runtime = intelligence_runtime
        self.logger = logger
        self._performance: dict[str, ToolPerformance] = {}
        self._recommendation_history: list[ToolRecommendation] = []

    def recommend_tool(self, task_description: str, available_tools: list[str] = None) -> ToolRecommendation:
        """Recommend a tool for a task."""
        if available_tools is None:
            available_tools = list(self._performance.keys())
        if not available_tools:
            available_tools = ["analyze_project", "read_file", "search_files"]
        scored_tools = []
        for tool_name in available_tools:
            perf = self._performance.get(tool_name)
            if perf:
                score = perf.success_rate * 0.6 + (1.0 / (perf.average_duration + 1.0)) * 0.4
            else:
                score = 0.5
            scored_tools.append((tool_name, score))
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        best_tool = scored_tools[0][0] if scored_tools else available_tools[0]
        alternatives = [t[0] for t in scored_tools[1:4]]
        recommendation = ToolRecommendation(
            tool_name=best_tool,
            confidence=scored_tools[0][1] if scored_tools else 0.5,
            reasoning=f"Recommended based on performance history",
            alternatives=alternatives,
        )
        self._recommendation_history.append(recommendation)
        return recommendation

    def record_tool_use(self, tool_name: str, success: bool, duration: float = 0.0) -> None:
        """Record a tool use for performance tracking."""
        if tool_name not in self._performance:
            self._performance[tool_name] = ToolPerformance(tool_name=tool_name)
        perf = self._performance[tool_name]
        perf.total_uses += 1
        if success:
            perf.successful_uses += 1
        else:
            perf.failed_uses += 1
        perf.success_rate = perf.successful_uses / perf.total_uses
        perf.average_duration = (perf.average_duration * (perf.total_uses - 1) + duration) / perf.total_uses
        perf.last_used = datetime.now().isoformat()

    def get_tool_performance(self, tool_name: str) -> Optional[ToolPerformance]:
        """Get performance metrics for a tool."""
        return self._performance.get(tool_name)

    def get_all_performance(self) -> dict[str, ToolPerformance]:
        """Get performance metrics for all tools."""
        return dict(self._performance)

    def get_optimization_metrics(self) -> dict[str, Any]:
        """Get optimization metrics."""
        total_tools = len(self._performance)
        total_uses = sum(p.total_uses for p in self._performance.values())
        avg_success_rate = sum(p.success_rate for p in self._performance.values()) / total_tools if total_tools > 0 else 0.0
        return {
            "total_tools_tracked": total_tools,
            "total_uses": total_uses,
            "average_success_rate": avg_success_rate,
            "recommendations_made": len(self._recommendation_history),
        }

    def clear_history(self) -> None:
        """Clear performance history."""
        self._performance = {}
        self._recommendation_history = []
