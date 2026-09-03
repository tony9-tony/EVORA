"""
Phase 18 — Native Goal Decomposition for EVORA.

Breaks complex goals into executable sub-tasks.

Supports:
  - Sequential decomposition
  - Parallel decomposition
  - Dependency tracking between subtasks
  - Priority assignment
  - Effort estimation
  - Reusability detection
  - Integration with IntelligenceRuntime planning
  - Integration with Agent execution

No independent authority system.
No security bypass.
Reuses existing abstractions:
  - Plan / PlanStep from planner
  - IntelligenceRuntime for reasoning
  - KnowledgeGraph for prior art
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

class DependencyType(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Subtask:
    """A decomposed sub-task."""
    subtask_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_goal_id: str = ""
    description: str = ""
    action_type: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    dependency_type: DependencyType = DependencyType.SEQUENTIAL
    priority: int = 0
    estimated_effort: float = 0.0
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[dict[str, Any]] = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "parent_goal_id": self.parent_goal_id,
            "description": self.description,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "dependency_type": self.dependency_type.value,
            "priority": self.priority,
            "estimated_effort": self.estimated_effort,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class DecomposedGoal:
    """A decomposed goal with subtasks."""
    goal_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_goal: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    overall_status: SubtaskStatus = SubtaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "original_goal": self.original_goal,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "overall_status": self.overall_status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Native Goal Decomposer
# ---------------------------------------------------------------------------

class NativeGoalDecomposer:
    """Decomposes complex goals into executable sub-tasks.

    Uses IntelligenceRuntime reasoning when available.
    Falls back to heuristic decomposition when runtime is absent.
    """

    def __init__(
        self,
        intelligence_runtime: Any = None,
        knowledge_graph: Any = None,
        logger: Optional[Any] = None,
    ):
        self.intelligence_runtime = intelligence_runtime
        self.knowledge_graph = knowledge_graph
        self.logger = logger
        self._decompositions: dict[str, DecomposedGoal] = {}

    def decompose(self, goal: str, context: dict[str, Any] = None) -> DecomposedGoal:
        """Decompose a goal into subtasks."""
        context = context or {}
        goal_id = uuid.uuid4().hex[:12]
        decomposed = DecomposedGoal(goal_id=goal_id, original_goal=goal)

        if self.intelligence_runtime is not None:
            try:
                import asyncio
                plan = asyncio.run(self.intelligence_runtime.plan(goal))
                if plan is not None and hasattr(plan, "steps"):
                    for idx, step in enumerate(plan.steps):
                        step_dict = step.to_dict() if hasattr(step, "to_dict") else step
                        subtask = Subtask(
                            parent_goal_id=goal_id,
                            description=step_dict.get("name", step_dict.get("description", "")),
                            action_type=step_dict.get("action_type", "unknown"),
                            parameters=step_dict.get("action_args", {}),
                            priority=idx,
                            estimated_effort=float(step_dict.get("estimated_effort", 1.0)),
                        )
                        decomposed.subtasks.append(subtask)
                    self._decompositions[goal_id] = decomposed
                    return decomposed
            except Exception:
                pass

        subtasks = self._heuristic_decompose(goal, goal_id)
        decomposed.subtasks = subtasks
        self._decompositions[goal_id] = decomposed
        return decomposed

    def _heuristic_decompose(self, goal: str, goal_id: str) -> list[Subtask]:
        """Heuristic decomposition fallback."""
        subtasks: list[Subtask] = []
        goal_lower = goal.lower()
        if "test" in goal_lower:
            subtasks.append(Subtask(
                parent_goal_id=goal_id,
                description="Identify test targets",
                action_type="analyze_project",
                parameters={"target": "tests"},
                priority=0,
                estimated_effort=1.0,
            ))
            subtasks.append(Subtask(
                parent_goal_id=goal_id,
                description="Run tests",
                action_type="run_tests",
                parameters={},
                priority=1,
                estimated_effort=2.0,
                dependencies=[subtasks[0].subtask_id] if subtasks else [],
                dependency_type=DependencyType.SEQUENTIAL,
            ))
        elif "refactor" in goal_lower:
            subtasks.append(Subtask(
                parent_goal_id=goal_id,
                description="Analyze code structure",
                action_type="analyze_code",
                parameters={"target": "project"},
                priority=0,
                estimated_effort=1.5,
            ))
            subtasks.append(Subtask(
                parent_goal_id=goal_id,
                description="Plan refactoring steps",
                action_type="plan",
                parameters={"goal": goal},
                priority=1,
                estimated_effort=1.0,
                dependencies=[subtasks[0].subtask_id] if subtasks else [],
                dependency_type=DependencyType.SEQUENTIAL,
            ))
            subtasks.append(Subtask(
                parent_goal_id=goal_id,
                description="Apply refactoring",
                action_type="edit_file",
                parameters={},
                priority=2,
                estimated_effort=3.0,
                dependencies=[subtasks[1].subtask_id] if len(subtasks) > 1 else [],
                dependency_type=DependencyType.SEQUENTIAL,
            ))
        else:
            subtasks.append(Subtask(
                parent_goal_id=goal_id,
                description=goal,
                action_type="analyze_project",
                parameters={"goal": goal},
                priority=0,
                estimated_effort=1.0,
            ))
        return subtasks

    def get_execution_order(self, decomposed: DecomposedGoal) -> list[list[Subtask]]:
        """Get execution order as batches of parallelizable subtasks."""
        completed: set[str] = set()
        batches: list[list[Subtask]] = []
        remaining = list(decomposed.subtasks)

        while remaining:
            ready = [
                s for s in remaining
                if all(d in completed for d in s.dependencies)
                and s.status not in (SubtaskStatus.COMPLETED, SubtaskStatus.SKIPPED)
            ]
            if not ready:
                break
            ready.sort(key=lambda s: s.priority)
            batches.append(ready)
            for s in ready:
                completed.add(s.subtask_id)
            remaining = [s for s in remaining if s.subtask_id not in completed]

        return batches

    def mark_subtask_completed(self, decomposed: DecomposedGoal, subtask_id: str, result: dict[str, Any] = None) -> bool:
        """Mark a subtask as completed."""
        for subtask in decomposed.subtasks:
            if subtask.subtask_id == subtask_id:
                subtask.status = SubtaskStatus.COMPLETED
                subtask.result = result
                subtask.completed_at = datetime.now().isoformat()
                self._update_overall_status(decomposed)
                return True
        return False

    def mark_subtask_failed(self, decomposed: DecomposedGoal, subtask_id: str, error: str = "") -> bool:
        """Mark a subtask as failed."""
        for subtask in decomposed.subtasks:
            if subtask.subtask_id == subtask_id:
                subtask.status = SubtaskStatus.FAILED
                subtask.error = error
                self._update_overall_status(decomposed)
                return True
        return False

    def _update_overall_status(self, decomposed: DecomposedGoal) -> None:
        """Update the overall status based on subtask states."""
        statuses = [s.status for s in decomposed.subtasks]
        if all(s == SubtaskStatus.COMPLETED for s in statuses):
            decomposed.overall_status = SubtaskStatus.COMPLETED
            decomposed.completed_at = datetime.now().isoformat()
        elif any(s == SubtaskStatus.FAILED for s in statuses):
            decomposed.overall_status = SubtaskStatus.FAILED
        elif any(s == SubtaskStatus.IN_PROGRESS for s in statuses):
            decomposed.overall_status = SubtaskStatus.IN_PROGRESS
        else:
            decomposed.overall_status = SubtaskStatus.PENDING

    def get_ready_subtasks(self, decomposed: DecomposedGoal) -> list[Subtask]:
        """Get subtasks ready for execution."""
        completed_ids = {s.subtask_id for s in decomposed.subtasks if s.status == SubtaskStatus.COMPLETED}
        return [
            s for s in decomposed.subtasks
            if s.status == SubtaskStatus.PENDING
            and all(d in completed_ids for d in s.dependencies)
        ]

    def get_next_subtask(self, decomposed: DecomposedGoal) -> Optional[Subtask]:
        """Get the next subtask to execute."""
        ready = self.get_ready_subtasks(decomposed)
        if not ready:
            return None
        return max(ready, key=lambda s: s.priority)

    def get_decomposition(self, goal_id: str) -> Optional[DecomposedGoal]:
        """Get a stored decomposition by ID."""
        return self._decompositions.get(goal_id)

    def get_metrics(self, decomposed: DecomposedGoal) -> dict[str, Any]:
        """Get metrics for a decomposed goal."""
        total = len(decomposed.subtasks)
        completed = sum(1 for s in decomposed.subtasks if s.status == SubtaskStatus.COMPLETED)
        failed = sum(1 for s in decomposed.subtasks if s.status == SubtaskStatus.FAILED)
        return {
            "goal_id": decomposed.goal_id,
            "total_subtasks": total,
            "completed": completed,
            "failed": failed,
            "pending": total - completed - failed,
            "progress": completed / total if total > 0 else 0.0,
            "overall_status": decomposed.overall_status.value,
        }
