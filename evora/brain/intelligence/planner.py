"""
Phase 10 — NativePlanner for EVORA native intelligence.

Knowledge-grounded planning without external model inference.
Uses KnowledgeGraph, MemoryService, ToolRegistry, and native reasoning.
No ModelManager dependency. No external model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class PlanStep:
    """A step in a native plan."""

    id: str
    name: str
    description: str
    action_type: str
    action_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    estimated_effort: str = "medium"
    validation: str = ""
    rollback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "depends_on": self.depends_on,
            "estimated_effort": self.estimated_effort,
            "validation": self.validation,
            "rollback": self.rollback,
        }


@dataclass
class NativePlan:
    """A plan created by native intelligence."""

    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    confidence: float = 0.0
    limitations: list[str] = field(default_factory=list)
    requires_approval: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "confidence": self.confidence,
            "limitations": self.limitations,
            "requires_approval": self.requires_approval,
            "metadata": self.metadata,
        }


class NativePlanner:
    """Knowledge-grounded planning without external model inference.

    Uses KnowledgeGraph, MemoryService, ToolRegistry, and native reasoning.
    No ModelManager dependency.
    No external model dependency.
    """

    def __init__(
        self,
        knowledge_graph: Any = None,
        memory_service: Any = None,
        tool_registry: Any = None,
        reasoning_engine: Any = None,
        logger: Optional[Logger] = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.memory_service = memory_service
        self.tool_registry = tool_registry
        self.reasoning_engine = reasoning_engine
        self.logger = logger

    async def plan(self, goal: str, constraints: list[str] = None) -> Optional[NativePlan]:
        """Create a plan using native capabilities only.

        Returns None if planning is not possible natively.
        """
        if not goal or not goal.strip():
            return None

        constraints = constraints or []
        limitations: list[str] = []

        # Step 1: Goal analysis
        objective, success_criteria = self._analyze_goal(goal)

        # Step 2: Precondition check
        preconditions = self._check_preconditions(goal, constraints)

        # Step 3: Knowledge retrieval
        patterns = self._retrieve_patterns(goal)
        past_plans = self._retrieve_past_plans(goal)

        # Step 4: Capability check
        can_plan, capability_limitations = self._check_capability(goal, patterns, past_plans)
        limitations.extend(capability_limitations)

        if not can_plan:
            return NativePlan(
                goal=goal,
                confidence=0.0,
                limitations=["Cannot plan this task with current native capabilities"] + limitations,
            )

        # Step 5: Generate candidate steps
        candidate_steps = self._generate_steps(goal, patterns, past_plans, preconditions)

        # Step 6: Determine dependencies
        steps_with_deps = self._determine_dependencies(candidate_steps)

        # Step 7: Assess risks
        risks = self._assess_risks(steps_with_deps)

        # Step 8: Structure ordered plan
        ordered_steps = self._order_steps(steps_with_deps)

        # Step 9: Validate plan
        valid, validation_errors = self._validate_plan(ordered_steps, constraints)
        if not valid:
            limitations.extend(validation_errors)

        # Step 10: Calculate confidence
        confidence = self._calculate_confidence(
            patterns=patterns,
            past_plans=past_plans,
            steps=ordered_steps,
            risks=risks,
        )

        # Build final plan
        plan_steps = []
        for i, step in enumerate(ordered_steps):
            plan_steps.append(PlanStep(
                id=step.get("id", f"step-{i}"),
                name=step.get("name", f"Step {i}"),
                description=step.get("description", ""),
                action_type=step.get("action_type", "unknown"),
                action_args=step.get("action_args", {}),
                depends_on=step.get("depends_on", []),
                estimated_effort=step.get("estimated_effort", "medium"),
                validation=step.get("validation", ""),
                rollback=step.get("rollback", ""),
            ))

        if self.logger:
            self.logger.plan(f"Created native plan for: {goal[:80]} ({len(plan_steps)} steps, confidence={confidence:.2f})")

        return NativePlan(
            goal=goal,
            steps=plan_steps,
            confidence=confidence,
            limitations=limitations,
            requires_approval=True,
            metadata={
                "objective": objective,
                "success_criteria": success_criteria,
                "preconditions": preconditions,
                "risks": risks,
                "pattern_count": len(patterns),
                "past_plan_count": len(past_plans),
            },
        )

    def _analyze_goal(self, goal: str) -> tuple[str, list[str]]:
        """Analyze goal into objective and success criteria."""
        objective = goal.strip()
        success_criteria = [f"Complete: {goal[:80]}"]
        return objective, success_criteria

    def _check_preconditions(self, goal: str, constraints: list[str]) -> list[str]:
        """Check preconditions for the goal."""
        preconditions = []

        if self.tool_registry is not None:
            try:
                available = list(self.tool_registry.list())
                preconditions.append(f"Available tools: {len(available)}")
            except Exception:
                preconditions.append("Tool registry unavailable")

        if self.knowledge_graph is not None:
            try:
                preconditions.append("Knowledge graph available")
            except Exception:
                preconditions.append("Knowledge graph unavailable")

        return preconditions

    def _retrieve_patterns(self, goal: str) -> list[dict[str, Any]]:
        """Retrieve known patterns from knowledge graph."""
        if self.knowledge_graph is None:
            return []
        try:
            nodes = self.knowledge_graph.query(goal, limit=5)
            return [n.to_dict() for n in nodes if n.type == "pattern"]
        except Exception:
            return []

    def _retrieve_past_plans(self, goal: str) -> list[dict[str, Any]]:
        """Retrieve past plans from memory."""
        if self.memory_service is None:
            return []
        try:
            memories = self.memory_service.retrieve_relevant(goal=goal, limit=5)
            return [m.to_dict() for m in memories] if memories else []
        except Exception:
            return []

    def _check_capability(self, goal: str, patterns: list, past_plans: list) -> tuple[bool, list[str]]:
        """Check if planning is possible natively."""
        limitations = []

        # Even without patterns, we can generate a basic analysis plan
        return True, limitations

    def _generate_steps(
        self,
        goal: str,
        patterns: list[dict[str, Any]],
        past_plans: list[dict[str, Any]],
        preconditions: list[str],
    ) -> list[dict[str, Any]]:
        """Generate candidate plan steps."""
        steps = []

        # Generate steps from patterns
        for i, pattern in enumerate(patterns[:3]):
            steps.append({
                "id": f"step-pattern-{i}",
                "name": f"Apply pattern: {pattern.get('content', 'unknown')[:50]}",
                "description": pattern.get("content", ""),
                "action_type": "apply_pattern",
                "action_args": {"pattern_id": pattern.get("id")},
                "estimated_effort": "medium",
                "validation": "Verify pattern application succeeded",
                "rollback": "Revert pattern changes",
            })

        # Generate steps from past plans
        for i, plan in enumerate(past_plans[:2]):
            plan_data = plan.get("data", {})
            if isinstance(plan_data, dict):
                plan_steps = plan_data.get("steps", [])
                for j, ps in enumerate(plan_steps[:3]):
                    steps.append({
                        "id": f"step-past-{i}-{j}",
                        "name": ps.get("name", f"Past step {j}"),
                        "description": ps.get("description", ""),
                        "action_type": ps.get("action_type", "unknown"),
                        "action_args": ps.get("action_args", {}),
                        "depends_on": ps.get("depends_on", []),
                        "estimated_effort": ps.get("estimated_effort", "medium"),
                        "validation": "Verify step completed",
                        "rollback": "Revert step changes",
                    })

        # If no steps generated, add default analysis step
        if not steps:
            steps.append({
                "id": "step-analyze",
                "name": "Analyze goal",
                "description": f"Analyze and understand: {goal[:80]}",
                "action_type": "analyze",
                "action_args": {"goal": goal},
                "estimated_effort": "low",
                "validation": "Goal understood",
                "rollback": "",
            })

        return steps

    def _determine_dependencies(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Determine dependencies between steps."""
        for i, step in enumerate(steps):
            if not step.get("depends_on"):
                step["depends_on"] = [steps[j]["id"] for j in range(i) if j < i]
        return steps

    def _assess_risks(self, steps: list[dict[str, Any]]) -> list[str]:
        """Assess risks in the plan."""
        risks = []
        for step in steps:
            if step.get("action_type") in ("execute_command", "modify_file", "git_commit"):
                risks.append(f"Step '{step['name']}' involves potentially risky operation")
        return risks

    def _order_steps(self, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Order steps respecting dependencies."""
        # Simple topological ordering
        ordered = []
        visited = set()
        step_map = {s["id"]: s for s in steps}

        def visit(step_id: str):
            if step_id in visited:
                return
            visited.add(step_id)
            step = step_map.get(step_id)
            if step:
                for dep in step.get("depends_on", []):
                    visit(dep)
                ordered.append(step)

        for step in steps:
            visit(step["id"])

        return ordered

    def _validate_plan(self, steps: list[dict[str, Any]], constraints: list[str]) -> tuple[bool, list[str]]:
        """Validate the plan."""
        errors = []

        if not steps:
            errors.append("Plan has no steps")
            return False, errors

        for step in steps:
            if not step.get("name"):
                errors.append(f"Step {step.get('id')} has no name")
            if not step.get("action_type"):
                errors.append(f"Step {step.get('id')} has no action_type")

        return len(errors) == 0, errors

    def _calculate_confidence(
        self,
        patterns: list,
        past_plans: list,
        steps: list,
        risks: list,
    ) -> float:
        """Calculate confidence in the plan."""
        confidence = 0.5

        # Boost based on patterns
        if patterns:
            confidence += 0.1 * min(len(patterns), 3)

        # Boost based on past plans
        if past_plans:
            confidence += 0.1 * min(len(past_plans), 2)

        # Boost based on steps
        if steps:
            confidence += 0.05 * min(len(steps), 5)

        # Penalize based on risks
        if risks:
            confidence -= 0.05 * min(len(risks), 3)

        return max(0.0, min(1.0, confidence))
