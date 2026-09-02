"""
Development planner for EVORA Phase 7.

Creates structured development plans from approved improvement candidates.
Plans contain ordered steps, dependencies, risks, tests, benchmarks,
and rollback strategies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from evora.logger import Logger
from evora.planner import Plan, PlanStep


@dataclass
class DevelopmentStep:
    """A step in a development plan."""

    id: str
    name: str
    description: str
    action_type: str
    action_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    estimated_effort: str = "medium"
    test_required: bool = True
    rollback_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "depends_on": self.depends_on,
            "estimated_effort": self.estimated_effort,
            "test_required": self.test_required,
            "rollback_action": self.rollback_action,
        }

    def to_plan_step(self) -> PlanStep:
        """Convert to a Phase 6 PlanStep."""
        return PlanStep(
            id=self.id,
            name=self.name,
            description=self.description,
            action_type=self.action_type,
            action_args=self.action_args,
            depends_on=self.depends_on,
            estimated_effort=self.estimated_effort,
        )


@dataclass
class DevelopmentPlan:
    """A structured development plan for an approved improvement."""

    id: str
    objective: str
    candidate_id: str
    steps: list[DevelopmentStep] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    tests_required: list[str] = field(default_factory=list)
    benchmark_criteria: dict[str, Any] = field(default_factory=dict)
    rollback_strategy: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "candidate_id": self.candidate_id,
            "steps": [s.to_dict() for s in self.steps],
            "risks": self.risks,
            "tests_required": self.tests_required,
            "benchmark_criteria": self.benchmark_criteria,
            "rollback_strategy": self.rollback_strategy,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_plan(self) -> Plan:
        """Convert to a Phase 6 Plan for execution."""
        return Plan(
            title=self.objective,
            description=f"Auto-generated development plan for {self.objective}",
            steps=[s.to_plan_step() for s in self.steps],
            raw_output="",
        )


class DevelopmentPlanner:
    """Creates structured development plans from improvement candidates."""

    def __init__(self, model_manager: Any, logger: Optional[Logger] = None):
        self.model_manager = model_manager
        self.logger = logger

    async def create_plan(self, candidate: Any, context: dict[str, Any]) -> DevelopmentPlan:
        """Create a development plan from an improvement candidate."""
        plan_id = f"plan-{uuid.uuid4().hex[:12]}"
        steps = self._generate_steps(candidate, context)
        tests = self._generate_tests(candidate, steps)
        benchmarks = self._generate_benchmarks(candidate)

        if self.logger:
            self.logger.plan(f"Created development plan: {plan_id} ({len(steps)} steps)")

        return DevelopmentPlan(
            id=plan_id,
            objective=candidate.title,
            candidate_id=candidate.id,
            steps=steps,
            risks=candidate.risks if hasattr(candidate, "risks") else [],
            tests_required=tests,
            benchmark_criteria=benchmarks,
            rollback_strategy=f"Rollback changes to {', '.join(candidate.affected_files) if hasattr(candidate, 'affected_files') and candidate.affected_files else 'affected files'}",
            metadata={"candidate": candidate.to_dict() if hasattr(candidate, "to_dict") else str(candidate)},
        )

    def _generate_steps(self, candidate: Any, context: dict[str, Any]) -> list[DevelopmentStep]:
        """Generate implementation steps from a candidate."""
        steps = []
        affected_files = getattr(candidate, "affected_files", [])

        steps.append(DevelopmentStep(
            id=f"step-{uuid.uuid4().hex[:8]}",
            name="Create rollback snapshot",
            description="Create a backup of affected files before modification",
            action_type="read_file",
            action_args={"path": affected_files[0]} if affected_files else {},
            estimated_effort="low",
            test_required=False,
            rollback_action="restore_from_snapshot",
        ))

        for file_path in affected_files:
            steps.append(DevelopmentStep(
                id=f"step-{uuid.uuid4().hex[:8]}",
                name=f"Modify {file_path}",
                description=f"Apply approved change to {file_path}",
                action_type="edit_file",
                action_args={"path": file_path},
                depends_on=[steps[0].id] if steps else [],
                estimated_effort=getattr(candidate, "estimated_scope", "medium"),
                test_required=True,
                rollback_action=f"restore_{file_path.replace('/', '_').replace('.', '_')}",
            ))

        steps.append(DevelopmentStep(
            id=f"step-{uuid.uuid4().hex[:8]}",
            name="Run tests",
            description="Execute test suite to verify changes",
            action_type="run_tests",
            action_args={},
            depends_on=[s.id for s in steps[1:]],
            estimated_effort="medium",
            test_required=True,
            rollback_action="",
        ))

        return steps

    def _generate_tests(self, candidate: Any, steps: list[DevelopmentStep]) -> list[str]:
        """Generate test requirements for the candidate."""
        tests = ["Run full pytest suite"]
        affected_files = getattr(candidate, "affected_files", [])

        for file_path in affected_files:
            if file_path.endswith(".py"):
                tests.append(f"Verify syntax: python -m py_compile {file_path}")

        validation = getattr(candidate, "validation_strategy", "")
        if validation:
            tests.append(validation)

        return tests

    def _generate_benchmarks(self, candidate: Any) -> dict[str, Any]:
        """Generate benchmark criteria for the candidate."""
        return {
            "metric": "test_pass_rate",
            "baseline": "unknown",
            "target": "100%",
            "measurement_method": "pytest --tb=short",
        }
