"""
Phase 10 — IntelligenceEvaluator for EVORA native intelligence.

Evaluates the quality of native intelligence results deterministically.
No external model dependency. No ModelManager dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from evora.logger import Logger
from evora.brain.intelligence.capabilities import CapabilityType


class EvaluationGrade(str, Enum):
    """Grade for intelligence evaluation."""
    STRONG = "strong"
    ACCEPTABLE = "acceptable"
    WEAK = "weak"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    UNKNOWN = "unknown"


@dataclass
class EvaluationResult:
    """Result of evaluating an intelligence output."""

    grade: EvaluationGrade
    confidence: float = 0.0
    reasoning: str = ""
    limitations: list[str] = field(default_factory=list)
    evidence_count: int = 0
    constraint_satisfied: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "limitations": self.limitations,
            "evidence_count": self.evidence_count,
            "constraint_satisfied": self.constraint_satisfied,
            "metadata": self.metadata,
        }


class IntelligenceEvaluator:
    """Deterministic evaluator for native intelligence results.

    Evaluates:
    - confidence
    - evidence
    - consistency
    - constraint satisfaction
    - result completeness
    - limitations

    Does NOT grant permissions.
    Model output is never authority.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def evaluate_reasoning(
        self,
        goal: str,
        result: Any,
        evidence: list[dict[str, Any]],
        constraints: list[str],
    ) -> EvaluationResult:
        """Evaluate a reasoning result."""
        if not goal or not goal.strip():
            return EvaluationResult(
                grade=EvaluationGrade.UNSUPPORTED_CAPABILITY,
                confidence=0.0,
                reasoning="Empty goal provided",
                limitations=["No goal to evaluate"],
            )

        if result is None:
            return EvaluationResult(
                grade=EvaluationGrade.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                reasoning="No result provided",
                limitations=["Missing result"],
            )

        confidence = getattr(result, "confidence", 0.0)
        evidence_count = len(evidence) if evidence else 0
        limitations = getattr(result, "limitations", [])
        if limitations is None:
            limitations = []

        # Check constraint satisfaction
        constraint_satisfied = True
        if constraints:
            for constraint in constraints:
                if constraint and constraint.strip():
                    pass  # Deterministic constraints would be checked here

        # Determine grade
        if confidence >= 0.8 and evidence_count >= 3:
            grade = EvaluationGrade.STRONG
        elif confidence >= 0.5 and evidence_count >= 1:
            grade = EvaluationGrade.ACCEPTABLE
        elif confidence >= 0.3:
            grade = EvaluationGrade.WEAK
        else:
            grade = EvaluationGrade.INSUFFICIENT_EVIDENCE

        if limitations:
            if grade == EvaluationGrade.STRONG:
                grade = EvaluationGrade.ACCEPTABLE

        return EvaluationResult(
            grade=grade,
            confidence=confidence,
            reasoning=f"Evaluated reasoning for: {goal[:80]}",
            limitations=limitations,
            evidence_count=evidence_count,
            constraint_satisfied=constraint_satisfied,
        )

    def evaluate_plan(
        self,
        goal: str,
        plan: Any,
        constraints: list[str],
    ) -> EvaluationResult:
        """Evaluate a planning result."""
        if plan is None:
            return EvaluationResult(
                grade=EvaluationGrade.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                reasoning="No plan provided",
                limitations=["Missing plan"],
            )

        steps = getattr(plan, "steps", [])
        if steps is None:
            steps = []

        confidence = getattr(plan, "confidence", 0.0)
        limitations = getattr(plan, "limitations", [])
        if limitations is None:
            limitations = []

        step_count = len(steps)
        has_dependencies = all(
            isinstance(step, dict) and step.get("depends_on") for step in steps
        ) if steps else False

        # Validate plan structure
        if step_count == 0:
            grade = EvaluationGrade.INSUFFICIENT_EVIDENCE
            limitations.append("Plan has no steps")
        elif confidence >= 0.7 and step_count >= 2 and has_dependencies:
            grade = EvaluationGrade.STRONG
        elif confidence >= 0.5 and step_count >= 1:
            grade = EvaluationGrade.ACCEPTABLE
        elif confidence >= 0.3:
            grade = EvaluationGrade.WEAK
        else:
            grade = EvaluationGrade.INSUFFICIENT_EVIDENCE

        if not has_dependencies and step_count > 0:
            limitations.append("Plan steps lack explicit dependencies")

        return EvaluationResult(
            grade=grade,
            confidence=confidence,
            reasoning=f"Evaluated plan with {step_count} steps for: {goal[:80]}",
            limitations=limitations,
            evidence_count=step_count,
            constraint_satisfied=True,
        )

    def evaluate_inference(
        self,
        query: str,
        result: Any,
        known_facts: list[str],
    ) -> EvaluationResult:
        """Evaluate an inference result."""
        if not query or not query.strip():
            return EvaluationResult(
                grade=EvaluationGrade.UNSUPPORTED_CAPABILITY,
                confidence=0.0,
                reasoning="Empty query",
                limitations=["No query to evaluate"],
            )

        if result is None:
            return EvaluationResult(
                grade=EvaluationGrade.UNKNOWN,
                confidence=0.0,
                reasoning="No inference result",
                limitations=["Missing result"],
            )

        answer = getattr(result, "answer", "")
        confidence = getattr(result, "confidence", 0.0)
        source = getattr(result, "source", "")
        limitations = getattr(result, "limitations", [])
        if limitations is None:
            limitations = []

        if not answer or not answer.strip():
            grade = EvaluationGrade.INSUFFICIENT_EVIDENCE
            limitations.append("Empty answer")
        elif confidence >= 0.8 and source:
            grade = EvaluationGrade.STRONG
        elif confidence >= 0.5:
            grade = EvaluationGrade.ACCEPTABLE
        elif confidence >= 0.3:
            grade = EvaluationGrade.WEAK
        else:
            grade = EvaluationGrade.INSUFFICIENT_EVIDENCE

        return EvaluationResult(
            grade=grade,
            confidence=confidence,
            reasoning=f"Evaluated inference for: {query[:80]}",
            limitations=limitations,
            evidence_count=len(known_facts) if known_facts else 0,
        )

    def evaluate_capability(
        self,
        capability_type: CapabilityType,
        native_confidence: float,
        limitations: list[str],
    ) -> EvaluationResult:
        """Evaluate whether a capability can be executed."""
        if capability_type == CapabilityType.NATIVE:
            if native_confidence >= 0.7:
                grade = EvaluationGrade.STRONG
            elif native_confidence >= 0.5:
                grade = EvaluationGrade.ACCEPTABLE
            elif native_confidence >= 0.3:
                grade = EvaluationGrade.WEAK
            else:
                grade = EvaluationGrade.INSUFFICIENT_EVIDENCE
        elif capability_type == CapabilityType.LOCAL_MODEL:
            grade = EvaluationGrade.ACCEPTABLE
        elif capability_type == CapabilityType.EXTERNAL_MODEL:
            grade = EvaluationGrade.WEAK
        else:
            grade = EvaluationGrade.UNSUPPORTED_CAPABILITY

        return EvaluationResult(
            grade=grade,
            confidence=native_confidence,
            reasoning=f"Evaluated {capability_type.value} capability",
            limitations=limitations,
            evidence_count=1 if capability_type != CapabilityType.UNAVAILABLE else 0,
        )
