"""
Evaluation system for EVORA Phase 2.

Assesses whether actions moved the task toward completion,
producing structured evaluation results that feed back into the decision loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from evora.logger import Logger
from evora.task import TaskState, Observation


class EvaluationOutcome(str, Enum):
    """Outcome of evaluating an action's result."""
    SUCCESS = "success"
    PROGRESS = "progress"
    FAILURE = "failure"
    NEEDS_INFO = "needs_information"
    NEEDS_APPROVAL = "needs_approval"
    BLOCKED = "blocked"


@dataclass
class EvaluationResult:
    """Result of evaluating an observation against the task state."""
    outcome: EvaluationOutcome
    confidence: float = 1.0
    reason: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "recommendations": self.recommendations,
        }


class Evaluator:
    """Evaluates observations to determine task progress."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def evaluate(self, state: TaskState, observation: Observation) -> EvaluationResult:
        """Evaluate an observation and produce a structured result.

        Considers:
        - Observation type and success
        - Current task status
        - Retry counts
        - Error history
        - Completion criteria
        """
        if state.is_complete or state.is_failed or state.is_cancelled:
            return self._eval_terminal(state, observation)

        obs_type = observation.type
        is_success = observation.success

        if obs_type == "error":
            return self._eval_error(state, observation)

        if obs_type == "command_failed" or obs_type == "action_failed" or obs_type == "build_failed":
            return self._eval_failure(state, observation)

        if obs_type == "test_failed":
            return self._eval_test_failure(state, observation)

        if obs_type == "test_passed":
            return self._eval_test_success(state, observation)

        if obs_type in ("file_created", "file_modified", "directory_created"):
            return self._eval_progress(state, observation)

        if obs_type == "command_success" or obs_type == "action_success":
            return self._eval_success(state, observation)

        if obs_type == "file_missing":
            return self._eval_blocked(state, observation)

        if obs_type == "approval_granted":
            return EvaluationResult(
                outcome=EvaluationOutcome.PROGRESS,
                confidence=1.0,
                reason="Approval granted, can proceed",
            )

        if obs_type == "approval_denied":
            return EvaluationResult(
                outcome=EvaluationOutcome.BLOCKED,
                confidence=1.0,
                reason="Approval denied by user",
                recommendations=["Ask user for clarification"],
            )

        if obs_type == "plan_created":
            return EvaluationResult(
                outcome=EvaluationOutcome.PROGRESS,
                confidence=1.0,
                reason="Plan created successfully",
            )

        return EvaluationResult(
            outcome=EvaluationOutcome.PROGRESS,
            confidence=0.5,
            reason=f"Observation type '{obs_type}' recorded",
        )

    def _eval_terminal(self, state: TaskState, obs: Observation) -> EvaluationResult:
        outcome = EvaluationOutcome.SUCCESS if state.is_complete else EvaluationOutcome.FAILURE
        if state.is_cancelled:
            outcome = EvaluationOutcome.BLOCKED
        return EvaluationResult(
            outcome=outcome,
            confidence=1.0,
            reason=f"Task in terminal state: {state.status}",
        )

    def _eval_error(self, state: TaskState, obs: Observation) -> EvaluationResult:
        return EvaluationResult(
            outcome=EvaluationOutcome.FAILURE,
            confidence=0.9,
            reason=f"Error observed: {obs.data.get('error', str(obs.data))}",
            recommendations=["Analyze the error and attempt a fix", "Reduce retry count"],
        )

    def _eval_failure(self, state: TaskState, obs: Observation) -> EvaluationResult:
        error_msg = obs.data.get("error", "")
        return_code = obs.data.get("return_code", -1)

        if "pytest" in obs.data.get("command", "").lower() or "test" in obs.type:
            return self._eval_test_failure(state, obs)

        return EvaluationResult(
            outcome=EvaluationOutcome.FAILURE,
            confidence=0.8,
            reason=f"Action failed (rc={return_code}): {error_msg[:100]}",
            recommendations=["Attempt automatic fix", "Provide more context"],
        )

    def _eval_test_failure(self, state: TaskState, obs: Observation) -> EvaluationResult:
        output = obs.data.get("output", "")
        return EvaluationResult(
            outcome=EvaluationOutcome.FAILURE,
            confidence=0.9,
            reason=f"Tests failed. Error: {obs.data.get('error', output[:200])[:200]}",
            recommendations=["Analyze test output for specific failures", "Fix code and retry tests"],
        )

    def _eval_test_success(self, state: TaskState, obs: Observation) -> EvaluationResult:
        remaining = len(state.remaining_steps)
        if remaining == 0:
            return EvaluationResult(
                outcome=EvaluationOutcome.SUCCESS,
                confidence=0.95,
                reason="All tests passed and no remaining steps",
            )
        return EvaluationResult(
            outcome=EvaluationOutcome.PROGRESS,
            confidence=0.8,
            reason=f"Tests passed, {remaining} steps remaining",
        )

    def _eval_progress(self, state: TaskState, obs: Observation) -> EvaluationResult:
        remaining = len(state.remaining_steps)
        return EvaluationResult(
            outcome=EvaluationOutcome.PROGRESS,
            confidence=0.9,
            reason=f"Action completed: {obs.type}. {remaining} steps remaining.",
        )

    def _eval_success(self, state: TaskState, obs: Observation) -> EvaluationResult:
        return EvaluationResult(
            outcome=EvaluationOutcome.PROGRESS,
            confidence=0.8,
            reason=f"Action succeeded: {obs.type}",
        )

    def _eval_blocked(self, state: TaskState, obs: Observation) -> EvaluationResult:
        return EvaluationResult(
            outcome=EvaluationOutcome.BLOCKED,
            confidence=0.7,
            reason=f"Action blocked: {obs.data.get('path', obs.type)}",
            recommendations=["Find alternate approach", "Provide missing resource"],
        )

    def evaluate_completion(self, state: TaskState) -> EvaluationResult:
        """Final completion check."""
        if state.remaining_steps:
            return EvaluationResult(
                outcome=EvaluationOutcome.PROGRESS,
                confidence=0.3,
                reason=f"Not complete: {len(state.remaining_steps)} steps remaining",
            )

        if state.test_results and not state.test_results[-1].passed:
            return EvaluationResult(
                outcome=EvaluationOutcome.FAILURE,
                confidence=0.9,
                reason="Tests failing",
            )

        if state.errors:
            recent_errors = [e for e in state.errors[-3:] if not state.is_complete]
            if recent_errors:
                return EvaluationResult(
                    outcome=EvaluationOutcome.PROGRESS,
                    confidence=0.6,
                    reason=f"Errors present but may be resolved: {recent_errors[0][:100]}",
                )

        return EvaluationResult(
            outcome=EvaluationOutcome.SUCCESS,
            confidence=0.95,
            reason="All steps complete, tests passing, no active errors",
        )
