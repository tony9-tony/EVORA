"""
Task state and decision abstractions for EVORA Phase 2.

Provides serializable data structures that track the full autonomous
agent loop state including goals, observations, decisions, and results.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional


@dataclass
class TestResult:
    """Result of a test execution."""
    command: str
    passed: bool
    output: str = ""
    error: str = ""
    return_code: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TestResult":
        return cls(**data)


@dataclass
class Observation:
    """A single observation from the environment after an action.

    Type values:
      file_created, file_modified, file_deleted, directory_created,
      command_success, command_failed, test_passed, test_failed,
      build_failed, file_missing, approval_granted, approval_denied,
      plan_created, error
    """
    type: str
    source: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(**data)


@dataclass
class Decision:
    """A structured decision from the decision engine.

    Action values:
      understand, analyze, plan, ask_approval, execute_tool,
      run_tests, fix_error, verify, report, done, cancel
    """
    action: str
    reason: str = ""
    tool: Optional[str] = None
    arguments: dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    risk_level: str = "safe"
    requires_approval: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Decision":
        return cls(**data)


@dataclass
class ActionResult:
    """Result of executing a tool or action."""
    success: bool
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str = ""
    observations: list[dict[str, Any]] = field(default_factory=list)
    return_code: int = 0
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionResult":
        return cls(**data)


@dataclass
class TaskState:
    """Complete serializable state of an autonomous agent task.

    This is the single source of truth for the agent loop. Every
    component reads from and writes to this state.
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request: str = ""
    goal: str = ""

    current_step: Optional[str] = None
    workspace: str = "."
    project_context: dict[str, Any] = field(default_factory=dict)

    plan: Optional[dict[str, Any]] = None
    plan_steps: list[dict[str, Any]] = field(default_factory=list)
    completed_steps: set[str] = field(default_factory=set)
    remaining_steps: list[dict[str, Any]] = field(default_factory=list)

    status: str = "idle"
    attempts: int = 0
    max_attempts: int = 10

    observations: list[Observation] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    test_results: list[TestResult] = field(default_factory=list)

    completion_criteria: list[str] = field(default_factory=list)
    final_result: str = ""
    elapsed: float = 0.0

    is_complete: bool = False
    is_failed: bool = False
    is_cancelled: bool = False

    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        if self.goal == "":
            self.goal = self.request

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)
        self.updated_at = datetime.now().isoformat()

    def add_decision(self, dec: Decision) -> None:
        self.decisions.append(dec)
        self.updated_at = datetime.now().isoformat()

    def add_action(self, tool: str, arguments: dict, result: ActionResult) -> None:
        self.actions.append({
            "tool": tool,
            "arguments": arguments,
            "success": result.success,
            "error": result.error,
            "timestamp": time.time(),
            "observations": result.observations,
        })
        self.updated_at = datetime.now().isoformat()

    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.updated_at = datetime.now().isoformat()

    def add_test_result(self, result: TestResult) -> None:
        self.test_results.append(result)
        self.updated_at = datetime.now().isoformat()

    def mark_complete(self, result: str = "") -> None:
        self.is_complete = True
        self.status = "completed"
        self.final_result = result
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, error: str = "") -> None:
        self.is_failed = True
        self.status = "failed"
        if error:
            self.final_result = error
        self.updated_at = datetime.now().isoformat()

    def mark_cancelled(self, reason: str = "") -> None:
        self.is_cancelled = True
        self.status = "cancelled"
        self.final_result = reason
        self.updated_at = datetime.now().isoformat()

    def increment_attempt(self) -> int:
        self.attempts += 1
        return self.attempts

    def exceeded_retry_limit(self) -> bool:
        return self.attempts >= self.max_attempts

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["completed_steps"] = list(self.completed_steps)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskState":
        completed_steps = data.pop("completed_steps", [])
        data["completed_steps"] = set(completed_steps)

        observations = data.get("observations")
        if observations is not None:
            data["observations"] = [
                obs if isinstance(obs, Observation) else Observation.from_dict(obs)
                for obs in observations
            ]

        decisions = data.get("decisions")
        if decisions is not None:
            data["decisions"] = [
                dec if isinstance(dec, Decision) else Decision.from_dict(dec)
                for dec in decisions
            ]

        test_results = data.get("test_results")
        if test_results is not None:
            data["test_results"] = [
                tr if isinstance(tr, TestResult) else TestResult.from_dict(tr)
                for tr in test_results
            ]

        return cls(**data)

    def snapshot(self) -> dict[str, Any]:
        """Lightweight snapshot for logging - no large data."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "current_step": self.current_step,
            "completed_steps": list(self.completed_steps),
            "remaining_steps": len(self.remaining_steps),
            "errors": len(self.errors),
            "test_results": len(self.test_results),
            "observations": len(self.observations),
        }
