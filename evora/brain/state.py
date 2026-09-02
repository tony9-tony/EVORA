"""
Phase 9 — Brain persistent internal state.

Wraps TaskState with EVORA Brain-specific metadata.
TaskState remains authoritative for task-level state.
BrainState adds Brain-specific context without duplicating TaskState.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class DevelopmentState(str, Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SystemStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class ResourceState:
    """Snapshot of resource usage."""
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    disk_usage_percent: float = 0.0
    active_provider: str = ""
    active_model: str = ""
    available_providers: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    workspace_dir: str = ""
    python_version: str = ""
    os_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "available_providers": self.available_providers,
            "available_tools": self.available_tools,
            "workspace_dir": self.workspace_dir,
            "python_version": self.python_version,
            "os_name": self.os_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class BrainState:
    """Structured persistent internal state for the EVORA Brain.

    Extends/wraps TaskState with Brain-specific metadata.
    TaskState remains authoritative for task-level state.
    """

    state_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    current_objective: str = ""
    current_task: str = ""
    active_plan: Optional[dict[str, Any]] = None
    development_state: DevelopmentState = DevelopmentState.IDLE
    known_constraints: list[str] = field(default_factory=list)
    active_tools: list[str] = field(default_factory=list)
    active_context: dict[str, Any] = field(default_factory=dict)
    recent_observations: list[dict[str, Any]] = field(default_factory=list)
    system_status: SystemStatus = SystemStatus.HEALTHY
    resource_state: ResourceState = field(default_factory=ResourceState)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    version: str = "phase9"

    MAX_RECENT_OBSERVATIONS = 50
    MAX_KNOWN_CONSTRAINTS = 100
    MAX_ACTIVE_TOOLS = 50

    def update_objective(self, objective: str) -> None:
        self.current_objective = objective
        self._touch()

    def update_task(self, task: str) -> None:
        self.current_task = task
        self._touch()

    def set_plan(self, plan: Optional[dict[str, Any]]) -> None:
        self.active_plan = plan
        self._touch()

    def set_development_state(self, state: DevelopmentState) -> None:
        self.development_state = state
        self._touch()

    def add_constraint(self, constraint: str) -> None:
        if constraint and constraint not in self.known_constraints:
            self.known_constraints.append(constraint)
            if len(self.known_constraints) > self.MAX_KNOWN_CONSTRAINTS:
                self.known_constraints = self.known_constraints[-self.MAX_KNOWN_CONSTRAINTS:]
            self._touch()

    def set_active_tools(self, tools: list[str]) -> None:
        self.active_tools = list(dict.fromkeys(tools))[: self.MAX_ACTIVE_TOOLS]
        self._touch()

    def add_observation(self, observation: dict[str, Any]) -> None:
        self.recent_observations.append(observation)
        if len(self.recent_observations) > self.MAX_RECENT_OBSERVATIONS:
            self.recent_observations = self.recent_observations[-self.MAX_RECENT_OBSERVATIONS:]
        self._touch()

    def set_resource_state(self, resource_state: ResourceState) -> None:
        self.resource_state = resource_state
        self._touch()

    def set_system_status(self, status: SystemStatus) -> None:
        self.system_status = status
        self._touch()

    def _touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "current_objective": self.current_objective,
            "current_task": self.current_task,
            "active_plan": self.active_plan,
            "development_state": self.development_state.value,
            "known_constraints": self.known_constraints,
            "active_tools": self.active_tools,
            "active_context": self.active_context,
            "recent_observations": self.recent_observations,
            "system_status": self.system_status.value,
            "resource_state": self.resource_state.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrainState":
        data = copy.deepcopy(data)
        data["development_state"] = DevelopmentState(data.get("development_state", "idle"))
        data["system_status"] = SystemStatus(data.get("system_status", "healthy"))
        resource_data = data.get("resource_state", {})
        if isinstance(resource_data, dict):
            data["resource_state"] = ResourceState.from_dict(resource_data)
        elif not isinstance(resource_data, ResourceState):
            data["resource_state"] = ResourceState()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def snapshot(self) -> dict[str, Any]:
        """Lightweight snapshot for logging."""
        return {
            "state_id": self.state_id,
            "development_state": self.development_state.value,
            "system_status": self.system_status.value,
            "current_objective": self.current_objective[:80] if self.current_objective else "",
            "current_task": self.current_task[:80] if self.current_task else "",
            "active_tools_count": len(self.active_tools),
            "recent_observations_count": len(self.recent_observations),
            "constraints_count": len(self.known_constraints),
        }

    def validate(self) -> list[str]:
        """Validate state integrity. Returns list of issues (empty if valid)."""
        issues: list[str] = []
        if not isinstance(self.state_id, str) or not self.state_id:
            issues.append("state_id must be a non-empty string")
        if not isinstance(self.current_objective, str):
            issues.append("current_objective must be a string")
        if not isinstance(self.current_task, str):
            issues.append("current_task must be a string")
        if not isinstance(self.known_constraints, list):
            issues.append("known_constraints must be a list")
        if not isinstance(self.active_tools, list):
            issues.append("active_tools must be a list")
        if not isinstance(self.recent_observations, list):
            issues.append("recent_observations must be a list")
        if not isinstance(self.active_context, dict):
            issues.append("active_context must be a dict")
        if not isinstance(self.resource_state, ResourceState):
            issues.append("resource_state must be a ResourceState")
        return issues
