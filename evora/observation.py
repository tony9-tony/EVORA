"""
Observation system for EVORA Phase 2.

Captures structured observations from tool execution results,
feeding them into the task state for the decision engine.
"""

from __future__ import annotations

from typing import Optional

from evora.logger import Logger
from evora.task import Observation, ActionResult, TaskState


class ObservationManager:
    """Captures and catalogs observations from action results."""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def observe_file_created(self, path: str, source: str = "write_file") -> Observation:
        obs = Observation(
            type="file_created",
            source=source,
            data={"path": path},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_file_modified(self, path: str, source: str = "edit_file") -> Observation:
        obs = Observation(
            type="file_modified",
            source=source,
            data={"path": path},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_file_deleted(self, path: str, source: str = "delete_file") -> Observation:
        obs = Observation(
            type="file_deleted",
            source=source,
            data={"path": path},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_directory_created(self, path: str, source: str = "create_directory") -> Observation:
        obs = Observation(
            type="directory_created",
            source=source,
            data={"path": path},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_command_success(self, command: str, output: str = "", return_code: int = 0) -> Observation:
        obs = Observation(
            type="command_success",
            source="execute_command",
            data={"command": command, "output": output, "return_code": return_code},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_command_failed(
        self, command: str, error: str = "", output: str = "", return_code: int = -1
    ) -> Observation:
        obs = Observation(
            type="command_failed",
            source="execute_command",
            data={
                "command": command,
                "output": output,
                "error": error,
                "return_code": return_code,
            },
            success=False,
        )
        self._log(obs)
        return obs

    def observe_test_passed(self, output: str = "", command: str = "") -> Observation:
        obs = Observation(
            type="test_passed",
            source="run_tests",
            data={"output": output, "command": command},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_test_failed(self, output: str = "", error: str = "", command: str = "") -> Observation:
        obs = Observation(
            type="test_failed",
            source="run_tests",
            data={"output": output, "error": error, "command": command},
            success=False,
        )
        self._log(obs)
        return obs

    def observe_build_failed(self, output: str = "", error: str = "") -> Observation:
        obs = Observation(
            type="build_failed",
            source="execute_command",
            data={"output": output, "error": error},
            success=False,
        )
        self._log(obs)
        return obs

    def observe_file_missing(self, path: str) -> Observation:
        obs = Observation(
            type="file_missing",
            source="read_file",
            data={"path": path},
            success=False,
        )
        self._log(obs)
        return obs

    def observe_plan_created(self, title: str, num_steps: int) -> Observation:
        obs = Observation(
            type="plan_created",
            source="planner",
            data={"title": title, "steps": num_steps},
            success=True,
        )
        self._log(obs)
        return obs

    def observe_approval(self, granted: bool, reason: str = "") -> Observation:
        obs = Observation(
            type="approval_granted" if granted else "approval_denied",
            source="approval_system",
            data={"granted": granted, "reason": reason},
            success=granted,
        )
        self._log(obs)
        return obs

    def observe_error(self, error: str, context: str = "") -> Observation:
        obs = Observation(
            type="error",
            source=context or "unknown",
            data={"error": error},
            success=False,
        )
        self._log(obs)
        return obs

    def from_action_result(self, result: ActionResult) -> list[Observation]:
        """Generate observations from an ActionResult."""
        observations: list[Observation] = []

        if result.success:
            if result.tool == "read_file":
                data = {"output_lines": len(result.output.splitlines())}
                if isinstance(result.data, dict):
                    data.update(result.data)
                observations.append(Observation(
                    type="file_read",
                    source=result.tool,
                    data=data,
                    success=True,
                ))
            elif result.tool in ("write_file", "create_directory"):
                path = result.arguments.get("path", "")
                if result.tool == "write_file":
                    observations.append(self.observe_file_created(path))
                else:
                    observations.append(self.observe_directory_created(path))
            elif result.tool == "edit_file":
                path = result.arguments.get("path", "")
                observations.append(self.observe_file_modified(path))
            elif result.tool == "execute_command":
                cmd = result.arguments.get("command", "")
                observations.append(self.observe_command_success(cmd, result.output, result.return_code))
            elif result.tool == "run_tests":
                observations.append(self.observe_test_passed(result.output))
            else:
                observations.append(Observation(
                    type="action_success",
                    source=result.tool,
                    data={"output": result.output[:200]},
                    success=True,
                ))
        else:
            if result.tool == "execute_command":
                cmd = result.arguments.get("command", "")
                observations.append(self.observe_command_failed(
                    cmd, result.error, result.output, result.return_code
                ))
            elif result.tool == "run_tests":
                observations.append(self.observe_test_failed(result.output, result.error))
            else:
                observations.append(Observation(
                    type="action_failed",
                    source=result.tool,
                    data={"error": result.error, "output": result.output[:200]},
                    success=False,
                ))

        for obs in observations:
            self._log(obs)

        return observations

    def _log(self, obs: Observation) -> None:
        if self.logger:
            detail = obs.data.get("path", obs.data.get("command", ""))
            status = "success" if obs.success else "failed"
            self.logger.observe(f"{obs.type}: {detail} ({status})")
