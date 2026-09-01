"""
Decision engine for EVORA Phase 2.

Determines what EVORA should do next based on the current task state
and observations, returning a structured Decision rather than arbitrary text.
"""

from __future__ import annotations

from typing import Optional

from evora.logger import Logger
from evora.task import TaskState, Decision, Observation


class DecisionEngine:
    """Determines the next action based on task state and observations.

    This is the core of the autonomous loop. Instead of a predetermined
    sequence, the engine examines the current state and decides dynamically.
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        max_retries: int = 3,
        auto_approve: bool = False,
    ):
        self.logger = logger
        self.max_retries = max_retries
        self.auto_approve = auto_approve
        self._retry_counts: dict[str, int] = {}

    def decide_next(self, state: TaskState) -> Decision:
        """Main entry point — determines what to do next based on current state.

        Implements a state machine that considers:
        - Current status
        - Plan steps remaining
        - Test results
        - Errors
        - Retry counts
        - Completion criteria
        """
        original_status = state.status

        if state.is_complete or state.is_failed or state.is_cancelled:
            return self._decide_report(state)

        if original_status == "idle" or not state.goal:
            return self._decide_understand(state)

        if original_status == "testing":
            if state.test_results and not state.test_results[-1].passed:
                return self._decide_fix(state)

        if not state.project_context:
            return self._decide_analyze(state)

        if not state.plan:
            return self._decide_plan(state)

        if original_status == "awaiting_approval":
            return self._decide_ask_approval(state)

        if not state.remaining_steps and not state.test_results:
            return self._decide_run_tests(state)

        if state.remaining_steps:
            return self._decide_execute(state)

        if state.exceeded_retry_limit():
            return self._decide_report(state)

        if self._check_completion(state):
            return self._decide_verify(state)

        return self._decide_done(state)

    def _decide_understand(self, state: TaskState) -> Decision:
        """Determine the goal from the user request."""
        if not state.request:
            return Decision(
                action="done",
                reason="No request provided",
                confidence=1.0,
            )

        state.goal = state.request
        self._log_decision("understand", f"Goal: {state.goal}")
        return Decision(
            action="understand",
            reason=f"Understanding request: {state.request[:100]}",
            tool=None,
            confidence=1.0,
        )

    def _decide_analyze(self, state: TaskState) -> Decision:
        """Decide to analyze the workspace."""
        self._log_decision("analyze", f"Analyzing workspace: {state.workspace}")
        return Decision(
            action="analyze",
            reason=f"Need to inspect workspace at {state.workspace} to understand project context",
            tool="list_directory",
            arguments={"path": state.workspace},
            confidence=1.0,
        )

    def _decide_plan(self, state: TaskState) -> Decision:
        """Decide to create a plan."""
        self._log_decision("plan", f"Planning for goal: {state.goal[:80]}")
        return Decision(
            action="plan",
            reason="No plan exists yet; need to decompose the request into steps",
            confidence=1.0,
        )

    def _decide_ask_approval(self, state: TaskState) -> Decision:
        """Present the plan and request approval."""
        self._log_decision("ask_approval", "Awaiting user approval for plan")
        return Decision(
            action="ask_approval",
            reason="Plan created, awaiting user approval before execution",
            requires_approval=True,
            confidence=1.0,
        )

    def _decide_execute(self, state: TaskState) -> Decision:
        """Decide to execute the next remaining step."""
        step = state.remaining_steps[0]
        step_name = step.get("name", step.get("action_type", "unknown"))
        action_type = step.get("action_type", "create_file")
        action_args = step.get("action_args", {})

        risk = "safe"
        requires_approval = False
        if action_type == "run_command":
            risk = "ask"
            requires_approval = not self.auto_approve
        elif action_type in ("run_tests",):
            risk = "ask"
            requires_approval = not self.auto_approve

        self._log_decision("execute", f"Executing step: {step_name} ({action_type})")

        tool_name = self._action_type_to_tool(action_type)
        return Decision(
            action="execute_tool",
            reason=f"Executing plan step '{step_name}'",
            tool=tool_name,
            arguments={**action_args, "action_type": action_type, "step_id": step.get("id", "")},
            expected_outcome=f"Step '{step_name}' should complete successfully",
            risk_level=risk,
            requires_approval=requires_approval,
            confidence=0.9,
        )

    def _decide_run_tests(self, state: TaskState) -> Decision:
        """Decide to run tests after plan steps are complete."""
        self._log_decision("test", "All plan steps complete, running tests")
        return Decision(
            action="run_tests",
            reason="All plan steps executed, need to verify with tests",
            tool="run_tests",
            arguments={},
            risk_level="ask",
            requires_approval=not self.auto_approve,
            confidence=0.9,
        )

    def _decide_fix(self, state: TaskState) -> Decision:
        """Decide to attempt a fix for test failures."""
        retry_key = "fix_attempt"
        count = self._retry_counts.get(retry_key, 0)
        self._retry_counts[retry_key] = count + 1

        if count >= self.max_retries:
            self._log_decision("fix", f"Max retries ({self.max_retries}) exceeded, reporting failure")
            return Decision(
                action="report",
                reason=f"Exceeded max retry limit ({self.max_retries}) for fixes",
                confidence=1.0,
            )

        self._log_decision("fix", f"Attempting fix (attempt {count + 1}/{self.max_retries})")
        return Decision(
            action="fix_error",
            reason=f"Tests failed, attempting automatic fix (attempt {count + 1}/{self.max_retries})",
            tool="run_command",
            arguments={
                "command": "python -m pytest -v",
                "context": "fix_attempt",
                "attempt": count + 1,
            },
            risk_level="ask",
            requires_approval=not self.auto_approve,
            confidence=0.7,
        )

    def _decide_verify(self, state: TaskState) -> Decision:
        """Decide to verify task completion."""
        self._log_decision("verify", "Task appears complete, running final verification")
        return Decision(
            action="verify",
            reason="Checking completion criteria",
            tool="run_tests",
            arguments={},
            confidence=0.8,
        )

    def _decide_report(self, state: TaskState) -> Decision:
        """Decide to report final results."""
        result = "completed" if state.is_complete else "failed" if state.is_failed else "cancelled"
        self._log_decision("report", f"Reporting task result: {result}")
        return Decision(
            action="report",
            reason=f"Task status: {result}",
            confidence=1.0,
        )

    def _decide_done(self, state: TaskState) -> Decision:
        """Decide the task is done."""
        self._log_decision("done", "Task completed successfully")
        return Decision(
            action="done",
            reason="Task completed successfully",
            confidence=1.0,
        )

    def _action_type_to_tool(self, action_type: str) -> str:
        """Map plan action types to tool names."""
        mapping = {
            "create_file": "write_file",
            "edit_file": "edit_file",
            "read_file": "read_file",
            "run_command": "execute_command",
            "run_tests": "execute_command",
            "analyze": "list_directory",
            "create_directory": "create_directory",
        }
        return mapping.get(action_type, "read_file")

    def _check_completion(self, state: TaskState) -> bool:
        """Check if all plan steps are done and tests pass."""
        if state.remaining_steps:
            return False
        if state.test_results and not state.test_results[-1].passed:
            return False
        return True

    def _log_decision(self, action: str, reason: str) -> None:
        if self.logger:
            self.logger.decide(f"Decision: {action} — {reason}")

    def reset_retry_count(self, key: str = "fix_attempt") -> None:
        """Reset retry counter after a successful action."""
        self._retry_counts.pop(key, None)
