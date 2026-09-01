"""
Approval system for EVORA.

Handles user interaction for approving/rejecting/modifying plans and
risky operations. Supports both interactive (CLI) and non-interactive
(auto-approve) modes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from evora.logger import Logger, Stage


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    CANCEL = "cancel"
    EXPLAIN = "explain"


@dataclass
class ApprovalRequest:
    """A request for user approval."""
    prompt: str
    level: str
    context: dict[str, Any] = field(default_factory=dict)
    options: list[str] = field(default_factory=list)


class ApprovalSystem:
    """Manages user approval flow for plans and risky operations."""

    def __init__(
        self,
        logger: Optional[Logger] = None,
        auto_approve: bool = False,
        auto_approve_level: str = "safe",
    ):
        self.logger = logger
        self.auto_approve = auto_approve
        self.auto_approve_level = auto_approve_level
        self._approval_callbacks: list = []

    def register_callback(self, callback):
        """Register a callback for programmatic approval (used in non-interactive mode)."""
        self._approval_callbacks.append(callback)

    def approve_plan(self, plan_text: str, plan_obj: Any = None) -> ApprovalDecision:
        """Present a plan and ask for approval."""
        if self.auto_approve:
            if self.logger:
                self.logger.ask(f"Auto-approved plan (auto_approve={self.auto_approve})")
            return ApprovalDecision.APPROVE

        for cb in self._approval_callbacks:
            result = cb(plan_text, plan_obj)
            if result is not None:
                return result

        if self.logger:
            self.logger.ask("Approval required for plan")

        print(f"\n{'=' * 60}")
        print(plan_text)
        print(f"{'=' * 60}")

        options = ["Approve", "Reject", "Modify", "Cancel", "Explain"]
        print("\nOptions:")
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")

        try:
            choice = input("\nYour choice [1]: ").strip()
        except EOFError:
            return ApprovalDecision.APPROVE

        if not choice:
            choice = "1"

        mapping = {
            "1": ApprovalDecision.APPROVE,
            "2": ApprovalDecision.REJECT,
            "3": ApprovalDecision.MODIFY,
            "4": ApprovalDecision.CANCEL,
            "5": ApprovalDecision.EXPLAIN,
        }

        return mapping.get(choice, ApprovalDecision.APPROVE)

    def approve_command(self, command: str, level: str = "ask", reason: str = "") -> bool:
        """Ask for approval before executing a risky command."""
        if level.lower() == "safe":
            return True

        if self.auto_approve and level.lower() <= self.auto_approve_level:
            if self.logger:
                self.logger.ask(f"Auto-approved {level} command: {command}")
            return True

        for cb in self._approval_callbacks:
            result = cb(command, level, reason)
            if result is not None:
                return bool(result)

        if self.logger:
            self.logger.ask(f"Approval required for command: {command}")

        display_level = level.upper() if isinstance(level, str) else str(level)
        print(f"\n[ASK] Command execution approval required ({display_level})")
        print(f"  Command: {command}")
        if reason:
            print(f"  Reason: {reason}")

        try:
            choice = input("\nApprove? [Y/n]: ").strip().lower()
        except EOFError:
            return True

        return choice in ("", "y", "yes")

    def get_modification(self, message: str = "Enter your modification:") -> str:
        """Get a free-text modification from the user."""
        print(f"\n[ASK] {message}")
        try:
            return input("> ").strip()
        except EOFError:
            return ""

    def explain_plan(self, plan_text: str) -> str:
        """Provide an explanation of the plan (user can then decide)."""
        print(f"\n[ASK] Plan Explanation:")
        print(f"  - This plan is generated based on your request.")
        print(f"  - Each step is an atomic action that EVORA will execute.")
        print(f"  - Steps with dependencies will be executed in order.")
        print(f"  - You can approve to proceed, modify to change, or reject to cancel.")
        print(f"  - Use 'modify' to provide a different approach or add constraints.")
        try:
            input("\nPress Enter to continue...")
        except EOFError:
            pass
        return "explanation provided"
