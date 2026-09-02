"""
Approval system for EVORA.

Handles user interaction for approving/rejecting/modifying plans and
risky operations. Supports both interactive (CLI) and non-interactive
(auto-approve) modes.
"""

import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from evora.logger import Logger, Stage


SECRET_KEY = os.urandom(32)


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


@dataclass
class ApprovalToken:
    """One-time approval token bound to a specific session, plan, and identity."""
    token_id: str
    session_id: str
    plan_id: str
    candidate_id: str
    approved_by: str
    issued_at: float
    nonce: str
    signature: str

    @staticmethod
    def _sign(session_id: str, plan_id: str, candidate_id: str, approved_by: str, issued_at: float, nonce: str) -> str:
        payload = f"{session_id}:{plan_id}:{candidate_id}:{approved_by}:{issued_at}:{nonce}"
        return hmac.new(SECRET_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    @classmethod
    def create(cls, session_id: str, plan_id: str, candidate_id: str, approved_by: str) -> "ApprovalToken":
        nonce = uuid.uuid4().hex[:16]
        issued_at = time.time()
        signature = cls._sign(session_id, plan_id, candidate_id, approved_by, issued_at, nonce)
        return cls(
            token_id=f"appr-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            plan_id=plan_id,
            candidate_id=candidate_id,
            approved_by=approved_by,
            issued_at=issued_at,
            nonce=nonce,
            signature=signature,
        )

    def verify(self, session_id: str, plan_id: str, candidate_id: str, ttl: float = 3600.0) -> bool:
        if self.session_id != session_id or self.plan_id != plan_id or self.candidate_id != candidate_id:
            return False
        if time.time() - self.issued_at > ttl:
            return False
        expected = self._sign(self.session_id, self.plan_id, self.candidate_id, self.approved_by, self.issued_at, self.nonce)
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "session_id": self.session_id,
            "plan_id": self.plan_id,
            "candidate_id": self.candidate_id,
            "approved_by": self.approved_by,
            "issued_at": self.issued_at,
            "nonce": self.nonce,
            "signature": self.signature,
        }


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
        self._issued_tokens: dict[str, ApprovalToken] = {}

    def register_callback(self, callback):
        """Register a callback for programmatic approval (used in non-interactive mode)."""
        self._approval_callbacks.append(callback)

    def _authoritative_decision(self, plan_text: str, plan_obj: Any = None) -> tuple[ApprovalDecision, Optional[str]]:
        if self.auto_approve:
            if self.logger:
                self.logger.ask(f"Auto-approved plan (auto_approve={self.auto_approve})")
            return ApprovalDecision.APPROVE, "auto_approve"

        for cb in self._approval_callbacks:
            result = cb(plan_text, plan_obj)
            if result is not None:
                if isinstance(result, ApprovalDecision):
                    return result, "callback"
                return ApprovalDecision(result), "callback"

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
            return ApprovalDecision.REJECT, "interactive"

        if not choice:
            choice = "1"

        mapping = {
            "1": ApprovalDecision.APPROVE,
            "2": ApprovalDecision.REJECT,
            "3": ApprovalDecision.MODIFY,
            "4": ApprovalDecision.CANCEL,
            "5": ApprovalDecision.EXPLAIN,
        }

        return mapping.get(choice, ApprovalDecision.REJECT), "interactive"

    def approve_plan(self, plan_text: str, plan_obj: Any = None) -> ApprovalDecision:
        decision, _ = self._authoritative_decision(plan_text, plan_obj)
        return decision

    def issue_approval_token(self, session_id: str, plan_id: str, candidate_id: str, approved_by: str) -> Optional[ApprovalToken]:
        decision, source = self._authoritative_decision("", None)
        if decision != ApprovalDecision.APPROVE:
            return None
        token = ApprovalToken.create(session_id, plan_id, candidate_id, approved_by)
        self._issued_tokens[token.token_id] = token
        return token

    def consume_approval_token(self, token_id: str, session_id: str, plan_id: str, candidate_id: str, ttl: float = 3600.0) -> bool:
        token = self._issued_tokens.pop(token_id, None)
        if token is None:
            return False
        return token.verify(session_id, plan_id, candidate_id, ttl)

    def approve_command(self, command: str, level: str = "ask", reason: str = "") -> bool:
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
            return False

        return choice in ("", "y", "yes")

    def get_modification(self, message: str = "Enter your modification:") -> str:
        print(f"\n[ASK] {message}")
        try:
            return input("> ").strip()
        except EOFError:
            return ""

    def explain_plan(self, plan_text: str) -> str:
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
