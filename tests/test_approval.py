"""
Tests for the EVORA approval system.
"""

import pytest

from evora.approval import ApprovalSystem, ApprovalDecision


class TestApprovalSystem:

    def test_auto_approve(self):
        system = ApprovalSystem(auto_approve=True)
        result = system.approve_plan("Test plan")
        assert result == ApprovalDecision.APPROVE

    def test_auto_approve_level(self):
        system = ApprovalSystem(auto_approve=True, auto_approve_level="ask")
        result = system.approve_command("pip install foo", level="safe")
        assert result is True

    def test_safe_command_auto(self):
        system = ApprovalSystem(auto_approve=True)
        result = system.approve_command("ls", level="safe")
        assert result is True

    def test_callback_registration(self):
        system = ApprovalSystem()
        called = []

        def callback(plan_text, plan_obj=None):
            called.append(plan_text)
            return ApprovalDecision.APPROVE

        system.register_callback(callback)
        result = system.approve_plan("Test plan")
        assert result == ApprovalDecision.APPROVE
        assert called == ["Test plan"]

    def test_decision_enum(self):
        assert ApprovalDecision.APPROVE.value == "approve"
        assert ApprovalDecision.REJECT.value == "reject"
        assert ApprovalDecision.MODIFY.value == "modify"
        assert ApprovalDecision.CANCEL.value == "cancel"
        assert ApprovalDecision.EXPLAIN.value == "explain"
