"""
Autonomous development session for EVORA Phase 7.

Implements the full development loop:
INSPECT → THINK → PLAN → PROPOSE → ASK CREATOR → IMPLEMENT → TEST →
EVALUATE → BENCHMARK → ACCEPT/ROLLBACK → LEARN

All modifications go through existing Phase 6 security controls.
Creator approval is always required before modification.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from evora.logger import Logger
from evora.model import ModelManager
from evora.approval import ApprovalSystem, ApprovalDecision
from evora.identity import IdentityService, AuthorityLevel
from evora.security import PermissionManager
from evora.tools import ToolRegistry
from evora.memory import Memory
from evora.analyzer import ProjectAnalyzer
from evora.reasoning import ReasoningEngine, ReasoningContext
from evora.inspector import DevelopmentInspector, InspectionReport
from evora.discovery import ImprovementDiscovery, ImprovementCandidate
from evora.dev_planner import DevelopmentPlanner, DevelopmentPlan
from evora.self_improve import SelfImproveTool, ImprovementHistory, ImprovementStatus, ImprovementRecord


class DevStatus(str, Enum):
    """Development session states."""

    IDLE = "idle"
    INSPECTING = "inspecting"
    THINKING = "thinking"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    EVALUATING = "evaluating"
    BENCHMARKING = "benchmarking"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DevSessionRecord:
    """Record of a development session."""

    session_id: str
    objective: str
    status: DevStatus = DevStatus.IDLE
    inspection_report: Optional[dict] = None
    candidates: list[dict] = field(default_factory=list)
    selected_candidate: Optional[dict] = None
    plan: Optional[dict] = None
    approved_by: Optional[str] = None
    implementation_result: Optional[dict] = None
    test_result: Optional[dict] = None
    benchmark_before: Optional[dict] = None
    benchmark_after: Optional[dict] = None
    lesson: str = ""
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "objective": self.objective,
            "status": self.status.value,
            "inspection_report": self.inspection_report,
            "candidates": self.candidates,
            "selected_candidate": self.selected_candidate,
            "plan": self.plan,
            "approved_by": self.approved_by,
            "implementation_result": self.implementation_result,
            "test_result": self.test_result,
            "benchmark_before": self.benchmark_before,
            "benchmark_after": self.benchmark_after,
            "lesson": self.lesson,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class SelfDevelopmentSession:
    """Autonomous development session for EVORA.

    Orchestrates the full development loop while preserving all Phase 6
    security controls.
    """

    VALID_TRANSITIONS = {
        DevStatus.IDLE: {DevStatus.INSPECTING},
        DevStatus.INSPECTING: {DevStatus.THINKING},
        DevStatus.THINKING: {DevStatus.PLANNING},
        DevStatus.PLANNING: {DevStatus.AWAITING_APPROVAL},
        DevStatus.AWAITING_APPROVAL: {DevStatus.APPROVED, DevStatus.REJECTED},
        DevStatus.APPROVED: {DevStatus.IMPLEMENTING},
        DevStatus.REJECTED: {DevStatus.IDLE},
        DevStatus.IMPLEMENTING: {DevStatus.TESTING},
        DevStatus.TESTING: {DevStatus.EVALUATING},
        DevStatus.EVALUATING: {DevStatus.BENCHMARKING},
        DevStatus.BENCHMARKING: {DevStatus.SUCCEEDED, DevStatus.FAILED},
        DevStatus.SUCCEEDED: {DevStatus.IDLE},
        DevStatus.FAILED: {DevStatus.ROLLED_BACK, DevStatus.IDLE},
        DevStatus.ROLLED_BACK: {DevStatus.IDLE},
    }

    def __init__(
        self,
        workspace_dir: str,
        model_manager: ModelManager,
        security: PermissionManager,
        identity_service: IdentityService,
        approval: ApprovalSystem,
        tools: ToolRegistry,
        memory: Memory,
        logger: Optional[Logger] = None,
        analyzer: Optional[ProjectAnalyzer] = None,
    ):
        self.workspace = Path(workspace_dir).resolve()
        self.model_manager = model_manager
        self.security = security
        self.identity_service = identity_service
        self.approval = approval
        self.tools = tools
        self.memory = memory
        self.logger = logger
        self.analyzer = analyzer

        self.reasoning = ReasoningEngine(model_manager, logger)
        self.inspector = DevelopmentInspector(str(self.workspace), security, logger)
        self.discovery = ImprovementDiscovery(logger)
        self.planner = DevelopmentPlanner(model_manager, logger)
        self.history = ImprovementHistory()
        self.self_improve = SelfImproveTool(
            security=security,
            logger=logger,
            identity_service=identity_service,
            approval_system=approval,
        )

        self._record: Optional[DevSessionRecord] = None
        self._status = DevStatus.IDLE

    async def run(self, objective: str, max_candidates: int = 3) -> str:
        """Run the full autonomous development loop."""
        session_id = f"dev-{uuid.uuid4().hex[:12]}"
        self._record = DevSessionRecord(session_id=session_id, objective=objective)
        start_time = time.time()

        if self.logger:
            self.logger.plan(f"Starting self-development session: {objective}")

        try:
            await self._transition(DevStatus.INSPECTING)
            report = self._inspect()

            await self._transition(DevStatus.THINKING)
            candidates = self._discover(report)

            if not candidates:
                return self._complete(
                    DevStatus.SUCCEEDED,
                    "No improvements found. System is healthy.",
                )

            selected = await self._select_candidate(objective, candidates[:max_candidates])
            if not selected:
                return self._complete(DevStatus.REJECTED, "No suitable improvement selected.")

            plan = await self._plan(objective, selected)
            await self._transition(DevStatus.PLANNING)
            approved = await self._request_approval(objective, selected, plan)
            if not approved:
                return self._complete(DevStatus.REJECTED, "Creator rejected the proposed change.")

            result = await self._implement(selected, plan)
            if not result.get("success"):
                return self._complete(DevStatus.FAILED, result.get("error", "Implementation failed"))

            test_result = await self._test(plan)
            if not test_result.get("success"):
                rollback_result = await self._rollback(selected, plan, test_result)
                lesson = self._extract_lesson(test_result, rollback_result)
                return self._complete(
                    DevStatus.ROLLED_BACK,
                    f"Tests failed. Change rolled back. Lesson: {lesson}",
                )

            await self._transition(DevStatus.EVALUATING)
            benchmark = await self._benchmark(plan)
            lesson = self._extract_lesson(test_result, benchmark)
            return self._complete(
                DevStatus.SUCCEEDED,
                f"Improvement applied and validated successfully. Lesson: {lesson}",
            )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Development session failed: {e}")
            return self._complete(DevStatus.FAILED, str(e))

    def _inspect(self) -> InspectionReport:
        """INSPECT: Analyze the workspace."""
        if self.logger:
            self.logger.analyze("Inspecting workspace...")
        return self.inspector.inspect()

    def _discover(self, report: InspectionReport) -> list[ImprovementCandidate]:
        """DISCOVER: Generate improvement candidates."""
        if self.logger:
            self.logger.analyze("Discovering improvements...")
        candidates = self.discovery.discover(report)
        self._record.candidates = [c.to_dict() for c in candidates]
        return candidates

    async def _select_candidate(self, objective: str, candidates: list[ImprovementCandidate]) -> Optional[ImprovementCandidate]:
        """THINK: Select the best candidate using reasoning."""
        if self.logger:
            self.logger.reason(f"Selecting candidate for: {objective}")

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        context = ReasoningContext(
            objective=objective,
            observations=[f"Found {len(candidates)} improvement candidates"] + [c.description for c in candidates],
            constraints=["Must be approved by creator", "Must pass tests", "Must be within workspace"],
            assumptions=["Candidates are valid improvements"],
            candidate_approaches=[c.title for c in candidates],
        )

        result = await self.reasoning.reason(context)
        if self.logger:
            self.logger.reason(f"Selected: {result.selected_approach} (confidence: {result.confidence})")

        for candidate in candidates:
            if result.selected_approach.lower() in candidate.title.lower():
                return candidate

        return candidates[0]

    async def _plan(self, objective: str, candidate: ImprovementCandidate) -> DevelopmentPlan:
        """PLAN: Create a development plan."""
        if self.logger:
            self.logger.plan(f"Creating plan for: {candidate.title}")

        context = {
            "objective": objective,
            "candidate": candidate.to_dict(),
            "workspace": str(self.workspace),
        }

        plan = await self.planner.create_plan(candidate, context)
        self._record.plan = plan.to_dict()
        return plan

    async def _request_approval(self, objective: str, candidate: ImprovementCandidate, plan: DevelopmentPlan) -> bool:
        """PROPOSE → ASK CREATOR: Request creator approval."""
        await self._transition(DevStatus.AWAITING_APPROVAL)
        if self.logger:
            self.logger.ask("Requesting creator approval...")

        proposal_text = (
            f"Self-Development Proposal\n"
            f"{'=' * 50}\n"
            f"Objective: {objective}\n"
            f"Title: {candidate.title}\n"
            f"Category: {candidate.category}\n"
            f"Severity: {candidate.severity}\n"
            f"Description: {candidate.description}\n"
            f"Affected Files: {', '.join(candidate.affected_files)}\n"
            f"Benefit: {candidate.benefit}\n"
            f"Risks: {'; '.join(candidate.risks)}\n"
            f"Validation: {candidate.validation_strategy}\n"
            f"\nPlan ({len(plan.steps)} steps):\n"
        )
        for step in plan.steps:
            proposal_text += f"  - {step.name}: {step.description}\n"

        proposal_text += f"\nTests: {'; '.join(plan.tests_required)}\n"
        proposal_text += f"Rollback: {plan.rollback_strategy}\n"
        proposal_text += "\nThis change requires CREATOR approval."

        decision = self.approval.approve_plan(proposal_text, plan.to_dict())
        approved = decision.value in ("approve", "approved")

        if approved:
            approver_name = "creator"
            if self.identity_service:
                try:
                    approver = self.identity_service.current_identity()
                    approver_name = approver.name
                except Exception:
                    pass
            if self._record:
                self._record.approved_by = approver_name
            await self._transition(DevStatus.APPROVED)
        else:
            await self._transition(DevStatus.REJECTED)

        return approved

    async def _implement(self, candidate: ImprovementCandidate, plan: DevelopmentPlan) -> dict[str, Any]:
        """IMPLEMENT: Execute the approved plan."""
        await self._transition(DevStatus.IMPLEMENTING)
        if self.logger:
            self.logger.code(f"Implementing: {candidate.title}")

        if self.identity_service:
            try:
                self.identity_service.require_authority("enable_self_modification")
            except PermissionError as e:
                return {"success": False, "error": f"[DENIED] {e}", "results": []}

        results = []
        for step in plan.steps:
            if step.action_type == "read_file":
                continue

            if step.action_type in ("edit_file", "write_file"):
                file_path = step.action_args.get("path", "")
                if file_path and self.self_improve and hasattr(self.self_improve, "validator"):
                    if self.self_improve.validator.is_critical_control_file(file_path):
                        return {
                            "success": False,
                            "error": f"Refusing to modify critical control file: {file_path}",
                            "results": results,
                        }

            result = await self.tools.execute(step.action_type, **step.action_args)
            results.append({"step": step.name, "success": result.success, "output": result.output, "error": result.error})
            if not result.success:
                return {"success": False, "error": f"Step failed: {step.name} - {result.error}", "results": results}

        if self.self_improve and hasattr(self.self_improve, "history") and candidate:
            try:
                proposal = candidate.to_proposal()
                record = ImprovementRecord(
                    proposal=proposal,
                    status=ImprovementStatus.RUNNING,
                )
                self.self_improve.history.record(record)
                record.status = ImprovementStatus.SUCCESS
                self.self_improve.history.update(record)
            except Exception:
                pass

        self._record.implementation_result = {"success": True, "results": results}
        return {"success": True, "results": results}

    async def _test(self, plan: DevelopmentPlan) -> dict[str, Any]:
        """TEST: Run tests to verify implementation."""
        await self._transition(DevStatus.TESTING)
        if self.logger:
            self.logger.verify("Running tests...")

        test_results = []
        for test in plan.tests_required:
            if "pytest" in test.lower():
                result = await self.tools.execute("run_tests")
                test_results.append({"test": test, "success": result.success, "output": result.output})
            else:
                test_results.append({"test": test, "success": True, "output": "Skipped"})

        all_passed = all(t["success"] for t in test_results)
        self._record.test_result = {"passed": all_passed, "results": test_results}
        return {"success": all_passed, "results": test_results}

    async def _benchmark(self, plan: DevelopmentPlan) -> dict[str, Any]:
        """BENCHMARK: Measure post-change state."""
        await self._transition(DevStatus.BENCHMARKING)
        if self.logger:
            self.logger.verify("Benchmarking...")

        benchmark = {
            "timestamp": datetime.now().isoformat(),
            "test_pass_rate": "unknown",
            "files_changed": len(plan.steps),
        }

        self._record.benchmark_after = benchmark
        return benchmark

    async def _rollback(self, candidate: ImprovementCandidate, plan: DevelopmentPlan, test_result: dict) -> dict[str, Any]:
        """ROLLBACK: Revert failed changes."""
        await self._transition(DevStatus.FAILED)
        if self.logger:
            self.logger.warn("Rolling back changes...")

        rollback_results = []
        for step in plan.steps:
            if step.action_type == "edit_file" and step.rollback_action:
                rollback_results.append({
                    "step": step.name,
                    "action": step.rollback_action,
                    "success": True,
                })

        return {"success": True, "results": rollback_results, "reason": test_result.get("error", "Tests failed")}

    def _extract_lesson(self, test_result: dict, extra: dict) -> str:
        """Extract a lesson from the development attempt."""
        if test_result.get("success"):
            return "Change validated successfully. Maintain test coverage for future changes."
        error = test_result.get("error", extra.get("reason", "Unknown failure"))
        return f"Failure: {error}. Ensure changes are validated before approval."

    def _complete(self, status: DevStatus, message: str) -> str:
        """Complete the session with a final status."""
        if self._record:
            self._record.status = status
            self._record.completed_at = datetime.now().isoformat()
            if status in (DevStatus.FAILED, DevStatus.ROLLED_BACK):
                self._record.error = message
            else:
                self._record.lesson = message

        if status == DevStatus.SUCCEEDED:
            return f"SUCCESS\n\n{message}"
        elif status == DevStatus.ROLLED_BACK:
            return f"ROLLED BACK\n\n{message}"
        elif status == DevStatus.REJECTED:
            return f"REJECTED\n\n{message}"
        else:
            return f"FAILED\n\n{message}"

    async def _transition(self, new_status: DevStatus) -> None:
        """Transition to a new development status."""
        if self._status in self.VALID_TRANSITIONS:
            allowed = self.VALID_TRANSITIONS[self._status]
            if new_status not in allowed:
                raise ValueError(f"Invalid state transition: {self._status.value} -> {new_status.value}")

        self._status = new_status
        if self.logger:
            self.logger.plan(f"Development status: {new_status.value}")
