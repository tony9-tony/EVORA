"""
Autonomous agent loop for EVORA Phase 2.

Implements the full DECIDE → ACT → OBSERVE → EVALUATE → REASON → DECIDE AGAIN cycle,
replacing the hardcoded linear pipeline with a dynamic decision-driven loop.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger
from evora.task import TaskState, Observation, Decision, ActionResult, TestResult
from evora.decision import DecisionEngine
from evora.observation import ObservationManager
from evora.evaluation import Evaluator, EvaluationOutcome, EvaluationResult
from evora.planner import Planner, Plan
from evora.approval import ApprovalSystem, ApprovalDecision
from evora.tools import ToolRegistry
from evora.memory import Memory
from evora.identity import IdentityService
from evora.security import PermissionManager, PermissionLevel
from evora.analyzer import ProjectAnalyzer


@dataclass
class AutonomousConfig:
    """Configuration for the autonomous agent."""
    max_retries: int = 3
    retry_delay: float = 2.0
    command_timeout: int = 60
    auto_approve: bool = False
    max_iterations: int = 50


class Phase(str):
    """Phase identifiers matching the logger stages."""
    UNDERSTAND = "UNDERSTAND"
    ANALYZE = "ANALYZE"
    PLAN = "PLAN"
    ASK = "ASK"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    EVALUATE = "EVALUATE"
    REASON = "REASON"
    VERIFY = "VERIFY"
    REPORT = "REPORT"
    MEMORY = "MEMORY"


class AutonomousAgent:
    """EVORA's Phase 2 autonomous agent.

    Uses the decision engine to determine the next action based on
    the current task state and observations, rather than following
    a predetermined sequence.
    """

    def __init__(
        self,
        model_manager: Any,
        planner: Planner,
        approval: ApprovalSystem,
        tools: ToolRegistry,
        memory: Memory,
        security: PermissionManager,
        logger: Logger,
        analyzer: ProjectAnalyzer,
        config: Optional[AutonomousConfig] = None,
        identity_service: Optional[IdentityService] = None,
        memory_service: Optional[Any] = None,
    ):
        self.model_manager = model_manager
        self.planner = planner
        self.approval = approval
        self.tools = tools
        self.memory = memory
        self.security = security
        self.logger = logger
        self.analyzer = analyzer
        self.config = config or AutonomousConfig()

        self.decision_engine = DecisionEngine(
            logger=logger,
            max_retries=self.config.max_retries,
            auto_approve=self.config.auto_approve,
        )
        self.observation_mgr = ObservationManager(logger=logger)
        self.evaluator = Evaluator(logger=logger)

        self._current_state: Optional[TaskState] = None
        self._current_plan: Optional[Plan] = None
        self._is_stopped = False

        # Phase 3: Identity and memory service integration
        self.identity_service = identity_service
        self.memory_service = memory_service

        if identity_service is None and memory_service is not None:
            # If memory_service was provided without identity, create a minimal one
            self.identity_service = memory_service.identity_service

    async def run(self, request: str, project_context: Optional[dict] = None) -> str:
        """Execute the full autonomous loop.

        GOAL → UNDERSTAND → ANALYZE → DECIDE → ACT → OBSERVE → EVALUATE → REASON → DECIDE AGAIN → ... → REPORT
        """
        start_time = time.time()

        state = TaskState(
            request=request,
            goal=request,
            workspace=self.security.workspace_dir,
            project_context=project_context or {},
            max_attempts=self.config.max_iterations,
        )

        if project_context is None and self.analyzer:
            analysis = self.analyzer.analyze()
            state.project_context = analysis.to_dict()

        self._current_state = state
        self._save_memory(state)

        # Phase 3: Retrieve relevant memory before acting
        if self.memory_service is not None:
            try:
                relevant = self.memory_service.retrieve_relevant(
                    goal=state.goal,
                    project=str(state.workspace),
                )
                if relevant:
                    state.project_context["relevant_memories"] = [
                        r.to_dict() for r in relevant
                    ]
                    self.logger.memory(
                        f"Retrieved {len(relevant)} relevant memories for goal: {state.goal[:80]}"
                    )
            except Exception as e:
                self.logger.memory(f"Memory retrieval skipped: {e}")

        self.logger.understand(f"Understood request: {request[:100]}")

        iteration = 0
        while not state.is_complete and not state.is_failed and not state.is_cancelled:
            if self._is_stopped:
                state.mark_cancelled("Agent stopped by user")
                break

            if iteration >= self.config.max_iterations:
                state.mark_failed(f"Exceeded max iterations ({self.config.max_iterations})")
                self.logger.error("Max iterations exceeded, stopping.")
                break

            iteration += 1
            self.logger.reason(f"--- Iteration {iteration} ---")

            decision = self.decision_engine.decide_next(state)
            state.add_decision(decision)

            if decision.requires_approval and not self._check_approval(decision):
                state.mark_cancelled("Cancelled by user during approval")
                self.logger.warn("Action cancelled by user")
                break

            result = await self._act(state, decision)
            state.add_action(decision.tool or "none", decision.arguments, result)

            observations = self._observe(state, result)

            eval_result = self.evaluator.evaluate(state, observations[-1] if observations else Observation(
                type="noop", source="agent", success=True
            ))
            state.updated_at = time.time()

            self._reason(state, decision, eval_result, observations)

            self._advance_state(state, decision, eval_result)

        state.elapsed = time.time() - start_time
        return self._report(state)

    def _observe(self, state: TaskState, result: ActionResult) -> list[Observation]:
        """Phase: OBSERVE — capture findings from the action result."""
        observations = self.observation_mgr.from_action_result(result)
        for obs in observations:
            state.add_observation(obs)
        return observations

    def _check_approval(self, decision: Decision) -> bool:
        """Check if an action requiring approval is approved."""
        if not decision.requires_approval:
            return True

        if self.config.auto_approve:
            self.logger.ask(f"Auto-approved action: {decision.action}")
            return True

        decision_text = f"Action: {decision.action}\nTool: {decision.tool}\nArgs: {decision.arguments}\nReason: {decision.reason}"
        result = self.approval.approve_plan(decision_text)

        obs = self.observation_mgr.observe_approval(
            granted=(result == ApprovalDecision.APPROVE),
            reason=f"Decision: {decision.action}",
        )

        if self._current_state:
            self._current_state.add_observation(obs)

        return result == ApprovalDecision.APPROVE

    async def _act(self, state: TaskState, decision: Decision) -> ActionResult:
        """Execute the action specified in a decision."""
        action = decision.action

        if action == "understand":
            return await self._act_understand(state, decision)

        elif action == "analyze":
            return await self._act_analyze(state, decision)

        elif action == "plan":
            return await self._act_plan(state, decision)

        elif action == "ask_approval":
            return await self._act_ask_approval(state, decision)

        elif action == "execute_tool":
            return await self._act_execute_tool(state, decision)

        elif action == "run_tests":
            return await self._act_run_tests(state, decision)

        elif action == "fix_error":
            return await self._act_fix_error(state, decision)

        elif action == "verify":
            return await self._act_verify(state, decision)

        elif action in ("report", "done"):
            return ActionResult(success=True, tool="report", output="Task completed")

        else:
            return ActionResult(success=False, tool="none", error=f"Unknown action: {action}")

    async def _act_understand(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: UNDERSTAND — parse the user's request."""
        self.logger.understand(f"Understanding request: {state.request[:100]}")
        state.goal = state.request
        state.status = "understood"
        return ActionResult(
            success=True,
            tool="understand",
            output=f"Goal established: {state.goal}",
        )

    async def _act_analyze(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: ANALYZE — inspect the workspace."""
        self.logger.analyze(f"Analyzing workspace: {state.workspace}")
        try:
            if self.analyzer:
                result = self.analyzer.analyze()
                state.project_context = result.to_dict()
                state.status = "analyzed"
                return ActionResult(
                    success=True,
                    tool="list_directory",
                    output=f"Analyzed project: {result.project_name}",
                    data=result.to_dict(),
                )
            return ActionResult(
                success=True,
                tool="list_directory",
                output="Analysis skipped (no analyzer)",
            )
        except Exception as e:
            return ActionResult(
                success=False,
                tool="list_directory",
                error=str(e),
            )

    async def _act_plan(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: PLAN — generate a structured plan."""
        self.logger.plan(f"Creating plan for: {state.goal[:80]}")

        try:
            plan = await self.planner.create_plan(state.goal, state.project_context)
            state.plan = plan.to_dict()
            state._plan = plan
            state.plan_steps = [s.to_dict() for s in plan.steps]
            state.remaining_steps = [s.to_dict() for s in plan.steps]
            state.completed_steps = set()
            state.status = "planned"

            obs = self.observation_mgr.observe_plan_created(plan.title, len(plan.steps))
            state.add_observation(obs)

            self.logger.plan(self.planner.format_plan(plan))

            return ActionResult(
                success=True,
                tool="planner",
                output=f"Plan created: '{plan.title}' with {len(plan.steps)} steps",
                data={"title": plan.title, "steps": len(plan.steps)},
            )
        except Exception as e:
            state.add_error(str(e))
            return ActionResult(
                success=False,
                tool="planner",
                error=str(e),
            )

    async def _act_ask_approval(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: ASK — present plan for approval."""
        self.logger.ask("Awaiting plan approval...")

        if self.config.auto_approve:
            self.logger.ask("Auto-approving plan")
            decision_obj = ApprovalDecision.APPROVE
        else:
            plan_text = self.planner.format_plan(state._plan) if state._plan else state.goal
            decision_obj = self.approval.approve_plan(plan_text)

        state.status = "executing"

        if decision_obj == ApprovalDecision.APPROVE:
            return ActionResult(
                success=True,
                tool="approval",
                output="Plan approved",
            )
        elif decision_obj == ApprovalDecision.REJECT or decision_obj == ApprovalDecision.CANCEL:
            state.mark_cancelled("Plan rejected by user")
            return ActionResult(
                success=False,
                tool="approval",
                error="Plan rejected by user",
            )
        else:
            return ActionResult(
                success=True,
                tool="approval",
                output="Plan requires modification",
            )

    async def _act_execute_tool(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: ACT — execute a tool based on the decision."""
        tool_name = decision.tool
        args = dict(decision.arguments)

        step_id = args.pop("step_id", None)
        args.pop("action_type", None)

        self.logger.code(f"Executing: {tool_name}({args})")

        try:
            result = await self.tools.execute(tool_name, **args)

            if result.success:
                state.increment_attempt()
                if step_id and step_id in [s.get("id") for s in state.remaining_steps]:
                    state.remaining_steps = [s for s in state.remaining_steps if s.get("id") != step_id]
                    state.completed_steps.add(step_id)
                    self.decision_engine.reset_retry_count()

            return ActionResult(
                success=result.success,
                tool=tool_name,
                arguments=args,
                output=result.output,
                error=result.error,
                observations=[o.to_dict() for o in result.data.get("observations", [])],
                return_code=result.data.get("returncode", 0),
            )
        except Exception as e:
            state.add_error(str(e))
            return ActionResult(
                success=False,
                tool=tool_name,
                arguments=args,
                error=str(e),
            )

    async def _act_run_tests(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: TEST — run the project test suite."""
        self.logger.test("Running tests...")

        test_cmd = "python -m pytest -v"
        result = await self.tools.execute("execute_command", command=test_cmd, timeout=300)

        test_result = TestResult(
            command=test_cmd,
            passed=result.success,
            output=result.output,
            error=result.error,
            return_code=result.data.get("returncode", 0),
        )
        state.add_test_result(test_result)

        if result.success:
            self.logger.success("Tests passed.")
        else:
            self.logger.error(f"Tests failed: {result.error[:200]}")
            state.add_error(f"Test failure: {result.error[:200]}")

        return ActionResult(
            success=result.success,
            tool="run_tests",
            arguments={"command": test_cmd},
            output=result.output,
            error=result.error,
            return_code=result.data.get("returncode", 0),
        )

    async def _act_fix_error(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: FIX — attempt to fix a failed action or test failure."""
        attempt = state.increment_attempt()
        self.logger.fix(f"Fix attempt {attempt}/{self.config.max_retries}")

        if attempt > self.config.max_retries:
            state.mark_failed(f"Max retries ({self.config.max_retries}) exceeded")
            return ActionResult(
                success=False,
                tool="fix_error",
                error="Max retries exceeded",
            )

        errors = state.errors[-5:] if state.errors else ["No specific error recorded"]
        error_summary = "\n".join(errors)

        fix_request = (
            f"The following errors occurred during task execution:\n{error_summary}\n\n"
            f"Goal: {state.goal}\n"
            f"Workspace: {state.workspace}\n\n"
            f"Suggest a fix. Output a plan with steps to resolve these issues."
        )

        try:
            fix_plan = await self.planner.create_plan(fix_request, state.project_context)

            state.plan_steps.extend([s.to_dict() for s in fix_plan.steps])
            state.remaining_steps = [s.to_dict() for s in state.remaining_steps] + [
                s.to_dict() for s in fix_plan.steps
            ]
            state.status = "fixing"

            return ActionResult(
                success=True,
                tool="planner",
                output=f"Fix plan created: '{fix_plan.title}' with {len(fix_plan.steps)} steps",
                data={"title": fix_plan.title, "steps": len(fix_plan.steps)},
            )
        except Exception as e:
            state.add_error(str(e))
            return ActionResult(
                success=False,
                tool="fix_error",
                error=str(e),
            )

    async def _act_verify(self, state: TaskState, decision: Decision) -> ActionResult:
        """Phase: VERIFY — final verification of task completion."""
        self.logger.success("Verifying task completion...")

        eval_result = self.evaluator.evaluate_completion(state)

        if eval_result.outcome == EvaluationOutcome.SUCCESS:
            return ActionResult(
                success=True,
                tool="verify",
                output="Task verified: all criteria met",
            )
        else:
            return ActionResult(
                success=False,
                tool="verify",
                error=f"Verification failed: {eval_result.reason}",
            )

    def _reason(
        self,
        state: TaskState,
        decision: Decision,
        eval_result: EvaluationResult,
        observations: list[Observation],
    ) -> None:
        """Phase: REASON — analyze the outcome and decide next steps."""
        self.logger.reason(
            f"Evaluated: {eval_result.outcome} ({eval_result.confidence:.0f}) — {eval_result.reason}"
        )

        if eval_result.outcome == EvaluationOutcome.FAILURE:
            if state.exceeded_retry_limit():
                state.mark_failed(f"Retry limit exceeded after {state.attempts} attempts")
                self.logger.error("Retry limit exceeded, marking task as failed")
            elif state.attempts > self.config.max_retries * 2:
                state.mark_failed("Too many total attempts, giving up")
                self.logger.error("Too many attempts, giving up")

    def _advance_state(
        self,
        state: TaskState,
        decision: Decision,
        eval_result: EvaluationResult,
    ) -> None:
        """Update state based on evaluation outcome."""
        state.updated_at = time.time()

        if eval_result.outcome == EvaluationOutcome.SUCCESS and not state.remaining_steps:
            if state.test_results and all(tr.passed for tr in state.test_results):
                state.mark_complete("All steps completed and tests passing")

    def _report(self, state: TaskState) -> str:
        """Phase: REPORT — generate final report."""
        elapsed = state.elapsed
        status = "COMPLETED" if state.is_complete else "FAILED" if state.is_failed else "CANCELLED"

        self.logger.info("=" * 60)
        if state.is_complete:
            self.logger.success(f"Task completed successfully in {elapsed:.1f}s")
        elif state.is_failed:
            self.logger.error(f"Task failed after {elapsed:.1f}s")
        elif state.is_cancelled:
            self.logger.warn(f"Task cancelled after {elapsed:.1f}s")

        report = (
            f"\n{'=' * 60}\n"
            f"  EVORA AUTONOMOUS TASK REPORT\n"
            f"{'=' * 60}\n\n"
            f"Status: {status}\n"
            f"Goal: {state.goal}\n"
            f"Steps completed: {len(state.completed_steps)}\n"
            f"Steps remaining: {len(state.remaining_steps)}\n"
            f"Total actions: {len(state.actions)}\n"
            f"Observations: {len(state.observations)}\n"
            f"Decisions made: {len(state.decisions)}\n"
            f"Errors: {len(state.errors)}\n"
            f"Test results: {sum(1 for t in state.test_results if t.passed)}/{len(state.test_results)} passed\n"
            f"Elapsed: {elapsed:.1f}s\n"
        )

        if state.errors:
            report += f"\nErrors:\n"
            for e in state.errors[-5:]:
                report += f"  - {e}\n"

        report += f"\n{'=' * 60}\n"

        self._save_memory(state)

        # Phase 3: Archive task outcome to long-term memory
        if self.memory_service is not None:
            try:
                self.memory_service.archive_task_outcome(state)
                self.memory_service.update_project_memory(state)
            except Exception as e:
                self.logger.memory(f"Memory archiving skipped: {e}")

        return report

    def _save_memory(self, state: TaskState) -> None:
        """Phase: MEMORY — save task and project memory."""
        try:
            from evora.memory import TaskEntry
            task_entry = self.memory.store.load_task(state.task_id)
            if task_entry is None:
                task_entry = TaskEntry.create(state.request, state.plan or {})
                task_entry.id = state.task_id

            if state.is_complete:
                task_entry.finish(state.final_result)
            elif state.is_failed:
                task_entry.fail(state.final_result)
            elif state.is_cancelled:
                task_entry.update_status("cancelled")

            task_entry.steps = state.actions
            task_entry.errors = state.errors
            task_entry.result = state.final_result
            task_entry.elapsed = state.elapsed

            self.memory.store.save_task(task_entry)
            self.memory.save_project()
        except Exception:
            pass

    async def stop(self) -> None:
        """Stop the agent gracefully."""
        self._is_stopped = True
        if self._current_state:
            self._current_state.mark_cancelled("Agent stopped by user")
