"""
Agent loop for EVORA.

The agent is the central orchestrator that:
1. Takes a user request
2. Delegates to the Planner to create a structured plan
3. Uses the ApprovalSystem to get user approval
4. Executes plan steps using the ToolRegistry
5. Runs tests and handles errors
6. Updates Memory with results

The agent loop handles:
    - Step execution with proper error handling
    - Retry logic with limits (no infinite loops)
    - Tool call orchestration
    - State tracking throughout execution
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from evora.logger import Logger, Stage
from evora.model import ModelManager, ChatRequest, Message, Role, ToolCall, ToolResult, ModelResponse
from evora.planner import Planner, Plan, PlanStep
from evora.approval import ApprovalSystem, ApprovalDecision
from evora.tools import ToolRegistry, ToolResult as ToolExecResult
from evora.memory import Memory, TaskEntry
from evora.security import PermissionManager, PermissionLevel
from evora.analyzer import ProjectAnalyzer, AnalysisResult


class AgentStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    TESTING = "testing"
    FIXING = "fixing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentConfig:
    max_retries: int = 3
    retry_delay: float = 2.0
    command_timeout: int = 60
    auto_approve: bool = False


class Agent:
    """EVORA's core agent that orchestrates the full workflow."""

    def __init__(
        self,
        model_manager: ModelManager,
        plan: Planner,
        approval: ApprovalSystem,
        tools: ToolRegistry,
        memory: Memory,
        security: PermissionManager,
        logger: Logger,
        config: Optional[AgentConfig] = None,
        analyzer: Optional[ProjectAnalyzer] = None,
    ):
        self.model_manager = model_manager
        self.plan = plan
        self.approval = approval
        self.tools = tools
        self.memory = memory
        self.security = security
        self.logger = logger
        self.config = config or AgentConfig()
        self.analyzer = analyzer
        self.status = AgentStatus.IDLE
        self._current_task: Optional[TaskEntry] = None
        self._completed_steps: set[str] = set()
        self._failed_steps: dict[str, str] = {}

    async def run(self, request: str, project_context: Optional[dict] = None) -> str:
        """Execute the full EVORA workflow: PLAN → ASK → CODE → TEST → FIX → REPORT → MEMORY."""
        start_time = time.time()
        self._current_task = self.memory.create_task(request, {})
        self.memory.save_task(self._current_task)

        try:
            plan = await self._plan(request, project_context)
            self._current_task.plan = plan.to_dict()
            self.memory.save_task(self._current_task)

            approved, plan = await self._ask(plan)
            if approved:
                success = await self._code(plan)
                if success:
                    tests_passed = await self._test()
                    if not tests_passed:
                        fixed = await self._fix(plan)
                    else:
                        fixed = True
                else:
                    fixed = False
            else:
                fixed = False
                self.status = AgentStatus.CANCELLED
                self._current_task.fail("Plan was not approved")
                self.memory.save_task(self._current_task)
                return "Task cancelled by user"

            report = self._report(success=success and fixed, plan=plan, elapsed=time.time() - start_time)

            self._current_task.elapsed = time.time() - start_time
            self.memory.save_task(self._current_task)
            self._save_memory(plan, success and fixed)

            return report

        except Exception as e:
            self.logger.error(f"Agent execution failed: {e}")
            self.status = AgentStatus.FAILED
            self._current_task.fail(str(e))
            self.memory.save_task(self._current_task)
            return f"Task failed: {e}"

    async def _plan(self, request: str, project_context: Optional[dict]) -> Plan:
        """Step 1: PLAN - Create a structured plan from the user request."""
        self.status = AgentStatus.PLANNING
        self.logger.plan("Creating implementation plan...")

        plan = await self.plan.create_plan(request, project_context)

        self.logger.plan(self.plan.format_plan(plan))

        if project_context and self.analyzer:
            for note in self._extract_conventions(project_context):
                self.memory.project.add_note(note)

        self._current_task.plan = plan.to_dict()
        return plan

    async def _ask(self, plan: Plan):
        """Step 2: ASK - Present the plan and wait for approval."""
        self.status = AgentStatus.AWAITING_APPROVAL
        self.logger.ask("Awaiting approval...")

        decision = self.approval.approve_plan(
            self.plan.format_plan(plan),
            plan_obj=plan.to_dict(),
        )

        if decision == ApprovalDecision.APPROVE:
            return True, plan
        elif decision == ApprovalDecision.MODIFY:
            modification = self.approval.get_modification("How should I modify the plan?")
            self.logger.info(f"User modification: {modification}")
            new_plan = await self.plan.create_plan(
                f"{plan.title}. User modification: {modification}",
            )
            return await self._ask(new_plan)
        elif decision == ApprovalDecision.EXPLAIN:
            self.approval.explain_plan(self.plan.format_plan(plan))
            return await self._ask(plan)
        elif decision == ApprovalDecision.REJECT:
            self.logger.warn("Plan rejected by user")
            return False, plan
        else:
            return False, plan

    async def _code(self, plan: Plan) -> bool:
        """Step 3: CODE - Execute plan steps using tools."""
        self.status = AgentStatus.EXECUTING
        self.logger.code("Executing plan...")

        for step in plan.steps:
            self.status = AgentStatus.EXECUTING
            self._current_task.steps.append({
                "step": step.name,
                "action_type": step.action_type,
                "status": "running",
            })

            success = await self._execute_step(step)
            self._completed_steps.add(step.id)

            idx = len(self._current_task.steps) - 1
            if idx >= 0:
                self._current_task.steps[idx]["status"] = "completed" if success else "failed"

            if not success:
                error_msg = self._current_task.steps[-1].get("error", "Unknown error")
                self._failed_steps[step.id] = error_msg
                self._current_task.fail(f"Step failed: {step.name} - {error_msg}")
                self.memory.save_task(self._current_task)
                return False

            self.memory.save_task(self._current_task)

        self.logger.code("All plan steps completed.")
        return True

    async def _execute_step(self, step: PlanStep) -> bool:
        """Execute a single plan step."""
        self.logger.code(f"Step '{step.name}': {step.action_type}")

        for attempt in range(self.config.max_retries):
            try:
                if step.action_type == "create_file":
                    return await self._tool_create_file(step)
                elif step.action_type == "edit_file":
                    return await self._tool_edit_file(step)
                elif step.action_type == "read_file":
                    return await self._tool_read_file(step)
                elif step.action_type == "run_command":
                    return await self._tool_run_command(step)
                elif step.action_type == "run_tests":
                    return await self._tool_run_tests(step)
                elif step.action_type == "analyze":
                    return await self._tool_analyze(step)
                elif step.action_type == "create_directory":
                    return await self._tool_create_dir(step)
                else:
                    self.logger.error(f"Unknown action type: {step.action_type}")
                    return False
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    self.logger.warn(f"Step failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    self.logger.error(f"Step failed after {self.config.max_retries} attempts: {e}")
                    return False

        return False

    async def _tool_create_file(self, step: PlanStep) -> bool:
        path = step.action_args.get("path", "")
        content = step.action_args.get("content", "")

        result = await self.tools.execute("write_file", path=path, content=content)
        if not result.success:
            self.logger.error(f"Failed to create file: {result.error}")
            return False
        return True

    async def _tool_edit_file(self, step: PlanStep) -> bool:
        path = step.action_args.get("path", "")
        old_string = step.action_args.get("old_string", "")
        new_string = step.action_args.get("new_string", "")
        replace_all = step.action_args.get("replace_all", False)

        result = await self.tools.execute("edit_file", path=path, old_string=old_string, new_string=new_string, replace_all=replace_all)
        if not result.success:
            self.logger.error(f"Failed to edit file: {result.error}")
            return False
        return True

    async def _tool_read_file(self, step: PlanStep) -> bool:
        path = step.action_args.get("path", "")
        result = await self.tools.execute("read_file", path=path)
        return result.success

    async def _tool_run_command(self, step: PlanStep) -> bool:
        command = step.action_args.get("command", "")
        result = await self.tools.execute("execute_command", command=command, timeout=self.config.command_timeout)
        if not result.success:
            self.logger.error(f"Command failed: {result.error}")
            return False
        return True

    async def _tool_run_tests(self, step: PlanStep) -> bool:
        command = step.action_args.get("command", "")
        result = await self.tools.execute("execute_command", command=command, timeout=300)
        return result.success

    async def _tool_analyze(self, step: PlanStep) -> bool:
        path = step.action_args.get("path", ".")
        result = await self.tools.execute("list_directory", path=path)
        return result.success

    async def _tool_create_dir(self, step: PlanStep) -> bool:
        path = step.action_args.get("path", "")
        result = await self.tools.execute("create_directory", path=path)
        return result.success

    async def _test(self) -> bool:
        """Step 4: TEST - Run tests."""
        self.status = AgentStatus.TESTING
        self.logger.test("Running tests...")

        test_cmd = "python -m pytest -v"
        result = await self.tools.execute("execute_command", command=test_cmd, timeout=120)

        if result.success:
            self.logger.success("All tests passed.")
            return True
        else:
            self.logger.error(f"Tests failed: {result.error}")
            return False

    async def _fix(self, plan: Plan) -> bool:
        """Step 5: FIX - Attempt to fix test failures."""
        self.status = AgentStatus.FIXING
        self.logger.fix("Analyzing test failures and attempting fixes...")

        if self._current_task and self._current_task.errors:
            for attempt in range(self.config.max_retries):
                self.logger.fix(f"Fix attempt {attempt + 1}/{self.config.max_retries}")

                fix_request = (
                    f"The plan '{plan.title}' has the following errors:\n"
                    f"{self._format_errors()}\n\n"
                    f"Suggest a fix. Output a plan with steps to fix the issues."
                )

                fix_plan = await self.plan.create_plan(fix_request)
                approved, fix_plan = await self._ask(fix_plan)

                if not approved:
                    self.logger.error("Fix plan was not approved")
                    return False

                success = await self._code(fix_plan)
                if success:
                    tests_passed = await self._test()
                    if tests_passed:
                        self.logger.success("Fix successful - all tests pass.")
                        return True
                self.logger.fix("Fix attempt did not resolve the issue.")

            self.logger.error(f"Could not fix after {self.config.max_retries} attempts.")
            return False

        self.logger.error("No error details available to fix.")
        return False

    def _format_errors(self) -> str:
        if not self._current_task:
            return "Unknown errors"
        return "\n".join(f"- {e}" for e in self._current_task.errors[-10:])

    def _report(self, success: bool, plan: Plan, elapsed: float) -> str:
        """Step 6: REPORT - Generate a final report."""
        self.logger.info("=" * 60)
        if success:
            self.logger.success(f"Task completed successfully in {elapsed:.1f}s")
        else:
            self.logger.error(f"Task failed after {elapsed:.1f}s")

        report = (
            f"\n{'=' * 60}\n"
            f"  EVORA TASK REPORT\n"
            f"{'=' * 60}\n\n"
            f"Status: {'COMPLETED' if success else 'FAILED'}\n"
            f"Plan: {plan.title}\n"
            f"Steps executed: {len(self._current_task.steps) if self._current_task else 0}\n"
            f"Errors: {len(self._current_task.errors) if self._current_task else 0}\n"
            f"Elapsed: {elapsed:.1f}s\n"
        )

        if self._current_task and self._current_task.steps:
            report += "\nStep results:\n"
            for s in self._current_task.steps:
                status = "✅" if s.get("status") == "completed" else "❌" if s.get("status") == "failed" else "⏳"
                report += f"  {status} {s.get('step', 'unknown')}\n"

        if self._current_task and self._current_task.errors:
            report += f"\nErrors:\n"
            for e in self._current_task.errors[-5:]:
                report += f"  - {e}\n"

        report += f"\n{'=' * 60}\n"
        return report

    def _save_memory(self, plan: Plan, success: bool):
        """Step 7: MEMORY - Save task and project memory."""
        if not self._current_task:
            return

        if success:
            self._current_task.add_memory(f"Successfully completed plan: {plan.title}")
            self.memory.project.add_learned(f"Completed: {plan.title}")
        else:
            self._current_task.add_memory(f"Failed to complete plan: {plan.title}")
            self.memory.project.add_note(f"Failed plan: {plan.title} - needs investigation")

        self.memory.save_task(self._current_task)
        self.memory.save_project()

    def _extract_conventions(self, project_context: dict) -> list[str]:
        """Extract project conventions from analysis results."""
        notes = []
        if project_context.get("languages"):
            langs = ", ".join(project_context["languages"].keys())
            notes.append(f"Project languages: {langs}")
        if project_context.get("frameworks"):
            notes.append(f"Frameworks: {', '.join(project_context['frameworks'])}")
        if project_context.get("build_system"):
            notes.append(f"Build system: {project_context['build_system']}")
        if project_context.get("test_command"):
            notes.append(f"Test command: {project_context['test_command']}")
        return notes

    async def stop(self):
        """Stop the agent gracefully."""
        self.status = AgentStatus.IDLE
        if self._current_task:
            self._current_task.update_status("cancelled")
            self.memory.save_task(self._current_task)
