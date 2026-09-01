"""
Planner module for EVORA.

Takes a user request and AI model analysis to produce a structured plan
with sequential steps. Each step has a name, description, and estimated
action type (e.g., create_file, edit_file, run_command).

The planner uses the AI model to decompose complex requests into
actionable tasks.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger
from evora.model import ModelManager, ChatRequest, Message, Role


@dataclass
class PlanStep:
    """A single step in an EVORA plan."""
    id: str
    name: str
    description: str
    action_type: str  # create_file, edit_file, run_command, read_file, run_tests, analyze
    action_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    estimated_effort: str = "low"  # low, medium, high

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "depends_on": self.depends_on,
            "estimated_effort": self.estimated_effort,
        }


@dataclass
class Plan:
    """A structured plan produced from a user request."""
    title: str
    description: str
    steps: list[PlanStep] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
        }

    def total_steps(self) -> int:
        return len(self.steps)


class Planner:
    """Generates structured plans from user requests using an AI model."""

    def __init__(self, model_manager: ModelManager, logger: Optional[Logger] = None):
        self.model_manager = model_manager
        self.logger = logger

    PLAN_SYSTEM_PROMPT = """
You are EVORA, a structured AI planning agent. Your job is to break down
user requests into clear, actionable implementation steps.

Given a user request, produce a JSON plan with the following structure:

{
  "title": "Brief plan title",
  "description": "One-paragraph summary of what will be accomplished",
  "steps": [
    {
      "id": "step-1",
      "name": "Descriptive step name",
      "description": "What this step does",
      "action_type": "create_file" | "edit_file" | "run_command" | "read_file" | "run_tests" | "analyze",
      "action_args": {"path": "...", "content": "...", "command": "..."},
      "depends_on": ["step-id-of-prerequisite"],
      "estimated_effort": "low" | "medium" | "high"
    }
  ]
}

Rules:
- Each step should be atomic and independently executable.
- Action args should match the action_type:
  - create_file: {"path": "path/to/file", "content": "file content"}
  - edit_file: {"path": "path/to/file", "old_string": "...", "new_string": "..."}
  - run_command: {"command": "shell command"}
  - read_file: {"path": "path/to/file"}
  - run_tests: {"command": "test command"}
  - analyze: {"path": "directory to analyze"}
- Do NOT use markdown code blocks. Output raw JSON only.
- Do NOT include explanations outside the JSON.
"""

    async def create_plan(self, request: str, project_context: Optional[dict] = None) -> Plan:
        """Create a structured plan from a user request."""

        if self.logger:
            self.logger.plan(f"Planning for request: {request[:100]}...")

        context_hint = ""
        if project_context:
            langs = project_context.get("languages", {})
            langs_str = ", ".join(f"{k} ({v:.0f}%)" for k, v in langs.items()) if langs else "unknown"
            context_hint = f"\nProject context:\n- Languages: {langs_str}\n- Frameworks: {', '.join(project_context.get('frameworks', []))}\n- Test command: {project_context.get('test_command', 'not detected')}\n"

        prompt = self.PLAN_SYSTEM_PROMPT + f"\n\nUser request:\n{request}\n{context_hint}"

        messages = [
            Message(role=Role.SYSTEM, content=self.PLAN_SYSTEM_PROMPT),
            Message(role=Role.USER, content=request + context_hint),
        ]

        request_obj = ChatRequest(messages=messages, max_tokens=4096, temperature=0.7)

        try:
            response = await self.model_manager.chat(request_obj)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Plan generation failed: {e}")
            return Plan(
                title="Error: Plan generation failed",
                description=f"Could not generate a plan: {e}",
                steps=[PlanStep(
                    id="error-1",
                    name="Retry",
                    description="Check model configuration and try again.",
                    action_type="analyze",
                    action_args={"path": "."},
                )],
                raw_output="",
            )

        plan = self._parse_plan(response.content)
        plan.raw_output = response.content

        if self.logger:
            self.logger.plan(f"Plan created: '{plan.title}' with {len(plan.steps)} steps")

        return plan

    def _parse_plan(self, raw_output: str) -> Plan:
        """Parse JSON plan output from the model."""
        import json

        cleaned = raw_output.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```")

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start >= 0 and end > start:
                    data = json.loads(cleaned[start:end+1])
                else:
                    raise ValueError("no JSON found")
            except (json.JSONDecodeError, ValueError):
                return Plan(
                    title="Fallback Plan",
                    description="Could not parse AI-plan. Using fallback.",
                    steps=[PlanStep(
                        id="step-1",
                        name="Read project",
                        description="Analyze the project to understand current state.",
                        action_type="analyze",
                        action_args={"path": "."},
                    )],
                    raw_output=cleaned,
                )

        steps = []
        for i, s in enumerate(data.get("steps", [])):
            step = PlanStep(
                id=s.get("id", f"step-{i+1}"),
                name=s.get("name", f"Step {i+1}"),
                description=s.get("description", ""),
                action_type=s.get("action_type", "create_file"),
                action_args=s.get("action_args", {}),
                depends_on=s.get("depends_on", []),
                estimated_effort=s.get("estimated_effort", "medium"),
            )
            steps.append(step)

        return Plan(
            title=data.get("title", "Generated Plan"),
            description=data.get("description", ""),
            steps=steps,
            raw_output=cleaned,
        )

    def format_plan(self, plan: Plan, include_details: bool = True) -> str:
        """Format a plan as human-readable text for display."""
        lines = [f"\n{'=' * 60}"]
        lines.append(f"  PLAN: {plan.title}")
        lines.append(f"{'=' * 60}")
        lines.append(f"\n  {plan.description}\n")

        if include_details and plan.steps:
            lines.append(f"  Steps ({len(plan.steps)} total):\n")
            for i, step in enumerate(plan.steps, 1):
                deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
                lines.append(f"  {i}. [{step.estimated_effort.upper()}] {step.name}{deps}")
                lines.append(f"     Action: {step.action_type}")
                lines.append(f"     {step.description}")
        else:
            lines.append(f"  Total steps: {len(plan.steps)}")

        lines.append(f"\n{'=' * 60}\n")
        return "\n".join(lines)
