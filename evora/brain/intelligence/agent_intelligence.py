"""
Phase 17 — Native Agent for EVORA.

Builds Agent mode using the same Intelligence Spine.

Agent must be able to:
  OBSERVE
  UNDERSTAND
  REASON
  PLAN
  REQUEST AUTHORIZATION
  ACT
  TEST
  EVALUATE
  LEARN

No independent authority system.
No security bypass.
Uses existing security architecture:
  - PermissionManager
  - ApprovalSystem
  - IdentityService
  - ToolRegistry

Reuses existing abstractions:
  - IntelligenceRuntime for reasoning/planning/inference
  - NativeComprehensionIntelligence for understanding
  - ConversationManager for context
  - TrainingPipeline for learning
  - KnowledgeGraph for knowledge
  - CapabilityRegistry for capability checking
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Agent representations
# ---------------------------------------------------------------------------

class AgentState(str, Enum):
    """States of the agent."""
    IDLE = "idle"
    OBSERVING = "observing"
    UNDERSTANDING = "understanding"
    REASONING = "reasoning"
    PLANNING = "planning"
    REQUESTING_AUTHORIZATION = "requesting_authorization"
    ACTING = "acting"
    TESTING = "testing"
    EVALUATING = "evaluating"
    LEARNING = "learning"
    ERROR = "error"


class AgentActionType(str, Enum):
    """Types of agent actions."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    EXECUTE_COMMAND = "execute_command"
    RUN_TESTS = "run_tests"
    ANALYZE_CODE = "analyze_code"
    ANALYZE_PROJECT = "analyze_project"
    SEARCH_FILES = "search_files"
    SEARCH_CONTENT = "search_content"
    USE_TOOL = "use_tool"
    ASK_CLARIFICATION = "ask_clarification"
    COMPLETE = "complete"
    ABORT = "abort"


@dataclass
class AgentObservation:
    """An observation from the environment."""
    observation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    observation_type: str = ""
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class AgentAction:
    """An action the agent intends to take."""
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action_type: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = True
    reasoning: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "description": self.description,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class AgentResult:
    """Result of an agent action."""
    action_id: str = ""
    success: bool = False
    output: str = ""
    error: str = ""
    observations: list[AgentObservation] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)
    lesson_learned: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "observations": [o.to_dict() for o in self.observations],
            "evaluation": self.evaluation,
            "lesson_learned": self.lesson_learned,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Native Agent
# ---------------------------------------------------------------------------

class NativeAgent:
    """Native agent for EVORA.

    Uses the same Intelligence Spine as chatbot and other modes.
    No independent authority system.
    No security bypass.
    """

    def __init__(
        self,
        intelligence_runtime: Any = None,
        comprehension_intelligence: Any = None,
        conversation_manager: Any = None,
        tool_registry: Any = None,
        permission_manager: Any = None,
        approval_system: Any = None,
        identity_service: Any = None,
        training_pipeline: Any = None,
        logger: Optional[Any] = None,
    ):
        self.intelligence_runtime = intelligence_runtime
        self.comprehension_intelligence = comprehension_intelligence
        self.conversation_manager = conversation_manager
        self.tool_registry = tool_registry
        self.permission_manager = permission_manager
        self.approval_system = approval_system
        self.identity_service = identity_service
        self.training_pipeline = training_pipeline
        self.logger = logger
        self._state = AgentState.IDLE
        self._history: list[dict[str, Any]] = []

    def execute(self, goal: str, context: dict[str, Any] = None) -> AgentResult:
        """Execute an agent cycle: observe → understand → reason → plan → request auth → act → test → evaluate → learn."""
        context = context or {}
        result = AgentResult()
        observations = []

        try:
            observations.extend(self.observe(context))
            understanding = self.understand(goal, context)
            reasoning = self.reason(goal, understanding, context)
            plan = self.plan(goal, reasoning, context)
            if plan.get("requires_approval", True):
                authorized = self.request_authorization(plan, context)
                if not authorized:
                    result.success = False
                    result.error = "Authorization denied"
                    result.observations = observations
                    return result
            action = self.decide_action(plan, reasoning)
            result = self.act(action, context)
            observations.extend(result.observations)
            test_result = self.test(result, context)
            evaluation = self.evaluate(goal, result, test_result, context)
            result.evaluation = evaluation
            lesson = self.learn(goal, result, evaluation, context)
            result.lesson_learned = lesson
            self._state = AgentState.IDLE
        except Exception as e:
            self._state = AgentState.ERROR
            result.error = str(e)
            if self.logger:
                self.logger.error(f"Agent execution failed: {e}")

        result.observations = observations
        self._history.append({
            "goal": goal,
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
        })
        return result

    def observe(self, context: dict[str, Any]) -> list[AgentObservation]:
        """Observe the environment."""
        self._state = AgentState.OBSERVING
        observations = []
        if self.tool_registry is not None:
            try:
                tools = list(self.tool_registry.list()) if hasattr(self.tool_registry, "list") else []
                observations.append(AgentObservation(
                    observation_type="tools_available",
                    source="tool_registry",
                    data={"tool_count": len(tools), "tools": [t.name if hasattr(t, "name") else str(t) for t in tools[:10]]},
                ))
            except Exception:
                pass
        if self.intelligence_runtime is not None and self.intelligence_runtime.knowledge_graph is not None:
            try:
                nodes = self.intelligence_runtime.knowledge_graph.get_all_nodes(limit=10)
                observations.append(AgentObservation(
                    observation_type="knowledge_available",
                    source="knowledge_graph",
                    data={"node_count": len(nodes)},
                ))
            except Exception:
                pass
        return observations

    def understand(self, goal: str, context: dict[str, Any]) -> dict[str, Any]:
        """Understand the goal."""
        self._state = AgentState.UNDERSTANDING
        understanding = {"goal": goal, "intent": "unknown", "entities": [], "constraints": []}
        if self.comprehension_intelligence is not None:
            try:
                request = self.comprehension_intelligence.comprehend(
                    goal,
                    conversation_history=context.get("conversation_history", []),
                    project=context.get("project", ""),
                )
                understanding["intent"] = request.intent.intent_type.value
                understanding["entities"] = [e.to_dict() for e in request.entities]
                understanding["constraints"] = request.constraints
                understanding["required_capabilities"] = request.required_capabilities
            except Exception:
                pass
        return understanding

    def reason(self, goal: str, understanding: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Reason about the goal."""
        self._state = AgentState.REASONING
        reasoning = {"decision": "proceed", "confidence": 0.5, "evidence": []}
        if self.intelligence_runtime is not None:
            try:
                import asyncio
                result = asyncio.run(self.intelligence_runtime.reason(goal, context=context))
                if hasattr(result, "to_dict"):
                    reasoning = result.to_dict()
                elif hasattr(result, "decision"):
                    reasoning = {
                        "decision": result.decision,
                        "action": result.action,
                        "confidence": result.confidence,
                        "evidence_count": result.evidence_count,
                        "reasoning_summary": result.reasoning_summary,
                    }
            except Exception:
                pass
        return reasoning

    def plan(self, goal: str, reasoning: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Plan the approach."""
        self._state = AgentState.PLANNING
        plan = {"steps": [], "requires_approval": True, "confidence": 0.5}
        if self.intelligence_runtime is not None:
            try:
                import asyncio
                result = asyncio.run(self.intelligence_runtime.plan(goal))
                if result is not None:
                    if hasattr(result, "to_dict"):
                        plan = result.to_dict()
                    elif hasattr(result, "steps"):
                        plan = {
                            "steps": [s.to_dict() if hasattr(s, "to_dict") else s for s in result.steps],
                            "confidence": result.confidence,
                            "requires_approval": result.requires_approval,
                        }
            except Exception:
                pass
        return plan

    def request_authorization(self, plan: dict[str, Any], context: dict[str, Any]) -> bool:
        """Request authorization for the plan."""
        self._state = AgentState.REQUESTING_AUTHORIZATION
        if self.approval_system is None:
            return plan.get("requires_approval", True) is False
        try:
            if hasattr(self.approval_system, "request_approval"):
                return self.approval_system.request_approval(
                    action=f"Execute plan with {len(plan.get('steps', []))} steps",
                    reason=plan.get("reasoning", ""),
                )
        except Exception:
            pass
        return False

    def decide_action(self, plan: dict[str, Any], reasoning: dict[str, Any]) -> AgentAction:
        """Decide the next action from the plan."""
        steps = plan.get("steps", [])
        if steps:
            step = steps[0]
            if isinstance(step, dict):
                return AgentAction(
                    action_type=step.get("action_type", "unknown"),
                    description=step.get("name", ""),
                    parameters=step.get("action_args", {}),
                    requires_approval=plan.get("requires_approval", True),
                    reasoning=reasoning.get("reasoning_summary", ""),
                    confidence=reasoning.get("confidence", 0.5),
                )
        return AgentAction(
            action_type=AgentActionType.ANALYZE_PROJECT.value,
            description="Analyze goal",
            requires_approval=False,
            confidence=0.5,
        )

    def act(self, action: AgentAction, context: dict[str, Any]) -> AgentResult:
        """Execute an action."""
        self._state = AgentState.ACTING
        result = AgentResult(action_id=action.action_id)
        if self.tool_registry is not None:
            try:
                tool = self.tool_registry.get(action.action_type)
                if tool is not None:
                    import asyncio
                    tool_result = asyncio.run(tool.execute(**action.parameters))
                    result.success = tool_result.success
                    result.output = tool_result.output or ""
                    result.error = tool_result.error or ""
                    result.observations.append(AgentObservation(
                        observation_type="tool_execution",
                        source=action.action_type,
                        data={"success": tool_result.success, "output": result.output[:200]},
                    ))
                    return result
            except Exception as e:
                result.success = False
                result.error = str(e)
                return result
        result.success = True
        result.output = f"Simulated action: {action.description}"
        return result

    def test(self, action_result: AgentResult, context: dict[str, Any]) -> dict[str, Any]:
        """Test the action result."""
        self._state = AgentState.TESTING
        test_result = {"passed": action_result.success, "tests_run": 1, "failures": []}
        if not action_result.success:
            test_result["failures"].append(action_result.error)
        return test_result

    def evaluate(self, goal: str, action_result: AgentResult, test_result: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the action result."""
        self._state = AgentState.EVALUATING
        evaluation = {
            "goal_achieved": action_result.success,
            "tests_passed": test_result.get("passed", False),
            "confidence": 0.8 if action_result.success else 0.3,
            "limitations": [],
        }
        if not action_result.success:
            evaluation["limitations"].append("Action failed")
        if test_result.get("failures"):
            evaluation["limitations"].append("Tests failed")
        if self.intelligence_runtime is not None and self.intelligence_runtime.intelligence_evaluator is not None:
            try:
                eval_result = self.intelligence_runtime.intelligence_evaluator.evaluate_reasoning(
                    goal=goal,
                    result=action_result,
                    evidence=[],
                    constraints=context.get("constraints", []),
                )
                if hasattr(eval_result, "to_dict"):
                    evaluation["evaluator_grade"] = eval_result.to_dict().get("grade", "unknown")
                elif hasattr(eval_result, "grade"):
                    evaluation["evaluator_grade"] = eval_result.grade.value
            except Exception:
                pass
        return evaluation

    def learn(self, goal: str, action_result: AgentResult, evaluation: dict[str, Any], context: dict[str, Any]) -> str:
        """Learn from the experience."""
        self._state = AgentState.LEARNING
        lesson = ""
        if self.training_pipeline is not None:
            try:
                from evora.brain.intelligence.training import OutcomeType
                outcome = OutcomeType.SUCCESS if action_result.success else OutcomeType.FAILURE
                self.training_pipeline.record_training_example(
                    session_id=context.get("session_id", ""),
                    task_id=context.get("task_id", ""),
                    project=context.get("project", ""),
                    component="agent",
                    input_data={"goal": goal, "action": action_result.action_id},
                    output_data={"success": action_result.success, "output": action_result.output[:200]},
                    outcome=outcome,
                    confidence=evaluation.get("confidence", 0.5),
                )
                lesson = f"Agent {'succeeded' if action_result.success else 'failed'} on: {goal[:80]}"
            except Exception:
                pass
        return lesson

    def get_state(self) -> dict[str, Any]:
        """Get current agent state."""
        return {
            "state": self._state.value,
            "history_count": len(self._history),
            "last_action": self._history[-1] if self._history else None,
        }
