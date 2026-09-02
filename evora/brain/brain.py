"""
Phase 9 — Brain controller for EVORA.

The Brain is the central coordination layer that orchestrates existing
capabilities without replacing them. It is intentionally thin.

Security boundary: the Brain can SUGGEST actions, but authorization
remains with the existing PermissionManager, ApprovalSystem,
IdentityService, and workspace confinement. Model-generated decisions
are NEVER equivalent to creator authorization.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger
from evora.brain.state import BrainState, DevelopmentState
from evora.brain.context import ContextBuilder, BrainContext
from evora.brain.self_model import SelfModel
from evora.brain.resources import ResourceMonitor


@dataclass
class BrainResponse:
    """Structured response from the Brain."""
    summary: str = ""
    decision: str = ""
    reasoning: str = ""
    suggested_tools: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "decision": self.decision,
            "reasoning": self.reasoning,
            "suggested_tools": self.suggested_tools,
            "observations": self.observations,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class BrainController:
    """EVORA Brain — provider-agnostic orchestration layer.

    The Brain coordinates existing capabilities without replacing them.
    It delegates to:
      - ContextBuilder for context construction
      - existing ReasoningEngine for reasoning
      - existing Planner for planning
      - existing DecisionEngine for decision generation
      - existing ToolRegistry for tool discovery
      - BrainState for persistent internal state
      - SelfModel for self-understanding
      - ResourceMonitor for resource awareness
    """

    def __init__(
        self,
        brain_state: Optional[BrainState] = None,
        self_model: Optional[SelfModel] = None,
        resource_monitor: Optional[ResourceMonitor] = None,
        context_builder: Optional[ContextBuilder] = None,
        model_manager: Any = None,
        reasoning_engine: Any = None,
        planner: Any = None,
        decision_engine: Any = None,
        tool_registry: Any = None,
        memory_service: Any = None,
        knowledge_base: Any = None,
        experience_store: Any = None,
        approval_system: Any = None,
        identity_service: Any = None,
        intelligence_runtime: Any = None,
        logger: Optional[Logger] = None,
    ):
        self.brain_state = brain_state or BrainState()
        self.self_model = self_model or SelfModel()
        self.resource_monitor = resource_monitor or ResourceMonitor(logger=logger)
        self.context_builder = context_builder or ContextBuilder(
            memory_service=memory_service,
            knowledge_base=knowledge_base,
            experience_store=experience_store,
            self_model=self.self_model,
            brain_state=self.brain_state,
            tool_registry=tool_registry,
            logger=logger,
        )
        self.model_manager = model_manager
        self.reasoning_engine = reasoning_engine
        self.planner = planner
        self.decision_engine = decision_engine
        self.tool_registry = tool_registry
        self.memory_service = memory_service
        self.knowledge_base = knowledge_base
        self.experience_store = experience_store
        self.approval_system = approval_system
        self.identity_service = identity_service
        self.intelligence_runtime = intelligence_runtime
        self.logger = logger

    def get_state(self) -> BrainState:
        """Return current Brain state."""
        return self.brain_state

    def update_state(self, **kwargs: Any) -> None:
        """Update Brain state fields."""
        for key, value in kwargs.items():
            if hasattr(self.brain_state, key):
                setattr(self.brain_state, key, value)
        if self.logger:
            self.logger.observe(f"Brain state updated: {self.brain_state.snapshot()}")

    def build_context(self, goal: str, project: str = "") -> BrainContext:
        """Build bounded context for inference."""
        return self.context_builder.build(goal=goal, project=project)

    async def reason(self, goal: str, context: Optional[BrainContext] = None) -> BrainResponse:
        """Reason about a goal using available context.

        Prefers native intelligence runtime when available.
        Falls back to external model provider.
        Returns a BrainResponse with confidence and summary.
        """
        if context is None:
            context = self.build_context(goal=goal)

        summary = ""
        reasoning = ""
        confidence = 0.0
        suggested_tools: list[dict[str, Any]] = []

        # Try native intelligence first
        if self.intelligence_runtime is not None:
            try:
                native_result = await self.intelligence_runtime.reason(goal, {})
                if native_result and getattr(native_result, "confidence", 0.0) > 0.3:
                    summary = getattr(native_result, "reasoning_summary", "")
                    reasoning = getattr(native_result, "decision", "")
                    confidence = getattr(native_result, "confidence", 0.0)
                    if self.logger:
                        self.logger.observe(f"Native reasoning: confidence={confidence:.2f}")
            except Exception as e:
                if self.logger:
                    self.logger.warn(f"Native reasoning failed: {e}")

        # Fallback to external model provider
        if not summary and self.reasoning_engine is not None and self.model_manager is not None and self.model_manager.active:
            try:
                from evora.reasoning import ReasoningContext
                ctx = ReasoningContext(
                    objective=goal,
                    observations=[context.to_prompt_context()],
                    constraints=list(self.brain_state.known_constraints),
                )
                result = await self.reasoning_engine.reason(ctx)
                summary = result.summary
                reasoning = result.selected_approach
                confidence = result.confidence
            except Exception as e:
                if self.logger:
                    self.logger.warn(f"Brain reasoning failed: {e}")
                summary = f"Reasoning unavailable: {e}"
                confidence = 0.0

        if not summary:
            summary = f"No reasoning available for: {goal}"
            confidence = 0.0

        if self.tool_registry is not None:
            try:
                available = list(self.tool_registry.list())
                suggested_tools = [
                    {"name": name, "description": tool.description}
                    for name, tool in available[:5]
                ]
            except Exception:
                pass

        self.brain_state.add_observation({
            "type": "reasoning",
            "goal": goal,
            "confidence": confidence,
            "timestamp": time.time(),
        })

        return BrainResponse(
            summary=summary,
            reasoning=reasoning,
            confidence=confidence,
            suggested_tools=suggested_tools,
            observations=self.brain_state.recent_observations[-5:],
        )

    async def plan(self, goal: str, project: str = "") -> Optional[dict[str, Any]]:
        """Create a plan for a goal.

        Prefers native intelligence runtime when available.
        Falls back to external model provider.
        """
        # Try native intelligence first
        if self.intelligence_runtime is not None:
            try:
                native_plan = await self.intelligence_runtime.plan(goal, [])
                if native_plan is not None and getattr(native_plan, "confidence", 0.0) > 0.3:
                    self.brain_state.set_plan(native_plan.to_dict())
                    self.brain_state.set_development_state(DevelopmentState.PLANNING)
                    return native_plan.to_dict()
            except Exception as e:
                if self.logger:
                    self.logger.warn(f"Native planning failed: {e}")

        # Fallback to external model provider
        if self.planner is None or self.model_manager is None or not self.model_manager.active:
            if self.logger:
                self.logger.warn("Cannot plan: no planner or no active model provider")
            return None

        try:
            plan = await self.planner.plan(goal)
            self.brain_state.set_plan(plan.to_dict())
            self.brain_state.set_development_state(DevelopmentState.PLANNING)
            return plan.to_dict()
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Brain planning failed: {e}")
            return None

    def decide(self, goal: str, context: Optional[BrainContext] = None) -> dict[str, Any]:
        """Decide next action based on current state using existing DecisionEngine."""
        if self.decision_engine is None:
            return {"action": "none", "reason": "no_decision_engine", "confidence": 0.0}

        try:
            from evora.task import TaskState, Decision
            state = TaskState(
                request=goal,
                goal=goal,
                status=self.brain_state.development_state.value,
            )
            decision = self.decision_engine.decide_next(state)
            self.brain_state.add_observation({
                "type": "decision",
                "action": decision.action,
                "reason": decision.reason,
                "confidence": decision.confidence,
                "timestamp": time.time(),
            })
            return decision.to_dict()
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Brain decision failed: {e}")
            return {"action": "error", "reason": str(e), "confidence": 0.0}

    def suggest_tools(self, goal: str, limit: int = 5) -> list[dict[str, Any]]:
        """Suggest tools relevant to a goal.

        IMPORTANT: Suggestions are NOT authorizations.
        Tool execution still requires PermissionManager + ApprovalSystem
        + IdentityService enforcement.
        """
        if self.tool_registry is None:
            return []

        try:
            available = list(self.tool_registry.list())
            goal_lower = goal.lower()
            keywords = goal_lower.split()
            scored: list[tuple[float, str, Any]] = []
            for name in available:
                tool = self.tool_registry.get(name)
                if tool is None:
                    continue
                desc = (tool.description or "").lower()
                score = sum(1 for kw in keywords if kw in desc)
                scored.append((score, name, tool))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"name": name, "description": tool.description, "score": score}
                for score, name, tool in scored[:limit]
                if score > 0
            ]
        except Exception:
            return []

    async def capture_experience(self, experience: Any) -> str:
        """Capture an experience via the experience store if available."""
        if self.experience_store is None or self.identity_service is None:
            return ""
        try:
            self.identity_service.require_authority("remember")
            sanitized = copy.deepcopy(experience)
            if hasattr(sanitized, "content") and isinstance(sanitized.content, str):
                from evora.memory import MemoryFilter
                sanitized.content = MemoryFilter.sanitize(sanitized.content)
            experience_id = self.experience_store.record(sanitized)
            self.brain_state.add_observation({
                "type": "experience_captured",
                "experience_id": experience_id,
                "timestamp": time.time(),
            })
            return experience_id
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Brain experience capture failed: {e}")
            return ""

    def get_self_description(self) -> str:
        """Return a concise self-description."""
        return self.self_model.describe()

    def get_resource_snapshot(self) -> dict[str, Any]:
        """Return current resource snapshot."""
        return self.resource_monitor.collect(
            model_manager=self.model_manager,
            tool_registry=self.tool_registry,
        ).to_dict()

    def serialize_state(self) -> str:
        """Serialize Brain state to JSON."""
        return json.dumps(self.brain_state.to_dict(), indent=2, ensure_ascii=False)

    def load_state(self, data: str) -> None:
        """Load Brain state from JSON."""
        try:
            parsed = json.loads(data)
            self.brain_state = BrainState.from_dict(parsed)
            if self.logger:
                self.logger.observe(f"Brain state loaded: {self.brain_state.state_id}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load Brain state: {e}")
            raise
