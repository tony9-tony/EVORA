"""
Phase 10 — NativeIntelligenceProvider for EVORA.

Implements ModelProvider interface so native intelligence can be
registered with ModelManager.

CRITICAL:
- Does NOT call ModelManager for inference
- Does NOT call external APIs
- Does NOT call itself recursively
- Uses IntelligenceRuntime which has NO ModelManager dependency
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Optional

from evora.logger import Logger
from evora.model import ModelProvider, ChatRequest, Message, ModelResponse, Role


class NativeIntelligenceProvider(ModelProvider):
    """EVORA's native intelligence provider.

    Implements ModelProvider so it can be registered with ModelManager.
    Uses IntelligenceRuntime internally.
    Does NOT call any external API.
    Does NOT call ModelManager for inference.
    """

    def __init__(self, runtime: Any, logger: Optional[Logger] = None):
        self._runtime = runtime
        self._logger = logger

    def name(self) -> str:
        return "native"

    def model(self) -> str:
        return "evora-native-intelligence"

    async def chat(self, request: ChatRequest) -> ModelResponse:
        """Process request using native intelligence.

        Never calls ModelManager.
        Never calls external APIs.
        """
        # Extract goal from request
        goal = self._extract_goal(request)

        # Determine capability needed
        capability = self._runtime.can_handle(self._classify_goal(goal))

        # Check if native can handle
        if capability.capability_type.value == "unavailable":
            return ModelResponse(
                content="",
                provider=self.name(),
                model=self.model(),
                raw={
                    "error": f"Capability unavailable: {goal[:80]}",
                    "type": "unavailable",
                    "capability": capability.name,
                },
            )

        # Route to appropriate native capability
        try:
            # Try reasoning first
            result = await self._runtime.reason(goal, {})
            if result and getattr(result, "confidence", 0.0) > 0.3:
                return ModelResponse(
                    content=getattr(result, "reasoning_summary", f"Reasoned about: {goal[:80]}"),
                    provider=self.name(),
                    model=self.model(),
                    raw={
                        "type": "native_result",
                        "confidence": getattr(result, "confidence", 0.0),
                        "capability": "reasoning",
                        "native": True,
                    },
                )

            # Try planning
            plan = await self._runtime.plan(goal, [])
            if plan and getattr(plan, "confidence", 0.0) > 0.3:
                return ModelResponse(
                    content=f"Plan created for: {goal[:80]} ({len(plan.steps)} steps)",
                    provider=self.name(),
                    model=self.model(),
                    raw={
                        "type": "native_result",
                        "confidence": getattr(plan, "confidence", 0.0),
                        "capability": "planning",
                        "native": True,
                    },
                )

            # Try inference
            inference = await self._runtime.infer(goal, {})
            if inference and getattr(inference, "confidence", 0.0) > 0.3:
                return ModelResponse(
                    content=getattr(inference, "answer", f"Inferred for: {goal[:80]}"),
                    provider=self.name(),
                    model=self.model(),
                    raw={
                        "type": "native_result",
                        "confidence": getattr(inference, "confidence", 0.0),
                        "capability": "inference",
                        "native": True,
                    },
                )

            # No native capability sufficient
            return ModelResponse(
                content="",
                provider=self.name(),
                model=self.model(),
                raw={
                    "error": "Native intelligence could not produce a confident result",
                    "type": "insufficient_confidence",
                    "capability": capability.name,
                    "native": True,
                },
            )

        except Exception as e:
            if self._logger:
                self._logger.warn(f"Native intelligence error: {e}")
            return ModelResponse(
                content="",
                provider=self.name(),
                model=self.model(),
                raw={
                    "error": f"Native intelligence error: {e}",
                    "type": "error",
                },
            )

    async def chat_stream(self, request: ChatRequest) -> AsyncGenerator[ModelResponse, None]:
        """Streaming not supported for native intelligence."""
        response = await self.chat(request)
        yield response

    def _extract_goal(self, request: ChatRequest) -> str:
        """Extract goal from chat request."""
        for msg in reversed(request.messages):
            if msg.role == Role.USER:
                return msg.content
        return ""

    def _classify_goal(self, goal: str) -> str:
        """Classify goal into capability type."""
        goal_lower = goal.lower()

        # Simple keyword-based classification
        if any(kw in goal_lower for kw in ["plan", "create plan", "steps", "how to"]):
            return "planning_known_patterns"
        elif any(kw in goal_lower for kw in ["reason", "think", "analyze", "why"]):
            return "simple_reasoning"
        elif any(kw in goal_lower for kw in ["what is", "tell me", "explain", "describe"]):
            return "known_fact_inference"
        elif any(kw in goal_lower for kw in ["tool", "use", "run", "execute"]):
            return "tool_suggestion"
        else:
            return "simple_reasoning"

    def close(self) -> None:
        """Close provider."""
        pass
