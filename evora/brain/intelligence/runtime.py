"""
Phase 10 — IntelligenceRuntime for EVORA native intelligence.

Coordinates native intelligence capabilities:
  - NativeReasoning
  - NativePlanner
  - InferenceEngine
  - KnowledgeGraph
  - IntelligenceEvaluator
  - CapabilityRegistry

NO ModelManager dependency.
NO external model dependency.
Native core works completely offline.
"""

from __future__ import annotations

from typing import Any, Optional

from evora.logger import Logger


class IntelligenceRuntime:
    """Coordinates native intelligence capabilities.

    NEVER calls ModelManager.
    NEVER calls external models.
    Native core works completely offline.
    """

    def __init__(
        self,
        native_reasoning: Any,
        native_planner: Any,
        inference_engine: Any,
        knowledge_graph: Any,
        intelligence_evaluator: Any,
        capability_registry: Any,
        logger: Optional[Logger] = None,
    ):
        self.native_reasoning = native_reasoning
        self.native_planner = native_planner
        self.inference_engine = inference_engine
        self.knowledge_graph = knowledge_graph
        self.intelligence_evaluator = intelligence_evaluator
        self.capability_registry = capability_registry
        self.logger = logger

    async def reason(self, goal: str, context: dict[str, Any] = None) -> Any:
        """Reason using native capabilities only.

        Returns ReasoningResult.
        Never calls ModelManager.
        """
        if not goal or not goal.strip():
            from evora.brain.intelligence.reasoning import ReasoningResult
            return ReasoningResult(
                decision="reject",
                action="none",
                confidence=0.0,
                reasoning_summary="Empty goal",
                limitations=["No goal provided"],
            )

        try:
            from evora.brain.intelligence.reasoning import ReasoningFacts

            # Build reasoning facts
            facts = ReasoningFacts(
                goal=goal,
                observations=context.get("observations", []) if context else [],
                constraints=context.get("constraints", []) if context else [],
                assumptions=context.get("assumptions", []) if context else [],
                evidence=context.get("evidence", []) if context else [],
            )

            result = await self.native_reasoning.reason(facts)
            return result

        except Exception as e:
            if self.logger:
                self.logger.warn(f"Native reasoning failed: {e}")
            from evora.brain.intelligence.reasoning import ReasoningResult
            return ReasoningResult(
                decision="error",
                action="none",
                confidence=0.0,
                reasoning_summary=f"Reasoning failed: {e}",
                limitations=[str(e)],
            )

    async def plan(self, goal: str, constraints: list[str] = None) -> Any:
        """Plan using native capabilities only.

        Returns NativePlan or None.
        Never calls ModelManager.
        """
        if not goal or not goal.strip():
            return None

        try:
            result = await self.native_planner.plan(goal, constraints)
            return result
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Native planning failed: {e}")
            return None

    async def infer(self, query: str, context: dict[str, Any] = None) -> Any:
        """Infer using rules and knowledge.

        Returns InferenceResult.
        Never calls ModelManager.
        """
        if not query or not query.strip():
            from evora.brain.intelligence.inference import InferenceResult
            return InferenceResult(
                answer="",
                confidence=0.0,
                source="",
                limitations=["Empty query"],
            )

        try:
            result = await self.inference_engine.infer(query, context)
            return result
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Native inference failed: {e}")
            from evora.brain.intelligence.inference import InferenceResult
            return InferenceResult(
                answer="",
                confidence=0.0,
                source="",
                limitations=[str(e)],
            )

    def evaluate(self, evaluation_type: str, **kwargs) -> Any:
        """Evaluate intelligence quality.

        Returns EvaluationResult.
        """
        try:
            if evaluation_type == "reasoning":
                return self.intelligence_evaluator.evaluate_reasoning(**kwargs)
            elif evaluation_type == "plan":
                return self.intelligence_evaluator.evaluate_plan(**kwargs)
            elif evaluation_type == "inference":
                return self.intelligence_evaluator.evaluate_inference(**kwargs)
            elif evaluation_type == "capability":
                return self.intelligence_evaluator.evaluate_capability(**kwargs)
            else:
                from evora.brain.intelligence.evaluation import EvaluationResult, EvaluationGrade
                return EvaluationResult(
                    grade=EvaluationGrade.UNKNOWN,
                    reasoning=f"Unknown evaluation type: {evaluation_type}",
                )
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Evaluation failed: {e}")
            from evora.brain.intelligence.evaluation import EvaluationResult, EvaluationGrade
            return EvaluationResult(
                grade=EvaluationGrade.UNKNOWN,
                reasoning=f"Evaluation failed: {e}",
            )

    def get_capabilities(self) -> list[Any]:
        """Get available capabilities."""
        if self.capability_registry is None:
            return []
        return self.capability_registry.list_all()

    def can_handle(self, task_type: str) -> Any:
        """Check if a task type can be handled natively."""
        if self.capability_registry is None:
            from evora.brain.intelligence.capabilities import CapabilityType, IntelligenceCapability
            return IntelligenceCapability(
                name=task_type,
                description="Unknown (no registry)",
                capability_type=CapabilityType.UNAVAILABLE,
            )
        return self.capability_registry.can_handle(task_type)

    def get_knowledge(self, concept: str, limit: int = 10) -> list[Any]:
        """Query knowledge graph."""
        if self.knowledge_graph is None:
            return []
        try:
            return self.knowledge_graph.query(concept, limit=limit)
        except Exception:
            return []

    def add_knowledge(self, node: Any) -> str:
        """Add knowledge node."""
        if self.knowledge_graph is None:
            return ""
        try:
            return self.knowledge_graph.add_node(node)
        except Exception:
            return ""
