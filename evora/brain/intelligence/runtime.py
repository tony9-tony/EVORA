"""
Phase 10 — IntelligenceRuntime for EVORA native intelligence.

Coordinates native intelligence capabilities:
  - NativeReasoning
  - NativePlanner
  - InferenceEngine
  - KnowledgeGraph
  - IntelligenceEvaluator
  - CapabilityRegistry
  - TrainingPipeline (Phase 11)
  - NativeCodingIntelligence (Phase 12)

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
        training_pipeline: Any = None,
        native_coding_intelligence: Any = None,
    ):
        self.native_reasoning = native_reasoning
        self.native_planner = native_planner
        self.inference_engine = inference_engine
        self.knowledge_graph = knowledge_graph
        self.intelligence_evaluator = intelligence_evaluator
        self.capability_registry = capability_registry
        self.logger = logger
        self.training_pipeline = training_pipeline
        self.native_coding_intelligence = native_coding_intelligence

    async def reason(self, goal: str, context: dict[str, Any] = None) -> Any:
        """Reason using native capabilities only.

        Returns ReasoningResult.
        Never calls ModelManager.
        Records training example if training pipeline is configured.
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

            facts = ReasoningFacts(
                goal=goal,
                observations=context.get("observations", []) if context else [],
                constraints=context.get("constraints", []) if context else [],
                assumptions=context.get("assumptions", []) if context else [],
                evidence=context.get("evidence", []) if context else [],
            )

            result = await self.native_reasoning.reason(facts)

            if self.training_pipeline is not None:
                try:
                    outcome = self._outcome_from_reasoning(result)
                    self.training_pipeline.record_training_example(
                        session_id=context.get("session_id", "") if context else "",
                        task_id=context.get("task_id", "") if context else "",
                        project=context.get("project", "") if context else "",
                        component="reasoning",
                        input_data={"goal": goal, "context": context or {}},
                        output_data={"result": result.to_dict() if hasattr(result, "to_dict") else str(result)},
                        outcome=outcome,
                        confidence=result.confidence,
                    )
                except Exception:
                    pass

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
        Records training example if training pipeline is configured.
        """
        if not goal or not goal.strip():
            return None

        try:
            result = await self.native_planner.plan(goal, constraints)

            if self.training_pipeline is not None and result is not None:
                try:
                    outcome = self._outcome_from_plan(result)
                    self.training_pipeline.record_training_example(
                        session_id="",
                        task_id="",
                        project="",
                        component="planner",
                        input_data={"goal": goal, "constraints": constraints or []},
                        output_data={"plan": result.to_dict() if hasattr(result, "to_dict") else str(result)},
                        outcome=outcome,
                        confidence=result.confidence,
                    )
                except Exception:
                    pass

            return result
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Native planning failed: {e}")
            return None

    async def infer(self, query: str, context: dict[str, Any] = None) -> Any:
        """Infer using rules and knowledge.

        Returns InferenceResult.
        Never calls ModelManager.
        Records training example if training pipeline is configured.
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

            if self.training_pipeline is not None:
                try:
                    outcome = self._outcome_from_inference(result)
                    self.training_pipeline.record_training_example(
                        session_id="",
                        task_id="",
                        project="",
                        component="inference",
                        input_data={"query": query, "context": context or {}},
                        output_data={"result": result.to_dict() if hasattr(result, "to_dict") else str(result)},
                        outcome=outcome,
                        confidence=result.confidence,
                    )
                except Exception:
                    pass

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

    def understand_file(self, path: str) -> dict[str, Any]:
        """Understand a source file structure."""
        if self.native_coding_intelligence is None:
            return {"error": "Coding intelligence not configured"}
        try:
            return self.native_coding_intelligence.understand_file(path)
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Code understanding failed: {e}")
            return {"error": str(e)}

    def detect_bugs(self, path: str) -> list[Any]:
        """Detect bugs in a source file."""
        if self.native_coding_intelligence is None:
            return []
        try:
            return self.native_coding_intelligence.detect_bugs(path)
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Bug detection failed: {e}")
            return []

    def generate_code(self, spec: dict[str, Any]) -> str:
        """Generate code from specification."""
        if self.native_coding_intelligence is None:
            return ""
        try:
            return self.native_coding_intelligence.generate_code(spec)
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Code generation failed: {e}")
            return ""

    def explain_code(self, path: str) -> Any:
        """Explain code structure."""
        if self.native_coding_intelligence is None:
            from evora.brain.intelligence.coding import CodeExplanation
            return CodeExplanation(summary="Coding intelligence not configured")
        try:
            return self.native_coding_intelligence.explain_code(path)
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Code explanation failed: {e}")
            from evora.brain.intelligence.coding import CodeExplanation
            return CodeExplanation(summary=f"Error: {e}")

    def generate_test(self, spec: dict[str, Any]) -> Any:
        """Generate test cases."""
        if self.native_coding_intelligence is None:
            from evora.brain.intelligence.coding import GeneratedTest
            return GeneratedTest(test_code="# Coding intelligence not configured")
        try:
            return self.native_coding_intelligence.generate_test(spec)
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Test generation failed: {e}")
            from evora.brain.intelligence.coding import GeneratedTest
            return GeneratedTest(test_code=f"# Error: {e}")

    def evaluate_patch(self, original: str, patched: str) -> Any:
        """Evaluate a patch."""
        if self.native_coding_intelligence is None:
            from evora.brain.intelligence.coding import PatchEvaluation
            return PatchEvaluation(confidence=0.0, reasoning="Coding intelligence not configured")
        try:
            return self.native_coding_intelligence.evaluate_patch(original, patched)
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Patch evaluation failed: {e}")
            from evora.brain.intelligence.coding import PatchEvaluation
            return PatchEvaluation(confidence=0.0, reasoning=f"Error: {e}")

    def _outcome_from_reasoning(self, result: Any) -> Any:
        """Derive outcome type from reasoning result."""
        from evora.brain.intelligence.training import OutcomeType
        if result is None:
            return OutcomeType.FAILURE
        confidence = getattr(result, "confidence", 0.0)
        limitations = getattr(result, "limitations", []) or []
        if not limitations and confidence >= 0.7:
            return OutcomeType.SUCCESS
        elif confidence >= 0.4:
            return OutcomeType.PARTIAL
        else:
            return OutcomeType.UNCERTAIN

    def _outcome_from_plan(self, plan: Any) -> Any:
        """Derive outcome type from plan result."""
        from evora.brain.intelligence.training import OutcomeType
        if plan is None:
            return OutcomeType.FAILURE
        confidence = getattr(plan, "confidence", 0.0)
        steps = getattr(plan, "steps", []) or []
        if confidence >= 0.6 and steps:
            return OutcomeType.SUCCESS
        elif steps:
            return OutcomeType.PARTIAL
        else:
            return OutcomeType.UNCERTAIN

    def _outcome_from_inference(self, result: Any) -> Any:
        """Derive outcome type from inference result."""
        from evora.brain.intelligence.training import OutcomeType
        if result is None:
            return OutcomeType.FAILURE
        confidence = getattr(result, "confidence", 0.0)
        answer = getattr(result, "answer", "") or ""
        if confidence >= 0.6 and answer.strip():
            return OutcomeType.SUCCESS
        elif confidence >= 0.3:
            return OutcomeType.PARTIAL
        else:
            return OutcomeType.UNCERTAIN
