"""
Phase 10 Integration — Native Intelligence Spine Factory.

Creates a shared intelligence spine that both Chatbot (interactive mode)
and NativeAgent (autonomous mode) use.

The spine wires together:
  - CapabilityRegistry (NATIVE vs MODEL-ENHANCED vs EXTERNAL vs UNAVAILABLE)
  - KnowledgeGraph (persistent knowledge)
  - NativeReasoning (decision-making from facts/constraints/evidence)
  - NativePlanner (knowledge-grounded planning)
  - InferenceEngine (rule-based inference)
  - IntelligenceEvaluator (evaluation of results)
  - TrainingPipeline (recorded experience → lessons)
  - NativeCodingIntelligence (AST-based code understanding/generation)
  - NativeComprehensionIntelligence (intent classification, entity extraction)
  - IntelligenceOrchestrator (capability routing)
  - IntelligenceRuntime (coordinator)
  - NativeIntelligenceProvider (ModelProvider wrapper for CLI compatibility)

No ModelManager dependency in the spine itself.
External models may be registered as explicit providers via the CLI.
"""

from __future__ import annotations

from typing import Any, Optional

from evora.logger import Logger
from evora.memory import Memory


# ---------------------------------------------------------------------------
# Native Spine Factory
# ---------------------------------------------------------------------------

class NativeSpineFactory:
    """Factory for the shared native intelligence spine.

    Creates all native components and wires them together so that
    Chatbot and Agent share the same reasoning, planning, memory,
    knowledge, and evaluation pipelines.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger("evora-native", "INFO", "")

    def create_spine(
        self,
        memory_service: Any = None,
        tool_registry: Any = None,
        identity_service: Any = None,
        permission_manager: Any = None,
        approval_system: Any = None,
        training_pipeline: Any = None,
        model_provider: Any = None,
        knowledge_graph: Any = None,
    ) -> dict[str, Any]:
        """Create and wire the complete native intelligence spine.

        Args:
            memory_service: EVORA MemoryService for memory retrieval
            tool_registry: EVORA ToolRegistry for tool execution
            identity_service: EVORA IdentityService for authority checks
            permission_manager: EVORA PermissionManager for workspace/security
            approval_system: EVORA ApprovalSystem for user approval
            training_pipeline: Optional training pipeline
            model_provider: Optional external model for enhancement (NOT required)
            knowledge_graph: Optional pre-existing knowledge graph

        Returns:
            dict with keys:
                - runtime: IntelligenceRuntime
                - orchestrator: IntelligenceOrchestrator
                - capability_registry: CapabilityRegistry
                - knowledge_graph: KnowledgeGraph
                - native_reasoning: NativeReasoning
                - native_planner: NativePlanner
                - inference_engine: InferenceEngine
                - intelligence_evaluator: IntelligenceEvaluator
                - native_coding_intelligence: NativeCodingIntelligence
                - comprehension_intelligence: NativeComprehensionIntelligence
                - native_chatbot: NativeChatbot
                - native_agent: NativeAgent
        """
        from evora.brain.intelligence.knowledge import KnowledgeGraph
        from evora.brain.intelligence.capabilities import CapabilityRegistry
        from evora.brain.intelligence.reasoning import NativeReasoning
        from evora.brain.intelligence.planner import NativePlanner
        from evora.brain.intelligence.inference import InferenceEngine
        from evora.brain.intelligence.evaluation import IntelligenceEvaluator
        from evora.brain.intelligence.coding import NativeCodingIntelligence
        from evora.brain.intelligence.comprehension import NativeComprehensionIntelligence
        from evora.brain.intelligence.orchestration import IntelligenceOrchestrator
        from evora.brain.intelligence.runtime import IntelligenceRuntime
        from evora.brain.intelligence.training import TrainingPipeline
        from evora.brain.intelligence.conversation import NativeChatbot, ConversationManager

        capability_registry = CapabilityRegistry(logger=self.logger)

        if knowledge_graph is None:
            knowledge_graph = KnowledgeGraph(
                memory_service=memory_service,
                logger=self.logger,
            )

        intelligence_evaluator = IntelligenceEvaluator(logger=self.logger)

        training_pipeline = TrainingPipeline(
            learning_engine=None,
            knowledge_graph=knowledge_graph,
            intelligence_evaluator=intelligence_evaluator,
            logger=self.logger,
        )

        native_reasoning = NativeReasoning(
            decision_engine=None,
            memory_service=memory_service,
            knowledge_graph=knowledge_graph,
            logger=self.logger,
        )

        native_planner = NativePlanner(
            knowledge_graph=knowledge_graph,
            memory_service=memory_service,
            tool_registry=tool_registry,
            reasoning_engine=native_reasoning,
            logger=self.logger,
        )

        inference_engine = InferenceEngine(
            knowledge_graph=knowledge_graph,
            memory_service=memory_service,
            logger=self.logger,
        )

        native_coding_intelligence = NativeCodingIntelligence(
            logger=self.logger,
        )

        comprehension_intelligence = NativeComprehensionIntelligence(
            capability_registry=capability_registry,
            knowledge_graph=knowledge_graph,
            memory_service=memory_service,
            logger=self.logger,
        )

        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            memory_service=memory_service,
            knowledge_graph=knowledge_graph,
            coding_intelligence=native_coding_intelligence,
            comprehension_intelligence=comprehension_intelligence,
            tool_registry=tool_registry,
            model_provider=model_provider,
            logger=self.logger,
        )

        runtime = IntelligenceRuntime(
            native_reasoning=native_reasoning,
            native_planner=native_planner,
            inference_engine=inference_engine,
            knowledge_graph=knowledge_graph,
            intelligence_evaluator=intelligence_evaluator,
            capability_registry=capability_registry,
            logger=self.logger,
            training_pipeline=training_pipeline,
            native_coding_intelligence=native_coding_intelligence,
            native_comprehension_intelligence=comprehension_intelligence,
            intelligence_orchestrator=orchestrator,
        )

        conversation_manager = ConversationManager(
            memory_service=memory_service,
            logger=self.logger,
        )

        from evora.brain.intelligence.agent_intelligence import NativeAgent
        native_agent = NativeAgent(
            intelligence_runtime=runtime,
            comprehension_intelligence=comprehension_intelligence,
            conversation_manager=conversation_manager,
            tool_registry=tool_registry,
            permission_manager=permission_manager,
            approval_system=approval_system,
            identity_service=identity_service,
            training_pipeline=training_pipeline,
            logger=self.logger,
        )

        native_chatbot = NativeChatbot(
            conversation_manager=conversation_manager,
            memory_service=memory_service,
            knowledge_graph=knowledge_graph,
            comprehension_intelligence=comprehension_intelligence,
            reasoning_engine=native_reasoning,
            training_pipeline=training_pipeline,
            logger=self.logger,
        )

        return {
            "runtime": runtime,
            "orchestrator": orchestrator,
            "capability_registry": capability_registry,
            "knowledge_graph": knowledge_graph,
            "native_reasoning": native_reasoning,
            "native_planner": native_planner,
            "inference_engine": inference_engine,
            "intelligence_evaluator": intelligence_evaluator,
            "native_coding_intelligence": native_coding_intelligence,
            "comprehension_intelligence": comprehension_intelligence,
            "native_chatbot": native_chatbot,
            "native_agent": native_agent,
            "conversation_manager": conversation_manager,
            "training_pipeline": training_pipeline,
        }

    def create_runtime(
        self,
        memory_service: Any = None,
        tool_registry: Any = None,
        identity_service: Any = None,
        permission_manager: Any = None,
        approval_system: Any = None,
        training_pipeline: Any = None,
        model_provider: Any = None,
        knowledge_graph: Any = None,
    ) -> Any:
        """Create just the IntelligenceRuntime (for NativeIntelligenceProvider)."""
        spine = self.create_spine(
            memory_service=memory_service,
            tool_registry=tool_registry,
            identity_service=identity_service,
            permission_manager=permission_manager,
            approval_system=approval_system,
            training_pipeline=training_pipeline,
            model_provider=model_provider,
            knowledge_graph=knowledge_graph,
        )
        return spine["runtime"]
