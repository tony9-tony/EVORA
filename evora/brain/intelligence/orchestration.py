"""
Phase 15 — Intelligence Orchestration for EVORA.

Builds a central capability-selection mechanism.

For every request EVORA should determine:
  - Can native intelligence solve this?
  - Can memory solve this?
  - Can knowledge solve this?
  - Can coding intelligence solve this?
  - Is a tool required?
  - Is a local model useful?
  - Is an external model useful?
  - Is the task unavailable?

Uses explicit result types:
  NATIVE_RESULT
  MODEL_ENHANCED_RESULT
  EXTERNAL_ONLY_RESULT
  UNAVAILABLE

Never silently pretend external output is native output.

Reuses existing abstractions:
  - CapabilityRegistry for capability classification
  - MemoryService for memory-based answers
  - KnowledgeGraph for knowledge-based answers
  - NativeCodingIntelligence for code tasks
  - NativeComprehensionIntelligence for understanding
  - ToolRegistry for tool execution
  - NativeIntelligenceProvider for model routing

No ModelManager dependency in the orchestrator itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ResultType(str, Enum):
    """Types of intelligence results."""
    NATIVE_RESULT = "native_result"
    MODEL_ENHANCED_RESULT = "model_enhanced_result"
    EXTERNAL_ONLY_RESULT = "external_only_result"
    UNAVAILABLE = "unavailable"


@dataclass
class OrchestrationDecision:
    """Decision from the orchestration layer."""
    result_type: ResultType
    capability_used: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    requires_model: bool = False
    requires_tool: bool = False
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_type": self.result_type.value,
            "capability_used": self.capability_used,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "requires_model": self.requires_model,
            "requires_tool": self.requires_tool,
            "requires_approval": self.requires_approval,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Intelligence Orchestrator
# ---------------------------------------------------------------------------

class IntelligenceOrchestrator:
    """Central capability-selection mechanism for EVORA.

    Determines the best way to handle a request:
    1. Check if native intelligence can solve it
    2. Check if memory has the answer
    3. Check if knowledge has the answer
    4. Check if coding intelligence can solve it
    5. Check if a tool is required
    6. Check if a local model would help
    7. Check if an external model is required
    8. Mark as unavailable if nothing can handle it

    Never silently pretends external output is native output.
    """

    def __init__(
        self,
        capability_registry: Any = None,
        memory_service: Any = None,
        knowledge_graph: Any = None,
        coding_intelligence: Any = None,
        comprehension_intelligence: Any = None,
        tool_registry: Any = None,
        model_provider: Any = None,
        logger: Optional[Any] = None,
    ):
        self.capability_registry = capability_registry
        self.memory_service = memory_service
        self.knowledge_graph = knowledge_graph
        self.coding_intelligence = coding_intelligence
        self.comprehension_intelligence = comprehension_intelligence
        self.tool_registry = tool_registry
        self.model_provider = model_provider
        self.logger = logger

    def orchestrate(self, request: dict[str, Any]) -> OrchestrationDecision:
        """Determine how to handle a request."""
        text = request.get("text", "") or request.get("goal", "") or ""
        context = request.get("context", {})
        task_type = request.get("task_type", "")

        if not text or not text.strip():
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                reasoning="Empty request",
                confidence=0.0,
            )

        intent = self._classify_intent(text)
        if self.logger:
            self.logger.observe(f"Orchestrating request: intent={intent.intent_type.value}, text={text[:80]}")

        memory_decision = self._try_memory(text, context)
        if memory_decision.result_type == ResultType.NATIVE_RESULT and memory_decision.confidence >= 0.7:
            return memory_decision

        knowledge_decision = self._try_knowledge(text, context)
        if knowledge_decision.result_type == ResultType.NATIVE_RESULT and knowledge_decision.confidence >= 0.7:
            return knowledge_decision

        coding_decision = self._try_coding(text, context)
        if coding_decision.result_type == ResultType.NATIVE_RESULT and coding_decision.confidence >= 0.6:
            return coding_decision

        capability_decision = self._try_capability(task_type, text, context)
        if capability_decision.result_type == ResultType.NATIVE_RESULT:
            return capability_decision

        tool_decision = self._try_tool(text, context)
        if tool_decision.result_type == ResultType.NATIVE_RESULT and tool_decision.confidence >= 0.5:
            return tool_decision

        model_decision = self._try_model(text, context, intent)
        if model_decision.result_type in (ResultType.MODEL_ENHANCED_RESULT, ResultType.EXTERNAL_ONLY_RESULT):
            return model_decision

        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="none",
            confidence=0.0,
            reasoning="No capability can handle this request",
        )

    def _classify_intent(self, text: str) -> Any:
        """Classify intent from text."""
        if self.comprehension_intelligence is not None:
            try:
                return self.comprehension_intelligence.classify_intent(text)
            except Exception:
                pass
        from evora.brain.intelligence.comprehension import IntentClassifier, IntentType
        classifier = IntentClassifier()
        return classifier.classify(text)

    def _try_memory(self, text: str, context: dict[str, Any]) -> OrchestrationDecision:
        """Try to answer from memory."""
        if self.memory_service is None:
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                capability_used="memory",
                reasoning="Memory service not available",
                confidence=0.0,
            )
        try:
            memories = self.memory_service.retrieve_relevant(goal=text, limit=3)
            if memories:
                best = memories[0]
                confidence = min(1.0, best.relevance_score if hasattr(best, "relevance_score") else 0.5)
                return OrchestrationDecision(
                    result_type=ResultType.NATIVE_RESULT,
                    capability_used="memory",
                    confidence=confidence,
                    reasoning=f"Found {len(memories)} relevant memory entries",
                )
        except Exception:
            pass
        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="memory",
            reasoning="No relevant memories found",
            confidence=0.0,
        )

    def _try_knowledge(self, text: str, context: dict[str, Any]) -> OrchestrationDecision:
        """Try to answer from knowledge graph."""
        if self.knowledge_graph is None:
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                capability_used="knowledge",
                reasoning="Knowledge graph not available",
                confidence=0.0,
            )
        try:
            nodes = self.knowledge_graph.query(text, limit=5)
            if nodes:
                confidence = max(0.0, min(1.0, nodes[0].confidence if hasattr(nodes[0], "confidence") else 0.5))
                return OrchestrationDecision(
                    result_type=ResultType.NATIVE_RESULT,
                    capability_used="knowledge",
                    confidence=confidence,
                    reasoning=f"Found {len(nodes)} relevant knowledge nodes",
                )
        except Exception:
            pass
        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="knowledge",
            reasoning="No relevant knowledge found",
            confidence=0.0,
        )

    def _try_coding(self, text: str, context: dict[str, Any]) -> OrchestrationDecision:
        """Try to handle as coding task."""
        if self.coding_intelligence is None:
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                capability_used="coding",
                reasoning="Coding intelligence not available",
                confidence=0.0,
            )
        try:
            capabilities = self.coding_intelligence.get_capabilities()
            for cap in capabilities:
                if cap.get("native") and cap.get("confidence", 0) >= 0.5:
                    return OrchestrationDecision(
                        result_type=ResultType.NATIVE_RESULT,
                        capability_used=cap["name"],
                        confidence=cap["confidence"],
                        reasoning=f"Coding intelligence can handle: {cap['name']}",
                    )
        except Exception:
            pass
        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="coding",
            reasoning="No coding capability matched",
            confidence=0.0,
        )

    def _try_capability(self, task_type: str, text: str, context: dict[str, Any]) -> OrchestrationDecision:
        """Try to handle via capability registry."""
        if self.capability_registry is None:
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                capability_used="registry",
                reasoning="Capability registry not available",
                confidence=0.0,
            )
        try:
            capability = self.capability_registry.can_handle(task_type or text[:50])
            if capability.capability_type.value == "native":
                return OrchestrationDecision(
                    result_type=ResultType.NATIVE_RESULT,
                    capability_used=capability.name,
                    confidence=capability.native_confidence,
                    reasoning=f"Native capability available: {capability.name}",
                )
            elif capability.capability_type.value == "external_model":
                return OrchestrationDecision(
                    result_type=ResultType.EXTERNAL_ONLY_RESULT,
                    capability_used=capability.name,
                    confidence=0.5,
                    reasoning=f"Requires external model: {capability.name}",
                    requires_model=True,
                )
            elif capability.capability_type.value == "local_model":
                return OrchestrationDecision(
                    result_type=ResultType.MODEL_ENHANCED_RESULT,
                    capability_used=capability.name,
                    confidence=0.6,
                    reasoning=f"Local model can enhance: {capability.name}",
                    requires_model=True,
                )
        except Exception:
            pass
        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="registry",
            reasoning="No matching capability in registry",
            confidence=0.0,
        )

    def _try_tool(self, text: str, context: dict[str, Any]) -> OrchestrationDecision:
        """Try to handle via tool execution."""
        if self.tool_registry is None:
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                capability_used="tool",
                reasoning="Tool registry not available",
                confidence=0.0,
            )
        try:
            available_tools = list(self.tool_registry.list()) if hasattr(self.tool_registry, "list") else []
            tool_keywords = {
                "read_file": ["read", "show", "display", "view"],
                "write_file": ["create", "write", "generate file"],
                "edit_file": ["edit", "modify", "change"],
                "execute_command": ["run", "execute", "command"],
                "search_files": ["find", "search", "locate"],
                "analyze_code": ["analyze", "inspect code", "review code"],
                "analyze_project": ["analyze project", "project structure"],
            }
            text_lower = text.lower()
            for tool_name in available_tools:
                tool_name_str = tool_name.name if hasattr(tool_name, "name") else str(tool_name)
                keywords = tool_keywords.get(tool_name_str, [])
                if any(kw in text_lower for kw in keywords):
                    return OrchestrationDecision(
                        result_type=ResultType.NATIVE_RESULT,
                        capability_used=tool_name_str,
                        confidence=0.7,
                        reasoning=f"Tool can handle: {tool_name_str}",
                        requires_tool=True,
                    )
        except Exception:
            pass
        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="tool",
            reasoning="No matching tool found",
            confidence=0.0,
        )

    def _try_model(self, text: str, context: dict[str, Any], intent: Any) -> OrchestrationDecision:
        """Try to handle via model enhancement."""
        if self.model_provider is None:
            return OrchestrationDecision(
                result_type=ResultType.UNAVAILABLE,
                capability_used="model",
                reasoning="No model provider available",
                confidence=0.0,
            )
        try:
            intent_type = intent.intent_type.value if hasattr(intent, "intent_type") else "unknown"
            if intent_type in ("query", "explain", "plan"):
                return OrchestrationDecision(
                    result_type=ResultType.MODEL_ENHANCED_RESULT,
                    capability_used="model_provider",
                    confidence=0.6,
                    reasoning="Model can enhance reasoning for this query type",
                    requires_model=True,
                )
            return OrchestrationDecision(
                result_type=ResultType.EXTERNAL_ONLY_RESULT,
                capability_used="model_provider",
                confidence=0.4,
                reasoning="External model required for this task",
                requires_model=True,
            )
        except Exception:
            pass
        return OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            capability_used="model",
            reasoning="Model provider error",
            confidence=0.0,
        )

    def get_orchestration_metrics(self) -> dict[str, Any]:
        """Get orchestration metrics."""
        return {
            "has_capability_registry": self.capability_registry is not None,
            "has_memory_service": self.memory_service is not None,
            "has_knowledge_graph": self.knowledge_graph is not None,
            "has_coding_intelligence": self.coding_intelligence is not None,
            "has_comprehension_intelligence": self.comprehension_intelligence is not None,
            "has_tool_registry": self.tool_registry is not None,
            "has_model_provider": self.model_provider is not None,
        }
