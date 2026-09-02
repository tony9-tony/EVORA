"""
Phase 10 — InferenceEngine for EVORA native intelligence.

Deterministic rule-based + knowledge-based inference.
No ModelManager dependency. No external model dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class InferenceRule:
    """A deterministic inference rule."""

    id: str
    condition: str
    consequence: str
    confidence: float = 1.0
    source: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "condition": self.condition,
            "consequence": self.consequence,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class InferenceResult:
    """Result of native inference."""

    answer: str = ""
    confidence: float = 0.0
    source: str = ""
    limitations: list[str] = field(default_factory=list)
    matched_rules: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "source": self.source,
            "limitations": self.limitations,
            "matched_rules": self.matched_rules,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


class InferenceEngine:
    """Rule-based + knowledge-based inference.

    No ModelManager dependency.
    No external model dependency.
    Uses KnowledgeGraph and MemoryService for evidence.
    """

    def __init__(
        self,
        knowledge_graph: Any = None,
        memory_service: Any = None,
        logger: Optional[Logger] = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.memory_service = memory_service
        self.logger = logger
        self._rules: list[InferenceRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default inference rules."""
        self._rules.extend([
            InferenceRule(
                id="rule-tools-available",
                condition="tools available for goal",
                consequence="tool_suggestion",
                confidence=0.9,
                source="system",
            ),
            InferenceRule(
                id="rule-knowledge-exists",
                condition="knowledge exists for topic",
                consequence="knowledge_retrieval",
                confidence=0.95,
                source="system",
            ),
            InferenceRule(
                id="rule-no-knowledge",
                condition="no knowledge for topic",
                consequence="insufficient_evidence",
                confidence=0.9,
                source="system",
            ),
            InferenceRule(
                id="rule-model-required",
                condition="complex reasoning required",
                consequence="external_model_needed",
                confidence=0.8,
                source="system",
            ),
        ])

    def add_rule(self, rule: InferenceRule) -> None:
        """Add an inference rule."""
        self._rules.append(rule)
        if self.logger:
            self.logger.observe(f"Added inference rule: {rule.id}")

    async def infer(self, query: str, context: dict[str, Any] = None) -> InferenceResult:
        """Perform inference using rules and knowledge.

        Returns InferenceResult with answer, confidence, and limitations.
        """
        if not query or not query.strip():
            return InferenceResult(
                answer="",
                confidence=0.0,
                source="",
                limitations=["Empty query"],
            )

        context = context or {}
        matched_rules = []
        evidence = []
        limitations = []
        confidence = 0.0

        # Step 1: Match rules
        for rule in self._rules:
            if self._evaluate_condition(rule.condition, query, context):
                matched_rules.append(rule.id)
                evidence.append({
                    "type": "rule",
                    "rule_id": rule.id,
                    "confidence": rule.confidence,
                })

        # Step 2: Query knowledge graph
        if self.knowledge_graph is not None:
            try:
                nodes = self.knowledge_graph.query(query, limit=5)
                for node in nodes:
                    evidence.append({
                        "type": "knowledge",
                        "node_id": node.id,
                        "content": node.content,
                        "confidence": node.confidence,
                    })
            except Exception:
                limitations.append("Knowledge graph query failed")

        # Step 3: Query memory
        if self.memory_service is not None:
            try:
                memories = self.memory_service.retrieve_relevant(goal=query, limit=3)
                for mem in memories:
                    evidence.append({
                        "type": "memory",
                        "content": mem.content,
                        "memory_type": mem.memory_type,
                    })
            except Exception:
                limitations.append("Memory retrieval failed")

        # Step 4: Determine answer and confidence
        answer = ""
        confidence = 0.0
        source = ""

        if evidence:
            # Use highest confidence evidence
            evidence.sort(key=lambda e: e.get("confidence", 0.0), reverse=True)
            best = evidence[0]

            if best.get("type") == "knowledge":
                answer = best.get("content", "")
                confidence = best.get("confidence", 0.0)
                source = "knowledge_graph"
            elif best.get("type") == "memory":
                answer = best.get("content", "")
                confidence = 0.7
                source = "memory"
            elif best.get("type") == "rule":
                answer = best.get("consequence", "")
                confidence = best.get("confidence", 0.0)
                source = "rule"
            else:
                answer = ""
                confidence = 0.0
                source = "unknown"
        else:
            limitations.append("No evidence available for inference")

        # Adjust confidence based on evidence count
        if evidence:
            confidence = min(1.0, confidence + 0.05 * min(len(evidence), 5))

        if self.logger:
            self.logger.observe(
                f"Inference: query={query[:50]}, confidence={confidence:.2f}, evidence={len(evidence)}"
            )

        return InferenceResult(
            answer=answer,
            confidence=confidence,
            source=source,
            limitations=limitations,
            matched_rules=matched_rules,
            evidence=evidence[:10],
        )

    def _evaluate_condition(self, condition: str, query: str, context: dict[str, Any]) -> bool:
        """Evaluate a rule condition against query and context."""
        # Simple deterministic condition evaluation
        # In a real system, this would be more sophisticated
        if not condition:
            return False

        condition_lower = condition.lower()
        query_lower = query.lower()

        # Check for keyword matches
        keywords = condition_lower.split()
        matches = sum(1 for kw in keywords if kw in query_lower)

        return matches >= len(keywords) // 2
