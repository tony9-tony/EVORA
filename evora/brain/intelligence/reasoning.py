"""
Phase 10 — NativeReasoning for EVORA native intelligence.

Model-independent reasoning using composable primitives.
No ModelManager dependency. No external model dependency.
Uses existing EVORA components:
  - DecisionEngine
  - MemoryService
  - LearningEngine
  - ObservationManager
  - KnowledgeGraph
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class ReasoningFacts:
    """Input facts for reasoning."""

    goal: str
    observations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Result of native reasoning."""

    decision: str = ""
    action: str = ""
    confidence: float = 0.0
    evidence_count: int = 0
    reasoning_summary: str = ""
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "action": self.action,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "reasoning_summary": self.reasoning_summary,
            "limitations": self.limitations,
            "metadata": self.metadata,
        }


class NativeReasoning:
    """Model-independent reasoning using composable primitives.

    No ModelManager dependency.
    No external model dependency.
    Uses existing EVORA components for evidence gathering.
    """

    def __init__(
        self,
        decision_engine: Any,
        memory_service: Any = None,
        learning_engine: Any = None,
        observation_manager: Any = None,
        knowledge_graph: Any = None,
        logger: Optional[Logger] = None,
    ):
        self.decision_engine = decision_engine
        self.memory_service = memory_service
        self.learning_engine = learning_engine
        self.observation_manager = observation_manager
        self.knowledge_graph = knowledge_graph
        self.logger = logger

    async def reason(self, facts: ReasoningFacts) -> ReasoningResult:
        """Reason about a goal using native capabilities only.

        Pipeline:
        1. Retrieve relevant memories
        2. Retrieve relevant lessons
        3. Retrieve relevant knowledge
        4. Generate candidate actions via DecisionEngine
        5. Evaluate candidates against constraints
        6. Score by evidence
        7. Select best action
        8. Return concise reasoning summary
        """
        if not facts.goal or not facts.goal.strip():
            return ReasoningResult(
                decision="reject",
                action="none",
                confidence=0.0,
                reasoning_summary="Empty goal provided",
                limitations=["No goal specified"],
            )

        # Step 1-3: Retrieve evidence
        memories = self._retrieve_memories(facts.goal)
        lessons = self._retrieve_lessons(facts.goal)
        knowledge = self._retrieve_knowledge(facts.goal)

        all_evidence = memories + lessons + knowledge + facts.evidence
        evidence_count = len(all_evidence)

        # Step 4: Generate candidate actions
        candidates = self._generate_candidates(facts, all_evidence)

        # Step 5-6: Evaluate and score
        scored_candidates = self._score_candidates(candidates, facts.constraints, all_evidence)

        # Step 7: Select best action
        best = self._select_best(scored_candidates)

        # Step 8: Build reasoning summary
        reasoning_summary = self._build_summary(facts.goal, best, evidence_count, all_evidence)
        limitations = self._determine_limitations(best, evidence_count, all_evidence)

        confidence = self._estimate_confidence(best, evidence_count, all_evidence)

        if self.logger:
            self.logger.observe(
                f"Native reasoning: goal={facts.goal[:80]}, confidence={confidence:.2f}, evidence={evidence_count}"
            )

        return ReasoningResult(
            decision=best.get("decision", "unknown"),
            action=best.get("action", "none"),
            confidence=confidence,
            evidence_count=evidence_count,
            reasoning_summary=reasoning_summary,
            limitations=limitations,
            metadata={
                "candidates_evaluated": len(candidates),
                "evidence_sources": {
                    "memories": len(memories),
                    "lessons": len(lessons),
                    "knowledge": len(knowledge),
                },
            },
        )

    def _retrieve_memories(self, goal: str) -> list[dict[str, Any]]:
        """Retrieve relevant memories."""
        if self.memory_service is None:
            return []
        try:
            memories = self.memory_service.retrieve_relevant(goal=goal, limit=5)
            return [m.to_dict() for m in memories] if memories else []
        except Exception:
            return []

    def _retrieve_lessons(self, goal: str) -> list[dict[str, Any]]:
        """Retrieve relevant lessons from learning engine."""
        if self.learning_engine is None:
            return []
        try:
            lessons = self.learning_engine.get_relevant_lessons(goal, limit=5)
            return [l.to_dict() for l in lessons] if lessons else []
        except Exception:
            return []

    def _retrieve_knowledge(self, goal: str) -> list[dict[str, Any]]:
        """Retrieve relevant knowledge from knowledge graph."""
        if self.knowledge_graph is None:
            return []
        try:
            nodes = self.knowledge_graph.query(goal, limit=5)
            return [n.to_dict() for n in nodes] if nodes else []
        except Exception:
            return []

    def _generate_candidates(self, facts: ReasoningFacts, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate candidate actions using DecisionEngine."""
        candidates = []

        # Use DecisionEngine if available
        if self.decision_engine is not None:
            try:
                from evora.task import TaskState
                state = TaskState(request=facts.goal, goal=facts.goal)
                decision = self.decision_engine.decide_next(state)
                candidates.append({
                    "decision": decision.action,
                    "action": decision.tool or decision.action,
                    "confidence": decision.confidence,
                    "source": "decision_engine",
                    "reason": decision.reason,
                })
            except Exception:
                pass

        # Generate tool-based candidates from evidence
        tool_names = set()
        for item in evidence:
            if isinstance(item, dict):
                tool = item.get("tool")
                if tool:
                    tool_names.add(tool)

        for tool in list(tool_names)[:3]:
            candidates.append({
                "decision": "execute_tool",
                "action": tool,
                "confidence": 0.5,
                "source": "evidence",
                "reason": f"Tool '{tool}' mentioned in evidence",
            })

        # If no candidates, add default
        if not candidates:
            candidates.append({
                "decision": "analyze",
                "action": "analyze",
                "confidence": 0.3,
                "source": "default",
                "reason": "No evidence found, default to analysis",
            })

        return candidates

    def _score_candidates(
        self,
        candidates: list[dict[str, Any]],
        constraints: list[str],
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score candidates against constraints and evidence."""
        scored = []
        for candidate in candidates:
            score = candidate.get("confidence", 0.0)

            # Boost score based on evidence
            evidence_matches = sum(
                1 for e in evidence
                if isinstance(e, dict) and candidate.get("action", "") in str(e).lower()
            )
            score += 0.1 * min(evidence_matches, 5)

            # Penalize if constraints violated
            for constraint in constraints:
                if constraint and candidate.get("action", "") in constraint.lower():
                    score -= 0.2

            candidate["score"] = max(0.0, min(1.0, score))
            scored.append(candidate)

        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored

    def _select_best(self, scored_candidates: list[dict[str, Any]]) -> dict[str, Any]:
        """Select the best candidate."""
        if not scored_candidates:
            return {
                "decision": "unknown",
                "action": "none",
                "confidence": 0.0,
                "score": 0.0,
            }
        return scored_candidates[0]

    def _build_summary(self, goal: str, best: dict[str, Any], evidence_count: int, evidence: list[dict[str, Any]]) -> str:
        """Build concise reasoning summary."""
        action = best.get("action", "none")
        score = best.get("score", 0.0)
        source = best.get("source", "unknown")

        if evidence_count == 0:
            return f"No evidence found for '{goal[:50]}'. Default action: {action} (score: {score:.2f})"

        return (
            f"Selected action '{action}' for '{goal[:50]}' "
            f"based on {evidence_count} evidence items (source: {source}, score: {score:.2f})"
        )

    def _determine_limitations(self, best: dict[str, Any], evidence_count: int, evidence: list[dict[str, Any]]) -> list[str]:
        """Determine limitations of the reasoning."""
        limitations = []

        if evidence_count == 0:
            limitations.append("No evidence available")
        elif evidence_count < 2:
            limitations.append("Limited evidence")

        if best.get("score", 0.0) < 0.5:
            limitations.append("Low confidence decision")

        if best.get("source") == "default":
            limitations.append("Default decision due to lack of evidence")

        return limitations

    def _estimate_confidence(self, best: dict[str, Any], evidence_count: int, evidence: list[dict[str, Any]]) -> float:
        """Estimate confidence in the reasoning."""
        base_confidence = best.get("score", 0.0)

        # Adjust based on evidence quality
        if evidence_count == 0:
            base_confidence *= 0.5
        elif evidence_count >= 3:
            base_confidence = min(1.0, base_confidence * 1.1)

        return max(0.0, min(1.0, base_confidence))
