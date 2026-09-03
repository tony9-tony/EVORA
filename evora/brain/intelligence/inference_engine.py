"""
Phase 25 — Native Inference Engine for EVORA.

Optimizes inference and decision-making.

Supports:
  - Rule-based inference
  - Pattern matching
  - Confidence scoring
  - Inference chaining
  - Decision optimization
  - Integration with IntelligenceRuntime
  - Integration with KnowledgeGraph
  - Integration with ReasoningEngine

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class InferenceType(str, Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"


class ConfidenceLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class InferenceRule:
    """An inference rule."""
    rule_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    condition: str = ""
    conclusion: str = ""
    confidence: float = 0.5
    inference_type: InferenceType = InferenceType.DEDUCTIVE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "inference_type": self.inference_type.value,
            "metadata": self.metadata,
        }


@dataclass
class InferenceResult:
    """Result of an inference."""
    result_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    inference_type: InferenceType = InferenceType.DEDUCTIVE
    conclusion: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    rules_used: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "result_id": self.result_id,
            "inference_type": self.inference_type.value,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "rules_used": self.rules_used,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Native Inference Engine
# ---------------------------------------------------------------------------

class NativeInferenceEngine:
    """Native inference engine for EVORA.

    Performs reasoning and decision-making.
    """

    def __init__(
        self,
        knowledge_graph: Any = None,
        intelligence_runtime: Any = None,
        logger: Optional[Any] = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.intelligence_runtime = intelligence_runtime
        self.logger = logger
        self._rules: dict[str, InferenceRule] = {}
        self._inference_history: list[InferenceResult] = []

    def add_rule(self, rule: InferenceRule) -> None:
        """Add an inference rule."""
        self._rules[rule.rule_id] = rule

    def infer(self, premises: list[str], inference_type: InferenceType = InferenceType.DEDUCTIVE) -> InferenceResult:
        """Perform inference from premises."""
        matched_rules = []
        for rule in self._rules.values():
            if rule.inference_type == inference_type:
                if any(premise.lower() in rule.condition.lower() for premise in premises):
                    matched_rules.append(rule)
        conclusion = ""
        confidence = 0.0
        evidence = []
        if matched_rules:
            best_rule = max(matched_rules, key=lambda r: r.confidence)
            conclusion = best_rule.conclusion
            confidence = best_rule.confidence
            evidence = [r.condition for r in matched_rules]
        else:
            conclusion = f"No inference possible from {len(premises)} premises"
            confidence = 0.1
        result = InferenceResult(
            inference_type=inference_type,
            conclusion=conclusion,
            confidence=confidence,
            evidence=evidence,
            rules_used=[r.rule_id for r in matched_rules],
        )
        self._inference_history.append(result)
        return result

    def match_pattern(self, pattern: str, text: str) -> dict[str, Any]:
        """Match a pattern against text."""
        import re
        matches = re.findall(pattern, text, re.IGNORECASE)
        return {
            "pattern": pattern,
            "text": text,
            "matches": matches,
            "match_count": len(matches),
            "confidence": min(1.0, len(matches) * 0.3),
        }

    def chain_inference(self, initial_premises: list[str], max_depth: int = 3) -> list[InferenceResult]:
        """Chain inferences to build reasoning chains."""
        results = []
        current_premises = initial_premises[:]
        for _ in range(max_depth):
            result = self.infer(current_premises)
            if result.confidence <= 0.1:
                break
            results.append(result)
            current_premises = [result.conclusion]
        return results

    def optimize_decision(self, options: list[dict[str, Any]], criteria: list[str] = None) -> dict[str, Any]:
        """Optimize a decision among multiple options."""
        criteria = criteria or ["confidence", "feasibility", "impact"]
        scored_options = []
        for option in options:
            score = sum(option.get(criterion, 0.5) for criterion in criteria) / len(criteria)
            scored_options.append({"option": option, "score": score})
        scored_options.sort(key=lambda x: x["score"], reverse=True)
        return {
            "best_option": scored_options[0] if scored_options else None,
            "all_options": scored_options,
            "criteria_used": criteria,
        }

    def get_inference_stats(self) -> dict[str, Any]:
        """Get inference statistics."""
        total = len(self._inference_history)
        by_type: dict[str, int] = {}
        for result in self._inference_history:
            by_type[result.inference_type.value] = by_type.get(result.inference_type.value, 0) + 1
        return {
            "total_inferences": total,
            "by_type": by_type,
            "rules_count": len(self._rules),
        }
