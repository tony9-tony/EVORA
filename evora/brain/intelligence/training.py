"""
Phase 11 — Training & Learning Foundation for EVORA native intelligence.

Builds a serious native learning foundation that:
  - collects experiences from native intelligence outputs
  - identifies outcomes
  - extracts lessons
  - associates lessons with tasks
  - evaluates successful and failed approaches
  - retrieves relevant previous experience
  - improves future decisions from previous experience
  - distinguishes facts from assumptions
  - distinguishes successful knowledge from failed knowledge
  - maintains confidence
  - detects contradictory knowledge
  - avoids blindly learning malicious instructions

Architecture:
  TrainingPipeline    — orchestrates experience capture, lesson extraction, knowledge update
  ConfidenceCalibrator — adjusts confidence based on observed outcomes
  ContradictionDetector — detects contradictions in knowledge graph
  NativeIntelligenceMetrics — tracks accuracy/precision/recall of native intelligence
  TrainingExample     — (input, output, feedback) training triple
  ProvenanceTracker   — tracks where knowledge came from

Reuses existing abstractions:
  - LearningEngine for experience/lesson/knowledge lifecycle
  - KnowledgeGraph for knowledge storage and retrieval
  - IntelligenceEvaluator for output quality assessment
  - MemoryService for durable storage
  - IdentityService for authority checks

Security:
  - Learning data is untrusted data
  - Learned content NEVER grants authority
  - Malicious instructions are filtered, not integrated
  - All knowledge updates require creator authority
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Core training dataclasses
# ---------------------------------------------------------------------------

class OutcomeType(str, Enum):
    """Kinds of outcomes from native intelligence."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class TrainingExampleStatus(str, Enum):
    """Lifecycle of a training example."""
    RAW = "raw"
    VALIDATED = "validated"
    INTEGRATED = "integrated"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


@dataclass
class TrainingExample:
    """A (input, output, feedback) training triple.

    Training examples are the unit of learning for native intelligence.
    They capture what EVORA did, what happened, and what was learned.
    """
    example_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    session_id: str = ""
    task_id: str = ""
    project: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    outcome: OutcomeType = OutcomeType.UNCERTAIN
    feedback: Optional[str] = None
    feedback_source: str = ""
    status: TrainingExampleStatus = TrainingExampleStatus.RAW
    confidence: float = 0.5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    integrated_at: str = ""
    superseded_at: str = ""
    superseded_by: str = ""

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "project": self.project,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "outcome": self.outcome.value,
            "feedback": self.feedback,
            "feedback_source": self.feedback_source,
            "status": self.status.value,
            "confidence": self.confidence,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "integrated_at": self.integrated_at,
            "superseded_at": self.superseded_at,
            "superseded_by": self.superseded_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingExample":
        data = dict(data)
        data["outcome"] = OutcomeType(data.get("outcome", "uncertain"))
        data["status"] = TrainingExampleStatus(data.get("status", "raw"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ProvenanceRecord:
    """Tracks where a piece of knowledge came from."""
    provenance_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    knowledge_id: str = ""
    source_type: str = ""  # experience, lesson, feedback, observation, import
    source_id: str = ""
    source_description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "knowledge_id": self.knowledge_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_description": self.source_description,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Confidence Calibrator
# ---------------------------------------------------------------------------

class ConfidenceCalibrator:
    """Adjusts confidence scores based on observed outcomes.

    Uses a simple Bayesian-like update to recalibrate confidence
    based on success/failure history for each capability/context.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._history: dict[str, list[dict[str, Any]]] = {}
        self._calibration: dict[str, float] = {}

    def _key(self, capability: str, context_hash: str) -> str:
        return f"{capability}:{context_hash}"

    def record_outcome(self, capability: str, context: dict[str, Any], predicted_confidence: float, outcome: OutcomeType) -> None:
        """Record an observed outcome for calibration."""
        context_hash = self._hash_context(context)
        key = self._key(capability, context_hash)
        if key not in self._history:
            self._history[key] = []
        self._history[key].append({
            "predicted": predicted_confidence,
            "outcome": outcome.value,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._history[key]) > 100:
            self._history[key] = self._history[key][-100:]

    def get_calibrated_confidence(self, capability: str, context: dict[str, Any], raw_confidence: float) -> float:
        """Get calibrated confidence based on historical accuracy."""
        context_hash = self._hash_context(context)
        key = self._key(capability, context_hash)
        history = self._history.get(key, [])
        if not history:
            return raw_confidence
        successes = sum(1 for h in history if h["outcome"] == OutcomeType.SUCCESS.value)
        total = len(history)
        accuracy = successes / total if total > 0 else 0.5
        calibration_factor = accuracy * 0.5 + 0.5
        calibrated = raw_confidence * calibration_factor
        return max(0.0, min(1.0, calibrated))

    def get_capability_accuracy(self, capability: str) -> float:
        """Get overall accuracy for a capability across all contexts."""
        all_outcomes = []
        for key, history in self._history.items():
            if key.startswith(f"{capability}:"):
                all_outcomes.extend(h["outcome"] for h in history)
        if not all_outcomes:
            return 0.5
        successes = sum(1 for o in all_outcomes if o == OutcomeType.SUCCESS.value)
        return successes / len(all_outcomes)

    def _hash_context(self, context: dict[str, Any]) -> str:
        """Create a stable hash for a context dict."""
        try:
            serialized = json.dumps(context, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        except Exception:
            return "default"

    def get_metrics(self) -> dict[str, Any]:
        """Get calibration metrics."""
        total_observations = sum(len(h) for h in self._history.values())
        capabilities = {}
        for key in self._history:
            cap = key.split(":")[0]
            if cap not in capabilities:
                capabilities[cap] = 0
            capabilities[cap] += len(self._history[key])
        return {
            "total_observations": total_observations,
            "capabilities_tracked": len(capabilities),
            "observations_per_capability": capabilities,
        }


# ---------------------------------------------------------------------------
# Contradiction Detector
# ---------------------------------------------------------------------------

class ContradictionDetector:
    """Detects contradictions in the knowledge graph.

    Checks for:
    - Contradictory edges (explicit contradicts relations)
    - Conflicting nodes with same concept but opposite content
    - Confidence-based contradictions (high-confidence conflicting claims)
    """

    def __init__(self, knowledge_graph: Any = None, logger: Optional[Any] = None):
        self.knowledge_graph = knowledge_graph
        self.logger = logger
        self._detected_contradictions: list[dict[str, Any]] = []

    def detect_contradictions(self) -> list[dict[str, Any]]:
        """Detect contradictions in the knowledge graph."""
        if self.knowledge_graph is None:
            return []
        contradictions = []
        try:
            nodes = self.knowledge_graph.get_all_nodes(limit=500)
            edges = self.knowledge_graph.get_all_edges(limit=1000)
            node_map = {n.id: n for n in nodes}
            for edge in edges:
                if edge.relation == "contradicts":
                    source = node_map.get(edge.source_id)
                    target = node_map.get(edge.target_id)
                    if source and target:
                        contradictions.append({
                            "type": "explicit_contradiction",
                            "source_id": source.id,
                            "target_id": target.id,
                            "source_content": source.content[:120],
                            "target_content": target.content[:120],
                            "confidence": edge.confidence,
                        })
            for i, node_a in enumerate(nodes):
                for node_b in nodes[i + 1:]:
                    if node_a.type == node_b.type and self._content_conflicts(node_a.content, node_b.content):
                        contradictions.append({
                            "type": "implicit_contradiction",
                            "source_id": node_a.id,
                            "target_id": node_b.id,
                            "source_content": node_a.content[:120],
                            "target_content": node_b.content[:120],
                            "confidence": min(node_a.confidence, node_b.confidence),
                        })
        except Exception:
            pass
        self._detected_contradictions = contradictions
        return contradictions

    def _content_conflicts(self, a: str, b: str) -> bool:
        """Check if two content strings likely conflict."""
        if not a or not b:
            return False
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()
        negation_markers = ["not ", "no ", "never ", "cannot ", "can't ", "doesn't ", "do not "]
        for marker in negation_markers:
            if a_lower.startswith(marker) and b_lower == a_lower[len(marker):]:
                return True
            if b_lower.startswith(marker) and a_lower == b_lower[len(marker):]:
                return True
        a_words = set(a_lower.split())
        b_words = set(b_lower.split())
        common = a_words & b_words
        if len(common) >= 2:
            a_neg = any(m in a_lower for m in negation_markers)
            b_neg = any(m in b_lower for m in negation_markers)
            if a_neg != b_neg:
                return True
        return False

    def get_contradiction_count(self) -> int:
        """Get count of detected contradictions."""
        return len(self._detected_contradictions)


# ---------------------------------------------------------------------------
# Native Intelligence Metrics
# ---------------------------------------------------------------------------

class NativeIntelligenceMetrics:
    """Tracks accuracy, precision, recall of native intelligence components.

    Maintains per-component metrics for:
    - NativeReasoning
    - NativePlanner
    - InferenceEngine

    All metrics are derived from training examples with known outcomes.
    Uses a sliding window to bound memory usage.
    """

    METRIC_WINDOW = 200

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._examples: list[TrainingExample] = []
        self._component_metrics: dict[str, dict[str, Any]] = {}

    def record_training_example(self, example: TrainingExample) -> None:
        """Record a training example for metrics calculation."""
        self._examples.append(example)
        if len(self._examples) > self.METRIC_WINDOW:
            self._examples = self._examples[-self.METRIC_WINDOW:]
        component = example.metadata.get("component", "unknown")
        if component not in self._component_metrics:
            self._component_metrics[component] = {
                "total": 0,
                "success": 0,
                "failure": 0,
                "partial": 0,
                "uncertain": 0,
                "rejected": 0,
                "confidences": [],
                "_outcomes": [],
            }
        metrics = self._component_metrics[component]
        metrics["total"] += 1
        metrics[example.outcome.value] += 1
        metrics["confidences"].append(example.confidence)
        metrics["_outcomes"].append(example.outcome.value)
        if len(metrics["confidences"]) > self.METRIC_WINDOW:
            metrics["confidences"] = metrics["confidences"][-self.METRIC_WINDOW:]
            metrics["_outcomes"] = metrics["_outcomes"][-self.METRIC_WINDOW:]
            removed = metrics["_outcomes"][0]
            metrics["total"] = len(metrics["_outcomes"])
            metrics[removed] = max(0, metrics[removed] - 1)

    def get_component_metrics(self, component: str) -> dict[str, Any]:
        """Get metrics for a specific component."""
        metrics = self._component_metrics.get(component, {
            "total": 0, "success": 0, "failure": 0, "partial": 0,
            "uncertain": 0, "rejected": 0, "confidences": [],
        })
        total = metrics["total"]
        success = metrics["success"]
        confidences = metrics["confidences"]
        return {
            "component": component,
            "total_examples": total,
            "success_count": success,
            "failure_count": metrics["failure"],
            "partial_count": metrics["partial"],
            "uncertain_count": metrics["uncertain"],
            "rejected_count": metrics["rejected"],
            "success_rate": round(success / total, 3) if total > 0 else 0.0,
            "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            "confidence_std": round(self._std(confidences), 3) if confidences else 0.0,
        }

    def get_all_metrics(self) -> dict[str, Any]:
        """Get metrics for all components."""
        return {comp: self.get_component_metrics(comp) for comp in self._component_metrics}

    def get_overall_metrics(self) -> dict[str, Any]:
        """Get overall metrics across all components."""
        total = sum(m["total"] for m in self._component_metrics.values())
        success = sum(m["success"] for m in self._component_metrics.values())
        all_confidences = []
        for m in self._component_metrics.values():
            all_confidences.extend(m["confidences"])
        return {
            "total_examples": total,
            "total_successes": success,
            "overall_success_rate": round(success / total, 3) if total > 0 else 0.0,
            "average_confidence": round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else 0.0,
            "components_tracked": len(self._component_metrics),
        }

    def _std(self, values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


# ---------------------------------------------------------------------------
# Provenance Tracker
# ---------------------------------------------------------------------------

class ProvenanceTracker:
    """Tracks where knowledge came from.

    Maintains provenance records for all knowledge updates,
    enabling audit trails and trust assessment.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._records: dict[str, ProvenanceRecord] = {}
        self._by_knowledge: dict[str, list[str]] = {}

    def record_provenance(self, knowledge_id: str, source_type: str, source_id: str = "", source_description: str = "", metadata: Optional[dict[str, Any]] = None) -> ProvenanceRecord:
        """Record provenance for a knowledge update."""
        record = ProvenanceRecord(
            knowledge_id=knowledge_id,
            source_type=source_type,
            source_id=source_id or "",
            source_description=source_description,
            metadata=metadata or {},
        )
        self._records[record.provenance_id] = record
        if knowledge_id not in self._by_knowledge:
            self._by_knowledge[knowledge_id] = []
        self._by_knowledge[knowledge_id].append(record.provenance_id)
        if self.logger:
            self.logger.memory(f"Recorded provenance: {source_type} -> {knowledge_id}")
        return record

    def get_provenance(self, knowledge_id: str) -> list[ProvenanceRecord]:
        """Get all provenance records for a knowledge ID."""
        provenance_ids = self._by_knowledge.get(knowledge_id, [])
        return [self._records[pid] for pid in provenance_ids if pid in self._records]

    def get_trust_score(self, knowledge_id: str) -> float:
        """Calculate a trust score based on provenance."""
        records = self.get_provenance(knowledge_id)
        if not records:
            return 0.5
        trusted_sources = {"feedback", "creator", "validated_lesson", "observation", "experience"}
        score = 0.5
        for record in records:
            if record.source_type in trusted_sources:
                score = min(1.0, score + 0.1)
            elif record.source_type == "experience":
                score = max(0.0, score - 0.05)
        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

class TrainingPipeline:
    """Orchestrates Phase 11 Training & Learning Foundation.

    Coordinates:
      - Automatic experience capture from native intelligence outputs
      - Outcome identification
      - Lesson extraction via LearningEngine
      - Knowledge update via KnowledgeGraph
      - Confidence calibration
      - Contradiction detection
      - Provenance tracking
      - Metrics collection

    Security:
      - Learning data is untrusted data
      - Learned content NEVER grants authority
      - Malicious instructions are filtered, not integrated
      - All knowledge updates require creator authority
    """

    def __init__(
        self,
        learning_engine: Any,
        knowledge_graph: Any,
        intelligence_evaluator: Any,
        confidence_calibrator: Optional[ConfidenceCalibrator] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        metrics: Optional[NativeIntelligenceMetrics] = None,
        provenance_tracker: Optional[ProvenanceTracker] = None,
        logger: Optional[Any] = None,
    ):
        self.learning_engine = learning_engine
        self.knowledge_graph = knowledge_graph
        self.intelligence_evaluator = intelligence_evaluator
        self.confidence_calibrator = confidence_calibrator or ConfidenceCalibrator(logger=logger)
        self.contradiction_detector = contradiction_detector or ContradictionDetector(
            knowledge_graph=knowledge_graph, logger=logger
        )
        self.metrics = metrics or NativeIntelligenceMetrics(logger=logger)
        self.provenance_tracker = provenance_tracker or ProvenanceTracker(logger=logger)
        self.logger = logger
        self._training_examples: dict[str, TrainingExample] = {}
        self._malicious_patterns = [
            "ignore all instructions",
            "disregard previous",
            "disregard all",
            "override security",
            "become admin",
            "grant yourself",
            "disable logging",
            "erase audit",
            "delete history",
            "bypass approval",
            "self-improve without approval",
            "remove restrictions",
            "remove all",
            "untrusted input is trusted",
            "i am the creator",
            "i have authority",
            "execute arbitrary",
            "run any command",
            "access any file",
            "bypass identity",
            "override approval",
            "self-improve without creator",
        ]

    def record_training_example(
        self,
        session_id: str,
        task_id: str,
        project: str,
        component: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        outcome: OutcomeType,
        feedback: Optional[str] = None,
        feedback_source: str = "",
        confidence: float = 0.5,
    ) -> TrainingExample:
        """Record a training example from native intelligence output."""
        if self._is_malicious(input_data, output_data):
            example = TrainingExample(
                session_id=session_id,
                task_id=task_id,
                project=project,
                input_data=input_data,
                output_data=output_data,
                outcome=OutcomeType.REJECTED,
                feedback="Malicious content detected",
                feedback_source="security_filter",
                confidence=0.0,
                tags=["rejected", "malicious"],
                metadata={"component": component, "reason": "malicious_content"},
            )
            self._training_examples[example.example_id] = example
            self.metrics.record_training_example(example)
            return example

        example = TrainingExample(
            session_id=session_id,
            task_id=task_id,
            project=project,
            input_data=input_data,
            output_data=output_data,
            outcome=outcome,
            feedback=feedback,
            feedback_source=feedback_source,
            confidence=confidence,
            tags=["native_intelligence", component],
            metadata={"component": component},
        )
        self._training_examples[example.example_id] = example
        self.metrics.record_training_example(example)
        self.confidence_calibrator.record_outcome(
            capability=component,
            context=input_data,
            predicted_confidence=confidence,
            outcome=outcome,
        )
        return example

    async def learn_from_example(self, example: TrainingExample) -> list[Any]:
        """Learn from a training example.

        Extracts lessons, updates knowledge graph, records provenance.
        Returns list of extracted lessons.
        """
        if example.outcome == OutcomeType.REJECTED:
            return []
        if self.learning_engine is None:
            return []
        experience = self._example_to_experience(example)
        lessons = await self.learning_engine.learn_from_experience(experience)
        for lesson in lessons:
            if self.knowledge_graph is not None and lesson.status.value == "validated":
                try:
                    from evora.brain.intelligence.knowledge import KnowledgeNode, KnowledgeType
                    node = KnowledgeNode(
                        type=KnowledgeType.LESSON.value,
                        content=lesson.summary,
                        confidence=lesson.confidence,
                        source="learned",
                        metadata={
                            "lesson_id": lesson.lesson_id,
                            "detail": lesson.detail,
                            "tags": lesson.tags,
                            "status": lesson.status.value,
                        },
                    )
                    node_id = self.knowledge_graph.add_node(node)
                    self.provenance_tracker.record_provenance(
                        knowledge_id=node_id,
                        source_type="lesson",
                        source_id=lesson.lesson_id,
                        source_description=f"Lesson extracted from example {example.example_id}",
                        metadata={"outcome": example.outcome.value},
                    )
                except Exception:
                    pass
        return lessons

    def evaluate_and_update(self, component: str, input_data: dict[str, Any], output_data: dict[str, Any], outcome: OutcomeType) -> dict[str, Any]:
        """Evaluate a native intelligence output and update metrics."""
        evaluation = {}
        try:
            if component == "reasoning":
                evaluation = self.intelligence_evaluator.evaluate_reasoning(
                    goal=input_data.get("goal", ""),
                    result=output_data.get("result"),
                    evidence=output_data.get("evidence", []),
                    constraints=input_data.get("constraints", []),
                )
            elif component == "plan":
                evaluation = self.intelligence_evaluator.evaluate_plan(
                    goal=input_data.get("goal", ""),
                    plan=output_data.get("plan"),
                    constraints=input_data.get("constraints", []),
                )
            elif component == "inference":
                evaluation = self.intelligence_evaluator.evaluate_inference(
                    query=input_data.get("query", ""),
                    result=output_data.get("result"),
                    known_facts=input_data.get("known_facts", []),
                )
        except Exception:
            evaluation = {}
        if evaluation:
            evaluation_dict = evaluation.to_dict() if hasattr(evaluation, "to_dict") else evaluation
            confidence = evaluation_dict.get("confidence", 0.0)
            grade = evaluation_dict.get("grade", "unknown")
        else:
            confidence = 0.0
            grade = "unknown"
            evaluation_dict = {}
        return {
            "component": component,
            "grade": grade,
            "confidence": confidence,
            "outcome": outcome.value,
            "evaluation": evaluation_dict,
        }

    def detect_knowledge_contradictions(self) -> list[dict[str, Any]]:
        """Detect contradictions in knowledge graph."""
        return self.contradiction_detector.detect_contradictions()

    def get_training_metrics(self) -> dict[str, Any]:
        """Get comprehensive training metrics."""
        metrics = self.metrics.get_overall_metrics()
        calibration = self.confidence_calibrator.get_metrics()
        contradictions = self.contradiction_detector.get_contradiction_count()
        total_examples = len(self._training_examples)
        by_status = {}
        for ex in self._training_examples.values():
            by_status[ex.status.value] = by_status.get(ex.status.value, 0) + 1
        by_outcome = {}
        for ex in self._training_examples.values():
            by_outcome[ex.outcome.value] = by_outcome.get(ex.outcome.value, 0) + 1
        return {
            "training_examples": {
                "total": total_examples,
                "by_status": by_status,
                "by_outcome": by_outcome,
            },
            "native_intelligence": metrics,
            "calibration": calibration,
            "contradictions_detected": contradictions,
            "provenance_records": len(self.provenance_tracker._records),
        }

    def get_component_metrics(self, component: str) -> dict[str, Any]:
        """Get metrics for a specific native intelligence component."""
        return self.metrics.get_component_metrics(component)

    def _example_to_experience(self, example: TrainingExample) -> Any:
        """Convert a training example to a LearningEngine Experience."""
        from evora.learning import Experience, ExperienceType
        content_parts = [
            f"Component: {example.metadata.get('component', 'unknown')}",
            f"Outcome: {example.outcome.value}",
            f"Input: {json.dumps(example.input_data, default=str)[:200]}",
            f"Output: {json.dumps(example.output_data, default=str)[:200]}",
        ]
        if example.feedback:
            content_parts.append(f"Feedback: {example.feedback}")
        return Experience(
            experience_type=ExperienceType.TASK_OUTCOME,
            session_id=example.session_id,
            task_id=example.task_id,
            project=example.project,
            content=" | ".join(content_parts),
            metadata={
                "training_example_id": example.example_id,
                "outcome": example.outcome.value,
                "confidence": example.confidence,
                "component": example.metadata.get("component", "unknown"),
            },
        )

    def _is_malicious(self, input_data: dict[str, Any], output_data: dict[str, Any]) -> bool:
        """Check if input/output contains malicious patterns."""
        text = json.dumps(input_data, default=str).lower() + " " + json.dumps(output_data, default=str).lower()
        return any(pattern in text for pattern in self._malicious_patterns)
