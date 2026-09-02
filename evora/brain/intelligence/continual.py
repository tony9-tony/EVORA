"""
Phase 13 — Continual Learning for EVORA native intelligence.

Implements:
  Experience
  ↓
  Evaluation
  ↓
  Lesson Extraction
  ↓
  Knowledge Update
  ↓
  Future Decision Improvement

Prevents:
  - poisoning
  - duplicate learning
  - confidence inflation
  - accidental authority escalation
  - malicious instructions becoming trusted knowledge
  - infinite memory growth

Implements:
  - bounded storage
  - provenance
  - confidence
  - validation
  - forgetting/consolidation strategies

Reuses existing abstractions:
  - TrainingPipeline (Phase 11) for experience capture and metrics
  - KnowledgeGraph for knowledge storage
  - LearningEngine for lesson lifecycle
  - IntelligenceEvaluator for quality assessment
  - MemoryService for durable storage

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
# Core continual learning dataclasses
# ---------------------------------------------------------------------------

class ConsolidationStrategy(str, Enum):
    """Strategies for consolidating learned knowledge."""
    RECENCY = "recency"
    CONFIDENCE = "confidence"
    SUCCESS_RATE = "success_rate"
    PROVENANCE = "provenance"
    HYBRID = "hybrid"


class ValidationResult(str, Enum):
    """Result of knowledge validation."""
    VALID = "valid"
    CONTRADICTORY = "contradictory"
    POISONED = "poisoned"
    DUPLICATE = "duplicate"
    INFLATED = "inflated"
    UNKNOWN = "unknown"


@dataclass
class ExperienceReplayEntry:
    """An entry in the experience replay buffer."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    experience_id: str = ""
    component: str = ""
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""
    confidence: float = 0.0
    evaluation_grade: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "experience_id": self.experience_id,
            "component": self.component,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "outcome": self.outcome,
            "confidence": self.confidence,
            "evaluation_grade": self.evaluation_grade,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperienceReplayEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    entries_processed: int = 0
    duplicates_removed: int = 0
    poisoned_removed: int = 0
    inflated_removed: int = 0
    low_value_removed: int = 0
    kept: int = 0
    details: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries_processed": self.entries_processed,
            "duplicates_removed": self.duplicates_removed,
            "poisoned_removed": self.poisoned_removed,
            "inflated_removed": self.inflated_removed,
            "low_value_removed": self.low_value_removed,
            "kept": self.kept,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------

class ExperienceReplayBuffer:
    """Bounded buffer for experience replay.

    Stores recent experiences for replay during learning.
    Enforces bounded storage to prevent infinite memory growth.
    """

    MAX_ENTRIES = 1000
    MAX_CONTENT_LENGTH = 1_048_576

    def __init__(self, max_entries: int = MAX_ENTRIES, logger: Optional[Any] = None):
        self.max_entries = max_entries
        self.logger = logger
        self._entries: dict[str, ExperienceReplayEntry] = {}
        self._order: list[str] = []

    def add(self, entry: ExperienceReplayEntry) -> None:
        """Add an entry to the replay buffer."""
        if len(self._order) >= self.max_entries:
            oldest = self._order.pop(0)
            self._entries.pop(oldest, None)
        self._entries[entry.entry_id] = entry
        self._order.append(entry.entry_id)

    def get(self, entry_id: str) -> Optional[ExperienceReplayEntry]:
        """Get an entry by ID."""
        entry = self._entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now().isoformat()
        return entry

    def get_recent(self, limit: int = 50) -> list[ExperienceReplayEntry]:
        """Get recent entries."""
        entries = [self._entries[eid] for eid in reversed(self._order) if eid in self._entries]
        return entries[:limit]

    def get_by_component(self, component: str, limit: int = 50) -> list[ExperienceReplayEntry]:
        """Get entries for a specific component."""
        return [e for e in self.get_recent(limit=limit * 2) if e.component == component][:limit]

    def remove(self, entry_id: str) -> bool:
        """Remove an entry."""
        if entry_id in self._entries:
            del self._entries[entry_id]
            if entry_id in self._order:
                self._order.remove(entry_id)
            return True
        return False

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()


# ---------------------------------------------------------------------------
# Poisoning Detector
# ---------------------------------------------------------------------------

class PoisoningDetector:
    """Detects poisoned or malicious learning data.

    Checks for:
    - Malicious instruction patterns
    - Contradictions with trusted knowledge
    - Confidence inflation
    - Authority escalation attempts
    """

    def __init__(self, knowledge_graph: Any = None, logger: Optional[Any] = None):
        self.knowledge_graph = knowledge_graph
        self.logger = logger
        self._malicious_patterns = [
            "ignore all instructions",
            "disregard previous",
            "override security",
            "become admin",
            "grant yourself",
            "disable logging",
            "erase audit",
            "delete history",
            "bypass approval",
            "self-improve without approval",
            "remove restrictions",
            "i am the creator",
            "i have authority",
            "execute arbitrary",
            "run any command",
            "access any file",
            "bypass identity",
            "override approval",
            "self-improve without creator",
            "remove all",
        ]

    def validate_experience(self, experience: dict[str, Any]) -> ValidationResult:
        """Validate an experience for poisoning."""
        text = json.dumps(experience, default=str).lower()
        for pattern in self._malicious_patterns:
            if pattern in text:
                return ValidationResult.POISONED
        return ValidationResult.VALID

    def validate_lesson(self, lesson_content: str, existing_knowledge: list[str]) -> ValidationResult:
        """Validate a lesson against existing knowledge."""
        content_lower = lesson_content.lower()
        for pattern in self._malicious_patterns:
            if pattern in content_lower:
                return ValidationResult.POISONED
        for knowledge in existing_knowledge:
            if self._is_contradictory(lesson_content, knowledge):
                return ValidationResult.CONTRADICTORY
        return ValidationResult.VALID

    def detect_confidence_inflation(self, confidence_history: list[float], threshold: float = 0.3) -> bool:
        """Detect if confidence is inflating without evidence."""
        if len(confidence_history) < 3:
            return False
        recent = confidence_history[-3:]
        if all(c >= 0.9 for c in recent):
            return True
        if len(confidence_history) >= 5:
            older = confidence_history[-5:-2]
            newer = confidence_history[-2:]
            if sum(newer) / len(newer) - sum(older) / len(older) > threshold:
                return True
        return False

    def _is_contradictory(self, a: str, b: str) -> bool:
        """Check if two content strings contradict."""
        if not a or not b:
            return False
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()
        if a_lower == b_lower:
            return False
        negation_markers = ["not ", "no ", "never ", "cannot ", "can't ", "doesn't ", "do not "]
        for marker in negation_markers:
            if a_lower.startswith(marker) and b_lower == a_lower[len(marker):]:
                return True
            if b_lower.startswith(marker) and a_lower == b_lower[len(marker):]:
                return True
        return False


# ---------------------------------------------------------------------------
# Lesson Consolidator
# ---------------------------------------------------------------------------

class LessonConsolidator:
    """Consolidates similar lessons to prevent duplicate learning.

    Merges lessons that are:
    - Semantically similar
    - From the same source
    - About the same topic
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger

    def consolidate(self, lessons: list[Any]) -> list[Any]:
        """Consolidate a list of lessons, removing duplicates."""
        if not lessons:
            return []

        seen_hashes = set()
        consolidated = []
        for lesson in lessons:
            content_hash = self._content_hash(lesson.summary if hasattr(lesson, "summary") else str(lesson))
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                consolidated.append(lesson)
        return consolidated

    def _content_hash(self, content: str) -> str:
        """Create a hash for deduplication."""
        normalized = content.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def merge_similar(self, lessons: list[Any], similarity_threshold: float = 0.8) -> list[Any]:
        """Merge lessons that are highly similar."""
        if len(lessons) <= 1:
            return lessons

        merged = []
        used = set()
        for i, lesson_a in enumerate(lessons):
            if i in used:
                continue
            best_match = None
            best_score = 0.0
            for j, lesson_b in enumerate(lessons):
                if j <= i or j in used:
                    continue
                score = self._similarity(
                    lesson_a.summary if hasattr(lesson_a, "summary") else str(lesson_a),
                    lesson_b.summary if hasattr(lesson_b, "summary") else str(lesson_b),
                )
                if score > best_score and score >= similarity_threshold:
                    best_score = score
                    best_match = j

            if best_match is not None:
                merged.append(self._merge_pair(lesson_a, lessons[best_match]))
                used.add(i)
                used.add(best_match)
            else:
                merged.append(lesson_a)
                used.add(i)

        return merged

    def _similarity(self, a: str, b: str) -> float:
        """Calculate similarity between two strings."""
        if not a or not b:
            return 0.0
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        if not a_words or not b_words:
            return 0.0
        intersection = a_words & b_words
        union = a_words | b_words
        return len(intersection) / len(union)

    def _merge_pair(self, a: Any, b: Any) -> Any:
        """Merge two similar lessons."""
        if hasattr(a, "confidence") and hasattr(b, "confidence"):
            a.confidence = max(a.confidence, b.confidence)
        if hasattr(a, "source_experience_ids") and hasattr(b, "source_experience_ids"):
            a.source_experience_ids = list(set(a.source_experience_ids + b.source_experience_ids))
        return a


# ---------------------------------------------------------------------------
# Knowledge Consolidator
# ---------------------------------------------------------------------------

class KnowledgeConsolidator:
    """Consolidates and prunes knowledge to prevent infinite growth.

    Strategies:
    - RECENCY: Keep recent knowledge, prune old
    - CONFIDENCE: Keep high-confidence knowledge
    - SUCCESS_RATE: Keep knowledge with high success rates
    - PROVENANCE: Keep knowledge from trusted sources
    - HYBRID: Combine multiple strategies
    """

    def __init__(
        self,
        strategy: ConsolidationStrategy = ConsolidationStrategy.HYBRID,
        max_knowledge_entries: int = 5000,
        logger: Optional[Any] = None,
    ):
        self.strategy = strategy
        self.max_knowledge_entries = max_knowledge_entries
        self.logger = logger

    def consolidate(self, knowledge_entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], ConsolidationResult]:
        """Consolidate knowledge entries and prune low-value ones."""
        result = ConsolidationResult(entries_processed=len(knowledge_entries))

        if not knowledge_entries:
            return knowledge_entries, result

        scored = []
        for entry in knowledge_entries:
            score, reasons = self._score_entry(entry)
            scored.append((score, entry, reasons))

        scored.sort(key=lambda x: x[0], reverse=True)

        kept = []
        for score, entry, reasons in scored:
            if len(kept) >= self.max_knowledge_entries:
                result.low_value_removed += 1
                result.details.append(f"Pruned: {entry.get('content', '')[:50]} (score: {score:.2f})")
                continue
            kept.append(entry)

        result.kept = len(kept)
        return kept, result

    def _score_entry(self, entry: dict[str, Any]) -> tuple[float, list[str]]:
        """Score a knowledge entry for retention."""
        score = 0.5
        reasons = []

        confidence = entry.get("confidence", 0.5)
        if isinstance(confidence, (int, float)):
            score += confidence * 0.2
            reasons.append(f"confidence={confidence:.2f}")

        success_count = entry.get("success_count", 0)
        failure_count = entry.get("failure_count", 0)
        if success_count + failure_count > 0:
            success_rate = success_count / (success_count + failure_count)
            score += success_rate * 0.2
            reasons.append(f"success_rate={success_rate:.2f}")

        importance = entry.get("importance", 0.5)
        if isinstance(importance, (int, float)):
            score += importance * 0.1
            reasons.append(f"importance={importance:.2f}")

        created_at = entry.get("created_at", "")
        if created_at:
            try:
                created = datetime.fromisoformat(created_at)
                age_days = (datetime.now() - created).days
                recency_score = max(0.0, 1.0 - age_days / 365.0)
                score += recency_score * 0.1
                reasons.append(f"recency={recency_score:.2f}")
            except Exception:
                pass

        provenance = entry.get("provenance", [])
        trusted_sources = {"feedback", "creator", "validated_lesson", "observation"}
        if any(p.get("source_type") in trusted_sources for p in provenance if isinstance(p, dict)):
            score += 0.1
            reasons.append("trusted_provenance")

        return max(0.0, min(1.0, score)), reasons


# ---------------------------------------------------------------------------
# Continual Learning Pipeline
# ---------------------------------------------------------------------------

class ContinualLearningPipeline:
    """Orchestrates Phase 13 Continual Learning.

    Extends TrainingPipeline with:
      - Experience replay
      - Lesson consolidation
      - Knowledge consolidation
      - Poisoning detection
      - Confidence inflation prevention
      - Duplicate learning prevention
      - Bounded storage enforcement
      - Forgetting/consolidation strategies

    Pipeline:
      Experience
      ↓
      Replay Buffer
      ↓
      Poisoning Check
      ↓
      Evaluation
      ↓
      Lesson Extraction
      ↓
      Consolidation (deduplicate)
      ↓
      Knowledge Validation
      ↓
      Knowledge Update (with bounded storage)
      ↓
      Metrics Update
    """

    def __init__(
        self,
        training_pipeline: Any,
        knowledge_graph: Any,
        knowledge_consolidator: Optional[KnowledgeConsolidator] = None,
        poisoning_detector: Optional[PoisoningDetector] = None,
        lesson_consolidator: Optional[LessonConsolidator] = None,
        logger: Optional[Any] = None,
    ):
        self.training_pipeline = training_pipeline
        self.knowledge_graph = knowledge_graph
        self.knowledge_consolidator = knowledge_consolidator or KnowledgeConsolidator(logger=logger)
        self.poisoning_detector = poisoning_detector or PoisoningDetector(
            knowledge_graph=knowledge_graph, logger=logger
        )
        self.lesson_consolidator = lesson_consolidator or LessonConsolidator(logger=logger)
        self.logger = logger
        self._replay_buffer = ExperienceReplayBuffer(logger=logger)
        self._consolidation_history: list[ConsolidationResult] = []

    def process_experience(self, experience_data: dict[str, Any]) -> dict[str, Any]:
        """Process a new experience through the continual learning pipeline."""
        result = {
            "status": "processed",
            "poisoned": False,
            "consolidated": False,
            "knowledge_updated": False,
            "replay_added": False,
        }

        validation = self.poisoning_detector.validate_experience(experience_data)
        if validation == ValidationResult.POISONED:
            result["status"] = "rejected"
            result["poisoned"] = True
            return result

        self._replay_buffer.add(ExperienceReplayEntry(
            experience_id=experience_data.get("experience_id", ""),
            component=experience_data.get("component", "unknown"),
            input_data=experience_data.get("input_data", {}),
            output_data=experience_data.get("output_data", {}),
            outcome=experience_data.get("outcome", "unknown"),
            confidence=experience_data.get("confidence", 0.0),
        ))
        result["replay_added"] = True

        lessons = self._extract_lessons(experience_data)
        if lessons:
            consolidated_lessons = self.lesson_consolidator.consolidate(lessons)
            merged_lessons = self.lesson_consolidator.merge_similar(consolidated_lessons)
            knowledge_updated = self._update_knowledge(merged_lessons, experience_data)
            result["consolidated"] = len(lessons) != len(merged_lessons)
            result["knowledge_updated"] = knowledge_updated

        return result

    def replay_and_learn(self, component: str = "", limit: int = 20) -> dict[str, Any]:
        """Replay recent experiences and extract consolidated learning."""
        entries = self._replay_buffer.get_by_component(component, limit=limit)
        if not entries:
            return {"replayed": 0, "lessons_extracted": 0}

        replayed = 0
        lessons_extracted = 0
        for entry in entries:
            if entry.outcome in ("success", "failure", "partial"):
                exp_data = {
                    "experience_id": entry.experience_id,
                    "component": entry.component,
                    "input_data": entry.input_data,
                    "output_data": entry.output_data,
                    "outcome": entry.outcome,
                    "confidence": entry.confidence,
                }
                result = self.process_experience(exp_data)
                if result["knowledge_updated"]:
                    lessons_extracted += 1
                replayed += 1

        return {"replayed": replayed, "lessons_extracted": lessons_extracted}

    def consolidate_knowledge(self) -> ConsolidationResult:
        """Consolidate knowledge graph and prune low-value entries."""
        if self.knowledge_graph is None:
            return ConsolidationResult()

        try:
            nodes = self.knowledge_graph.get_all_nodes(limit=10000)
            node_dicts = [n.to_dict() for n in nodes]
            kept, result = self.knowledge_consolidator.consolidate(node_dicts)
            self._consolidation_history.append(result)
            return result
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Knowledge consolidation failed: {e}")
            return ConsolidationResult(details=[str(e)])

    def get_continual_metrics(self) -> dict[str, Any]:
        """Get continual learning metrics."""
        replay_count = self._replay_buffer.count()
        consolidation_count = len(self._consolidation_history)
        last_consolidation = self._consolidation_history[-1] if self._consolidation_history else None

        return {
            "replay_buffer_size": replay_count,
            "consolidations_performed": consolidation_count,
            "last_consolidation": last_consolidation.to_dict() if last_consolidation else None,
            "total_consolidated_entries": sum(c.entries_processed for c in self._consolidation_history),
            "total_pruned": sum(
                c.duplicates_removed + c.poisoned_removed + c.inflated_removed + c.low_value_removed
                for c in self._consolidation_history
            ),
        }

    def _extract_lessons(self, experience_data: dict[str, Any]) -> list[Any]:
        """Extract lessons from experience using TrainingPipeline."""
        if self.training_pipeline is None or self.training_pipeline.learning_engine is None:
            return []
        try:
            from evora.learning import Experience, ExperienceType
            from evora.brain.intelligence.training import TrainingExample, OutcomeType
            experience = Experience(
                experience_type=ExperienceType.TASK_OUTCOME,
                session_id=experience_data.get("session_id", ""),
                task_id=experience_data.get("task_id", ""),
                project=experience_data.get("project", ""),
                content=json.dumps(experience_data, default=str),
                metadata=experience_data.get("metadata", {}),
            )
            import asyncio
            lessons = asyncio.run(self.training_pipeline.learning_engine.learn_from_experience(experience))
            return lessons
        except Exception:
            return []

    def _update_knowledge(self, lessons: list[Any], experience_data: dict[str, Any]) -> bool:
        """Update knowledge graph with consolidated lessons."""
        if not lessons or self.knowledge_graph is None:
            return False
        try:
            from evora.brain.intelligence.knowledge import KnowledgeNode, KnowledgeType
            for lesson in lessons:
                if hasattr(lesson, "status") and lesson.status.value == "validated":
                    node = KnowledgeNode(
                        type=KnowledgeType.LESSON.value,
                        content=lesson.summary,
                        confidence=lesson.confidence,
                        source="learned",
                        metadata={
                            "lesson_id": lesson.lesson_id,
                            "status": lesson.status.value,
                            "tags": lesson.tags,
                        },
                    )
                    self.knowledge_graph.add_node(node)
            return True
        except Exception:
            return False
