"""
Phase 8 — Advanced Learning layer for EVORA.

Provides structured experience capture, lesson extraction, feedback handling,
knowledge formation, and measurable learning improvement.

Architecture:
  Experience  — ephemeral observation tied to a specific task/session
  Lesson      — extracted insight from one or more experiences
  Feedback    — explicit correction from creator/user on a lesson
  Knowledge   — durable, validated understanding ready for reuse

Separation of concerns:
  - Experience/knowledge are distinct from configuration and code
  - Experiences are ephemeral and scoped to a session/project
  - Knowledge is durable, auditable, and authority-gated
  - Learning never bypasses creator approval for self-modification

Reuses existing abstractions:
  - MemoryService / LongTermMemoryEntry for durable storage
  - ReasoningEngine for structured lesson extraction
  - ApprovalSystem for creator feedback on lessons
  - IdentityService for authority checks
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from evora.logger import Logger
from evora.memory import MemoryFilter


# ---------------------------------------------------------------------------
# Core learning dataclasses
# ---------------------------------------------------------------------------

class ExperienceType(str, Enum):
    """Kinds of experiences EVORA can capture."""
    TASK_OUTCOME = "task_outcome"
    SELF_DEVELOPMENT = "self_development"
    TOOL_EXECUTION = "tool_execution"
    MODEL_INTERACTION = "model_interaction"
    APPROVAL_FEEDBACK = "approval_feedback"
    ERROR_RECOVERY = "error_recovery"


class LessonStatus(str, Enum):
    """Lifecycle of a lesson."""
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    INTEGRATED = "integrated"
    SUPERSEDED = "superseded"


class FeedbackType(str, Enum):
    """Kinds of creator feedback on lessons."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    CRITICISM = "criticism"


@dataclass
class Experience:
    """Ephemeral observation tied to a specific task/session.

    Experiences are raw material for learning. They are not durable knowledge
    and are subject to retention policies.
    """
    experience_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    experience_type: ExperienceType = ExperienceType.TASK_OUTCOME
    session_id: str = ""
    task_id: str = ""
    project: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.experience_id,
            "experience_type": self.experience_type.value,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "project": self.project,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        data = dict(data)
        data["experience_type"] = ExperienceType(data.get("experience_type", "task_outcome"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Lesson:
    """Extracted insight from one or more experiences.

    Lessons are proposals for durable knowledge. They require validation
    before becoming knowledge.
    """
    lesson_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_experience_ids: list[str] = field(default_factory=list)
    summary: str = ""
    detail: str = ""
    tags: list[str] = field(default_factory=list)
    status: LessonStatus = LessonStatus.PROPOSED
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    validated_at: str = ""
    rejected_at: str = ""
    rejection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "source_experience_ids": self.source_experience_ids,
            "summary": self.summary,
            "detail": self.detail,
            "tags": self.tags,
            "status": self.status.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "validated_at": self.validated_at,
            "rejected_at": self.rejected_at,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lesson":
        data = dict(data)
        data["status"] = LessonStatus(data.get("status", "proposed"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Feedback:
    """Explicit creator/user feedback on a lesson."""
    feedback_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    lesson_id: str = ""
    feedback_type: FeedbackType = FeedbackType.APPROVE
    content: str = ""
    provided_by: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "lesson_id": self.lesson_id,
            "feedback_type": self.feedback_type.value,
            "content": self.content,
            "provided_by": self.provided_by,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Feedback":
        data = dict(data)
        data["feedback_type"] = FeedbackType(data.get("feedback_type", "approve"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Knowledge:
    """Durable, validated understanding ready for retrieval and reuse.

    Knowledge is the output of successful learning. It is stored via
    MemoryService as a LongTermMemoryEntry for persistence and retrieval.
    """
    knowledge_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_lesson_ids: list[str] = field(default_factory=list)
    content: str = ""
    memory_type: str = "learning"
    importance: float = 0.5
    project: str = ""
    tags: list[str] = field(default_factory=list)
    pinned: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_applied: str = ""
    application_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_id": self.knowledge_id,
            "source_lesson_ids": self.source_lesson_ids,
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "project": self.project,
            "tags": self.tags,
            "pinned": self.pinned,
            "created_at": self.created_at,
            "last_applied": self.last_applied,
            "application_count": self.application_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Knowledge":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def record_application(self, success: bool) -> None:
        self.application_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_applied = datetime.now().isoformat()
        self.importance = min(1.0, max(0.0, self.importance + (0.05 if success else -0.1)))

    @property
    def success_rate(self) -> float:
        if self.application_count == 0:
            return 0.0
        return self.success_count / self.application_count


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------

class ExperienceStore:
    """Ephemeral store for recent experiences.

    Retention policy: experiences older than retention_days are purged.
    Not intended as long-term durable storage.
    """
    MAX_CONTENT_LENGTH = 1_048_576
    MAX_METADATA_LENGTH = 262_144

    def __init__(self, data_dir: str, retention_days: int = 30):
        self.data_dir = Path(data_dir)
        self.experiences_dir = self.data_dir / "experiences"
        self.experiences_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days

    def _path(self, experience_id: str) -> Path:
        if not isinstance(experience_id, str) or not experience_id:
            raise ValueError("experience_id must be a non-empty string")
        path = (self.experiences_dir / f"{experience_id}.json").resolve()
        if path.parent != self.experiences_dir.resolve():
            raise ValueError("experience_id must not escape the experience store")
        return path

    def record(self, experience: Experience) -> str:
        if not isinstance(experience.content, str) or len(experience.content) > self.MAX_CONTENT_LENGTH:
            raise ValueError("experience content exceeds the maximum size")
        metadata_size = len(json.dumps(experience.metadata, ensure_ascii=False))
        if metadata_size > self.MAX_METADATA_LENGTH:
            raise ValueError("experience metadata exceeds the maximum size")
        path = self._path(experience.experience_id)
        with open(path, "w", encoding="utf-8") as f:
            data = experience.to_dict()
            data["content"] = MemoryFilter.sanitize(experience.content)
            data["metadata"] = MemoryFilter.sanitize_dict(experience.metadata)
            json.dump(data, f, indent=2, ensure_ascii=False)
        return experience.experience_id

    def get(self, experience_id: str) -> Optional[Experience]:
        path = self._path(experience_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return Experience.from_dict(json.load(f))

    def list_recent(self, limit: int = 50, project: str = "") -> list[Experience]:
        experiences = []
        for path in sorted(self.experiences_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                exp = Experience.from_dict(data)
                if project and exp.project != project:
                    continue
                experiences.append(exp)
                if len(experiences) >= limit:
                    break
            except Exception:
                continue
        return experiences

    def purge_old(self) -> int:
        cutoff = time.time() - (self.retention_days * 24 * 3600)
        purged = 0
        for path in list(self.experiences_dir.glob("*.json")):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    purged += 1
            except OSError:
                continue
        return purged

    def count(self, project: str = "") -> int:
        return len(self.list_recent(limit=99999, project=project))


class KnowledgeBase:
    """Durable knowledge store backed by MemoryService.

    All knowledge is stored as LongTermMemoryEntry with memory_type='learning'
    or 'instruction'. This preserves the existing memory architecture while
    adding Phase 8 semantics.
    """

    def __init__(
        self,
        memory_service: Any,
        logger: Optional[Logger] = None,
        identity_service: Any = None,
    ):
        self.memory_service = memory_service
        self.logger = logger
        self.identity_service = identity_service or getattr(memory_service, "identity_service", None)

    def store_knowledge(self, knowledge: Knowledge) -> str:
        if self.identity_service is None:
            raise PermissionError("Creator authority is required to store knowledge")
        self.identity_service.require_authority("enable_self_modification")
        entry = self.memory_service.remember(
            content=knowledge.content,
            memory_type=knowledge.memory_type,
            importance=knowledge.importance,
            project=knowledge.project or None,
            tags=knowledge.tags,
            pinned=knowledge.pinned,
        )
        if self.logger:
            self.logger.memory(f"Stored knowledge: {knowledge.content[:80]}")
        return entry.id

    def retrieve_relevant(self, goal: str, project: str = "", limit: int = 10) -> list[dict]:
        results = self.memory_service.retrieve_relevant(
            goal=goal,
            project=project or None,
            memory_types=["learning", "instruction"],
            limit=limit,
        )
        return [r.to_dict() for r in results]

    def record_feedback_on_knowledge(self, knowledge_id: str, feedback: Feedback) -> None:
        if self.identity_service is None:
            raise PermissionError("Creator authority is required to modify knowledge")
        self.identity_service.require_authority("enable_self_modification")
        entry = self.memory_service.store.load_ltm_entry(knowledge_id)
        if entry is None:
            return
        if feedback.feedback_type == FeedbackType.APPROVE:
            entry.importance = min(1.0, entry.importance + 0.05)
        elif feedback.feedback_type in (FeedbackType.REJECT, FeedbackType.CRITICISM):
            entry.importance = max(0.0, entry.importance - 0.1)
        self.memory_service.store.save_ltm_entry(entry)


# ---------------------------------------------------------------------------
# Lesson extraction
# ---------------------------------------------------------------------------

class LessonExtractor:
    """Extract structured lessons from raw experiences.

    Uses lightweight heuristics + optional model reasoning to convert
    raw observations into actionable lessons.
    """

    FAILURE_INDICATORS = [
        "error", "failed", "failure", "exception", "traceback",
        "rejected", "denied", "timeout", "rolled back",
    ]

    SUCCESS_INDICATORS = [
        "success", "passed", "approved", "completed", "validated",
    ]

    def __init__(self, model_manager: Any = None, logger: Optional[Logger] = None):
        self.model_manager = model_manager
        self.logger = logger

    def extract(self, experience: Experience) -> list[Lesson]:
        """Extract one or more lessons from a single experience."""
        lessons = []
        content_lower = experience.content.lower()
        is_failure = any(ind in content_lower for ind in self.FAILURE_INDICATORS)
        is_success = any(ind in content_lower for ind in self.SUCCESS_INDICATORS)

        if is_failure and not is_success:
            lessons.append(self._extract_failure_lesson(experience))
        elif is_success and not is_failure:
            lessons.append(self._extract_success_lesson(experience))
        else:
            lessons.append(self._extract_neutral_lesson(experience))

        if self.logger:
            self.logger.memory(f"Extracted {len(lessons)} lesson(s) from experience {experience.experience_id}")
        return lessons

    def _extract_failure_lesson(self, experience: Experience) -> Lesson:
        summary = f"Failure observed: {experience.content[:120]}"
        detail = (
            f"Experience {experience.experience_id} ({experience.experience_type.value}) "
            f"reported a failure. Investigate root cause before repeating similar actions."
        )
        return Lesson(
            source_experience_ids=[experience.experience_id],
            summary=summary,
            detail=detail,
            tags=["failure", experience.experience_type.value, experience.project],
            status=LessonStatus.PROPOSED,
            confidence=0.6,
            metadata={"source": "heuristic", "experience_type": experience.experience_type.value},
        )

    def _extract_success_lesson(self, experience: Experience) -> Lesson:
        summary = f"Success pattern: {experience.content[:120]}"
        detail = (
            f"Experience {experience.experience_id} ({experience.experience_type.value}) "
            f"succeeded. Consider generalizing this pattern for future tasks."
        )
        return Lesson(
            source_experience_ids=[experience.experience_id],
            summary=summary,
            detail=detail,
            tags=["success", experience.experience_type.value, experience.project],
            status=LessonStatus.PROPOSED,
            confidence=0.7,
            metadata={"source": "heuristic", "experience_type": experience.experience_type.value},
        )

    def _extract_neutral_lesson(self, experience: Experience) -> Lesson:
        summary = f"Observation: {experience.content[:120]}"
        detail = (
            f"Experience {experience.experience_id} ({experience.experience_type.value}) "
            f"recorded. Review for potential learning value."
        )
        return Lesson(
            source_experience_ids=[experience.experience_id],
            summary=summary,
            detail=detail,
            tags=["observation", experience.experience_type.value, experience.project],
            status=LessonStatus.PROPOSED,
            confidence=0.4,
            metadata={"source": "heuristic", "experience_type": experience.experience_type.value},
        )

    async def extract_with_reasoning(self, experience: Experience, context: dict[str, Any]) -> list[Lesson]:
        """Use model reasoning to extract lessons (preferred when model is available)."""
        if self.model_manager is None:
            return self.extract(experience)

        try:
            from evora.reasoning import ReasoningEngine, ReasoningContext
            engine = ReasoningEngine(self.model_manager, self.logger)
            ctx = ReasoningContext(
                objective=f"Extract a lesson from this experience: {experience.content[:200]}",
                observations=[
                    f"Experience type: {experience.experience_type.value}",
                    f"Project: {experience.project or 'unknown'}",
                    f"Session: {experience.session_id}",
                    f"Metadata: {json.dumps(experience.metadata)[:500]}",
                ],
                constraints=[
                    "Lesson must be actionable",
                    "Lesson must be generalizable beyond this single event",
                    "Do not store secrets or sensitive data",
                ],
            )
            result = await engine.reason(ctx)
            if result.confidence >= 0.5 and result.next_action != "abort":
                lesson = Lesson(
                    source_experience_ids=[experience.experience_id],
                    summary=result.summary,
                    detail=result.selected_approach,
                    tags=["success" if "success" in result.summary.lower() else "failure", experience.experience_type.value, experience.project],
                    status=LessonStatus.PROPOSED,
                    confidence=result.confidence,
                    metadata={"source": "model", "reasoning_metadata": result.metadata},
                )
                return [lesson]
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Model-based lesson extraction failed: {e}")

        return self.extract(experience)


# ---------------------------------------------------------------------------
# Learning engine
# ---------------------------------------------------------------------------

class LearningEngine:
    """Orchestrates Phase 8 Advanced Learning.

    Coordinates:
      - Experience capture from sessions
      - Lesson extraction
      - Feedback handling
      - Knowledge formation
      - Learning evaluation
      - Knowledge retrieval
    """

    def __init__(
        self,
        experience_store: ExperienceStore,
        knowledge_base: KnowledgeBase,
        lesson_extractor: LessonExtractor,
        memory_service: Any = None,
        approval_system: Any = None,
        identity_service: Any = None,
        logger: Optional[Logger] = None,
    ):
        self.experience_store = experience_store
        self.knowledge_base = knowledge_base
        self.lesson_extractor = lesson_extractor
        self.memory_service = memory_service
        self.approval_system = approval_system
        self.identity_service = identity_service
        self.logger = logger
        self._pending_lessons: dict[str, Lesson] = {}

    def capture_experience(self, experience: Experience) -> str:
        """Record a raw experience for later learning."""
        if self.identity_service is None:
            return ""
        try:
            self.identity_service.require_authority("remember")
        except PermissionError:
            if self.logger:
                self.logger.warn("Experience capture denied: insufficient authority")
            return ""

        sanitized_content = MemoryFilter.sanitize(experience.content)
        if sanitized_content != experience.content:
            experience.content = sanitized_content

        experience_id = self.experience_store.record(experience)
        if self.logger:
            self.logger.memory(f"Captured experience: {experience_id} ({experience.experience_type.value})")
        return experience_id

    async def learn_from_experience(self, experience: Experience) -> list[Lesson]:
        """Capture experience and extract lessons from it."""
        experience_id = self.capture_experience(experience)
        if not experience_id:
            return []

        lessons = await self.lesson_extractor.extract_with_reasoning(experience, {})
        for lesson in lessons:
            if experience.project:
                lesson.metadata.setdefault("project", experience.project)
            self._pending_lessons[lesson.lesson_id] = lesson
            if self.logger:
                self.logger.memory(f"Lesson proposed: {lesson.summary[:80]} (confidence={lesson.confidence:.2f})")
        return lessons

    def provide_feedback(self, lesson_id: str, feedback: Feedback) -> bool:
        """Apply creator feedback to a proposed lesson."""
        if self.identity_service is None:
            return False
        try:
            approver = self.identity_service.require_authority("enable_self_modification")
        except PermissionError:
            if self.logger:
                self.logger.warn("Lesson feedback denied: creator authority required")
            return False
        if feedback.lesson_id != lesson_id:
            return False
        feedback.provided_by = approver.name
        lesson = self._pending_lessons.get(lesson_id)
        if lesson is None:
            return False

        if feedback.feedback_type == FeedbackType.APPROVE:
            lesson.status = LessonStatus.VALIDATED
            lesson.validated_at = datetime.now().isoformat()
        elif feedback.feedback_type == FeedbackType.REJECT:
            lesson.status = LessonStatus.REJECTED
            lesson.rejected_at = datetime.now().isoformat()
            lesson.rejection_reason = feedback.content
        elif feedback.feedback_type == FeedbackType.MODIFY:
            lesson.detail = feedback.content
            lesson.status = LessonStatus.PROPOSED
        elif feedback.feedback_type == FeedbackType.CRITICISM:
            lesson.confidence = max(0.0, lesson.confidence - 0.2)
            lesson.metadata["criticism"] = feedback.content

        self._pending_lessons[lesson_id] = lesson
        if self.logger:
            self.logger.memory(f"Feedback applied to lesson {lesson_id}: {feedback.feedback_type.value}")
        return True

    def integrate_lesson(self, lesson_id: str, memory_type: str = "learning") -> Optional[str]:
        """Integrate a validated lesson into durable knowledge."""
        lesson = self._pending_lessons.get(lesson_id)
        if lesson is None:
            return None
        if lesson.status != LessonStatus.VALIDATED:
            return None

        if self.identity_service is None:
            return None
        try:
            self.identity_service.require_authority("enable_self_modification")
        except PermissionError:
            if self.logger:
                self.logger.warn("Knowledge integration denied: insufficient authority")
            return None

        knowledge = Knowledge(
            source_lesson_ids=[lesson.lesson_id],
            content=lesson.summary,
            memory_type=memory_type,
            importance=lesson.confidence,
            project=lesson.metadata.get("project", ""),
            tags=lesson.tags,
            metadata={"detail": lesson.detail, "source": "lesson_integration"},
        )
        knowledge_id = self.knowledge_base.store_knowledge(knowledge)
        lesson.status = LessonStatus.INTEGRATED
        self._pending_lessons[lesson_id] = lesson

        if self.logger:
            self.logger.memory(f"Lesson {lesson_id} integrated as knowledge {knowledge_id}")
        return knowledge_id

    def retrieve_relevant_knowledge(self, goal: str, project: str = "", limit: int = 10) -> list[dict]:
        """Retrieve knowledge relevant to a goal."""
        return self.knowledge_base.retrieve_relevant(goal=goal, project=project, limit=limit)

    def evaluate_learning(self, project: str = "") -> dict[str, Any]:
        """Evaluate the current state of learning."""
        experiences = self.experience_store.list_recent(limit=200, project=project)
        knowledge = self.knowledge_base.retrieve_relevant(goal="", project=project, limit=200)

        by_type: dict[str, int] = {}
        for exp in experiences:
            t = exp.experience_type.value
            by_type[t] = by_type.get(t, 0) + 1

        lessons_proposed = sum(1 for l in self._pending_lessons.values() if l.status == LessonStatus.PROPOSED)
        lessons_validated = sum(1 for l in self._pending_lessons.values() if l.status == LessonStatus.VALIDATED)
        lessons_integrated = sum(1 for l in self._pending_lessons.values() if l.status == LessonStatus.INTEGRATED)
        lessons_rejected = sum(1 for l in self._pending_lessons.values() if l.status == LessonStatus.REJECTED)

        total_apps = sum(k.get("application_count", 0) for k in knowledge)
        total_success = sum(k.get("success_count", 0) for k in knowledge)
        success_rate = (total_success / total_apps) if total_apps > 0 else 0.0

        return {
            "experiences_captured": len(experiences),
            "experiences_by_type": by_type,
            "pending_lessons": lessons_proposed,
            "validated_lessons": lessons_validated,
            "integrated_lessons": lessons_integrated,
            "rejected_lessons": lessons_rejected,
            "knowledge_entries": len(knowledge),
            "knowledge_success_rate": round(success_rate, 3),
            "knowledge_total_applications": total_apps,
        }
