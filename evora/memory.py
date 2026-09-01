"""
Memory system for EVORA.

Provides persistent storage for:
    - Task memory: conversation history and results for individual tasks
    - Project memory: project-level context, learnings, and configurations
    - Long-term memory: accumulated knowledge across projects
    - Long-term memory entries: retrievable memories with importance scoring

Storage is file-based (JSON) for simplicity and replaceable later.

Phase 3 additions:
    - LongTermMemoryEntry with importance, recency, pinning, project scoping
    - MemoryFilter for secret/sensitive value sanitization
    - MemoryRetriever for relevance-scored retrieval (no vector DB)
    - MemoryService for user-controlled remember/forget operations
    - Extended ProjectMemory with languages, frameworks, important_files, etc.
    - Identity integration via IdentityService (authority checks)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evora.logger import Logger


# ---------------------------------------------------------------------------
# TASK MEMORY (Phase 1 — preserved, backward compatible)
# ---------------------------------------------------------------------------

@dataclass
class TaskEntry:
    id: str
    request: str
    plan: dict[str, Any]
    steps: list[dict[str, Any]]
    result: str
    status: str  # pending, running, completed, failed, cancelled
    timestamp: str
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)
    memories: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, request: str, plan: dict) -> "TaskEntry":
        return cls(
            id=str(uuid.uuid4()),
            request=request,
            plan=plan,
            steps=[],
            result="",
            status="pending",
            timestamp=datetime.now().isoformat(),
        )

    def update_status(self, status: str):
        self.status = status

    def add_step(self, step: str, result: str, status: str = "completed", error: str = None):
        entry = {
            "step": step,
            "result": result,
            "status": status,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            entry["error"] = error
            self.errors.append(f"{step}: {error}")
        self.steps.append(entry)

    def add_memory(self, memory: str):
        if memory not in self.memories:
            self.memories.append(memory)

    def finish(self, result: str):
        self.result = result
        self.status = "completed"

    def fail(self, error: str):
        self.result = error
        self.status = "failed"
        self.errors.append(error)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# PROJECT MEMORY (Phase 1 — extended for Phase 3)
# ---------------------------------------------------------------------------

@dataclass
class ProjectMemory:
    project_name: str
    workspace_dir: str
    created_at: str
    last_active: str
    notes: list[str] = field(default_factory=list)
    conventions: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    learned: list[str] = field(default_factory=list)

    # Phase 3 additions (all with defaults for backward compatibility)
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    important_files: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    architecture_notes: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    previous_completed_tasks: list[str] = field(default_factory=list)
    important_technical_decisions: list[str] = field(default_factory=list)

    @classmethod
    def create(cls, project_name: str, workspace_dir: str) -> "ProjectMemory":
        return cls(
            project_name=project_name,
            workspace_dir=workspace_dir,
            created_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
        )

    def add_note(self, note: str):
        if note not in self.notes:
            self.notes.append(note)

    def add_learned(self, item: str):
        if item not in self.learned:
            self.learned.append(item)

    def add_language(self, language: str):
        if language not in self.languages:
            self.languages.append(language)

    def add_framework(self, framework: str):
        if framework not in self.frameworks:
            self.frameworks.append(framework)

    def add_important_file(self, path: str):
        if path not in self.important_files:
            self.important_files.append(path)

    def add_build_command(self, command: str):
        if command not in self.build_commands:
            self.build_commands.append(command)

    def add_architecture_note(self, note: str):
        if note not in self.architecture_notes:
            self.architecture_notes.append(note)

    def add_known_issue(self, issue: str):
        if issue not in self.known_issues:
            self.known_issues.append(issue)

    def add_technical_decision(self, decision: str):
        if decision not in self.important_technical_decisions:
            self.important_technical_decisions.append(decision)

    def add_completed_task(self, task_summary: str):
        if task_summary not in self.previous_completed_tasks:
            self.previous_completed_tasks.append(task_summary)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# LONG-TERM MEMORY (Phase 3 — new)
# ---------------------------------------------------------------------------

@dataclass
class LongTermMemoryEntry:
    """A single long-term memory entry with importance, recency, and scoping."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: str = "preference"  # preference, decision, learning, instruction
    content: str = ""
    created_at: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 – 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    project: Optional[str] = None  # scoped to project, or None for global
    pinned: bool = False
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongTermMemoryEntry":
        return cls(**data)

    def score_relevance(self, keywords: list[str], project: Optional[str] = None) -> float:
        """Compute a simple relevance score for this entry.

        Factors:
          - Keyword overlap (30%)
          - Project match (25%)
          - Type relevance (20% — always 1.0 if no type filter)
          - Recency (15%)
          - Importance (10%)
        """
        if not keywords:
            keyword_score = 0.5
        else:
            content_lower = self.content.lower()
            matched = sum(1 for kw in keywords if kw.lower() in content_lower)
            keyword_score = matched / len(keywords) if keywords else 0.0

        if project:
            project_score = 1.0 if self.project == project else 0.0
        else:
            project_score = 0.5  # neutral if no project context

        type_score = 1.0  # neutral if no type filter

        # Recency: more recent = higher score
        now = time.time()
        age_seconds = now - self.created_at
        # Score decays over time: 1.0 for very recent, ~0.0 for > 30 days
        recency_score = max(0.0, 1.0 - (age_seconds / (30 * 24 * 3600)))

        importance_score = self.importance

        score = (
            0.30 * keyword_score
            + 0.25 * project_score
            + 0.20 * type_score
            + 0.15 * recency_score
            + 0.10 * importance_score
        )
        return min(1.0, max(0.0, score))

    def record_access(self) -> None:
        """Update access tracking when this memory is retrieved."""
        self.last_accessed = time.time()
        self.access_count += 1


# ---------------------------------------------------------------------------
# SECRET FILTERING (Phase 3 — new)
# ---------------------------------------------------------------------------

class MemoryFilter:
    """Filters sensitive data before persisting to memory.

    Ensures API keys, passwords, tokens, and private keys are never
    stored in memory.
    """

    SENSITIVE_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}"),
        re.compile(r"['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        re.compile(r"['\"]?password['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        re.compile(r"['\"]?passwd['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        re.compile(r"['\"]?secret['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        re.compile(r"['\"]?token['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY"),
        re.compile(r"aws_access_key_id", re.IGNORECASE),
        re.compile(r"aws_secret_access_key", re.IGNORECASE),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{36}", re.IGNORECASE),
        re.compile(r"github_pat_[A-Za-z0-9_]{22,}", re.IGNORECASE),
        re.compile(r"Bearer\s+[A-Za-z0-9._-]+"),
        re.compile(r"['\"]?authorization['\"]?\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE),
    ]

    SENSITIVE_KEY_NAMES = {
        "password", "passwd", "pwd", "api_key", "apikey", "api-key",
        "secret", "token", "access_token", "secret_token", "private_key",
        "privatekey", "authorization", "auth_token",
    }

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Replace secrets with [REDACTED] in text."""
        redacted = text
        for pattern in cls.SENSITIVE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    @classmethod
    def contains_secrets(cls, text: str) -> bool:
        """Check if text contains any sensitive patterns."""
        return any(p.search(text) for p in cls.SENSITIVE_PATTERNS)

    @classmethod
    def _is_sensitive_key(cls, key: str) -> bool:
        """Check if a dict key name indicates a sensitive value."""
        key_lower = key.lower().strip().strip("'\"")
        return key_lower in cls.SENSITIVE_KEY_NAMES

    @classmethod
    def sanitize_dict(cls, data: Any) -> Any:
        """Recursively sanitize a dict/list structure.

        Also redacts values when dict keys match sensitive key names
        (e.g., {"password": "mysecret"} → {"password": "[REDACTED]"}).
        """
        if isinstance(data, str):
            return cls.sanitize(data)
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                # If key is sensitive, redact the value
                if cls._is_sensitive_key(k):
                    if isinstance(v, str):
                        result[k] = "[REDACTED]"
                    elif isinstance(v, dict):
                        result[k] = {ik: "[REDACTED]" for ik in v}
                    elif isinstance(v, list):
                        result[k] = ["[REDACTED]" for _ in v]
                    else:
                        result[k] = "[REDACTED]"
                else:
                    result[k] = cls.sanitize_dict(v)
            return result
        if isinstance(data, list):
            return [cls.sanitize_dict(item) for item in data]
        return data


# ---------------------------------------------------------------------------
# MEMORY STORE (Phase 1 — extended for Phase 3)
# ---------------------------------------------------------------------------

class MemoryStore:
    """File-based memory store for EVORA.

    Storage layout:
        memory_dir/
            tasks/       — TaskEntry JSON files (TaskState task_id as filename)
            projects/    — ProjectMemory JSON files (project_name sanitized)
            global/      — Global key-value JSON files
            longterm/    — LongTermMemoryEntry JSON files (Phase 3)
            identities/  — Identity JSON configs (managed by IdentityStore)
    """

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._tasks_dir = self.memory_dir / "tasks"
        self._projects_dir = self.memory_dir / "projects"
        self._global_dir = self.memory_dir / "global"
        self._longterm_dir = self.memory_dir / "longterm"
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        self._global_dir.mkdir(parents=True, exist_ok=True)
        self._longterm_dir.mkdir(parents=True, exist_ok=True)

    def save_task(self, entry: TaskEntry) -> str:
        entry_path = self._tasks_dir / f"{entry.id}.json"
        with open(entry_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        return str(entry_path)

    def load_task(self, task_id: str) -> Optional[TaskEntry]:
        entry_path = self._tasks_dir / f"{task_id}.json"
        if not entry_path.exists():
            return None
        with open(entry_path, "r") as f:
            data = json.load(f)
        return TaskEntry(**data)

    def list_tasks(self, limit: int = 50) -> list[dict]:
        tasks = []
        for path in sorted(self._tasks_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            with open(path, "r") as f:
                tasks.append(json.load(f))
            if len(tasks) >= limit:
                break
        return tasks

    def save_project_memory(self, pm: ProjectMemory) -> str:
        project_path = self._projects_dir / f"{self._safe_name(pm.project_name)}.json"
        with open(project_path, "w") as f:
            json.dump(pm.to_dict(), f, indent=2)
        return str(project_path)

    def load_project_memory(self, project_name: str) -> Optional[ProjectMemory]:
        project_path = self._projects_dir / f"{self._safe_name(project_name)}.json"
        if not project_path.exists():
            return None
        with open(project_path, "r") as f:
            data = json.load(f)
        return ProjectMemory(**data)

    def save_global_memory(self, key: str, value: Any) -> None:
        path = self._global_dir / f"{self._safe_name(key)}.json"
        with open(path, "w") as f:
            json.dump({"key": key, "value": value, "updated": datetime.now().isoformat()}, f, indent=2)

    def load_global_memory(self, key: str) -> Optional[Any]:
        path = self._global_dir / f"{self._safe_name(key)}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("value")

    def _safe_name(self, name: str) -> str:
        safe = "".join(c for c in name if c.isalnum() or c in "-_")
        return safe or "unnamed"

    def search_memories(self, query: str, limit: int = 20) -> list[dict]:
        results = []
        query_lower = query.lower()

        for path in self._tasks_dir.glob("*.json"):
            with open(path, "r") as f:
                data = json.load(f)
            if query_lower in data.get("request", "").lower() or query_lower in data.get("result", "").lower():
                results.append({"type": "task", **data})

        for path in self._projects_dir.glob("*.json"):
            with open(path, "r") as f:
                data = json.load(f)
            if query_lower in data.get("project_name", "").lower() or any(query_lower in n.lower() for n in data.get("notes", [])):
                results.append({"type": "project", **data})

        return results[:limit]

    # --- Phase 3: Long-term memory methods ---

    def save_ltm_entry(self, entry: LongTermMemoryEntry) -> str:
        """Save a long-term memory entry, applying secret filtering."""
        sanitized = LongTermMemoryEntry(
            id=entry.id,
            memory_type=entry.memory_type,
            content=MemoryFilter.sanitize(entry.content),
            created_at=entry.created_at,
            importance=entry.importance,
            last_accessed=entry.last_accessed,
            access_count=entry.access_count,
            project=str(entry.project) if entry.project is not None else None,
            pinned=entry.pinned,
            tags=list(entry.tags) if entry.tags else [],
        )
        entry_path = self._longterm_dir / f"{self._safe_name(entry.id)}.json"
        with open(entry_path, "w") as f:
            json.dump(sanitized.to_dict(), f, indent=2, ensure_ascii=False)
        return str(entry_path)

    def load_ltm_entry(self, entry_id: str) -> Optional[LongTermMemoryEntry]:
        entry_path = self._longterm_dir / f"{self._safe_name(entry_id)}.json"
        if not entry_path.exists():
            return None
        with open(entry_path, "r") as f:
            data = json.load(f)
        return LongTermMemoryEntry.from_dict(data)

    def list_ltm_entries(
        self,
        project: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[LongTermMemoryEntry]:
        """List long-term memory entries, optionally filtered by project/type."""
        entries = []
        for path in sorted(self._longterm_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                if self.logger:
                    self.logger.memory(f"Skipping invalid LTM file: {path.name}")
                continue
            entry = LongTermMemoryEntry.from_dict(data)
            if project is not None and entry.project != project:
                continue
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break
        return entries

    def delete_ltm_entry(self, entry_id: str) -> bool:
        """Delete a long-term memory entry by ID."""
        entry_path = self._longterm_dir / f"{self._safe_name(entry_id)}.json"
        if entry_path.exists():
            entry_path.unlink()
            return True
        return False

    def delete_all_ltm(
        self,
        project: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> int:
        """Delete all long-term memory entries matching the filters."""
        count = 0
        for path in list(self._longterm_dir.glob("*.json")):
            with open(path, "r") as f:
                data = json.load(f)
            entry = LongTermMemoryEntry.from_dict(data)
            if project is not None and entry.project != project:
                continue
            if memory_type is not None and entry.memory_type != memory_type:
                continue
            path.unlink()
            count += 1
        return count

    def search_ltm(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 20,
    ) -> list[LongTermMemoryEntry]:
        """Search long-term memory entries by content text."""
        query_lower = query.lower()
        entries = []
        for path in sorted(self._longterm_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            with open(path, "r") as f:
                data = json.load(f)
            entry = LongTermMemoryEntry.from_dict(data)
            if project is not None and entry.project != project:
                continue
            if query_lower in entry.content.lower() or any(query_lower in t.lower() for t in entry.tags):
                entries.append(entry)
                if len(entries) >= limit:
                    break
        return entries


# ---------------------------------------------------------------------------
# MEMORY RETRIEVER (Phase 3 — new)
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A retrieved memory with its relevance score."""
    entry: LongTermMemoryEntry
    score: float

    def to_dict(self) -> dict[str, Any]:
        d = self.entry.to_dict()
        d["score"] = self.score
        return d


class MemoryRetriever:
    """Selects relevant memories based on query criteria.

    Uses simple keyword overlap, project matching, recency, and importance
    for scoring. No vector database required — designed so semantic retrieval
    can be added later by replacing this class.
    """

    def __init__(self, store: Optional[MemoryStore] = None, logger: Optional[Logger] = None):
        self.store = store
        self.logger = logger

    def _ensure_store(self, memory_dir: Optional[str] = None) -> MemoryStore:
        if self.store is None:
            if memory_dir is None:
                from evora.config import load_config
                config = load_config()
                memory_dir = config.memory_dir
            self.store = MemoryStore(memory_dir)
        return self.store

    def retrieve(
        self,
        goal: str = "",
        project: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        limit: int = 10,
        include_pinned: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories with relevance scoring.

        Scoring factors (simple, no vector DB):
        - Keyword overlap with goal (30%)
        - Project match (25%)
        - Type relevance (20% — 1.0 if no filter or type matches)
        - Recency — updated_at (15%)
        - Importance field (10%)
        - Pinned entries always included at top
        """
        store = self._ensure_store()

        keywords = goal.lower().split() if goal else []

        all_entries = store.list_ltm_entries(project=project, limit=200)

        pinned: list[RetrievalResult] = []
        scored: list[RetrievalResult] = []

        for entry in all_entries:
            if entry.pinned and include_pinned:
                pinned.append(RetrievalResult(entry=entry, score=1.0))
                entry.record_access()
                continue

            type_score = 1.0 if memory_types is None else (
                1.0 if entry.memory_type in memory_types else 0.0
            )
            if type_score == 0.0:
                continue

            # Temporarily adjust scoring by type filter
            relevance = entry.score_relevance(keywords, project)
            # Adjust for type match (if filtering)
            relevance = relevance * (1.0 if type_score >= 1.0 else 0.3)

            scored.append(RetrievalResult(entry=entry, score=relevance))

        scored.sort(key=lambda r: r.score, reverse=True)

        # Update access tracking for retrieved entries
        for r in scored[:limit]:
            r.entry.record_access()

        result = pinned + scored[:limit]
        result.sort(key=lambda r: r.score, reverse=True)

        if self.logger:
            self.logger.memory(f"Retrieved {len(result)} relevant memories")

        return result[:limit]


# ---------------------------------------------------------------------------
# MEMORY SERVICE (Phase 3 — new, user-controlled operations)
# ---------------------------------------------------------------------------

class MemoryService:
    """High-level user-controlled memory operations.

    Bridges task memory, project memory, long-term memory, and identity.
    Enforces authority checks before destructive operations.
    """

    def __init__(
        self,
        memory: Optional["Memory"] = None,
        store: Optional[MemoryStore] = None,
        identity_service: Optional[Any] = None,
        logger: Optional[Logger] = None,
        memory_dir: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        if memory is not None:
            self.memory = memory
            self.store = memory.store
        elif store is not None:
            self.store = store
            self.memory = Memory(self.store.memory_dir, project_name or "default")
        else:
            from evora.config import load_config
            config = load_config()
            mem_dir = memory_dir or config.memory_dir
            self.memory = Memory(mem_dir, project_name or "default")
            self.store = self.memory.store

        self.identity_service = identity_service
        self.logger = logger
        self._retriever = MemoryRetriever(self.store, logger)

    # --- remember ---

    def remember(
        self,
        content: str,
        memory_type: str = "preference",
        importance: float = 0.5,
        project: Optional[str] = None,
        tags: Optional[list[str]] = None,
        pinned: bool = False,
    ) -> LongTermMemoryEntry:
        """Store a memory explicitly. Requires USER authority minimum.

        Applies secret filtering before persistence.
        """
        if self.identity_service:
            self.identity_service.require_authority("remember")

        sanitized_content = MemoryFilter.sanitize(content)

        entry = LongTermMemoryEntry(
            memory_type=memory_type,
            content=sanitized_content,
            importance=min(1.0, max(0.0, importance)),
            project=project,
            pinned=pinned,
            tags=tags or [],
        )

        self.store.save_ltm_entry(entry)

        if self.logger:
            self.logger.memory(
                f"Remembered memory (type={memory_type}, project={project or 'global'}, pinned={pinned})"
            )

        return entry

    # --- forget ---

    def forget(self, entry_id: str) -> bool:
        """Delete a long-term memory entry by ID (supports partial prefix match).
        Requires ADMIN authority."""
        if self.identity_service:
            self.identity_service.require_authority("forget")

        # Try exact match first
        if self.store.delete_ltm_entry(entry_id):
            return True

        # Try partial prefix match
        entries = self.store.list_ltm_entries(limit=1000)
        for entry in entries:
            if entry.id.startswith(entry_id):
                return self.store.delete_ltm_entry(entry.id)

        return False

    def forget_all(
        self,
        memory_type: Optional[str] = None,
        project: Optional[str] = None,
    ) -> int:
        """Delete all matching long-term memory entries. Requires ADMIN authority."""
        if self.identity_service:
            self.identity_service.require_authority("forget")

        return self.store.delete_all_ltm(project=project, memory_type=memory_type)

    # --- what do you remember ---

    def list_memories(
        self,
        project: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[LongTermMemoryEntry]:
        """List long-term memory entries."""
        if self.identity_service:
            self.identity_service.require_authority("list_memories")

        return self.store.list_ltm_entries(
            project=project, memory_type=memory_type, limit=limit
        )

    def search_memories(
        self,
        query: str,
        project: Optional[str] = None,
        limit: int = 20,
    ) -> list[LongTermMemoryEntry]:
        """Search long-term memory entries by content."""
        return self.store.search_ltm(query=query, project=project, limit=limit)

    # --- retrieval ---

    def retrieve_relevant(
        self,
        goal: str = "",
        project: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories using the retriever."""
        return self._retriever.retrieve(
            goal=goal,
            project=project,
            memory_types=memory_types,
            limit=limit,
        )

    # --- task lifecycle integration ---

    def archive_task_outcome(self, state: Any) -> None:
        """Save task decisions, errors, and test_results as long-term memory.

        Applies secret filtering. This is called after a task completes
        to preserve learnings without leaking sensitive data.
        """
        from evora.task import TaskState

        if not isinstance(state, TaskState):
            return

        # Archive decisions
        for decision in state.decisions:
            decision_text = f"Decision: {decision.action} — {decision.reason}"
            memory = LongTermMemoryEntry(
                memory_type="decision",
                content=MemoryFilter.sanitize(decision_text),
                importance=0.7,
                project=state.workspace,
                tags=[decision.action, "task-outcome"],
            )
            self.store.save_ltm_entry(memory)

        # Archive errors (sanitized)
        for error in state.errors:
            memory = LongTermMemoryEntry(
                memory_type="learning",
                content=f"Error encountered: {MemoryFilter.sanitize(error)}",
                importance=0.6,
                project=state.workspace,
                tags=["error", "task-outcome"],
            )
            self.store.save_ltm_entry(memory)

        # Archive test results
        for tr in state.test_results:
            status = "passed" if tr.passed else "failed"
            memory = LongTermMemoryEntry(
                memory_type="learning",
                content=MemoryFilter.sanitize(f"Test {status}: {tr.command[:200]}"),
                importance=0.5,
                project=state.workspace,
                tags=["test", status],
            )
            self.store.save_ltm_entry(memory)

        # Archive final result
        if state.final_result:
            memory = LongTermMemoryEntry(
                memory_type=state.status if state.status in ("completed", "failed", "cancelled") else "learning",
                content=f"Task '{state.request[:200]}' final result: {MemoryFilter.sanitize(state.final_result)[:500]}",
                importance=0.8,
                project=state.workspace,
                tags=["task-completion", state.status],
            )
            self.store.save_ltm_entry(memory)

        if self.logger:
            self.logger.memory(f"Archived task outcome for '{state.request[:80]}'")

    # --- project memory management ---

    def update_project_memory(self, state: Any) -> None:
        """Update project memory with task learnings.

        Respects project isolation — only updates the project matching
        the current workspace.
        """
        from evora.task import TaskState

        if not isinstance(state, TaskState):
            return

        project_name = Path(state.workspace).name
        project_mem = self.memory.store.load_project_memory(project_name)
        if project_mem is None:
            project_mem = ProjectMemory.create(project_name, state.workspace)

        # Record this task as a completed (or failed) task
        status_label = "completed" if state.is_complete else "failed" if state.is_failed else "cancelled"
        project_mem.add_completed_task(f"{status_label}: {state.request[:200]}")

        # Extract conventions from project_context
        context = state.project_context
        if isinstance(context, dict):
            if context.get("languages"):
                for lang in context["languages"].keys():
                    project_mem.add_language(lang)
            if context.get("frameworks"):
                for fw in context.get("frameworks", []):
                    project_mem.add_framework(fw)
            if context.get("build_system"):
                project_mem.add_build_command(context["build_system"])
            if context.get("test_command"):
                project_mem.add_build_command(context["test_command"])
            if context.get("entry_points"):
                for ep in context.get("entry_points", []):
                    project_mem.add_important_file(ep)

        # Record important technical decisions
        for decision in state.decisions:
            if decision.action == "execute_tool" and decision.confidence > 0.8:
                project_mem.add_technical_decision(
                    f"Used {decision.tool} for: {decision.reason[:200]}"
                )

        # Record known issues
        for error in state.errors:
            project_mem.add_known_issue(MemoryFilter.sanitize(error)[:500])

        project_mem.last_active = datetime.now().isoformat()

        self.store.save_project_memory(project_mem)

        if self.logger:
            self.logger.memory(f"Updated project memory for '{project_name}'")

    # --- clearing ---

    def clear_task_memory(self) -> int:
        """Delete all task entries for the current project."""
        project_name = self.memory.project_name
        count = 0
        for path in list(self.store._tasks_dir.glob("*.json")):
            with open(path, "r") as f:
                data = json.load(f)
            # Match by project in plan or request context
            # We delete all task entries since they are task-scoped
            path.unlink()
            count += 1
        return count

    def clear_project_memory(self) -> bool:
        """Delete all project + long-term memory for the current project."""
        if self.identity_service:
            self.identity_service.require_authority("clear_project_memory")

        project_name = self.memory.project_name
        safe_name = self.store._safe_name(project_name)

        # Delete project memory file
        project_path = self.store._projects_dir / f"{safe_name}.json"
        deleted_project = project_path.exists()
        if deleted_project:
            project_path.unlink()

        # Delete long-term memory entries scoped to this project
        deleted_ltm = self.store.delete_all_ltm(project=project_name)

        if self.logger:
            self.logger.memory(
                f"Cleared project memory: {project_name} "
                f"(project file deleted={deleted_project}, ltm entries deleted={deleted_ltm})"
            )

        return deleted_project or deleted_ltm > 0


# ---------------------------------------------------------------------------
# MEMORY FACADE (Phase 1 — extended with Phase 3 helpers)
# ---------------------------------------------------------------------------

class Memory:
    """EVORA memory interface - combines task, project, and global memory.

    Extended in Phase 3 to support long-term memory and memory service
    integration. All new functionality is optional — backward compatibility
    is preserved: Memory(memory_dir, project_name) still works as before.
    """

    def __init__(self, memory_dir: str, project_name: str = "default"):
        self.store = MemoryStore(memory_dir)
        self.project_name = project_name
        self._project_cache: Optional[ProjectMemory] = None

    @property
    def project(self) -> ProjectMemory:
        if self._project_cache is None:
            self._project_cache = self.store.load_project_memory(self.project_name)
            if self._project_cache is None:
                self._project_cache = ProjectMemory.create(self.project_name, str(Path.cwd()))
                self.store.save_project_memory(self._project_cache)
        return self._project_cache

    def save_project(self):
        self.project.last_active = datetime.now().isoformat()
        self.store.save_project_memory(self.project)

    def create_task(self, request: str, plan: dict) -> TaskEntry:
        return TaskEntry.create(request, plan)

    def save_task(self, entry: TaskEntry):
        self.store.save_task(entry)

    def load_task(self, task_id: str) -> Optional[TaskEntry]:
        return self.store.load_task(task_id)

    def search(self, query: str, limit: int = 20) -> list[dict]:
        return self.store.search_memories(query, limit)

    # --- Phase 3 convenience wrappers ---

    @property
    def longterm_dir(self) -> Path:
        """Path to the long-term memory directory."""
        return self.store._longterm_dir

    def get_memory_service(
        self,
        identity_service: Optional[Any] = None,
        logger: Optional[Logger] = None,
    ) -> MemoryService:
        """Create a MemoryService backed by this Memory instance."""
        return MemoryService(
            memory=self,
            identity_service=identity_service,
            logger=logger,
        )
