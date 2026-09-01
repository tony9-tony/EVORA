"""
Memory system for EVORA.

Provides persistent storage for:
    - Task memory: conversation history and results for individual tasks
    - Project memory: project-level context, learnings, and configurations
    - Long-term memory: accumulated knowledge across projects

Storage is file-based (JSON) for simplicity and replaceable later.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


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

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryStore:
    """File-based memory store for EVORA."""

    def __init__(self, memory_dir: str):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._tasks_dir = self.memory_dir / "tasks"
        self._projects_dir = self.memory_dir / "projects"
        self._global_dir = self.memory_dir / "global"
        self._tasks_dir.mkdir(parents=True, exist_ok=True)
        self._projects_dir.mkdir(parents=True, exist_ok=True)
        self._global_dir.mkdir(parents=True, exist_ok=True)

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


class Memory:
    """EVORA memory interface - combines task, project, and global memory."""

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
