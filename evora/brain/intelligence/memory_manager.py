"""
Phase 23 — Native Memory Manager for EVORA.

Manages conversation and task memory efficiently.

Supports:
  - Short-term memory (recent context)
  - Long-term memory (persistent knowledge)
  - Memory pruning
  - Memory retrieval by relevance
  - Memory consolidation
  - Integration with ConversationManager
  - Integration with TrainingPipeline
  - Integration with KnowledgeGraph

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryItem:
    """A memory item."""
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    memory_type: MemoryType = MemoryType.SHORT_TERM
    priority: MemoryPriority = MemoryPriority.MEDIUM
    content: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    expires_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "content": self.content,
            "context": self.context,
            "tags": self.tags,
            "importance": self.importance,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "expires_at": self.expires_at,
        }


@dataclass
class MemoryQuery:
    """A memory query."""
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    query_text: str = ""
    memory_types: list[MemoryType] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    limit: int = 10
    min_importance: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "memory_types": [mt.value for mt in self.memory_types],
            "tags": self.tags,
            "limit": self.limit,
            "min_importance": self.min_importance,
            "context": self.context,
        }


# ---------------------------------------------------------------------------
# Native Memory Manager
# ---------------------------------------------------------------------------

class NativeMemoryManager:
    """Native memory manager for EVORA.

    Manages short-term and long-term memory.
    """

    def __init__(
        self,
        conversation_manager: Any = None,
        training_pipeline: Any = None,
        knowledge_graph: Any = None,
        logger: Optional[Any] = None,
    ):
        self.conversation_manager = conversation_manager
        self.training_pipeline = training_pipeline
        self.knowledge_graph = knowledge_graph
        self.logger = logger
        self._short_term: dict[str, MemoryItem] = {}
        self._long_term: dict[str, MemoryItem] = {}
        self._max_short_term = 100
        self._max_long_term = 1000

    def store(self, content: str, memory_type: MemoryType = MemoryType.SHORT_TERM, priority: MemoryPriority = MemoryPriority.MEDIUM, importance: float = 0.5, tags: list[str] = None, context: dict[str, Any] = None, ttl: float = 0.0) -> MemoryItem:
        """Store a memory item."""
        tags = tags or []
        context = context or {}
        item = MemoryItem(
            memory_type=memory_type,
            priority=priority,
            content=content,
            context=context,
            tags=tags,
            importance=importance,
        )
        if ttl > 0:
            item.expires_at = datetime.now().isoformat()
        if memory_type == MemoryType.SHORT_TERM:
            self._short_term[item.item_id] = item
            self._prune_short_term()
        else:
            self._long_term[item.item_id] = item
            self._prune_long_term()
        return item

    def retrieve(self, query: MemoryQuery) -> list[MemoryItem]:
        """Retrieve memory items matching a query."""
        results = []
        memory_store = self._short_term if MemoryType.SHORT_TERM in query.memory_types or not query.memory_types else {}
        memory_store.update(self._long_term if MemoryType.LONG_TERM in query.memory_types or not query.memory_types else {})
        query_text_lower = query.query_text.lower()
        for item in memory_store.values():
            if query_text_lower and query_text_lower not in item.content.lower():
                continue
            if query.min_importance > 0 and item.importance < query.min_importance:
                continue
            if query.tags and not any(tag in item.tags for tag in query.tags):
                continue
            item.access_count += 1
            item.last_accessed = datetime.now().isoformat()
            results.append(item)
            if len(results) >= query.limit:
                break
        return results

    def retrieve_relevant(self, query_text: str, limit: int = 5) -> list[MemoryItem]:
        """Retrieve relevant memory items by text."""
        query = MemoryQuery(query_text=query_text, limit=limit)
        return self.retrieve(query)

    def forget(self, item_id: str) -> bool:
        """Forget a memory item."""
        if item_id in self._short_term:
            del self._short_term[item_id]
            return True
        if item_id in self._long_term:
            del self._long_term[item_id]
            return True
        return False

    def consolidate(self) -> dict[str, Any]:
        """Consolidate memories from short-term to long-term."""
        consolidated = 0
        for item in list(self._short_term.values()):
            if item.importance >= 0.7:
                item.memory_type = MemoryType.LONG_TERM
                self._long_term[item.item_id] = item
                del self._short_term[item.item_id]
                consolidated += 1
        return {"consolidated_count": consolidated, "short_term_remaining": len(self._short_term), "long_term_total": len(self._long_term)}

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "total_count": len(self._short_term) + len(self._long_term),
            "max_short_term": self._max_short_term,
            "max_long_term": self._max_long_term,
        }

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self._short_term = {}

    def clear_long_term(self) -> None:
        """Clear long-term memory."""
        self._long_term = {}

    def _prune_short_term(self) -> None:
        """Prune short-term memory if over limit."""
        if len(self._short_term) > self._max_short_term:
            sorted_items = sorted(self._short_term.values(), key=lambda x: x.importance)
            for item in sorted_items[:len(self._short_term) - self._max_short_term]:
                del self._short_term[item.item_id]

    def _prune_long_term(self) -> None:
        """Prune long-term memory if over limit."""
        if len(self._long_term) > self._max_long_term:
            sorted_items = sorted(self._long_term.values(), key=lambda x: x.importance)
            for item in sorted_items[:len(self._long_term) - self._max_long_term]:
                del self._long_term[item.item_id]
