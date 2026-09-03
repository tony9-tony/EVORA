"""
Phase 22 — Native Knowledge Consolidation for EVORA.

Consolidates and organizes knowledge from various sources.

Supports:
  - Knowledge deduplication
  - Knowledge merging
  - Confidence scoring
  - Source tracking
  - Knowledge pruning
  - Integration with KnowledgeGraph
  - Integration with TrainingPipeline
  - Integration with MemoryService

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

class KnowledgeSource(str, Enum):
    TRAINING = "training"
    EXPERIENCE = "experience"
    REFLECTION = "reflection"
    EXTERNAL = "external"
    USER = "user"


class ConsolidationStrategy(str, Enum):
    MERGE = "merge"
    REPLACE = "replace"
    KEEP_BOTH = "keep_both"
    DISCARD = "discard"


@dataclass
class KnowledgeEntry:
    """A knowledge entry."""
    entry_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    source: KnowledgeSource = KnowledgeSource.TRAINING
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    last_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "source": self.source.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
        }


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    success: bool = False
    merged_count: int = 0
    discarded_count: int = 0
    kept_count: int = 0
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "merged_count": self.merged_count,
            "discarded_count": self.discarded_count,
            "kept_count": self.kept_count,
            "errors": self.errors,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Native Knowledge Consolidation
# ---------------------------------------------------------------------------

class NativeKnowledgeConsolidation:
    """Native knowledge consolidation for EVORA.

    Consolidates knowledge from various sources.
    """

    def __init__(
        self,
        knowledge_graph: Any = None,
        training_pipeline: Any = None,
        memory_service: Any = None,
        logger: Optional[Any] = None,
    ):
        self.knowledge_graph = knowledge_graph
        self.training_pipeline = training_pipeline
        self.memory_service = memory_service
        self.logger = logger
        self._knowledge_base: dict[str, KnowledgeEntry] = {}

    def add_knowledge(self, content: str, source: KnowledgeSource = KnowledgeSource.TRAINING, confidence: float = 0.5, tags: list[str] = None) -> KnowledgeEntry:
        """Add knowledge to the consolidation base."""
        tags = tags or []
        entry = KnowledgeEntry(
            content=content,
            source=source,
            confidence=confidence,
            tags=tags,
        )
        self._knowledge_base[entry.entry_id] = entry
        if self.knowledge_graph is not None:
            try:
                self.knowledge_graph.add_node(content, metadata={"source": source.value, "confidence": confidence})
            except Exception:
                pass
        return entry

    def consolidate(self, strategy: ConsolidationStrategy = ConsolidationStrategy.MERGE) -> ConsolidationResult:
        """Consolidate the knowledge base."""
        result = ConsolidationResult()
        entries = list(self._knowledge_base.values())
        if not entries:
            result.success = True
            return result
        seen_contents: dict[str, KnowledgeEntry] = {}
        for entry in entries:
            content_key = entry.content.lower().strip()
            if content_key in seen_contents:
                existing = seen_contents[content_key]
                if strategy == ConsolidationStrategy.MERGE:
                    existing.confidence = max(existing.confidence, entry.confidence)
                    existing.usage_count += entry.usage_count
                    existing.updated_at = datetime.now().isoformat()
                    result.merged_count += 1
                elif strategy == ConsolidationStrategy.REPLACE:
                    existing.confidence = entry.confidence
                    existing.updated_at = datetime.now().isoformat()
                    result.merged_count += 1
                elif strategy == ConsolidationStrategy.DISCARD:
                    result.discarded_count += 1
                else:
                    result.kept_count += 1
            else:
                seen_contents[content_key] = entry
                result.kept_count += 1
        result.success = True
        return result

    def prune_low_confidence(self, threshold: float = 0.3) -> ConsolidationResult:
        """Prune low-confidence knowledge entries."""
        result = ConsolidationResult()
        to_remove = [eid for eid, entry in self._knowledge_base.items() if entry.confidence < threshold]
        for eid in to_remove:
            del self._knowledge_base[eid]
            result.discarded_count += 1
        result.success = True
        return result

    def get_knowledge(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        return self._knowledge_base.get(entry_id)

    def search_knowledge(self, query: str, tags: list[str] = None) -> list[KnowledgeEntry]:
        """Search knowledge by query and tags."""
        tags = tags or []
        results = []
        query_lower = query.lower()
        for entry in self._knowledge_base.values():
            if query_lower in entry.content.lower():
                if not tags or any(tag in entry.tags for tag in tags):
                    results.append(entry)
        return results

    def record_usage(self, entry_id: str) -> bool:
        """Record usage of a knowledge entry."""
        entry = self._knowledge_base.get(entry_id)
        if entry:
            entry.usage_count += 1
            entry.last_used = datetime.now().isoformat()
            return True
        return False

    def get_consolidation_metrics(self) -> dict[str, Any]:
        """Get consolidation metrics."""
        total = len(self._knowledge_base)
        by_source: dict[str, int] = {}
        for entry in self._knowledge_base.values():
            by_source[entry.source.value] = by_source.get(entry.source.value, 0) + 1
        return {
            "total_entries": total,
            "by_source": by_source,
        }
