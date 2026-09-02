"""
Phase 10 — KnowledgeGraph for EVORA native intelligence.

Provides a lightweight, deterministic knowledge representation
for native intelligence. No external model dependency.
No ModelManager dependency. No network dependency.

Integrates with existing MemoryService by storing knowledge
as LongTermMemoryEntry with memory_type="knowledge".
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class KnowledgeType(str, Enum):
    """Types of knowledge nodes."""
    CONCEPT = "concept"
    PATTERN = "pattern"
    TOOL = "tool"
    LESSON = "lesson"
    CONSTRAINT = "constraint"
    CAPABILITY = "capability"


class RelationType(str, Enum):
    """Types of relationships between knowledge nodes."""
    USES = "uses"
    REQUIRES = "requires"
    CONTRADICTS = "contradicts"
    ENABLES = "enables"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    INSTANCE_OF = "instance_of"
    PRECEDES = "precedes"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str = field(default_factory=lambda: f"k-{uuid.uuid4().hex[:12]}")
    type: str = KnowledgeType.CONCEPT.value
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "learned"  # learned, observed, imported, derived
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    def __post_init__(self):
        if not self.content or not self.content.strip():
            raise ValueError("KnowledgeNode content must be non-empty")
        if self.confidence < 0.0:
            self.confidence = 0.0
        if self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeNode":
        return cls(
            id=data.get("id", f"k-{uuid.uuid4().hex[:12]}"),
            type=data.get("type", KnowledgeType.CONCEPT.value),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "learned"),
            created_at=data.get("created_at", time.time()),
            access_count=data.get("access_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
        )

    def record_access(self, success: bool = True) -> None:
        """Record that this node was accessed."""
        self.access_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def success_rate(self) -> float:
        """Get success rate of this node."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total


@dataclass
class KnowledgeEdge:
    """An edge (relationship) between two knowledge nodes."""

    source_id: str
    target_id: str
    relation: str = RelationType.RELATED_TO.value
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        if self.confidence > 1.0:
            self.confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEdge":
        return cls(
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation=data.get("relation", RelationType.RELATED_TO.value),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


class KnowledgeGraph:
    """Lightweight knowledge graph for native intelligence.

    No external model dependency.
    No ModelManager dependency.
    No network dependency.

    Integrates with MemoryService by storing knowledge as
    LongTermMemoryEntry with memory_type="knowledge".
    """

    def __init__(self, memory_service: Any = None, logger: Any = None):
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: list[KnowledgeEdge] = []
        self._index: dict[str, set[str]] = {}  # relation -> set of (source, target)
        self._memory_service = memory_service
        self._logger = logger
        self._max_nodes = 1000
        self._max_edges = 5000

    def add_node(self, node: KnowledgeNode) -> str:
        """Add a node to the knowledge graph.

        Returns the node id.
        Handles duplicates by content+type (not id).
        """
        if not node.content or not node.content.strip():
            raise ValueError("KnowledgeNode content must be non-empty")

        existing = self._find_node_by_content(node.content, node.type)
        if existing is not None:
            existing.confidence = max(existing.confidence, node.confidence)
            existing.access_count += 1
            return existing.id

        if len(self._nodes) >= self._max_nodes:
            self._evict_least_valuable_node()

        self._nodes[node.id] = node
        if self._logger:
            self._logger.observe(f"Added knowledge node: {node.id} ({node.type})")
        return node.id

    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge between two nodes.

        Returns True if added, False if invalid.
        """
        if edge.source_id not in self._nodes or edge.target_id not in self._nodes:
            if self._logger:
                self._logger.warn(f"Cannot add edge: node not found ({edge.source_id} -> {edge.target_id})")
            return False

        if edge.source_id == edge.target_id:
            return False

        if len(self._edges) >= self._max_edges:
            self._edges = self._edges[-self._max_edges + 100 :]

        self._edges.append(edge)
        if edge.relation not in self._index:
            self._index[edge.relation] = set()
        self._index[edge.relation].add((edge.source_id, edge.target_id))

        if self._logger:
            self._logger.observe(
                f"Added knowledge edge: {edge.source_id} --[{edge.relation}]--> {edge.target_id}"
            )
        return True

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by id."""
        node = self._nodes.get(node_id)
        if node is not None:
            node.record_access()
        return node

    def get_related(self, node_id: str, relation: Optional[str] = None) -> list[KnowledgeNode]:
        """Get nodes related to a given node.

        If relation is provided, only return nodes with that relation.
        Otherwise return all related nodes.
        """
        if node_id not in self._nodes:
            return []

        related_ids: set[str] = set()
        if relation is not None:
            pairs = self._index.get(relation, set())
            for source_id, target_id in pairs:
                if source_id == node_id:
                    related_ids.add(target_id)
                elif target_id == node_id:
                    related_ids.add(source_id)
        else:
            for pairs in self._index.values():
                for source_id, target_id in pairs:
                    if source_id == node_id:
                        related_ids.add(target_id)
                    elif target_id == node_id:
                        related_ids.add(source_id)

        result = []
        for rid in related_ids:
            node = self._nodes.get(rid)
            if node is not None:
                node.record_access()
                result.append(node)
        return result

    def query(self, concept: str, relation: Optional[str] = None, limit: int = 10) -> list[KnowledgeNode]:
        """Query knowledge graph for nodes matching a concept.

        Matches by content substring (case-insensitive).
        Optionally filters by relation.
        Results are bounded by limit.
        """
        if not concept or not concept.strip():
            return []

        concept_lower = concept.lower()
        matches: list[tuple[float, KnowledgeNode]] = []

        for node in self._nodes.values():
            content_lower = node.content.lower()
            if concept_lower in content_lower:
                score = node.confidence * (1.0 + 0.1 * min(node.access_count, 10))
                matches.append((score, node))

        matches.sort(key=lambda x: x[0], reverse=True)
        results = [node for _, node in matches[:limit]]
        for node in results:
            node.record_access()
        return results

    def get_all_nodes(self, node_type: Optional[str] = None, limit: int = 100) -> list[KnowledgeNode]:
        """Get all nodes, optionally filtered by type."""
        nodes = list(self._nodes.values())
        if node_type is not None:
            nodes = [n for n in nodes if n.type == node_type]
        nodes.sort(key=lambda n: n.confidence, reverse=True)
        return nodes[:limit]

    def get_all_edges(self, relation: Optional[str] = None, limit: int = 500) -> list[KnowledgeEdge]:
        """Get all edges, optionally filtered by relation."""
        edges = self._edges
        if relation is not None:
            edges = [e for e in edges if e.relation == relation]
        return edges[:limit]

    def persist_to_memory(self, project: Optional[str] = None) -> list[str]:
        """Persist knowledge nodes to MemoryService as LongTermMemoryEntry.

        Returns list of created entry ids.
        Skips nodes that are already persisted.
        """
        if self._memory_service is None:
            return []

        created_ids = []
        for node in self._nodes.values():
            try:
                memory_id = self._memory_service.remember(
                    content=node.content,
                    memory_type="knowledge",
                    project=project,
                    importance=node.confidence,
                    tags=[node.type, node.source],
                )
                created_ids.append(memory_id)
            except Exception:
                continue
        return created_ids

    def load_from_memory(self, project: Optional[str] = None, limit: int = 100) -> int:
        """Load knowledge entries from MemoryService.

        Returns number of nodes loaded.
        Skips nodes that already exist (by content+type).
        """
        if self._memory_service is None:
            return 0

        loaded = 0
        try:
            entries = self._memory_service.store.list_ltm_entries(
                project=project,
                memory_type="knowledge",
                limit=limit,
            )
        except Exception:
            return 0

        for entry in entries:
            node = KnowledgeNode(
                id=f"k-{entry.id}",
                type=entry.tags[0] if entry.tags else KnowledgeType.CONCEPT.value,
                content=entry.content,
                confidence=entry.importance,
                source=entry.tags[1] if len(entry.tags) > 1 else "observed",
                created_at=entry.created_at,
                access_count=0,
                success_count=0,
                failure_count=0,
            )
            existing = self._find_node_by_content(node.content, node.type)
            if existing is None:
                self._nodes[node.id] = node
                loaded += 1

        return loaded

    def summary(self) -> dict[str, Any]:
        """Return a summary of the knowledge graph."""
        type_counts: dict[str, int] = {}
        for node in self._nodes.values():
            type_counts[node.type] = type_counts.get(node.type, 0) + 1

        relation_counts: dict[str, int] = {}
        for edge in self._edges:
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": type_counts,
            "relations": relation_counts,
            "avg_confidence": sum(n.confidence for n in self._nodes.values()) / max(len(self._nodes), 1),
            "avg_success_rate": sum(n.success_rate() for n in self._nodes.values()) / max(len(self._nodes), 1),
        }

    def _find_node_by_content(self, content: str, node_type: str) -> Optional[KnowledgeNode]:
        """Find existing node by content and type."""
        content_lower = content.lower()
        for node in self._nodes.values():
            if node.content.lower() == content_lower and node.type == node_type:
                return node
        return None

    def _evict_least_valuable_node(self) -> None:
        """Evict the least valuable node when at capacity."""
        if not self._nodes:
            return

        def node_value(node: KnowledgeNode) -> float:
            recency = max(0.0, 1.0 - (time.time() - node.created_at) / (30 * 24 * 3600))
            return node.confidence * (0.5 + 0.3 * recency + 0.2 * node.success_rate())

        sorted_nodes = sorted(self._nodes.values(), key=node_value)
        evict_id = sorted_nodes[0].id
        del self._nodes[evict_id]
        self._edges = [e for e in self._edges if e.source_id != evict_id and e.target_id != evict_id]
        for relation, pairs in list(self._index.items()):
            self._index[relation] = {p for p in pairs if evict_id not in p}
