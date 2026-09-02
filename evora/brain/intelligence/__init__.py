"""
Phase 10 — Native Intelligence package.

Provides EVORA's own cognitive capabilities independent of external models.
"""

from evora.brain.intelligence.knowledge import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeType,
    RelationType,
)

__all__ = [
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeType",
    "RelationType",
]
