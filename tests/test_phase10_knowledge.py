"""
Phase 10 — KnowledgeGraph tests.

Verifies:
* node creation
* node retrieval
* edge creation
* relationship queries
* missing-node behavior
* duplicate handling
* malformed input handling
* bounded retrieval
* offline operation
* no ModelManager dependency
"""

import time
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeType,
    RelationType,
)


@pytest.fixture
def empty_graph():
    return KnowledgeGraph()


@pytest.fixture
def graph_with_memory_service():
    memory_service = MagicMock()
    memory_service.remember = MagicMock(return_value="mem-123")
    memory_service.store.list_ltm_entries = MagicMock(return_value=[])
    return KnowledgeGraph(memory_service=memory_service)


class TestKnowledgeNode:
    """Test KnowledgeNode dataclass."""

    def test_node_creation_defaults(self):
        node = KnowledgeNode(content="test concept")
        assert node.content == "test concept"
        assert node.type == KnowledgeType.CONCEPT.value
        assert node.confidence == 1.0
        assert node.source == "learned"
        assert node.access_count == 0

    def test_node_creation_custom(self):
        node = KnowledgeNode(
            content="test pattern",
            type=KnowledgeType.PATTERN.value,
            confidence=0.8,
            source="observed",
        )
        assert node.type == KnowledgeType.PATTERN.value
        assert node.confidence == 0.8
        assert node.source == "observed"

    def test_node_confidence_clamped(self):
        node = KnowledgeNode(content="test", confidence=1.5)
        assert node.confidence == 1.0
        node2 = KnowledgeNode(content="test", confidence=-0.5)
        assert node2.confidence == 0.0

    def test_node_serialization_roundtrip(self):
        node = KnowledgeNode(
            content="test concept",
            type=KnowledgeType.LESSON.value,
            confidence=0.9,
            metadata={"key": "value"},
        )
        data = node.to_dict()
        restored = KnowledgeNode.from_dict(data)
        assert restored.content == "test concept"
        assert restored.type == KnowledgeType.LESSON.value
        assert restored.confidence == 0.9
        assert restored.metadata == {"key": "value"}

    def test_node_record_access(self):
        node = KnowledgeNode(content="test")
        node.record_access(success=True)
        assert node.access_count == 1
        assert node.success_count == 1
        assert node.failure_count == 0

        node.record_access(success=False)
        assert node.access_count == 2
        assert node.success_count == 1
        assert node.failure_count == 1

    def test_node_success_rate(self):
        node = KnowledgeNode(content="test")
        assert node.success_rate() == 1.0

        node.record_access(success=True)
        node.record_access(success=True)
        node.record_access(success=False)
        assert node.success_rate() == pytest.approx(2 / 3)

    def test_node_empty_content_rejected(self):
        with pytest.raises(ValueError):
            KnowledgeNode(content="")


class TestKnowledgeEdge:
    """Test KnowledgeEdge dataclass."""

    def test_edge_creation(self):
        edge = KnowledgeEdge(
            source_id="node-1",
            target_id="node-2",
            relation=RelationType.USES.value,
        )
        assert edge.source_id == "node-1"
        assert edge.target_id == "node-2"
        assert edge.relation == RelationType.USES.value
        assert edge.confidence == 1.0

    def test_edge_confidence_clamped(self):
        edge = KnowledgeEdge(source_id="a", target_id="b", confidence=2.0)
        assert edge.confidence == 1.0
        edge2 = KnowledgeEdge(source_id="a", target_id="b", confidence=-0.5)
        assert edge2.confidence == 0.0

    def test_edge_serialization_roundtrip(self):
        edge = KnowledgeEdge(
            source_id="node-1",
            target_id="node-2",
            relation=RelationType.REQUIRES.value,
            confidence=0.7,
            metadata={"reason": "dependency"},
        )
        data = edge.to_dict()
        restored = KnowledgeEdge.from_dict(data)
        assert restored.source_id == "node-1"
        assert restored.target_id == "node-2"
        assert restored.relation == RelationType.REQUIRES.value
        assert restored.confidence == 0.7
        assert restored.metadata == {"reason": "dependency"}


class TestKnowledgeGraphBasics:
    """Test basic KnowledgeGraph operations."""

    def test_add_and_get_node(self, empty_graph):
        node = KnowledgeNode(content="test concept", type=KnowledgeType.CONCEPT.value)
        node_id = empty_graph.add_node(node)
        retrieved = empty_graph.get_node(node_id)
        assert retrieved is not None
        assert retrieved.content == "test concept"
        assert retrieved.type == KnowledgeType.CONCEPT.value

    def test_get_missing_node(self, empty_graph):
        assert empty_graph.get_node("nonexistent") is None

    def test_add_edge_success(self, empty_graph):
        node1 = KnowledgeNode(content="concept A")
        node2 = KnowledgeNode(content="concept B")
        id1 = empty_graph.add_node(node1)
        id2 = empty_graph.add_node(node2)
        result = empty_graph.add_edge(KnowledgeEdge(source_id=id1, target_id=id2))
        assert result is True

    def test_add_edge_missing_node(self, empty_graph):
        node1 = KnowledgeNode(content="concept A")
        id1 = empty_graph.add_node(node1)
        result = empty_graph.add_edge(KnowledgeEdge(source_id=id1, target_id="missing"))
        assert result is False

    def test_add_edge_same_node(self, empty_graph):
        node1 = KnowledgeNode(content="concept A")
        id1 = empty_graph.add_node(node1)
        result = empty_graph.add_edge(KnowledgeEdge(source_id=id1, target_id=id1))
        assert result is False


class TestDuplicateHandling:
    """Test duplicate handling in KnowledgeGraph."""

    def test_duplicate_content_merged(self, empty_graph):
        node1 = KnowledgeNode(content="test concept", type=KnowledgeType.CONCEPT.value, confidence=0.8)
        node2 = KnowledgeNode(content="test concept", type=KnowledgeType.CONCEPT.value, confidence=0.9)
        id1 = empty_graph.add_node(node1)
        id2 = empty_graph.add_node(node2)
        assert id1 == id2
        assert len(empty_graph._nodes) == 1
        retrieved = empty_graph.get_node(id1)
        assert retrieved.confidence == 0.9

    def test_different_content_not_merged(self, empty_graph):
        node1 = KnowledgeNode(content="concept A", type=KnowledgeType.CONCEPT.value)
        node2 = KnowledgeNode(content="concept B", type=KnowledgeType.CONCEPT.value)
        id1 = empty_graph.add_node(node1)
        id2 = empty_graph.add_node(node2)
        assert id1 != id2
        assert len(empty_graph._nodes) == 2


class TestRelationshipQueries:
    """Test relationship queries."""

    def test_get_related_by_relation(self, empty_graph):
        node_a = KnowledgeNode(content="A", type=KnowledgeType.CONCEPT.value)
        node_b = KnowledgeNode(content="B", type=KnowledgeType.TOOL.value)
        node_c = KnowledgeNode(content="C", type=KnowledgeType.CONCEPT.value)
        id_a = empty_graph.add_node(node_a)
        id_b = empty_graph.add_node(node_b)
        id_c = empty_graph.add_node(node_c)
        empty_graph.add_edge(KnowledgeEdge(source_id=id_a, target_id=id_b, relation=RelationType.USES.value))
        empty_graph.add_edge(KnowledgeEdge(source_id=id_a, target_id=id_c, relation=RelationType.REQUIRES.value))

        uses_related = empty_graph.get_related(id_a, RelationType.USES.value)
        assert len(uses_related) == 1
        assert uses_related[0].content == "B"

        requires_related = empty_graph.get_related(id_a, RelationType.REQUIRES.value)
        assert len(requires_related) == 1
        assert requires_related[0].content == "C"

    def test_get_related_all_relations(self, empty_graph):
        node_a = KnowledgeNode(content="A")
        node_b = KnowledgeNode(content="B")
        id_a = empty_graph.add_node(node_a)
        id_b = empty_graph.add_node(node_b)
        empty_graph.add_edge(KnowledgeEdge(source_id=id_a, target_id=id_b, relation=RelationType.USES.value))

        all_related = empty_graph.get_related(id_a)
        assert len(all_related) == 1
        assert all_related[0].content == "B"

    def test_get_related_missing_node(self, empty_graph):
        assert empty_graph.get_related("nonexistent") == []
        assert empty_graph.get_related("nonexistent", RelationType.USES.value) == []


class TestQuery:
    """Test knowledge graph queries."""

    def test_query_by_concept(self, empty_graph):
        node1 = KnowledgeNode(content="use pytest for testing", type=KnowledgeType.PATTERN.value)
        node2 = KnowledgeNode(content="use unittest for testing", type=KnowledgeType.PATTERN.value)
        node3 = KnowledgeNode(content="deploy with docker", type=KnowledgeType.PATTERN.value)
        empty_graph.add_node(node1)
        empty_graph.add_node(node2)
        empty_graph.add_node(node3)
        results = empty_graph.query("pytest")
        assert len(results) == 1
        assert results[0].content == "use pytest for testing"

    def test_query_case_insensitive(self, empty_graph):
        node = KnowledgeNode(content="Use Pytest for Testing")
        empty_graph.add_node(node)
        results = empty_graph.query("pytest")
        assert len(results) == 1
        results2 = empty_graph.query("PYTEST")
        assert len(results2) == 1

    def test_query_bounded(self, empty_graph):
        for i in range(20):
            empty_graph.add_node(KnowledgeNode(content=f"pytest pattern {i}"))
        results = empty_graph.query("pytest", limit=5)
        assert len(results) == 5

    def test_query_empty_concept(self, empty_graph):
        assert empty_graph.query("") == []
        assert empty_graph.query("  ") == []

    def test_query_no_matches(self, empty_graph):
        empty_graph.add_node(KnowledgeNode(content="something else"))
        assert empty_graph.query("nonexistent") == []


class TestMalformedInput:
    """Test safe handling of malformed data."""

    def test_add_node_empty_content_rejected(self, empty_graph):
        with pytest.raises(ValueError):
            empty_graph.add_node(KnowledgeNode(content=""))

    def test_add_edge_malformed_ids(self, empty_graph):
        node = KnowledgeNode(content="test")
        node_id = empty_graph.add_node(node)
        result = empty_graph.add_edge(KnowledgeEdge(source_id=node_id, target_id="nonexistent"))
        assert result is False

    def test_query_empty_concept(self, empty_graph):
        assert empty_graph.query("") == []

    def test_get_related_empty_node_id(self, empty_graph):
        assert empty_graph.get_related("") == []


class TestBoundedRetrieval:
    """Test bounded retrieval."""

    def test_get_all_nodes_bounded(self, empty_graph):
        for i in range(200):
            empty_graph.add_node(KnowledgeNode(content=f"node {i}"))
        all_nodes = empty_graph.get_all_nodes(limit=100)
        assert len(all_nodes) == 100

    def test_get_all_edges_bounded(self, empty_graph):
        nodes = []
        for i in range(10):
            node = KnowledgeNode(content=f"node {i}")
            nodes.append(empty_graph.add_node(node))
        for i in range(200):
            empty_graph.add_edge(KnowledgeEdge(
                source_id=nodes[i % 10],
                target_id=nodes[(i + 1) % 10],
            ))
        edges = empty_graph.get_all_edges(limit=100)
        assert len(edges) == 100

    def test_query_bounded(self, empty_graph):
        for i in range(50):
            empty_graph.add_node(KnowledgeNode(content=f"test concept {i}"))
        results = empty_graph.query("test", limit=10)
        assert len(results) == 10


class TestOfflineOperation:
    """Test offline operation (no ModelManager, no network)."""

    def test_no_model_manager_dependency(self, empty_graph):
        """KnowledgeGraph has no ModelManager dependency."""
        assert not hasattr(empty_graph, "model_manager")
        assert not hasattr(empty_graph, "_model_manager")

    def test_no_network_calls(self, empty_graph):
        """KnowledgeGraph does not make network calls."""
        import evora.brain.intelligence.knowledge as kg_mod
        source = kg_mod.__file__
        with open(source, "r") as f:
            code = f.read()
        forbidden = ["requests", "aiohttp", "httpx", "urllib", "openai", "anthropic", "ollama"]
        for term in forbidden:
            assert term not in code.lower(), f"KnowledgeGraph must not use {term}"

    def test_offline_operations_work(self, empty_graph):
        """All operations work offline."""
        node1 = KnowledgeNode(content="pattern A", type=KnowledgeType.PATTERN.value)
        node2 = KnowledgeNode(content="pattern B", type=KnowledgeType.PATTERN.value)
        id1 = empty_graph.add_node(node1)
        id2 = empty_graph.add_node(node2)
        empty_graph.add_edge(KnowledgeEdge(source_id=id1, target_id=id2, relation=RelationType.USES.value))
        assert empty_graph.get_node(id1) is not None
        assert empty_graph.get_node(id2) is not None
        related = empty_graph.get_related(id1)
        assert len(related) == 1
        results = empty_graph.query("pattern")
        assert len(results) == 2


class TestMemoryIntegration:
    """Test integration with MemoryService."""

    def test_persist_to_memory(self, graph_with_memory_service):
        node = KnowledgeNode(content="test knowledge", type=KnowledgeType.LESSON.value)
        graph_with_memory_service.add_node(node)
        created_ids = graph_with_memory_service.persist_to_memory(project="testproject")
        assert len(created_ids) == 1
        graph_with_memory_service._memory_service.remember.assert_called_once()

    def test_persist_requires_memory_service(self, empty_graph):
        node = KnowledgeNode(content="test")
        empty_graph.add_node(node)
        created_ids = empty_graph.persist_to_memory()
        assert created_ids == []

    def test_load_from_memory(self, graph_with_memory_service):
        from evora.memory import LongTermMemoryEntry
        entry = LongTermMemoryEntry(
            content="loaded knowledge",
            memory_type="knowledge",
            tags=["concept", "learned"],
        )
        graph_with_memory_service._memory_service.store.list_ltm_entries.return_value = [entry]
        loaded = graph_with_memory_service.load_from_memory(project="testproject", limit=10)
        assert loaded == 1
        assert len(graph_with_memory_service._nodes) == 1

    def test_load_from_memory_no_service(self, empty_graph):
        loaded = empty_graph.load_from_memory()
        assert loaded == 0


class TestSummary:
    """Test knowledge graph summary."""

    def test_empty_summary(self, empty_graph):
        summary = empty_graph.summary()
        assert summary["total_nodes"] == 0
        assert summary["total_edges"] == 0

    def test_summary_with_data(self, empty_graph):
        node1 = KnowledgeNode(content="A", type=KnowledgeType.CONCEPT.value)
        node2 = KnowledgeNode(content="B", type=KnowledgeType.TOOL.value)
        id1 = empty_graph.add_node(node1)
        id2 = empty_graph.add_node(node2)
        empty_graph.add_edge(KnowledgeEdge(source_id=id1, target_id=id2))
        summary = empty_graph.summary()
        assert summary["total_nodes"] == 2
        assert summary["total_edges"] == 1
        assert summary["node_types"]["concept"] == 1
        assert summary["node_types"]["tool"] == 1


class TestEviction:
    """Test node eviction when at capacity."""

    def test_eviction_when_at_capacity(self):
        graph = KnowledgeGraph()
        graph._max_nodes = 3
        node1 = KnowledgeNode(content="node 1")
        node2 = KnowledgeNode(content="node 2")
        node3 = KnowledgeNode(content="node 3")
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        assert len(graph._nodes) == 3
        node4 = KnowledgeNode(content="node 4")
        graph.add_node(node4)
        assert len(graph._nodes) == 3
        assert node4.id in graph._nodes
