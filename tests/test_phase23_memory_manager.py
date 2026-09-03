"""
Phase 23 — Native Memory Manager tests.

Verifies:
1. MemoryItem has correct structure
2. MemoryQuery has correct structure
3. MemoryType enum exists
4. MemoryPriority enum exists
5. NativeMemoryManager initializes
6. NativeMemoryManager stores short-term memory
7. NativeMemoryManager stores long-term memory
8. NativeMemoryManager retrieves memory
9. NativeMemoryManager retrieves relevant memory
10. NativeMemoryManager forgets memory
11. NativeMemoryManager consolidates memory
12. NativeMemoryManager returns stats
13. NativeMemoryManager clears short-term
14. NativeMemoryManager clears long-term
15. NativeMemoryManager prunes over limit
16. Memory integrates with ConversationManager
17. No ModelManager dependency
18. No external dependencies
19. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.memory_manager import (
    MemoryItem,
    MemoryPriority,
    MemoryQuery,
    MemoryType,
    NativeMemoryManager,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_manager():
    return NativeMemoryManager(logger=MagicMock())


@pytest.fixture
def memory_with_conversation():
    cm = MagicMock()
    return NativeMemoryManager(conversation_manager=cm, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestMemoryItem
# ---------------------------------------------------------------------------

class TestMemoryItem:
    """Test MemoryItem."""

    def test_default_item(self):
        item = MemoryItem()
        assert item.item_id != ""
        assert item.memory_type == MemoryType.SHORT_TERM

    def test_item_to_dict(self):
        item = MemoryItem(content="Test memory", importance=0.8, tags=["test"])
        data = item.to_dict()
        assert data["content"] == "Test memory"
        assert data["importance"] == 0.8


# ---------------------------------------------------------------------------
# TestMemoryQuery
# ---------------------------------------------------------------------------

class TestMemoryQuery:
    """Test MemoryQuery."""

    def test_default_query(self):
        query = MemoryQuery()
        assert query.query_id != ""
        assert query.limit == 10

    def test_query_to_dict(self):
        query = MemoryQuery(query_text="test", limit=5, min_importance=0.5)
        data = query.to_dict()
        assert data["query_text"] == "test"
        assert data["limit"] == 5


# ---------------------------------------------------------------------------
# TestMemoryTypeEnum
# ---------------------------------------------------------------------------

class TestMemoryTypeEnum:
    """Test MemoryType enum."""

    def test_memory_types_exist(self):
        assert MemoryType.SHORT_TERM is not None
        assert MemoryType.LONG_TERM is not None
        assert MemoryType.WORKING is not None
        assert MemoryType.EPISODIC is not None
        assert MemoryType.SEMANTIC is not None


# ---------------------------------------------------------------------------
# TestMemoryPriorityEnum
# ---------------------------------------------------------------------------

class TestMemoryPriorityEnum:
    """Test MemoryPriority enum."""

    def test_priority_values(self):
        assert MemoryPriority.LOW.value == "low"
        assert MemoryPriority.MEDIUM.value == "medium"
        assert MemoryPriority.HIGH.value == "high"
        assert MemoryPriority.CRITICAL.value == "critical"


# ---------------------------------------------------------------------------
# TestNativeMemoryManager
# ---------------------------------------------------------------------------

class TestNativeMemoryManager:
    """Test NativeMemoryManager."""

    def test_memory_manager_initializes(self, memory_manager):
        assert memory_manager is not None

    def test_store_short_term(self, memory_manager):
        item = memory_manager.store("Test memory", memory_type=MemoryType.SHORT_TERM)
        assert item.item_id != ""
        assert item.content == "Test memory"

    def test_store_long_term(self, memory_manager):
        item = memory_manager.store("Persistent memory", memory_type=MemoryType.LONG_TERM)
        assert item.memory_type == MemoryType.LONG_TERM

    def test_store_with_importance(self, memory_manager):
        item = memory_manager.store("Important", importance=0.9)
        assert item.importance == 0.9

    def test_store_with_tags(self, memory_manager):
        item = memory_manager.store("Tagged", tags=["tag1", "tag2"])
        assert "tag1" in item.tags

    def test_retrieve_memory(self, memory_manager):
        memory_manager.store("Python", tags=["python"])
        results = memory_manager.retrieve(MemoryQuery(query_text="python"))
        assert len(results) > 0

    def test_retrieve_relevant(self, memory_manager):
        memory_manager.store("Python is great", tags=["python"])
        results = memory_manager.retrieve_relevant("python", limit=5)
        assert len(results) > 0

    def test_retrieve_empty(self, memory_manager):
        results = memory_manager.retrieve(MemoryQuery(query_text="nonexistent"))
        assert len(results) == 0

    def test_forget_memory(self, memory_manager):
        item = memory_manager.store("Test")
        result = memory_manager.forget(item.item_id)
        assert result is True
        results = memory_manager.retrieve(MemoryQuery(query_text="Test"))
        assert len(results) == 0

    def test_forget_missing(self, memory_manager):
        result = memory_manager.forget("nonexistent")
        assert result is False

    def test_consolidate_memory(self, memory_manager):
        memory_manager.store("Important", importance=0.9, memory_type=MemoryType.SHORT_TERM)
        result = memory_manager.consolidate()
        assert "consolidated_count" in result

    def test_get_stats(self, memory_manager):
        memory_manager.store("Test 1")
        memory_manager.store("Test 2", memory_type=MemoryType.LONG_TERM)
        stats = memory_manager.get_memory_stats()
        assert stats["total_count"] == 2

    def test_clear_short_term(self, memory_manager):
        memory_manager.store("Test 1")
        memory_manager.store("Test 2", memory_type=MemoryType.LONG_TERM)
        memory_manager.clear_short_term()
        stats = memory_manager.get_memory_stats()
        assert stats["short_term_count"] == 0
        assert stats["long_term_count"] == 1

    def test_clear_long_term(self, memory_manager):
        memory_manager.store("Test 1")
        memory_manager.store("Test 2", memory_type=MemoryType.LONG_TERM)
        memory_manager.clear_long_term()
        stats = memory_manager.get_memory_stats()
        assert stats["long_term_count"] == 0


# ---------------------------------------------------------------------------
# TestMemoryPruning
# ---------------------------------------------------------------------------

class TestMemoryPruning:
    """Test memory pruning."""

    def test_prune_short_term_over_limit(self, memory_manager):
        for i in range(110):
            memory_manager.store(f"Memory {i}", importance=float(i) / 110.0)
        stats = memory_manager.get_memory_stats()
        assert stats["short_term_count"] <= memory_manager._max_short_term

    def test_prune_long_term_over_limit(self, memory_manager):
        for i in range(1100):
            memory_manager.store(f"Memory {i}", importance=float(i) / 1100.0, memory_type=MemoryType.LONG_TERM)
        stats = memory_manager.get_memory_stats()
        assert stats["long_term_count"] <= memory_manager._max_long_term


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 23 security boundaries."""

    def test_no_model_manager_in_memory(self):
        import evora.brain.intelligence.memory_manager as mem_mod
        source = Path(mem_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.memory_manager as mem_mod
        source = Path(mem_mod.__file__).read_text(encoding="utf-8")
        import_section = False
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_section = True
            elif import_section and stripped and not stripped.startswith("#"):
                break
            if import_section:
                for forbidden in ["openai", "anthropic", "ollama", "requests", "aiohttp", "httpx", "urllib", "socket"]:
                    assert forbidden not in stripped.lower(), f"Found forbidden dependency: {forbidden}"


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 23 works offline."""

    def test_memory_manager_works_offline(self, memory_manager):
        item = memory_manager.store("offline memory")
        assert item is not None

    def test_retrieve_offline(self, memory_manager):
        memory_manager.store("offline test")
        results = memory_manager.retrieve_relevant("offline")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 23 architecture readiness."""

    def test_native_memory_manager_exists(self):
        from evora.brain.intelligence.memory_manager import NativeMemoryManager
        assert NativeMemoryManager is not None

    def test_memory_item_exists(self):
        from evora.brain.intelligence.memory_manager import MemoryItem
        assert MemoryItem is not None

    def test_memory_query_exists(self):
        from evora.brain.intelligence.memory_manager import MemoryQuery
        assert MemoryQuery is not None

    def test_memory_type_enum_exists(self):
        from evora.brain.intelligence.memory_manager import MemoryType
        assert MemoryType.SHORT_TERM is not None
        assert MemoryType.LONG_TERM is not None

    def test_memory_priority_enum_exists(self):
        from evora.brain.intelligence.memory_manager import MemoryPriority
        assert MemoryPriority.LOW is not None
        assert MemoryPriority.HIGH is not None

    def test_memory_manager_reuses_conversation_manager(self, memory_with_conversation):
        assert memory_with_conversation.conversation_manager is not None
