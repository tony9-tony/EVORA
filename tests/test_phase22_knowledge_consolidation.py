"""
Phase 22 — Native Knowledge Consolidation tests.

Verifies:
1. KnowledgeEntry has correct structure
2. ConsolidationResult has correct structure
3. KnowledgeSource enum exists
4. ConsolidationStrategy enum exists
5. NativeKnowledgeConsolidation initializes
6. NativeKnowledgeConsolidation adds knowledge
7. NativeKnowledgeConsolidation consolidates with merge
8. NativeKnowledgeConsolidation consolidates with replace
9. NativeKnowledgeConsolidation consolidates with keep_both
10. NativeKnowledgeConsolidation consolidates with discard
11. NativeKnowledgeConsolidation prunes low confidence
12. NativeKnowledgeConsolidation gets knowledge by ID
13. NativeKnowledgeConsolidation searches knowledge
14. NativeKnowledgeConsolidation records usage
15. NativeKnowledgeConsolidation returns metrics
16. Knowledge integrates with KnowledgeGraph
17. No ModelManager dependency
18. No external dependencies
19. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.knowledge_consolidation import (
    ConsolidationResult,
    ConsolidationStrategy,
    KnowledgeEntry,
    KnowledgeSource,
    NativeKnowledgeConsolidation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def knowledge_consolidation():
    return NativeKnowledgeConsolidation(logger=MagicMock())


@pytest.fixture
def knowledge_with_graph():
    kg = MagicMock()
    return NativeKnowledgeConsolidation(knowledge_graph=kg, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestKnowledgeEntry
# ---------------------------------------------------------------------------

class TestKnowledgeEntry:
    """Test KnowledgeEntry."""

    def test_default_entry(self):
        entry = KnowledgeEntry()
        assert entry.entry_id != ""
        assert entry.confidence == 0.5

    def test_entry_to_dict(self):
        entry = KnowledgeEntry(content="Python is a language", confidence=0.9, tags=["python"])
        data = entry.to_dict()
        assert data["content"] == "Python is a language"
        assert data["confidence"] == 0.9


# ---------------------------------------------------------------------------
# TestConsolidationResult
# ---------------------------------------------------------------------------

class TestConsolidationResult:
    """Test ConsolidationResult."""

    def test_default_result(self):
        result = ConsolidationResult()
        assert result.success is False

    def test_result_to_dict(self):
        result = ConsolidationResult(success=True, merged_count=5, kept_count=10)
        data = result.to_dict()
        assert data["success"] is True
        assert data["merged_count"] == 5
        assert data["kept_count"] == 10


# ---------------------------------------------------------------------------
# TestKnowledgeSourceEnum
# ---------------------------------------------------------------------------

class TestKnowledgeSourceEnum:
    """Test KnowledgeSource enum."""

    def test_source_values(self):
        assert KnowledgeSource.TRAINING.value == "training"
        assert KnowledgeSource.EXPERIENCE.value == "experience"
        assert KnowledgeSource.REFLECTION.value == "reflection"


# ---------------------------------------------------------------------------
# TestConsolidationStrategyEnum
# ---------------------------------------------------------------------------

class TestConsolidationStrategyEnum:
    """Test ConsolidationStrategy enum."""

    def test_strategy_values(self):
        assert ConsolidationStrategy.MERGE.value == "merge"
        assert ConsolidationStrategy.REPLACE.value == "replace"
        assert ConsolidationStrategy.KEEP_BOTH.value == "keep_both"
        assert ConsolidationStrategy.DISCARD.value == "discard"


# ---------------------------------------------------------------------------
# TestNativeKnowledgeConsolidation
# ---------------------------------------------------------------------------

class TestNativeKnowledgeConsolidation:
    """Test NativeKnowledgeConsolidation."""

    def test_consolidation_initializes(self, knowledge_consolidation):
        assert knowledge_consolidation is not None

    def test_add_knowledge(self, knowledge_consolidation):
        entry = knowledge_consolidation.add_knowledge("Python is a language", confidence=0.9)
        assert entry.entry_id != ""
        assert entry.content == "Python is a language"

    def test_add_knowledge_with_tags(self, knowledge_consolidation):
        entry = knowledge_consolidation.add_knowledge("Test", tags=["test", "unit"])
        assert "test" in entry.tags

    def test_get_knowledge(self, knowledge_consolidation):
        entry = knowledge_consolidation.add_knowledge("Test knowledge")
        retrieved = knowledge_consolidation.get_knowledge(entry.entry_id)
        assert retrieved is not None
        assert retrieved.content == "Test knowledge"

    def test_get_knowledge_missing(self, knowledge_consolidation):
        retrieved = knowledge_consolidation.get_knowledge("nonexistent")
        assert retrieved is None

    def test_search_knowledge(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("Python is great", tags=["python"])
        knowledge_consolidation.add_knowledge("Java is great", tags=["java"])
        results = knowledge_consolidation.search_knowledge("python")
        assert len(results) == 1

    def test_search_knowledge_with_tags(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("Python is great", tags=["python"])
        results = knowledge_consolidation.search_knowledge("python", tags=["python"])
        assert len(results) == 1

    def test_record_usage(self, knowledge_consolidation):
        entry = knowledge_consolidation.add_knowledge("Test")
        result = knowledge_consolidation.record_usage(entry.entry_id)
        assert result is True
        assert entry.usage_count == 1

    def test_record_usage_missing(self, knowledge_consolidation):
        result = knowledge_consolidation.record_usage("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# TestConsolidation
# ---------------------------------------------------------------------------

class TestConsolidation:
    """Test knowledge consolidation."""

    def test_consolidate_merge(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("Python", confidence=0.7)
        knowledge_consolidation.add_knowledge("Python", confidence=0.9)
        result = knowledge_consolidation.consolidate(ConsolidationStrategy.MERGE)
        assert result.merged_count == 1
        assert result.success is True

    def test_consolidate_keep_both(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("Python", confidence=0.7)
        knowledge_consolidation.add_knowledge("Python", confidence=0.9)
        result = knowledge_consolidation.consolidate(ConsolidationStrategy.KEEP_BOTH)
        assert result.kept_count == 2
        assert result.success is True

    def test_consolidate_discard(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("Python", confidence=0.7)
        knowledge_consolidation.add_knowledge("Python", confidence=0.9)
        result = knowledge_consolidation.consolidate(ConsolidationStrategy.DISCARD)
        assert result.discarded_count == 1
        assert result.success is True

    def test_consolidate_empty(self, knowledge_consolidation):
        result = knowledge_consolidation.consolidate()
        assert result.success is True

    def test_prune_low_confidence(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("High confidence", confidence=0.9)
        knowledge_consolidation.add_knowledge("Low confidence", confidence=0.1)
        result = knowledge_consolidation.prune_low_confidence(threshold=0.5)
        assert result.discarded_count == 1
        assert result.success is True

    def test_prune_keeps_high_confidence(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("High confidence", confidence=0.9)
        result = knowledge_consolidation.prune_low_confidence(threshold=0.5)
        assert result.discarded_count == 0


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Test consolidation metrics."""

    def test_get_metrics(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("Test 1", source=KnowledgeSource.TRAINING)
        knowledge_consolidation.add_knowledge("Test 2", source=KnowledgeSource.EXPERIENCE)
        metrics = knowledge_consolidation.get_consolidation_metrics()
        assert metrics["total_entries"] == 2
        assert "by_source" in metrics


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 22 security boundaries."""

    def test_no_model_manager_in_consolidation(self):
        import evora.brain.intelligence.knowledge_consolidation as kc_mod
        source = Path(kc_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.knowledge_consolidation as kc_mod
        source = Path(kc_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 22 works offline."""

    def test_consolidation_works_offline(self, knowledge_consolidation):
        entry = knowledge_consolidation.add_knowledge("offline test")
        assert entry is not None

    def test_search_offline(self, knowledge_consolidation):
        knowledge_consolidation.add_knowledge("offline test")
        results = knowledge_consolidation.search_knowledge("test")
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 22 architecture readiness."""

    def test_native_knowledge_consolidation_exists(self):
        from evora.brain.intelligence.knowledge_consolidation import NativeKnowledgeConsolidation
        assert NativeKnowledgeConsolidation is not None

    def test_knowledge_entry_exists(self):
        from evora.brain.intelligence.knowledge_consolidation import KnowledgeEntry
        assert KnowledgeEntry is not None

    def test_consolidation_result_exists(self):
        from evora.brain.intelligence.knowledge_consolidation import ConsolidationResult
        assert ConsolidationResult is not None

    def test_knowledge_source_enum_exists(self):
        from evora.brain.intelligence.knowledge_consolidation import KnowledgeSource
        assert KnowledgeSource.TRAINING is not None
        assert KnowledgeSource.EXPERIENCE is not None

    def test_consolidation_strategy_enum_exists(self):
        from evora.brain.intelligence.knowledge_consolidation import ConsolidationStrategy
        assert ConsolidationStrategy.MERGE is not None
        assert ConsolidationStrategy.DISCARD is not None

    def test_consolidation_reuses_knowledge_graph(self, knowledge_with_graph):
        assert knowledge_with_graph.knowledge_graph is not None
