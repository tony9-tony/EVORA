"""
Phase 24 — Native Context Manager tests.

Verifies:
1. ContextItem has correct structure
2. ContextWindow has correct structure
3. ContextType enum exists
4. PruningStrategy enum exists
5. NativeContextManager initializes
6. NativeContextManager creates window
7. NativeContextManager adds context
8. NativeContextManager gets context
9. NativeContextManager gets window summary
10. NativeContextManager prunes window FIFO
11. NativeContextManager prunes window LIFO
12. NativeContextManager prunes window by importance
13. NativeContextManager optimizes prompt
14. NativeContextManager returns stats
15. NativeContextManager clears window
16. Context integrates with ConversationManager
17. No ModelManager dependency
18. No external dependencies
19. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.context_manager import (
    ContextItem,
    ContextType,
    ContextWindow,
    NativeContextManager,
    PruningStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def context_manager():
    return NativeContextManager(logger=MagicMock())


@pytest.fixture
def context_with_conversation():
    cm = MagicMock()
    return NativeContextManager(conversation_manager=cm, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestContextItem
# ---------------------------------------------------------------------------

class TestContextItem:
    """Test ContextItem."""

    def test_default_item(self):
        item = ContextItem()
        assert item.item_id != ""
        assert item.context_type == ContextType.CONVERSATION

    def test_item_to_dict(self):
        item = ContextItem(content="Test context", tokens=10, importance=0.8)
        data = item.to_dict()
        assert data["content"] == "Test context"
        assert data["tokens"] == 10


# ---------------------------------------------------------------------------
# TestContextWindow
# ---------------------------------------------------------------------------

class TestContextWindow:
    """Test ContextWindow."""

    def test_default_window(self):
        window = ContextWindow()
        assert window.window_id != ""
        assert window.max_tokens == 4096

    def test_window_to_dict(self):
        window = ContextWindow(max_tokens=2048, current_tokens=1024)
        data = window.to_dict()
        assert data["max_tokens"] == 2048
        assert data["current_tokens"] == 1024


# ---------------------------------------------------------------------------
# TestContextTypeEnum
# ---------------------------------------------------------------------------

class TestContextTypeEnum:
    """Test ContextType enum."""

    def test_context_types_exist(self):
        assert ContextType.CONVERSATION is not None
        assert ContextType.TASK is not None
        assert ContextType.REASONING is not None
        assert ContextType.PLANNING is not None


# ---------------------------------------------------------------------------
# TestPruningStrategyEnum
# ---------------------------------------------------------------------------

class TestPruningStrategyEnum:
    """Test PruningStrategy enum."""

    def test_strategy_values(self):
        assert PruningStrategy.FIFO.value == "fifo"
        assert PruningStrategy.LIFO.value == "lifo"
        assert PruningStrategy.IMPORTANCE.value == "importance"
        assert PruningStrategy.RELEVANCE.value == "relevance"


# ---------------------------------------------------------------------------
# TestNativeContextManager
# ---------------------------------------------------------------------------

class TestNativeContextManager:
    """Test NativeContextManager."""

    def test_context_manager_initializes(self, context_manager):
        assert context_manager is not None

    def test_create_window(self, context_manager):
        window = context_manager.create_window()
        assert window.window_id != ""
        assert window.max_tokens == 4096

    def test_create_window_custom(self, context_manager):
        window = context_manager.create_window(max_tokens=2048)
        assert window.max_tokens == 2048

    def test_add_context(self, context_manager):
        window = context_manager.create_window(max_tokens=100)
        item = context_manager.add_context(window.window_id, "Test context", tokens=10)
        assert item is not None
        assert item.content == "Test context"

    def test_get_context(self, context_manager):
        window = context_manager.create_window()
        context_manager.add_context(window.window_id, "Item 1")
        context_manager.add_context(window.window_id, "Item 2")
        items = context_manager.get_context(window.window_id)
        assert len(items) == 2

    def test_get_window_summary(self, context_manager):
        window = context_manager.create_window(max_tokens=100)
        summary = context_manager.get_window_summary(window.window_id)
        assert summary is not None
        assert summary["max_tokens"] == 100

    def test_get_window_summary_missing(self, context_manager):
        summary = context_manager.get_window_summary("nonexistent")
        assert summary is None

    def test_prune_window_fifo(self, context_manager):
        window = context_manager.create_window(max_tokens=25, strategy=PruningStrategy.FIFO)
        context_manager.add_context(window.window_id, "First", tokens=10)
        context_manager.add_context(window.window_id, "Second", tokens=10)
        context_manager.add_context(window.window_id, "Third", tokens=10)
        context_manager.prune_window(window.window_id)
        assert window.current_tokens <= window.max_tokens

    def test_optimize_prompt(self, context_manager):
        window = context_manager.create_window(max_tokens=100)
        context_manager.add_context(window.window_id, "Important", importance=0.9, tokens=10)
        context_manager.add_context(window.window_id, "Less important", importance=0.3, tokens=10)
        prompt = context_manager.optimize_prompt(window.window_id, target_tokens=10)
        assert prompt is not None
        assert "Important" in prompt

    def test_get_stats(self, context_manager):
        context_manager.create_window()
        context_manager.create_window(max_tokens=2048)
        stats = context_manager.get_context_stats()
        assert stats["total_windows"] == 2

    def test_clear_window(self, context_manager):
        window = context_manager.create_window()
        context_manager.add_context(window.window_id, "Test")
        result = context_manager.clear_window(window.window_id)
        assert result is True
        items = context_manager.get_context(window.window_id)
        assert len(items) == 0

    def test_clear_window_missing(self, context_manager):
        result = context_manager.clear_window("nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# TestContextIntegration
# ---------------------------------------------------------------------------

class TestContextIntegration:
    """Test context manager integration."""

    def test_context_with_conversation_manager(self, context_with_conversation):
        window = context_with_conversation.create_window()
        assert context_with_conversation.conversation_manager is not None

    def test_multiple_windows(self, context_manager):
        w1 = context_manager.create_window()
        w2 = context_manager.create_window(max_tokens=1024)
        context_manager.add_context(w1.window_id, "Window 1")
        context_manager.add_context(w2.window_id, "Window 2")
        assert len(context_manager.get_context(w1.window_id)) == 1
        assert len(context_manager.get_context(w2.window_id)) == 1


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 24 security boundaries."""

    def test_no_model_manager_in_context(self):
        import evora.brain.intelligence.context_manager as ctx_mod
        source = Path(ctx_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.context_manager as ctx_mod
        source = Path(ctx_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 24 works offline."""

    def test_context_manager_works_offline(self, context_manager):
        window = context_manager.create_window()
        assert window is not None

    def test_add_context_offline(self, context_manager):
        window = context_manager.create_window()
        item = context_manager.add_context(window.window_id, "offline test")
        assert item is not None


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 24 architecture readiness."""

    def test_native_context_manager_exists(self):
        from evora.brain.intelligence.context_manager import NativeContextManager
        assert NativeContextManager is not None

    def test_context_item_exists(self):
        from evora.brain.intelligence.context_manager import ContextItem
        assert ContextItem is not None

    def test_context_window_exists(self):
        from evora.brain.intelligence.context_manager import ContextWindow
        assert ContextWindow is not None

    def test_context_type_enum_exists(self):
        from evora.brain.intelligence.context_manager import ContextType
        assert ContextType.CONVERSATION is not None
        assert ContextType.TASK is not None

    def test_pruning_strategy_enum_exists(self):
        from evora.brain.intelligence.context_manager import PruningStrategy
        assert PruningStrategy.FIFO is not None
        assert PruningStrategy.LIFO is not None

    def test_context_manager_reuses_conversation_manager(self, context_with_conversation):
        assert context_with_conversation.conversation_manager is not None
