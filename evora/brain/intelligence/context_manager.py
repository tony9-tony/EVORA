"""
Phase 24 — Native Context Manager for EVORA.

Manages context windows and prompt optimization.

Supports:
  - Context window management
  - Context summarization
  - Prompt optimization
  - Token counting
  - Context pruning
  - Integration with ConversationManager
  - Integration with IntelligenceRuntime

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

class ContextType(str, Enum):
    CONVERSATION = "conversation"
    TASK = "task"
    REASONING = "reasoning"
    PLANNING = "planning"
    EXECUTION = "execution"


class PruningStrategy(str, Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    IMPORTANCE = "importance"
    RELEVANCE = "relevance"


@dataclass
class ContextItem:
    """A context item."""
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    context_type: ContextType = ContextType.CONVERSATION
    content: str = ""
    tokens: int = 0
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "context_type": self.context_type.value,
            "content": self.content,
            "tokens": self.tokens,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class ContextWindow:
    """A context window."""
    window_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    max_tokens: int = 4096
    current_tokens: int = 0
    items: list[ContextItem] = field(default_factory=list)
    strategy: PruningStrategy = PruningStrategy.RELEVANCE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
            "item_count": len(self.items),
            "strategy": self.strategy.value,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Native Context Manager
# ---------------------------------------------------------------------------

class NativeContextManager:
    """Native context manager for EVORA.

    Manages context windows and prompt optimization.
    """

    def __init__(
        self,
        conversation_manager: Any = None,
        intelligence_runtime: Any = None,
        logger: Optional[Any] = None,
    ):
        self.conversation_manager = conversation_manager
        self.intelligence_runtime = intelligence_runtime
        self.logger = logger
        self._windows: dict[str, ContextWindow] = {}
        self._default_max_tokens = 4096

    def create_window(self, window_id: str = None, max_tokens: int = None, strategy: PruningStrategy = None) -> ContextWindow:
        """Create a new context window."""
        window = ContextWindow(
            window_id=window_id or uuid.uuid4().hex[:12],
            max_tokens=max_tokens or self._default_max_tokens,
            strategy=strategy or PruningStrategy.RELEVANCE,
        )
        self._windows[window.window_id] = window
        return window

    def add_context(self, window_id: str, content: str, context_type: ContextType = ContextType.CONVERSATION, importance: float = 0.5, tokens: int = None) -> Optional[ContextItem]:
        """Add context to a window."""
        window = self._windows.get(window_id)
        if window is None:
            return None
        if tokens is None:
            tokens = len(content.split())
        item = ContextItem(
            context_type=context_type,
            content=content,
            tokens=tokens,
            importance=importance,
        )
        window.items.append(item)
        window.current_tokens += tokens
        self._prune_window(window)
        return item

    def get_context(self, window_id: str) -> list[ContextItem]:
        """Get all context items in a window."""
        window = self._windows.get(window_id)
        return list(window.items) if window else []

    def get_window_summary(self, window_id: str) -> Optional[dict[str, Any]]:
        """Get a summary of a context window."""
        window = self._windows.get(window_id)
        if window is None:
            return None
        return {
            "window_id": window.window_id,
            "max_tokens": window.max_tokens,
            "current_tokens": window.current_tokens,
            "remaining_tokens": window.max_tokens - window.current_tokens,
            "item_count": len(window.items),
            "strategy": window.strategy.value,
            "utilization": window.current_tokens / window.max_tokens if window.max_tokens > 0 else 0.0,
        }

    def prune_window(self, window_id: str) -> bool:
        """Prune a window to fit within token limits."""
        window = self._windows.get(window_id)
        if window is None:
            return False
        return self._prune_window(window)

    def _prune_window(self, window: ContextWindow) -> bool:
        """Prune items from a window if over limit."""
        if window.current_tokens <= window.max_tokens:
            return False
        if window.strategy == PruningStrategy.FIFO:
            while window.current_tokens > window.max_tokens and window.items:
                removed = window.items.pop(0)
                window.current_tokens -= removed.tokens
        elif window.strategy == PruningStrategy.LIFO:
            while window.current_tokens > window.max_tokens and window.items:
                removed = window.items.pop()
                window.current_tokens -= removed.tokens
        elif window.strategy == PruningStrategy.IMPORTANCE:
            sorted_items = sorted(window.items, key=lambda x: x.importance)
            while window.current_tokens > window.max_tokens and sorted_items:
                removed = sorted_items.pop(0)
                window.items.remove(removed)
                window.current_tokens -= removed.tokens
        elif window.strategy == PruningStrategy.RELEVANCE:
            sorted_items = sorted(window.items, key=lambda x: x.importance)
            while window.current_tokens > window.max_tokens and sorted_items:
                removed = sorted_items.pop(0)
                window.items.remove(removed)
                window.current_tokens -= removed.tokens
        return True

    def optimize_prompt(self, window_id: str, target_tokens: int = None) -> Optional[str]:
        """Optimize a prompt by summarizing context."""
        window = self._windows.get(window_id)
        if window is None:
            return None
        target_tokens = target_tokens or window.max_tokens
        if window.current_tokens <= target_tokens:
            return "\n".join(item.content for item in window.items)
        sorted_items = sorted(window.items, key=lambda x: x.importance, reverse=True)
        selected = []
        total_tokens = 0
        for item in sorted_items:
            if total_tokens + item.tokens <= target_tokens:
                selected.append(item)
                total_tokens += item.tokens
        return "\n".join(item.content for item in selected)

    def get_context_stats(self) -> dict[str, Any]:
        """Get context statistics."""
        total_windows = len(self._windows)
        total_tokens = sum(w.current_tokens for w in self._windows.values())
        total_max = sum(w.max_tokens for w in self._windows.values())
        return {
            "total_windows": total_windows,
            "total_tokens": total_tokens,
            "total_max_tokens": total_max,
            "utilization": total_tokens / total_max if total_max > 0 else 0.0,
        }

    def clear_window(self, window_id: str) -> bool:
        """Clear a context window."""
        window = self._windows.get(window_id)
        if window is None:
            return False
        window.items = []
        window.current_tokens = 0
        return True
