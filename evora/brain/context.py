"""
Phase 9 — Context construction for the EVORA Brain.

Builds bounded, relevant context for inference by retrieving:
  - long-term memories
  - validated knowledge
  - relevant experiences
  - current brain state
  - self-model summary

Does NOT blindly inject the entire memory database.
Maintains distinction between memory, knowledge, experience,
current state, and self-model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class BrainContext:
    """Structured context for Brain inference."""

    goal: str = ""
    project: str = ""
    recent_memories: list[dict[str, Any]] = field(default_factory=list)
    relevant_knowledge: list[dict[str, Any]] = field(default_factory=list)
    relevant_experiences: list[dict[str, Any]] = field(default_factory=list)
    current_state: dict[str, Any] = field(default_factory=dict)
    self_model_summary: dict[str, Any] = field(default_factory=dict)
    active_tools: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        """Render a concise prompt context string."""
        parts = []
        if self.goal:
            parts.append(f"Goal: {self.goal}")
        if self.project:
            parts.append(f"Project: {self.project}")
        if self.recent_memories:
            parts.append(f"Recent memories ({len(self.recent_memories)}):")
            for m in self.recent_memories[:3]:
                parts.append(f"  - {m.get('content', '')[:120]}")
        if self.relevant_knowledge:
            parts.append(f"Relevant knowledge ({len(self.relevant_knowledge)}):")
            for k in self.relevant_knowledge[:3]:
                parts.append(f"  - {k.get('content', '')[:120]}")
        if self.current_state:
            parts.append(f"Current state: {self.current_state.get('development_state', 'idle')}")
        if self.active_tools:
            parts.append(f"Available tools: {', '.join(self.active_tools[:10])}")
        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")
        return "\n".join(parts) if parts else "No context available."


class ContextBuilder:
    """Builds bounded, relevant context for Brain inference."""

    def __init__(
        self,
        memory_service: Any = None,
        knowledge_base: Any = None,
        experience_store: Any = None,
        self_model: Any = None,
        brain_state: Any = None,
        tool_registry: Any = None,
        logger: Optional[Logger] = None,
    ):
        self.memory_service = memory_service
        self.knowledge_base = knowledge_base
        self.experience_store = experience_store
        self.self_model = self_model
        self.brain_state = brain_state
        self.tool_registry = tool_registry
        self.logger = logger

    def build(self, goal: str, project: str = "") -> BrainContext:
        """Build context for a given goal."""
        ctx = BrainContext(goal=goal, project=project)

        if self.brain_state is not None:
            ctx.current_state = self.brain_state.to_dict()
            ctx.active_tools = list(self.brain_state.active_tools)
            ctx.constraints = list(self.brain_state.known_constraints)

        if self.memory_service is not None:
            try:
                memories = self.memory_service.retrieve_relevant(
                    goal=goal,
                    project=project or None,
                    limit=5,
                )
                ctx.recent_memories = [m.to_dict() for m in memories]
            except Exception:
                pass

        if self.knowledge_base is not None:
            try:
                ctx.relevant_knowledge = self.knowledge_base.retrieve_relevant(
                    goal=goal,
                    project=project or "",
                    limit=5,
                )
            except Exception:
                pass

        if self.experience_store is not None:
            try:
                experiences = self.experience_store.list_recent(limit=5, project=project)
                ctx.relevant_experiences = [e.to_dict() for e in experiences]
            except Exception:
                pass

        if self.self_model is not None:
            try:
                ctx.self_model_summary = self.self_model.to_dict()
            except Exception:
                pass

        if self.tool_registry is not None and not ctx.active_tools:
            try:
                ctx.active_tools = list(self.tool_registry.list())
            except Exception:
                pass

        if self.logger:
            self.logger.observe(
                f"Built context: {len(ctx.recent_memories)} memories, "
                f"{len(ctx.relevant_knowledge)} knowledge, "
                f"{len(ctx.relevant_experiences)} experiences"
            )

        return ctx
