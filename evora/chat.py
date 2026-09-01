"""
Core chat session for EVORA.

Provides a reusable ChatSession that wires together model, identity,
memory, and reasoning so the CLI and any future UI share the same path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from evora.config import load_config
from evora.identity import IdentityService
from evora.logger import Logger
from evora.memory import Memory, MemoryService
from evora.model import Message, Role


class ChatSession:
    """Stateful conversational session backed by EVORA's infrastructure."""

    def __init__(self, config=None, logger=None, provider_override: Optional[str] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.logger = logger or Logger("evora-chat", config.log_level, config.log_file)
        self.provider_override = provider_override

        # Lazy imports to avoid circular dependency with cli.py
        from evora.cli import _build_model_manager, _chat_turn
        self._build_model_manager = _build_model_manager
        self._chat_turn = _chat_turn

        self.manager = _build_model_manager(config, self.logger, provider_override=provider_override)
        self.identity_service = IdentityService(identity_dir=config.identity_dir)
        self.memory = Memory(config.memory_dir, project_name=Path(config.workspace_dir).name)
        self.memory_service = MemoryService(
            memory=self.memory,
            identity_service=self.identity_service,
            logger=self.logger,
        )
        self.workspace_name = Path(config.workspace_dir).name
        self._closed = False

        current_identity = self.identity_service.current_identity()
        self.system_prompt = (
            "You are EVORA, an AI coding assistant. "
            f"Current identity: {current_identity.name} "
            f"(authority: {current_identity.authority.value}). "
            "Be helpful, concise, and direct."
        )
        self.messages: list[Message] = [Message(role=Role.SYSTEM, content=self.system_prompt)]

    async def process_message(self, user_input: str) -> dict:
        """Process a single user message and return the response payload."""
        response = await self._chat_turn(
            self.manager,
            self.messages,
            user_input,
            self.memory_service,
            self.workspace_name,
            self.logger,
        )
        active = self.manager.active
        identity = self.identity_service.current_identity()
        memory_count = 0
        try:
            memory_count = len(self.memory_service.list_memories(project=self.workspace_name, limit=100))
        except Exception:
            pass
        return {
            "response": response.content,
            "provider": active.name() if active else "none",
            "model": active.model() if active else "none",
            "identity": identity.name,
            "authority": identity.authority.value,
            "memory_count": memory_count,
        }

    async def stream_message(self, user_input: str):
        """Process a chat message and stream the response token-by-token.

        Yields event dicts:
            {"type": "content", "content": "chunk text"}
            {"type": "tool", "name": "...", "output": "...", "error": "..."}
            {"type": "error", "error": "..."}
        """
        messages = list(self.messages)
        messages.append(Message(role=Role.USER, content=user_input))
        request_messages = list(messages)

        context = ""
        try:
            if self.memory_service is not None:
                relevant = self.memory_service.retrieve_relevant(
                    goal=user_input,
                    project=self.workspace_name,
                    limit=5,
                )
                if relevant:
                    context = "\n".join([f"- {r.entry.content[:200]}" for r in relevant[:5]])
        except Exception as e:
            self.logger.debug(f"Memory retrieval skipped: {e}")

        if context:
            request_messages.insert(-1, Message(role=Role.SYSTEM, content=f"Relevant memories:\n{context}"))

        from evora.model import ChatRequest
        request = ChatRequest(messages=request_messages, max_tokens=4096, temperature=0.7, stream=True)

        active = self.manager.active
        full_content = ""
        try:
            async for chunk in active.chat_stream(request):
                if chunk.content:
                    full_content += chunk.content
                    yield {"type": "content", "content": chunk.content}

                if chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        yield {"type": "tool", "name": tc.name, "output": json.dumps(tc.arguments)}
        except Exception as e:
            yield {"type": "error", "error": str(e)}
            return

        self.messages = messages
        self.messages.append(Message(role=Role.ASSISTANT, content=full_content))
        yield {"type": "done", "response": full_content}

    def clear(self) -> None:
        self.messages = [Message(role=Role.SYSTEM, content=self.system_prompt)]

    def status(self) -> dict:
        active = self.manager.active
        identity = self.identity_service.current_identity()
        memory_count = 0
        try:
            memory_count = len(self.memory_service.list_memories(project=self.workspace_name, limit=100))
        except Exception:
            pass
        return {
            "provider": active.name() if active else "none",
            "model": active.model() if active else "none",
            "identity": identity.name,
            "authority": identity.authority.value,
            "workspace": self.config.workspace_dir,
            "memory_count": memory_count,
        }

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.manager.close()
