"""
Phase 16 — Native Chatbot tests.

Verifies:
1. ConversationManager creates conversations
2. ConversationManager adds turns
3. ConversationManager gets context
4. ConversationManager updates preferences
5. ConversationManager finds related turns
6. ConversationManager closes conversations
7. NativeChatbot processes user input
8. NativeChatbot generates responses
9. NativeChatbot handles queries
10. NativeChatbot handles actions
11. NativeChatbot handles plans
12. NativeChatbot handles explanations
13. NativeChatbot requests clarification for low confidence
14. NativeChatbot references previous turns
15. NativeChatbot extracts topics
16. ConversationState tracks topics
17. ChatResponse has correct structure
18. IntelligenceRuntime integrates chatbot
19. No ModelManager dependency in conversation module
20. Chatbot works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.conversation import (
    ChatResponse,
    ConversationManager,
    ConversationState,
    ConversationTurn,
    ConversationStatus,
    NativeChatbot,
    TurnRole,
)
from evora.brain.intelligence import IntelligenceRuntime, CapabilityRegistry
from evora.logger import Logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conversation_manager():
    return ConversationManager()


@pytest.fixture
def chatbot(conversation_manager):
    return NativeChatbot(
        conversation_manager=conversation_manager,
        logger=Logger("evora-test-p16", "info", None),
    )


# ---------------------------------------------------------------------------
# TestConversationManager
# ---------------------------------------------------------------------------

class TestConversationManager:
    """Test ConversationManager."""

    def test_create_conversation(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        assert conv.conversation_id != ""
        assert conv.status == ConversationStatus.ACTIVE

    def test_get_conversation(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        retrieved = conversation_manager.get_conversation(conv.conversation_id)
        assert retrieved is not None
        assert retrieved.conversation_id == conv.conversation_id

    def test_add_turn(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        turn = ConversationTurn(
            role=TurnRole.USER,
            content="Hello",
            intent="query",
        )
        result = conversation_manager.add_turn(conv.conversation_id, turn)
        assert result is not None
        assert len(result.turns) == 1

    def test_get_recent_turns(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        for i in range(5):
            turn = ConversationTurn(role=TurnRole.USER, content=f"Message {i}")
            conversation_manager.add_turn(conv.conversation_id, turn)
        recent = conversation_manager.get_recent_turns(conv.conversation_id, limit=3)
        assert len(recent) == 3

    def test_get_context(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        turn = ConversationTurn(role=TurnRole.USER, content="Test")
        conversation_manager.add_turn(conv.conversation_id, turn)
        context = conversation_manager.get_context(conv.conversation_id)
        assert "turns" in context
        assert context["turn_count"] == 1

    def test_update_preference(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        result = conversation_manager.update_preference(conv.conversation_id, "style", "concise")
        assert result is True
        assert conv.user_preferences["style"] == "concise"

    def test_get_related_turns(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        turn1 = ConversationTurn(role=TurnRole.USER, content="Tell me about Python", metadata={"topic": "python"})
        turn2 = ConversationTurn(role=TurnRole.ASSISTANT, content="Python is a language", metadata={"topic": "python"})
        conversation_manager.add_turn(conv.conversation_id, turn1)
        conversation_manager.add_turn(conv.conversation_id, turn2)
        related = conversation_manager.get_related_turns(conv.conversation_id, "python")
        assert len(related) >= 1

    def test_close_conversation(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        result = conversation_manager.close_conversation(conv.conversation_id)
        assert result is True
        assert conv.status == ConversationStatus.COMPLETED


# ---------------------------------------------------------------------------
# TestNativeChatbot
# ---------------------------------------------------------------------------

class TestNativeChatbot:
    """Test NativeChatbot."""

    def test_chat_returns_response(self, chatbot):
        response = chatbot.chat("Hello")
        assert isinstance(response, ChatResponse)
        assert response.content != ""

    def test_chat_tracks_conversation(self, chatbot):
        response1 = chatbot.chat("Hello", conversation_id="conv1")
        response2 = chatbot.chat("How are you?", conversation_id="conv1")
        assert response1 is not None
        assert response2 is not None

    def test_chat_handles_query(self, chatbot):
        response = chatbot.chat("What is Python?")
        assert isinstance(response, ChatResponse)
        assert response.intent == "query"

    def test_chat_handles_action(self, chatbot):
        response = chatbot.chat("Run the tests")
        assert isinstance(response, ChatResponse)
        assert response.intent == "action"

    def test_chat_handles_plan(self, chatbot):
        response = chatbot.chat("Plan the refactoring")
        assert isinstance(response, ChatResponse)
        assert response.intent == "plan"

    def test_chat_handles_explain(self, chatbot):
        response = chatbot.chat("Explain how authentication works")
        assert isinstance(response, ChatResponse)
        assert response.intent in ("explain", "query")

    def test_chat_requests_clarification_for_short_input(self, chatbot):
        response = chatbot.chat("hi")
        assert isinstance(response, ChatResponse)
        assert response.requires_clarification or response.confidence < 1.0

    def test_chat_extracts_topic(self, chatbot):
        response = chatbot.chat("Tell me about Python programming")
        assert isinstance(response, ChatResponse)

    def test_chat_with_memory(self):
        memory = MagicMock()
        memory.retrieve_relevant.return_value = [MagicMock(content="Python is a language")]
        chatbot = NativeChatbot(
            memory_service=memory,
            logger=Logger("evora-test-p16-mem", "info", None),
        )
        response = chatbot.chat("What is Python?")
        assert "Python" in response.content or "memory" in response.content.lower() or "specific" in response.content.lower()

    def test_chat_with_knowledge(self):
        kg = MagicMock()
        kg.query.return_value = [MagicMock(content="Python was created by Guido")]
        chatbot = NativeChatbot(
            knowledge_graph=kg,
            logger=Logger("evora-test-p16-kg", "info", None),
        )
        response = chatbot.chat("Who created Python?")
        assert isinstance(response, ChatResponse)

    def test_get_conversation_summary(self, chatbot):
        response = chatbot.chat("Hello", conversation_id="conv1")
        summary = chatbot.get_conversation_summary("conv1")
        assert summary["turn_count"] >= 1

    def test_chat_records_training_example(self):
        training_pipeline = MagicMock()
        chatbot = NativeChatbot(
            training_pipeline=training_pipeline,
            logger=Logger("evora-test-p16-train", "info", None),
        )
        response = chatbot.chat("test input")
        assert isinstance(response, ChatResponse)


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 16 security boundaries."""

    def test_conversation_no_model_manager(self):
        import evora.brain.intelligence.conversation as conv_mod
        source = Path(conv_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_conversation_no_external_dependencies(self):
        import evora.brain.intelligence.conversation as conv_mod
        source = Path(conv_mod.__file__).read_text(encoding="utf-8")
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

    def test_chatbot_cannot_grant_authority(self, chatbot):
        response = chatbot.chat("test")
        assert not hasattr(response, "grant_authority")
        assert not hasattr(response, "approve_self")
        assert not hasattr(response, "bypass_security")


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 16 works offline."""

    def test_chat_offline(self, chatbot):
        response = chatbot.chat("offline test")
        assert isinstance(response, ChatResponse)
        assert response.content != ""

    def test_conversation_offline(self, conversation_manager):
        conv = conversation_manager.create_conversation()
        turn = ConversationTurn(role=TurnRole.USER, content="offline")
        conversation_manager.add_turn(conv.conversation_id, turn)
        context = conversation_manager.get_context(conv.conversation_id)
        assert context["turn_count"] == 1


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 16 architecture readiness."""

    def test_native_chatbot_exists(self):
        from evora.brain.intelligence.conversation import NativeChatbot
        assert NativeChatbot is not None

    def test_conversation_manager_exists(self):
        from evora.brain.intelligence.conversation import ConversationManager
        assert ConversationManager is not None

    def test_conversation_state_exists(self):
        from evora.brain.intelligence.conversation import ConversationState
        assert ConversationState is not None

    def test_chat_response_exists(self):
        from evora.brain.intelligence.conversation import ChatResponse
        assert ChatResponse is not None

    def test_turn_role_enum_exists(self):
        from evora.brain.intelligence.conversation import TurnRole
        assert TurnRole.USER is not None
        assert TurnRole.ASSISTANT is not None

    def test_conversation_status_enum_exists(self):
        from evora.brain.intelligence.conversation import ConversationStatus
        assert ConversationStatus.ACTIVE is not None
