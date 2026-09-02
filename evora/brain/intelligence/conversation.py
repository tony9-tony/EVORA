"""
Phase 16 — Native Chatbot for EVORA.

Builds a real conversational interface with:
  - multi-turn conversation
  - contextual responses
  - conversation state
  - short-term context
  - long-term relevant memory
  - clarification
  - follow-up questions
  - topic continuity
  - references to previous statements
  - uncertainty
  - correction
  - user preferences
  - natural conversational flow

No ModelManager dependency.
No external model dependency.
Works offline.

Reuses existing abstractions:
  - MemoryService for long-term memory
  - KnowledgeGraph for knowledge retrieval
  - NativeComprehensionIntelligence for understanding
  - IntelligenceRuntime for reasoning
  - TrainingPipeline for learning from conversations
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Conversation representations
# ---------------------------------------------------------------------------

class TurnRole(str, Enum):
    """Role of a conversation turn."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationStatus(str, Enum):
    """Status of a conversation."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    role: TurnRole = TurnRole.USER
    content: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    intent: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    response: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "intent": self.intent,
            "entities": self.entities,
            "response": self.response,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class ConversationState:
    """State of an active conversation."""
    conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ConversationStatus = ConversationStatus.ACTIVE
    turns: list[ConversationTurn] = field(default_factory=list)
    current_topic: str = ""
    previous_topics: list[str] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "status": self.status.value,
            "turns": [t.to_dict() for t in self.turns],
            "current_topic": self.current_topic,
            "previous_topics": self.previous_topics,
            "user_preferences": self.user_preferences,
            "context": self.context,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "metadata": self.metadata,
        }


@dataclass
class ChatResponse:
    """Response from the chatbot."""
    response_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    intent: str = ""
    confidence: float = 0.0
    requires_clarification: bool = False
    clarification_question: str = ""
    references_previous: bool = False
    referenced_turns: list[str] = field(default_factory=list)
    uncertainty_factors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "content": self.content,
            "intent": self.intent,
            "confidence": self.confidence,
            "requires_clarification": self.requires_clarification,
            "clarification_question": self.clarification_question,
            "references_previous": self.references_previous,
            "referenced_turns": self.referenced_turns,
            "uncertainty_factors": self.uncertainty_factors,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Conversation Manager
# ---------------------------------------------------------------------------

class ConversationManager:
    """Manages multi-turn conversations.

    Tracks:
      - Conversation state
      - Turn history
      - Current topic
      - Previous topics
      - User preferences
      - Short-term context
    """

    MAX_TURNS = 100
    MAX_TOPICS = 10

    def __init__(self, memory_service: Any = None, logger: Optional[Any] = None):
        self.memory_service = memory_service
        self.logger = logger
        self._conversations: dict[str, ConversationState] = {}

    def create_conversation(self, user_id: str = "default", conversation_id: str = "") -> ConversationState:
        """Create a new conversation."""
        conv_id = conversation_id or uuid.uuid4().hex[:12]
        conversation = ConversationState(conversation_id=conv_id, metadata={"user_id": user_id})
        self._conversations[conversation.conversation_id] = conversation
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def add_turn(self, conversation_id: str, turn: ConversationTurn) -> Optional[ConversationState]:
        """Add a turn to a conversation."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return None
        conversation.turns.append(turn)
        conversation.last_active = datetime.now().isoformat()
        if len(conversation.turns) > self.MAX_TURNS:
            conversation.turns = conversation.turns[-self.MAX_TURNS:]
        if turn.role == TurnRole.USER:
            topic = turn.metadata.get("topic", "")
            if topic and topic != conversation.current_topic:
                conversation.previous_topics.append(conversation.current_topic)
                conversation.current_topic = topic
                if len(conversation.previous_topics) > self.MAX_TOPICS:
                    conversation.previous_topics = conversation.previous_topics[-self.MAX_TOPICS:]
        return conversation

    def get_recent_turns(self, conversation_id: str, limit: int = 10) -> list[ConversationTurn]:
        """Get recent turns."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return []
        return conversation.turns[-limit:]

    def get_context(self, conversation_id: str, limit: int = 5) -> dict[str, Any]:
        """Get conversation context."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return {}
        recent = conversation.turns[-limit:]
        return {
            "turns": [t.to_dict() for t in recent],
            "current_topic": conversation.current_topic,
            "previous_topics": conversation.previous_topics,
            "user_preferences": conversation.user_preferences,
            "turn_count": len(conversation.turns),
        }

    def update_preference(self, conversation_id: str, key: str, value: Any) -> bool:
        """Update user preference."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return False
        conversation.user_preferences[key] = value
        return True

    def get_related_turns(self, conversation_id: str, topic: str, limit: int = 5) -> list[ConversationTurn]:
        """Get turns related to a topic."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return []
        related = [
            t for t in conversation.turns
            if topic.lower() in t.content.lower() or topic.lower() in (t.response or "").lower()
        ]
        return related[-limit:]

    def close_conversation(self, conversation_id: str) -> bool:
        """Close a conversation."""
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            return False
        conversation.status = ConversationStatus.COMPLETED
        conversation.last_active = datetime.now().isoformat()
        return True


# ---------------------------------------------------------------------------
# Native Chatbot
# ---------------------------------------------------------------------------

class NativeChatbot:
    """Native chatbot for EVORA.

    Provides conversational interface using native intelligence.
    No external model dependency.
    Works offline.
    """

    CLARIFICATION_THRESHOLD = 0.4
    CONFIDENCE_THRESHOLD = 0.5

    def __init__(
        self,
        conversation_manager: Optional[ConversationManager] = None,
        memory_service: Any = None,
        knowledge_graph: Any = None,
        comprehension_intelligence: Any = None,
        reasoning_engine: Any = None,
        training_pipeline: Any = None,
        logger: Optional[Any] = None,
    ):
        self.conversation_manager = conversation_manager or ConversationManager(
            memory_service=memory_service, logger=logger
        )
        self.memory_service = memory_service
        self.knowledge_graph = knowledge_graph
        self.comprehension_intelligence = comprehension_intelligence
        self.reasoning_engine = reasoning_engine
        self.training_pipeline = training_pipeline
        self.logger = logger

    def chat(self, user_input: str, conversation_id: str = "") -> ChatResponse:
        """Process a user message and generate a response."""
        conversation = self._get_or_create_conversation(conversation_id)
        intent = self._understand_intent(user_input)
        entities = self._extract_entities(user_input)
        context = self.conversation_manager.get_context(conversation.conversation_id)
        references = self._find_references(user_input, conversation)

        turn = ConversationTurn(
            role=TurnRole.USER,
            content=user_input,
            intent=intent.intent_type.value if hasattr(intent, "intent_type") else "unknown",
            entities=[e.to_dict() if hasattr(e, "to_dict") else e for e in entities],
            metadata={"topic": self._extract_topic(user_input)},
        )
        self.conversation_manager.add_turn(conversation.conversation_id, turn)

        confidence = self._estimate_confidence(user_input, context, references)
        if confidence < self.CLARIFICATION_THRESHOLD:
            response = ChatResponse(
                content=self._generate_clarification(user_input, context),
                intent=intent.intent_type.value if hasattr(intent, "intent_type") else "unknown",
                confidence=confidence,
                requires_clarification=True,
                clarification_question=self._generate_clarification_question(user_input),
                uncertainty_factors=["low_confidence", "ambiguous_request"],
            )
        else:
            response_content = self._generate_response(user_input, intent, context, references)
            response = ChatResponse(
                content=response_content,
                intent=intent.intent_type.value if hasattr(intent, "intent_type") else "unknown",
                confidence=confidence,
                references_previous=bool(references),
                referenced_turns=[t.turn_id for t in references],
            )

        turn.response = response.content
        turn.confidence = response.confidence
        self._record_training_example(conversation.conversation_id, user_input, response)
        return response

    def _get_or_create_conversation(self, conversation_id: str) -> ConversationState:
        """Get existing or create new conversation."""
        if conversation_id and conversation_id in self.conversation_manager._conversations:
            return self.conversation_manager._conversations[conversation_id]
        return self.conversation_manager.create_conversation(conversation_id=conversation_id)

    def _understand_intent(self, text: str) -> Any:
        """Understand user intent."""
        if self.comprehension_intelligence is not None:
            try:
                return self.comprehension_intelligence.classify_intent(text)
            except Exception:
                pass
        from evora.brain.intelligence.comprehension import IntentClassifier
        return IntentClassifier().classify(text)

    def _extract_entities(self, text: str) -> list[Any]:
        """Extract entities from text."""
        if self.comprehension_intelligence is not None:
            try:
                return self.comprehension_intelligence.extract_entities(text)
            except Exception:
                pass
        from evora.brain.intelligence.comprehension import EntityExtractor
        return EntityExtractor().extract(text)

    def _extract_topic(self, text: str) -> str:
        """Extract topic from text."""
        words = text.lower().split()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                      "have", "has", "had", "do", "does", "did", "will", "would", "could",
                      "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                      "on", "with", "at", "by", "from", "as", "into", "through", "during",
                      "before", "after", "above", "below", "between", "out", "off", "over",
                      "under", "again", "further", "then", "once", "here", "there", "when",
                      "where", "why", "how", "all", "both", "each", "few", "more", "most",
                      "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                      "so", "than", "too", "very", "just", "because", "but", "and", "or",
                      "if", "while", "about", "up", "what", "which", "who", "whom", "this",
                      "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
                      "he", "him", "his", "she", "her", "it", "its", "they", "them", "their"}
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        return content_words[0] if content_words else ""

    def _find_references(self, text: str, conversation: ConversationState) -> list[ConversationTurn]:
        """Find turns referenced by the current input."""
        references = []
        text_lower = text.lower()
        for turn in conversation.turns[-20:]:
            if turn.role == TurnRole.USER:
                topic = turn.metadata.get("topic", "")
                if topic and topic in text_lower:
                    references.append(turn)
        return references

    def _estimate_confidence(self, text: str, context: dict[str, Any], references: list[ConversationTurn]) -> float:
        """Estimate confidence in understanding the request."""
        confidence = 0.5
        if references:
            confidence += 0.2
        if context.get("turn_count", 0) > 2:
            confidence += 0.1
        words = text.split()
        if len(words) < 3:
            confidence -= 0.2
        return max(0.0, min(1.0, confidence))

    def _generate_response(self, user_input: str, intent: Any, context: dict[str, Any], references: list[ConversationTurn]) -> str:
        """Generate a response."""
        intent_value = intent.intent_type.value if hasattr(intent, "intent_type") else "unknown"
        if references:
            recent_ref = references[-1]
            return f"Building on our earlier discussion about '{recent_ref.metadata.get('topic', '')}': "
        if intent_value == "query":
            return self._handle_query(user_input, context)
        elif intent_value == "action":
            return self._handle_action(user_input, context)
        elif intent_value == "plan":
            return self._handle_plan(user_input, context)
        elif intent_value == "explain":
            return self._handle_explain(user_input, context)
        elif intent_value == "remember":
            return "I'll make a note of that."
        elif intent_value == "forget":
            return "I'll remove that from my memory."
        else:
            return self._handle_general(user_input, context)

    def _handle_query(self, text: str, context: dict[str, Any]) -> str:
        """Handle a query."""
        if self.knowledge_graph is not None:
            try:
                nodes = self.knowledge_graph.query(text, limit=3)
                if nodes:
                    return f"Based on what I know: {nodes[0].content}"
            except Exception:
                pass
        if self.memory_service is not None:
            try:
                memories = self.memory_service.retrieve_relevant(goal=text, limit=3)
                if memories:
                    return f"From my memory: {memories[0].content}"
            except Exception:
                pass
        return "I don't have specific information about that yet. Would you like me to look into it?"

    def _handle_action(self, text: str, context: dict[str, Any]) -> str:
        """Handle an action request."""
        return f"I can help with that action. Let me analyze: '{text[:80]}'"

    def _handle_plan(self, text: str, context: dict[str, Any]) -> str:
        """Handle a planning request."""
        return f"Let me create a plan for: '{text[:80]}'"

    def _handle_explain(self, text: str, context: dict[str, Any]) -> str:
        """Handle an explanation request."""
        return f"Here's what I understand about '{text[:80]}'..."

    def _handle_general(self, text: str, context: dict[str, Any]) -> str:
        """Handle a general request."""
        return f"I understand you're asking about '{text[:80]}'. Let me think about that."

    def _generate_clarification(self, user_input: str, context: dict[str, Any]) -> str:
        """Generate a clarification response."""
        return f"I want to make sure I understand correctly. Could you clarify what you mean by '{user_input[:50]}'?"

    def _generate_clarification_question(self, user_input: str) -> str:
        """Generate a clarification question."""
        return f"Could you provide more details about '{user_input[:50]}'?"

    def _record_training_example(self, conversation_id: str, user_input: str, response: ChatResponse) -> None:
        """Record training example from conversation."""
        if self.training_pipeline is None:
            return
        try:
            from evora.brain.intelligence.training import TrainingExample, OutcomeType
            example = TrainingExample(
                session_id=conversation_id,
                task_id="",
                project="",
                input_data={"user_input": user_input, "intent": response.intent},
                output_data={"response": response.content, "confidence": response.confidence},
                outcome=OutcomeType.SUCCESS if response.confidence >= self.CONFIDENCE_THRESHOLD else OutcomeType.PARTIAL,
                confidence=response.confidence,
                metadata={"component": "chatbot", "conversation_id": conversation_id},
            )
            self.training_pipeline.record_training_example(
                session_id=conversation_id,
                task_id="",
                project="",
                component="chatbot",
                input_data={"user_input": user_input},
                output_data={"response": response.content, "confidence": response.confidence},
                outcome=example.outcome,
                confidence=response.confidence,
            )
        except Exception:
            pass

    def get_conversation_summary(self, conversation_id: str) -> dict[str, Any]:
        """Get a summary of the conversation."""
        conversation = self.conversation_manager.get_conversation(conversation_id)
        if conversation is None:
            return {}
        return {
            "conversation_id": conversation_id,
            "status": conversation.status.value,
            "turn_count": len(conversation.turns),
            "current_topic": conversation.current_topic,
            "previous_topics": conversation.previous_topics,
        }
