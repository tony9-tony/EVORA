"""
Phase 14 — Native Intelligence Expansion for EVORA.

Builds structured internal representations for:
  - intent
  - entities
  - context
  - constraints
  - goals
  - priorities
  - ambiguity
  - conversation history
  - task state
  - relationships
  - cause/effect
  - known patterns
  - uncertainty

EVORA should be able to take a natural request and convert it into:
  Intent
  Goal
  Context
  Constraints
  Required capabilities
  Plan
  Action candidates
  Evaluation criteria

No ModelManager dependency.
No external model dependency.
Works offline.

Reuses existing abstractions:
  - CapabilityRegistry for capability matching
  - KnowledgeGraph for pattern retrieval
  - MemoryService for context/history
  - NativePlanner for plan generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Structured representations
# ---------------------------------------------------------------------------

class IntentType(str, Enum):
    """Kinds of intents EVORA can classify."""
    QUERY = "query"
    ACTION = "action"
    PLAN = "plan"
    LEARN = "learn"
    REMEMBER = "remember"
    FORGET = "forget"
    EXPLAIN = "explain"
    ANALYZE = "analyze"
    CREATE = "create"
    MODIFY = "modify"
    DELETE = "delete"
    EXECUTE = "execute"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    """Priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AmbiguityLevel(str, Enum):
    """Level of ambiguity in a request."""
    CLEAR = "clear"
    SLIGHTLY_AMBIGUOUS = "slightly_ambiguous"
    AMBIGUOUS = "ambiguous"
    VERY_AMBIGUOUS = "very_ambiguous"


@dataclass
class Entity:
    """An extracted entity."""
    entity_id: str = ""
    entity_type: str = ""  # file, function, class, project, concept, tool, person
    name: str = ""
    value: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class Intent:
    """Classified intent."""
    intent_type: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_type": self.intent_type.value,
            "confidence": self.confidence,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class Context:
    """Conversation/project context."""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    project_context: dict[str, Any] = field(default_factory=dict)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    active_task: str = ""
    recent_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_history": self.conversation_history,
            "project_context": self.project_context,
            "user_preferences": self.user_preferences,
            "active_task": self.active_task,
            "recent_actions": self.recent_actions,
            "metadata": self.metadata,
        }


@dataclass
class NaturalRequest:
    """Structured representation of a natural request."""
    raw_input: str = ""
    intent: Intent = field(default_factory=lambda: Intent())
    goal: str = ""
    entities: list[Entity] = field(default_factory=list)
    context: Context = field(default_factory=Context)
    constraints: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    plan_candidates: list[dict[str, Any]] = field(default_factory=list)
    evaluation_criteria: list[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    ambiguity: AmbiguityLevel = AmbiguityLevel.CLEAR
    uncertainty_factors: list[str] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    cause_effect: list[dict[str, Any]] = field(default_factory=list)
    known_patterns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_input": self.raw_input,
            "intent": self.intent.to_dict(),
            "goal": self.goal,
            "entities": [e.to_dict() for e in self.entities],
            "context": self.context.to_dict(),
            "constraints": self.constraints,
            "required_capabilities": self.required_capabilities,
            "plan_candidates": self.plan_candidates,
            "evaluation_criteria": self.evaluation_criteria,
            "priority": self.priority.value,
            "ambiguity": self.ambiguity.value,
            "uncertainty_factors": self.uncertainty_factors,
            "relationships": self.relationships,
            "cause_effect": self.cause_effect,
            "known_patterns": self.known_patterns,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------

class IntentClassifier:
    """Classifies user intent from natural language.

    Uses keyword-based classification with confidence scoring.
    No external model dependency.
    """

    INTENT_PATTERNS = {
        IntentType.QUERY: [
            "what is", "what are", "how does", "how do", "why is", "why are",
            "when is", "when are", "where is", "where are", "who is", "who are",
            "can you explain", "tell me about", "describe", "explain",
        ],
        IntentType.ACTION: [
            "do", "perform", "execute", "run", "apply", "implement",
            "carry out", "take action", "make it happen",
        ],
        IntentType.PLAN: [
            "plan", "plan out", "create a plan", "make a plan", "how should i",
            "what steps", "roadmap", "strategy for", "approach to",
        ],
        IntentType.LEARN: [
            "learn", "study", "practice", "understand", "master", "improve at",
        ],
        IntentType.REMEMBER: [
            "remember", "don't forget", "keep in mind", "note that", "save this",
            "store this", "memorize",
        ],
        IntentType.FORGET: [
            "forget", "remove from memory", "don't remember", "delete from memory",
        ],
        IntentType.EXPLAIN: [
            "explain", "clarify", "break down", "walk through", "describe how",
            "help me understand",
        ],
        IntentType.ANALYZE: [
            "analyze", "examine", "inspect", "review", "assess", "evaluate",
            "audit", "check", "investigate",
        ],
        IntentType.CREATE: [
            "create", "make", "build", "generate", "write", "compose",
            "design", "develop", "new",
        ],
        IntentType.MODIFY: [
            "modify", "change", "update", "edit", "alter", "revise",
            "refactor", "improve", "fix",
        ],
        IntentType.DELETE: [
            "delete", "remove", "drop", "erase", "clear", "purge",
        ],
        IntentType.EXECUTE: [
            "run", "execute", "start", "launch", "trigger", "invoke",
            "call", "perform",
        ],
    }

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger

    def classify(self, text: str) -> Intent:
        """Classify intent from text."""
        text_lower = text.lower().strip()
        best_intent = IntentType.UNKNOWN
        best_score = 0.0

        for intent_type, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                pattern_words = pattern.split()
                if all(word in text_lower.split() for word in pattern_words):
                    score += 1
            if score > best_score:
                best_score = score
                best_intent = intent_type

        confidence = min(1.0, best_score / 2.0) if best_score > 0 else 0.1
        if confidence < 0.2:
            best_intent = IntentType.UNKNOWN
            confidence = 0.1

        description = f"Classified as {best_intent.value} with confidence {confidence:.2f}"
        return Intent(
            intent_type=best_intent,
            confidence=confidence,
            description=description,
        )


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """Extracts entities from natural language.

    Uses regex-based extraction for common entity types.
    """

    ENTITY_PATTERNS = {
        "file": [
            r"[\w\-/\\]+\.py\b",
            r"[\w\-/\\]+\.js\b",
            r"[\w\-/\\]+\.ts\b",
            r"[\w\-/\\]+\.go\b",
            r"[\w\-/\\]+\.rs\b",
            r"[\w\-/\\]+\.java\b",
            r"[\w\-/\\]+\.c\b",
            r"[\w\-/\\]+\.cpp\b",
        ],
        "function": [
            r"function\s+(\w+)",
            r"def\s+(\w+)",
            r"method\s+(\w+)",
        ],
        "class": [
            r"class\s+(\w+)",
        ],
        "url": [
            r"https?://[^\s]+",
        ],
        "path": [
            r"(?:\./|/)[\w\-/\\]+",
        ],
        "number": [
            r"\b\d+\.?\d*\b",
        ],
    }

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        entities = []
        seen = set()

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    value = match.group(0)
                    if value not in seen:
                        seen.add(value)
                        name = match.group(1) if match.lastindex is not None and match.lastindex >= 1 else value
                        entities.append(Entity(
                            entity_id=f"e-{len(entities)+1:03d}",
                            entity_type=entity_type,
                            name=name,
                            value=value,
                            confidence=0.8 if entity_type in ("file", "url", "path") else 0.6,
                        ))

        return entities


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

class ContextBuilder:
    """Builds context from conversation history and project state.

    Combines:
      - Recent conversation turns
      - Project memory
      - Active task
      - User preferences
    """

    def __init__(self, memory_service: Any = None, logger: Optional[Any] = None):
        self.memory_service = memory_service
        self.logger = logger

    def build(self, conversation_history: list[dict[str, Any]], project: str = "") -> Context:
        """Build context from available information."""
        recent_actions = []
        for turn in conversation_history[-5:]:
            if turn.get("role") == "assistant":
                action = turn.get("action", "")
                if action:
                    recent_actions.append(action)

        project_context = {}
        if self.memory_service is not None:
            try:
                memories = self.memory_service.retrieve_relevant(goal=project or "", limit=5)
                project_context = {
                    "recent_memories": [m.to_dict() for m in memories] if memories else []
                }
            except Exception:
                pass

        return Context(
            conversation_history=conversation_history[-10:],
            project_context=project_context,
            recent_actions=recent_actions,
        )

    def detect_ambiguity(self, text: str, context: Context) -> AmbiguityLevel:
        """Detect ambiguity level in a request."""
        ambiguity_indicators = [
            "maybe", "perhaps", "possibly", "might", "could", "not sure",
            "i think", "probably", "somehow", "something", "somehow",
        ]
        text_lower = text.lower()
        indicator_count = sum(1 for ind in ambiguity_indicators if ind in text_lower)

        question_marks = text.count("?")
        if indicator_count >= 2 or question_marks >= 2:
            return AmbiguityLevel.VERY_AMBIGUOUS
        elif indicator_count == 1 or question_marks == 1:
            return AmbiguityLevel.AMBIGUOUS
        elif indicator_count == 0 and question_marks == 0:
            return AmbiguityLevel.CLEAR
        else:
            return AmbiguityLevel.SLIGHTLY_AMBIGUOUS


# ---------------------------------------------------------------------------
# Request Comprehender
# ---------------------------------------------------------------------------

class RequestComprehender:
    """Converts natural requests into structured representations.

    Produces:
      - Intent classification
      - Goal extraction
      - Entity extraction
      - Context building
      - Constraint identification
      - Capability matching
      - Ambiguity detection
      - Uncertainty analysis
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        context_builder: Optional[ContextBuilder] = None,
        capability_registry: Any = None,
        knowledge_graph: Any = None,
        memory_service: Any = None,
        logger: Optional[Any] = None,
    ):
        self.intent_classifier = intent_classifier or IntentClassifier(logger=logger)
        self.entity_extractor = entity_extractor or EntityExtractor(logger=logger)
        self.context_builder = context_builder or ContextBuilder(
            memory_service=memory_service, logger=logger
        )
        self.capability_registry = capability_registry
        self.knowledge_graph = knowledge_graph
        self.memory_service = memory_service
        self.logger = logger

    def comprehend(self, text: str, conversation_history: list[dict[str, Any]] = None, project: str = "") -> NaturalRequest:
        """Comprehend a natural request and return structured representation."""
        conversation_history = conversation_history or []

        intent = self.intent_classifier.classify(text)
        entities = self.entity_extractor.extract(text)
        context = self.context_builder.build(conversation_history, project=project)
        ambiguity = self.context_builder.detect_ambiguity(text, context)

        goal = self._extract_goal(text, intent, entities)
        constraints = self._identify_constraints(text, context)
        required_capabilities = self._match_capabilities(intent, goal, entities)
        plan_candidates = self._generate_plan_candidates(goal, required_capabilities)
        evaluation_criteria = self._define_evaluation_criteria(intent, goal)
        uncertainty_factors = self._identify_uncertainty(text, intent, ambiguity)
        relationships = self._extract_relationships(entities, text)
        cause_effect = self._extract_cause_effect(text)
        known_patterns = self._retrieve_patterns(goal)

        priority = self._determine_priority(intent, goal, ambiguity)

        return NaturalRequest(
            raw_input=text,
            intent=intent,
            goal=goal,
            entities=entities,
            context=context,
            constraints=constraints,
            required_capabilities=required_capabilities,
            plan_candidates=plan_candidates,
            evaluation_criteria=evaluation_criteria,
            priority=priority,
            ambiguity=ambiguity,
            uncertainty_factors=uncertainty_factors,
            relationships=relationships,
            cause_effect=cause_effect,
            known_patterns=known_patterns,
        )

    def _extract_goal(self, text: str, intent: Intent, entities: list[Entity]) -> str:
        """Extract the goal from text."""
        text_clean = text.strip()
        if len(text_clean) > 200:
            text_clean = text_clean[:200] + "..."
        return text_clean

    def _identify_constraints(self, text: str, context: Context) -> list[str]:
        """Identify constraints from text and context."""
        constraints = []
        constraint_keywords = [
            "without", "without using", "don't use", "avoid", "must not",
            "cannot", "should not", "do not", "no ", "restrict",
            "limit", "only", "require", "requires", "needs",
        ]
        text_lower = text.lower()
        for keyword in constraint_keywords:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                snippet = text[idx:idx + 100].strip()
                constraints.append(snippet)
        return constraints[:5]

    def _match_capabilities(self, intent: Intent, goal: str, entities: list[Entity]) -> list[str]:
        """Match required capabilities based on intent and goal."""
        capabilities = []
        if self.capability_registry is None:
            return capabilities
        try:
            capability = self.capability_registry.can_handle(intent.intent_type.value)
            if capability.capability_type.value != "unavailable":
                capabilities.append(capability.name)
        except Exception:
            pass
        return capabilities

    def _generate_plan_candidates(self, goal: str, capabilities: list[str]) -> list[dict[str, Any]]:
        """Generate plan candidates."""
        candidates = []
        candidates.append({
            "approach": "analyze",
            "description": f"Analyze: {goal[:80]}",
            "confidence": 0.7,
            "capabilities_required": capabilities,
        })
        if capabilities:
            candidates.append({
                "approach": "direct",
                "description": f"Direct execution using: {', '.join(capabilities[:3])}",
                "confidence": 0.5,
                "capabilities_required": capabilities,
            })
        return candidates

    def _define_evaluation_criteria(self, intent: Intent, goal: str) -> list[str]:
        """Define evaluation criteria for the request."""
        criteria = [f"Complete the goal: {goal[:80]}"]
        if intent.intent_type in (IntentType.CREATE, IntentType.MODIFY):
            criteria.append("Code should be syntactically correct")
        if intent.intent_type == IntentType.ANALYZE:
            criteria.append("Analysis should be comprehensive")
        return criteria

    def _identify_uncertainty(self, text: str, intent: Intent, ambiguity: AmbiguityLevel) -> list[str]:
        """Identify uncertainty factors."""
        factors = []
        if intent.confidence < 0.5:
            factors.append("Low intent confidence")
        if ambiguity in (AmbiguityLevel.AMBIGUOUS, AmbiguityLevel.VERY_AMBIGUOUS):
            factors.append("Ambiguous request")
        if "?" in text and text.count("?") > 1:
            factors.append("Multiple questions in request")
        return factors

    def _extract_relationships(self, entities: list[Entity], text: str) -> list[dict[str, Any]]:
        """Extract relationships between entities."""
        relationships = []
        for i, entity_a in enumerate(entities):
            for j, entity_b in enumerate(entities):
                if i >= j:
                    continue
                if entity_a.value.lower() in text.lower() and entity_b.value.lower() in text.lower():
                    relationships.append({
                        "source": entity_a.name,
                        "target": entity_b.name,
                        "relation": "mentioned_together",
                        "confidence": 0.5,
                    })
        return relationships[:10]

    def _extract_cause_effect(self, text: str) -> list[dict[str, Any]]:
        """Extract cause-effect relationships from text."""
        cause_effect = []
        cause_keywords = ["because", "since", "due to", "as a result", "therefore", "consequently"]
        text_lower = text.lower()
        for keyword in cause_keywords:
            if keyword in text_lower:
                idx = text_lower.find(keyword)
                cause_effect.append({
                    "keyword": keyword,
                    "position": idx,
                    "snippet": text[max(0, idx - 20):idx + 50].strip(),
                })
        return cause_effect[:5]

    def _retrieve_patterns(self, goal: str) -> list[str]:
        """Retrieve known patterns from knowledge graph."""
        if self.knowledge_graph is None:
            return []
        try:
            nodes = self.knowledge_graph.query(goal, limit=5)
            return [n.content for n in nodes if hasattr(n, "content") and n.content]
        except Exception:
            return []

    def _determine_priority(self, intent: Intent, goal: str, ambiguity: AmbiguityLevel) -> Priority:
        """Determine priority of the request."""
        if intent.intent_type in (IntentType.ACTION, IntentType.EXECUTE, IntentType.CREATE):
            if ambiguity == AmbiguityLevel.CLEAR:
                return Priority.HIGH
            return Priority.MEDIUM
        if intent.intent_type in (IntentType.QUERY, IntentType.EXPLAIN):
            return Priority.LOW
        return Priority.MEDIUM


# ---------------------------------------------------------------------------
# Native Comprehension Intelligence
# ---------------------------------------------------------------------------

class NativeComprehensionIntelligence:
    """Native comprehension intelligence for EVORA.

    Provides:
      - Intent classification
      - Entity extraction
      - Context building
      - Ambiguity detection
      - Structured request representation
      - Goal extraction
      - Constraint identification
      - Capability matching
      - Plan candidate generation
      - Uncertainty analysis

    No ModelManager dependency.
    No external model dependency.
    Works offline.
    """

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        context_builder: Optional[ContextBuilder] = None,
        capability_registry: Any = None,
        knowledge_graph: Any = None,
        memory_service: Any = None,
        logger: Optional[Any] = None,
    ):
        self.intent_classifier = intent_classifier or IntentClassifier(logger=logger)
        self.entity_extractor = entity_extractor or EntityExtractor(logger=logger)
        self.context_builder = context_builder or ContextBuilder(
            memory_service=memory_service, logger=logger
        )
        self.capability_registry = capability_registry
        self.knowledge_graph = knowledge_graph
        self.memory_service = memory_service
        self.logger = logger
        self._request_comprehender = RequestComprehender(
            intent_classifier=self.intent_classifier,
            entity_extractor=self.entity_extractor,
            context_builder=self.context_builder,
            capability_registry=capability_registry,
            knowledge_graph=knowledge_graph,
            memory_service=memory_service,
            logger=logger,
        )

    def comprehend(self, text: str, conversation_history: list[dict[str, Any]] = None, project: str = "") -> NaturalRequest:
        """Comprehend a natural request."""
        if self.logger:
            self.logger.observe(f"Comprehending request: {text[:80]}")
        return self._request_comprehender.comprehend(text, conversation_history, project)

    def classify_intent(self, text: str) -> Intent:
        """Classify intent from text."""
        return self.intent_classifier.classify(text)

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        return self.entity_extractor.extract(text)

    def build_context(self, conversation_history: list[dict[str, Any]], project: str = "") -> Context:
        """Build context."""
        return self.context_builder.build(conversation_history, project=project)

    def get_capabilities(self) -> list[dict[str, Any]]:
        """Get comprehension capabilities."""
        return [
            {"name": "intent_classification", "native": True, "confidence": 0.7},
            {"name": "entity_extraction", "native": True, "confidence": 0.6},
            {"name": "context_building", "native": True, "confidence": 0.7},
            {"name": "ambiguity_detection", "native": True, "confidence": 0.6},
            {"name": "goal_extraction", "native": True, "confidence": 0.7},
            {"name": "constraint_identification", "native": True, "confidence": 0.5},
            {"name": "capability_matching", "native": True, "confidence": 0.8},
        ]
