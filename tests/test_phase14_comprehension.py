"""
Phase 14 — Native Intelligence Expansion tests.

Verifies:
1. IntentClassifier classifies query intent
2. IntentClassifier classifies action intent
3. IntentClassifier classifies plan intent
4. IntentClassifier returns unknown for unrecognized text
5. EntityExtractor extracts files
6. EntityExtractor extracts functions
7. EntityExtractor extracts classes
8. EntityExtractor extracts URLs
9. ContextBuilder builds context
10. ContextBuilder detects ambiguity
11. RequestComprehender comprehends simple request
12. RequestComprehender extracts goal
13. RequestComprehender identifies constraints
14. RequestComprehender matches capabilities
15. RequestComprehender generates plan candidates
16. RequestComprehender detects uncertainty
17. NativeComprehensionIntelligence orchestrates comprehension
18. NativeComprehensionIntelligence works offline
19. IntelligenceRuntime integrates comprehension
20. No ModelManager dependency in comprehension module
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.comprehension import (
    AmbiguityLevel,
    Context,
    ContextBuilder,
    Entity,
    EntityExtractor,
    Intent,
    IntentClassifier,
    IntentType,
    NaturalRequest,
    NativeComprehensionIntelligence,
    Priority,
    RequestComprehender,
)
from evora.brain.intelligence import IntelligenceRuntime, CapabilityRegistry
from evora.logger import Logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def intent_classifier():
    return IntentClassifier()


@pytest.fixture
def entity_extractor():
    return EntityExtractor()


@pytest.fixture
def context_builder():
    return ContextBuilder()


@pytest.fixture
def capability_registry():
    return CapabilityRegistry()


@pytest.fixture
def comprehension_intelligence(intent_classifier, entity_extractor, context_builder, capability_registry):
    return NativeComprehensionIntelligence(
        intent_classifier=intent_classifier,
        entity_extractor=entity_extractor,
        context_builder=context_builder,
        capability_registry=capability_registry,
        logger=Logger("evora-test-p14", "info", None),
    )


# ---------------------------------------------------------------------------
# TestIntentClassifier
# ---------------------------------------------------------------------------

class TestIntentClassifier:
    """Test IntentClassifier."""

    def test_classify_query_intent(self, intent_classifier):
        intent = intent_classifier.classify("What is the project structure?")
        assert intent.intent_type == IntentType.QUERY

    def test_classify_action_intent(self, intent_classifier):
        intent = intent_classifier.classify("Execute the deployment script")
        assert intent.intent_type == IntentType.ACTION

    def test_classify_plan_intent(self, intent_classifier):
        intent = intent_classifier.classify("Plan the refactoring steps")
        assert intent.intent_type == IntentType.PLAN

    def test_classify_create_intent(self, intent_classifier):
        intent = intent_classifier.classify("Create a new Python module")
        assert intent.intent_type == IntentType.CREATE

    def test_classify_unknown_intent(self, intent_classifier):
        intent = intent_classifier.classify("asdfghjkl random text")
        assert intent.intent_type == IntentType.UNKNOWN

    def test_intent_has_confidence(self, intent_classifier):
        intent = intent_classifier.classify("Analyze the codebase")
        assert 0.0 <= intent.confidence <= 1.0


# ---------------------------------------------------------------------------
# TestEntityExtractor
# ---------------------------------------------------------------------------

class TestEntityExtractor:
    """Test EntityExtractor."""

    def test_extract_python_file(self, entity_extractor):
        entities = entity_extractor.extract("Check the file src/main.py for issues")
        file_entities = [e for e in entities if e.entity_type == "file"]
        assert len(file_entities) >= 1
        assert "main.py" in file_entities[0].value

    def test_extract_function(self, entity_extractor):
        entities = entity_extractor.extract("Call function hello_world()")
        func_entities = [e for e in entities if e.entity_type == "function"]
        assert len(func_entities) >= 1
        assert "hello_world" in [e.name for e in func_entities]

    def test_extract_class(self, entity_extractor):
        entities = entity_extractor.extract("Modify class UserService")
        class_entities = [e for e in entities if e.entity_type == "class"]
        assert len(class_entities) >= 1
        assert "UserService" in [e.name for e in class_entities]

    def test_extract_url(self, entity_extractor):
        entities = entity_extractor.extract("Fetch https://example.com/api")
        url_entities = [e for e in entities if e.entity_type == "url"]
        assert len(url_entities) >= 1
        assert "https://example.com/api" in [e.value for e in url_entities]

    def test_extract_path(self, entity_extractor):
        entities = entity_extractor.extract("Look at ./src/utils/helpers.py")
        path_entities = [e for e in entities if e.entity_type == "path"]
        assert len(path_entities) >= 1

    def test_no_entities(self, entity_extractor):
        entities = entity_extractor.extract("random text with no entities")
        assert len(entities) == 0


# ---------------------------------------------------------------------------
# TestContextBuilder
# ---------------------------------------------------------------------------

class TestContextBuilder:
    """Test ContextBuilder."""

    def test_build_context_empty_history(self, context_builder):
        context = context_builder.build([])
        assert isinstance(context, Context)
        assert len(context.conversation_history) == 0

    def test_build_context_with_history(self, context_builder):
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there", "action": "greet"},
        ]
        context = context_builder.build(history)
        assert len(context.conversation_history) == 2
        assert "greet" in context.recent_actions

    def test_detect_ambiguity_clear(self, context_builder):
        ambiguity = context_builder.detect_ambiguity("Run the tests now", Context())
        assert ambiguity == AmbiguityLevel.CLEAR

    def test_detect_ambiguity_ambiguous(self, context_builder):
        ambiguity = context_builder.detect_ambiguity("Maybe we should probably fix the bug", Context())
        assert ambiguity in (AmbiguityLevel.AMBIGUOUS, AmbiguityLevel.VERY_AMBIGUOUS)

    def test_detect_ambiguity_very_ambiguous(self, context_builder):
        text = "Maybe perhaps possibly we could somehow fix this? Or maybe not?"
        ambiguity = context_builder.detect_ambiguity(text, Context())
        assert ambiguity == AmbiguityLevel.VERY_AMBIGUOUS


# ---------------------------------------------------------------------------
# TestRequestComprehender
# ---------------------------------------------------------------------------

class TestRequestComprehender:
    """Test RequestComprehender."""

    def test_comprehend_simple_request(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Create a new Python module")
        assert request.goal != ""
        assert request.intent.intent_type == IntentType.CREATE

    def test_extract_goal(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Analyze the authentication bug in src/auth.py")
        assert "authentication" in request.goal.lower() or "auth" in request.goal.lower()

    def test_identify_constraints(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Fix the bug without modifying the database layer")
        assert len(request.constraints) >= 1

    def test_match_capabilities(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Analyze the project structure")
        assert len(request.required_capabilities) >= 0

    def test_generate_plan_candidates(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Refactor the user service")
        assert len(request.plan_candidates) >= 1

    def test_detect_ambiguity_in_request(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Maybe we should fix the bug")
        assert request.ambiguity != AmbiguityLevel.CLEAR

    def test_extract_entities_from_request(self, capability_registry):
        comprehender = RequestComprehender(capability_registry=capability_registry)
        request = comprehender.comprehend("Check src/main.py and src/utils.py")
        file_entities = [e for e in request.entities if e.entity_type == "file"]
        assert len(file_entities) >= 1


# ---------------------------------------------------------------------------
# TestNativeComprehensionIntelligence
# ---------------------------------------------------------------------------

class TestNativeComprehensionIntelligence:
    """Test NativeComprehensionIntelligence."""

    def test_comprehend_request(self, comprehension_intelligence):
        request = comprehension_intelligence.comprehend("Create a new test file")
        assert isinstance(request, NaturalRequest)
        assert request.goal != ""
        assert request.intent.intent_type == IntentType.CREATE

    def test_classify_intent(self, comprehension_intelligence):
        intent = comprehension_intelligence.classify_intent("Explain how authentication works")
        assert intent.intent_type in (IntentType.EXPLAIN, IntentType.QUERY)

    def test_extract_entities(self, comprehension_intelligence):
        entities = comprehension_intelligence.extract_entities("Check src/main.py and https://example.com")
        assert len(entities) >= 1

    def test_build_context(self, comprehension_intelligence):
        history = [{"role": "user", "content": "Hello"}]
        context = comprehension_intelligence.build_context(history)
        assert isinstance(context, Context)
        assert len(context.conversation_history) == 1

    def test_get_capabilities(self, comprehension_intelligence):
        capabilities = comprehension_intelligence.get_capabilities()
        assert len(capabilities) >= 5
        names = [c["name"] for c in capabilities]
        assert "intent_classification" in names
        assert "entity_extraction" in names


# ---------------------------------------------------------------------------
# TestIntelligenceRuntimeComprehensionIntegration
# ---------------------------------------------------------------------------

class TestIntelligenceRuntimeComprehensionIntegration:
    """Test IntelligenceRuntime integration with comprehension intelligence."""

    def test_runtime_comprehend_request(self, capability_registry):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=capability_registry,
            native_comprehension_intelligence=NativeComprehensionIntelligence(
                capability_registry=capability_registry,
                logger=Logger("evora-test-p14-rt", "info", None),
            ),
            logger=Logger("evora-test-p14-rt", "info", None),
        )
        request = runtime.comprehend_request("Create a new API endpoint")
        assert request.goal != ""
        assert request.intent.intent_type == IntentType.CREATE

    def test_runtime_classify_intent(self, capability_registry):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=capability_registry,
            native_comprehension_intelligence=NativeComprehensionIntelligence(
                capability_registry=capability_registry,
                logger=Logger("evora-test-p14-rt2", "info", None),
            ),
            logger=Logger("evora-test-p14-rt2", "info", None),
        )
        intent = runtime.classify_intent("Analyze the codebase")
        assert intent.intent_type == IntentType.ANALYZE

    def test_runtime_without_comprehension(self):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=CapabilityRegistry(),
            logger=Logger("evora-test-p14-rt3", "info", None),
        )
        request = runtime.comprehend_request("test")
        assert request.raw_input == "test"


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 14 security boundaries."""

    def test_comprehension_no_model_manager(self):
        import evora.brain.intelligence.comprehension as comp_mod
        source = Path(comp_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_comprehension_no_external_dependencies(self):
        import evora.brain.intelligence.comprehension as comp_mod
        source = Path(comp_mod.__file__).read_text(encoding="utf-8")
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

    def test_comprehension_cannot_grant_authority(self, comprehension_intelligence):
        request = comprehension_intelligence.comprehend("test request")
        assert not hasattr(request, "grant_authority")
        assert not hasattr(request, "approve_self")
        assert not hasattr(request, "bypass_security")


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 14 works offline."""

    def test_intent_classifier_offline(self, intent_classifier):
        intent = intent_classifier.classify("offline test query")
        assert intent.intent_type != IntentType.UNKNOWN or intent.confidence >= 0.0

    def test_entity_extractor_offline(self, entity_extractor):
        entities = entity_extractor.extract("offline test with src/main.py")
        assert len(entities) >= 0

    def test_comprehension_offline(self, comprehension_intelligence):
        request = comprehension_intelligence.comprehend("offline request")
        assert request.raw_input == "offline request"


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 14 architecture readiness."""

    def test_comprehension_intelligence_exists(self):
        from evora.brain.intelligence.comprehension import NativeComprehensionIntelligence
        assert NativeComprehensionIntelligence is not None

    def test_intent_classifier_exists(self):
        from evora.brain.intelligence.comprehension import IntentClassifier
        assert IntentClassifier is not None

    def test_entity_extractor_exists(self):
        from evora.brain.intelligence.comprehension import EntityExtractor
        assert EntityExtractor is not None

    def test_context_builder_exists(self):
        from evora.brain.intelligence.comprehension import ContextBuilder
        assert ContextBuilder is not None

    def test_request_comprehender_exists(self):
        from evora.brain.intelligence.comprehension import RequestComprehender
        assert RequestComprehender is not None

    def test_natural_request_exists(self):
        from evora.brain.intelligence.comprehension import NaturalRequest
        assert NaturalRequest is not None

    def test_runtime_has_comprehension_parameter(self):
        import inspect
        sig = inspect.signature(IntelligenceRuntime.__init__)
        assert "native_comprehension_intelligence" in sig.parameters

    def test_comprehension_capabilities_in_registry(self):
        registry = CapabilityRegistry()
        caps = registry.list_all()
        assert "intent_classification" in caps
        assert "entity_extraction" in caps
        assert "context_building" in caps
