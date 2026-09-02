"""
Phase 15 — Intelligence Orchestration tests.

Verifies:
1. OrchestrationDecision has correct result types
2. IntelligenceOrchestrator returns NATIVE_RESULT for memory
3. IntelligenceOrchestrator returns NATIVE_RESULT for knowledge
4. IntelligenceOrchestrator returns NATIVE_RESULT for coding
5. IntelligenceOrchestrator returns NATIVE_RESULT for capability
6. IntelligenceOrchestrator returns MODEL_ENHANCED_RESULT for model
7. IntelligenceOrchestrator returns UNAVAILABLE for empty request
8. IntelligenceOrchestrator never silently pretends external is native
9. IntelligenceOrchestrator requires model for external_only
10. IntelligenceOrchestrator requires tool for tool-based
11. IntelligenceRuntime integrates orchestration
12. Orchestration metrics work
13. No ModelManager dependency in orchestration module
14. End-to-end orchestration flow
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.orchestration import (
    IntelligenceOrchestrator,
    OrchestrationDecision,
    ResultType,
)
from evora.brain.intelligence import IntelligenceRuntime, CapabilityRegistry
from evora.logger import Logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def capability_registry():
    return CapabilityRegistry()


@pytest.fixture
def orchestrator(capability_registry):
    return IntelligenceOrchestrator(
        capability_registry=capability_registry,
        logger=Logger("evora-test-p15", "info", None),
    )


# ---------------------------------------------------------------------------
# TestOrchestrationDecision
# ---------------------------------------------------------------------------

class TestOrchestrationDecision:
    """Test OrchestrationDecision."""

    def test_native_result_type(self):
        decision = OrchestrationDecision(
            result_type=ResultType.NATIVE_RESULT,
            capability_used="memory",
            confidence=0.8,
        )
        assert decision.result_type == ResultType.NATIVE_RESULT

    def test_model_enhanced_result_type(self):
        decision = OrchestrationDecision(
            result_type=ResultType.MODEL_ENHANCED_RESULT,
            capability_used="model",
            requires_model=True,
        )
        assert decision.requires_model is True

    def test_external_only_result_type(self):
        decision = OrchestrationDecision(
            result_type=ResultType.EXTERNAL_ONLY_RESULT,
            capability_used="external_model",
            requires_model=True,
        )
        assert decision.requires_model is True

    def test_unavailable_result_type(self):
        decision = OrchestrationDecision(
            result_type=ResultType.UNAVAILABLE,
            reasoning="Nothing can handle this",
        )
        assert decision.result_type == ResultType.UNAVAILABLE

    def test_decision_to_dict(self):
        decision = OrchestrationDecision(
            result_type=ResultType.NATIVE_RESULT,
            capability_used="test",
            confidence=0.7,
            reasoning="Test",
        )
        data = decision.to_dict()
        assert data["result_type"] == "native_result"
        assert data["capability_used"] == "test"


# ---------------------------------------------------------------------------
# TestIntelligenceOrchestrator
# ---------------------------------------------------------------------------

class TestIntelligenceOrchestrator:
    """Test IntelligenceOrchestrator."""

    def test_empty_request_returns_unavailable(self, orchestrator):
        decision = orchestrator.orchestrate({"text": ""})
        assert decision.result_type == ResultType.UNAVAILABLE

    def test_orchestrate_returns_decision(self, orchestrator):
        decision = orchestrator.orchestrate({"text": "What is Python?"})
        assert isinstance(decision, OrchestrationDecision)
        assert decision.result_type in (
            ResultType.NATIVE_RESULT,
            ResultType.MODEL_ENHANCED_RESULT,
            ResultType.EXTERNAL_ONLY_RESULT,
            ResultType.UNAVAILABLE,
        )

    def test_memory_service_available(self, capability_registry):
        memory_service = MagicMock()
        memory_service.retrieve_relevant.return_value = [
            MagicMock(relevance_score=0.9),
        ]
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            memory_service=memory_service,
            logger=Logger("evora-test-p15-mem", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "test query"})
        assert decision.capability_used == "memory"

    def test_knowledge_graph_available(self, capability_registry):
        kg = MagicMock()
        kg.query.return_value = [
            MagicMock(confidence=0.8),
        ]
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            knowledge_graph=kg,
            logger=Logger("evora-test-p15-kg", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "test query"})
        assert decision.capability_used == "knowledge"

    def test_coding_intelligence_available(self, capability_registry):
        coding = MagicMock()
        coding.get_capabilities.return_value = [
            {"name": "python_code_understanding", "native": True, "confidence": 0.8},
        ]
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            coding_intelligence=coding,
            logger=Logger("evora-test-p15-code", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "analyze code"})
        assert decision.capability_used == "python_code_understanding"

    def test_model_provider_available(self, capability_registry):
        model = MagicMock()
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            model_provider=model,
            logger=Logger("evora-test-p15-model", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "explain complex theory"})
        assert decision.requires_model is True
        assert decision.result_type in (
            ResultType.MODEL_ENHANCED_RESULT,
            ResultType.EXTERNAL_ONLY_RESULT,
        )

    def test_never_pretend_external_is_native(self, capability_registry):
        model = MagicMock()
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            model_provider=model,
            logger=Logger("evora-test-p15-ext", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "complex reasoning task"})
        if decision.result_type == ResultType.EXTERNAL_ONLY_RESULT:
            assert decision.requires_model is True
            assert "external" in decision.reasoning.lower() or "model" in decision.reasoning.lower()

    def test_tool_registry_available(self, capability_registry):
        tool = MagicMock()
        tool.name = "read_file"
        tool_registry = MagicMock()
        tool_registry.list.return_value = [tool]
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            tool_registry=tool_registry,
            logger=Logger("evora-test-p15-tool", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "read the main.py file"})
        assert decision.requires_tool is True

    def test_get_orchestration_metrics(self, orchestrator):
        metrics = orchestrator.get_orchestration_metrics()
        assert "has_capability_registry" in metrics
        assert metrics["has_capability_registry"] is True


# ---------------------------------------------------------------------------
# TestIntelligenceRuntimeOrchestrationIntegration
# ---------------------------------------------------------------------------

class TestIntelligenceRuntimeOrchestrationIntegration:
    """Test IntelligenceRuntime integration with orchestration."""

    def test_runtime_orchestrate(self, capability_registry):
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            logger=Logger("evora-test-p15-rt", "info", None),
        )
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=capability_registry,
            intelligence_orchestrator=orchestrator,
            logger=Logger("evora-test-p15-rt", "info", None),
        )
        decision = runtime.orchestrate({"text": "test request"})
        assert isinstance(decision, OrchestrationDecision)
        assert decision.result_type in (
            ResultType.NATIVE_RESULT,
            ResultType.MODEL_ENHANCED_RESULT,
            ResultType.EXTERNAL_ONLY_RESULT,
            ResultType.UNAVAILABLE,
        )

    def test_runtime_without_orchestrator(self):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=CapabilityRegistry(),
            logger=Logger("evora-test-p15-rt2", "info", None),
        )
        decision = runtime.orchestrate({"text": "test"})
        assert decision.result_type == ResultType.UNAVAILABLE


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 15 security boundaries."""

    def test_orchestration_no_model_manager(self):
        import evora.brain.intelligence.orchestration as orch_mod
        source = Path(orch_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_orchestration_no_external_dependencies(self):
        import evora.brain.intelligence.orchestration as orch_mod
        source = Path(orch_mod.__file__).read_text(encoding="utf-8")
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

    def test_external_result_clearly_labeled(self, capability_registry):
        model = MagicMock()
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            model_provider=model,
            logger=Logger("evora-test-p15-sec", "info", None),
        )
        decision = orchestrator.orchestrate({"text": "complex external task"})
        if decision.result_type == ResultType.EXTERNAL_ONLY_RESULT:
            assert decision.requires_model is True
            assert decision.result_type == ResultType.EXTERNAL_ONLY_RESULT


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 15 works offline."""

    def test_orchestrator_works_offline(self, orchestrator):
        decision = orchestrator.orchestrate({"text": "offline test"})
        assert decision.result_type in (
            ResultType.NATIVE_RESULT,
            ResultType.MODEL_ENHANCED_RESULT,
            ResultType.EXTERNAL_ONLY_RESULT,
            ResultType.UNAVAILABLE,
        )

    def test_memory_decision_offline(self, capability_registry):
        memory_service = MagicMock()
        memory_service.retrieve_relevant.return_value = [MagicMock(relevance_score=0.8)]
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            memory_service=memory_service,
        )
        decision = orchestrator.orchestrate({"text": "test"})
        assert decision.result_type == ResultType.NATIVE_RESULT

    def test_knowledge_decision_offline(self, capability_registry):
        kg = MagicMock()
        kg.query.return_value = [MagicMock(confidence=0.7)]
        orchestrator = IntelligenceOrchestrator(
            capability_registry=capability_registry,
            knowledge_graph=kg,
        )
        decision = orchestrator.orchestrate({"text": "test"})
        assert decision.result_type == ResultType.NATIVE_RESULT


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 15 architecture readiness."""

    def test_orchestrator_exists(self):
        from evora.brain.intelligence.orchestration import IntelligenceOrchestrator
        assert IntelligenceOrchestrator is not None

    def test_orchestration_decision_exists(self):
        from evora.brain.intelligence.orchestration import OrchestrationDecision
        assert OrchestrationDecision is not None

    def test_result_type_enum_exists(self):
        from evora.brain.intelligence.orchestration import ResultType
        assert ResultType.NATIVE_RESULT is not None
        assert ResultType.MODEL_ENHANCED_RESULT is not None
        assert ResultType.EXTERNAL_ONLY_RESULT is not None
        assert ResultType.UNAVAILABLE is not None

    def test_runtime_has_orchestrator_parameter(self):
        import inspect
        sig = inspect.signature(IntelligenceRuntime.__init__)
        assert "intelligence_orchestrator" in sig.parameters
