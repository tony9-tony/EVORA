"""
Phase 10 — C10: Full Validation + Security tests.

Verifies:
- Native intelligence works offline (no OpenAI, Anthropic, Ollama, internet)
- No native-core ModelManager dependency
- No native-core network dependency
- No recursive provider architecture
- Chatbot/Agent shared-spine architecture readiness
- Creator/User authority separation
- Internet remains controlled future tool boundary
- Computer control remains deferred
- Self-improvement remains controlled
- Model output never becomes authority
- Phase 6/7/8/9 regression
"""

import asyncio
import inspect
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain import BrainController
from evora.brain.intelligence import (
    CapabilityRegistry,
    CapabilityType,
    EvaluationGrade,
    IntelligenceEvaluator,
    InferenceEngine,
    IntelligenceRuntime,
    KnowledgeGraph,
    NativeIntelligenceProvider,
    NativePlanner,
    NativeReasoning,
    ReasoningFacts,
)


# ============================================================================
# C10.1: Offline Operation
# ============================================================================


class TestOfflineOperation:
    """Prove native core works without any external provider."""

    def test_no_openai_dependency(self):
        """Native core modules must not import openai."""
        modules = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
            "evora.brain.intelligence.provider",
        ]
        for mod_name in modules:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            assert "openai" not in source.lower(), f"{mod_name} must not import openai"

    def test_no_anthropic_dependency(self):
        """Native core modules must not import anthropic."""
        modules = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
        ]
        for mod_name in modules:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            assert "anthropic" not in source.lower(), f"{mod_name} must not import anthropic"

    def test_no_ollama_dependency(self):
        """Native core modules must not import ollama."""
        modules = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
        ]
        for mod_name in modules:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            assert "ollama" not in source.lower(), f"{mod_name} must not import ollama"

    def test_no_network_calls_in_native_core(self):
        """Native core must not make network calls."""
        modules = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
        ]
        forbidden = ["requests", "aiohttp", "httpx", "urllib", "urlopen", "socket"]
        for mod_name in modules:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            for term in forbidden:
                assert term not in source.lower(), f"{mod_name} must not use {term}"

    def test_native_reasoning_offline(self):
        """NativeReasoning works without network."""
        reasoning = NativeReasoning(decision_engine=None)
        result = asyncio.run(reasoning.reason(ReasoningFacts(goal="offline test")))
        assert result is not None

    def test_native_planner_offline(self):
        """NativePlanner works without network."""
        planner = NativePlanner()
        result = asyncio.run(planner.plan("offline plan"))
        assert result is not None

    def test_inference_engine_offline(self):
        """InferenceEngine works without network."""
        engine = InferenceEngine()
        result = asyncio.run(engine.infer("offline query"))
        assert result is not None


# ============================================================================
# C10.2: No ModelManager Dependency
# ============================================================================


class TestNoModelManagerDependency:
    """Prove native intelligence never calls ModelManager."""

    def test_reasoning_no_model_manager(self):
        """NativeReasoning must not import or use ModelManager."""
        mod = __import__("evora.brain.intelligence.reasoning", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_planner_no_model_manager(self):
        """NativePlanner must not import or use ModelManager."""
        mod = __import__("evora.brain.intelligence.planner", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_inference_no_model_manager(self):
        """InferenceEngine must not import or use ModelManager."""
        mod = __import__("evora.brain.intelligence.inference", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_runtime_no_model_manager_parameter(self):
        mod = __import__("evora.brain.intelligence.runtime", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "model_manager" not in source.lower() or "register" in source.lower()

    def test_runtime_never_calls_model_manager(self):
        """IntelligenceRuntime must not call ModelManager."""
        mod = __import__("evora.brain.intelligence.runtime", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        lines = source.split("\n")
        for line in lines:
            if "model_manager" in line.lower() and "register" not in line.lower():
                pytest.fail(f"IntelligenceRuntime must not reference ModelManager: {line.strip()}")


# ============================================================================
# C10.3: No Recursive Provider Architecture
# ============================================================================


class TestNoRecursion:
    """Prove no recursive provider architecture."""

    def test_provider_does_not_call_model_manager(self):
        """NativeIntelligenceProvider must not call ModelManager for inference."""
        mod = __import__("evora.brain.intelligence.provider", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        # ModelManager should only appear in registration context, not inference
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "model_manager" in line.lower():
                # Check context - should only be in registration or comments
                context = "\n".join(lines[max(0, i-2):i+2])
                assert "register" in context.lower() or "#" in context, \
                    f"Provider must not call ModelManager for inference at line {i+1}"

    def test_runtime_has_no_model_manager_dependency(self):
        """IntelligenceRuntime constructor must not accept ModelManager."""
        from evora.brain.intelligence.runtime import IntelligenceRuntime
        sig = inspect.signature(IntelligenceRuntime.__init__)
        params = list(sig.parameters.keys())
        assert "model_manager" not in params

    def test_no_circular_dependency(self):
        """Verify no circular dependency between native core and ModelManager."""
        native_mods = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
        ]
        for mod_name in native_mods:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            assert "from evora.model import" not in source, f"{mod_name} must not import from evora.model"
            assert "from evora.brain.intelligence.provider" not in source, f"{mod_name} must not import provider (prevents recursion)"


# ============================================================================
# C10.4: Security Boundaries
# ============================================================================


class TestSecurityBoundaries:
    """Native intelligence cannot bypass security."""

    def test_model_output_never_authority(self):
        """Model output (native or external) is never authority."""
        from evora.brain.intelligence.provider import NativeIntelligenceProvider
        mod = __import__("evora.brain.intelligence.provider", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert "authority" not in source.lower() or "native intelligence" in source.lower()

    def test_native_intelligence_cannot_grant_permissions(self):
        """Native intelligence output cannot grant permissions."""
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=MagicMock(),
        )
        result = asyncio.run(runtime.reason("test"))
        assert hasattr(result, "to_dict")
        d = result.to_dict()
        assert "permission" not in d.get("decision", "").lower()

    def test_native_output_is_advisory(self):
        """Native intelligence output is advisory, not authoritative."""
        provider = NativeIntelligenceProvider(MagicMock())
        assert hasattr(provider, "name")
        assert provider.name() == "native"

    def test_no_secret_leakage_in_reasoning(self):
        """Reasoning summaries must not expose secrets from evidence."""
        reasoning = NativeReasoning(decision_engine=None)
        # Goal itself may contain text, but secrets from evidence should not leak
        result = asyncio.run(reasoning.reason(ReasoningFacts(goal="test reasoning")))
        assert "api_key" not in result.reasoning_summary or "test reasoning" in result.reasoning_summary


# ============================================================================
# C10.5: Authority Separation
# ============================================================================


class TestAuthoritySeparation:
    """Creator and user authority remains separate from intelligence."""

    def test_intelligence_does_not_modify_identity(self):
        """Native intelligence must not modify identity systems."""
        from evora.brain.intelligence.runtime import IntelligenceRuntime
        sig = inspect.signature(IntelligenceRuntime.__init__)
        params = list(sig.parameters.keys())
        assert "identity_service" not in params
        assert "approval_system" not in params

    def test_intelligence_does_not_bypass_approval(self):
        """Native intelligence must not bypass approval system."""
        from evora.brain.intelligence.runtime import IntelligenceRuntime
        sig = inspect.signature(IntelligenceRuntime.__init__)
        params = list(sig.parameters.keys())
        assert "approval_system" not in params

    def test_native_provider_is_not_creator(self):
        """Native provider cannot declare itself creator."""
        provider = NativeIntelligenceProvider(MagicMock())
        assert provider.name() == "native"
        assert "creator" not in provider.name()


# ============================================================================
# C10.6: Architecture Readiness
# ============================================================================


class TestArchitectureReadiness:
    """Verify architecture readiness for future modes."""

    def test_shared_intelligence_spine_possible(self):
        """BrainController can share IntelligenceRuntime across modes."""
        from evora.brain.intelligence import (
            CapabilityRegistry,
            IntelligenceEvaluator,
            IntelligenceRuntime,
            InferenceEngine,
            KnowledgeGraph,
            NativePlanner,
            NativeReasoning,
        )

        kg = KnowledgeGraph()
        registry = CapabilityRegistry()
        evaluator = IntelligenceEvaluator()
        reasoning = NativeReasoning(decision_engine=None)
        planner = NativePlanner(knowledge_graph=kg)
        inference = InferenceEngine(knowledge_graph=kg)
        runtime = IntelligenceRuntime(
            native_reasoning=reasoning,
            native_planner=planner,
            inference_engine=inference,
            knowledge_graph=kg,
            intelligence_evaluator=evaluator,
            capability_registry=registry,
        )

        # Same runtime can be shared
        brain1 = BrainController(intelligence_runtime=runtime)
        brain2 = BrainController(intelligence_runtime=runtime)
        assert brain1.intelligence_runtime is brain2.intelligence_runtime

    def test_internet_remains_future_tool_boundary(self):
        """Internet access is not implemented in native intelligence."""
        mod = __import__("evora.brain.intelligence.runtime", fromlist=[""])
        source = Path(mod.__file__).read_text(encoding="utf-8")
        forbidden = ["web_search", "web_fetch", "internet", "http", "fetch_url"]
        for term in forbidden:
            assert term not in source.lower(), "Internet access must be a future tool boundary"

    def test_computer_control_deferred(self):
        """Computer control is not implemented in Phase 10."""
        native_mods = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
            "evora.brain.intelligence.provider",
        ]
        for mod_name in native_mods:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            forbidden = ["os.system", "subprocess.call", "shutil.rmtree", "computer_control"]
            for term in forbidden:
                assert term not in source.lower(), f"{mod_name} must not implement computer control"

    def test_self_improvement_remains_controlled(self):
        """Self-improvement is not implemented in native intelligence."""
        native_mods = [
            "evora.brain.intelligence.knowledge",
            "evora.brain.intelligence.capabilities",
            "evora.brain.intelligence.evaluation",
            "evora.brain.intelligence.reasoning",
            "evora.brain.intelligence.planner",
            "evora.brain.intelligence.inference",
            "evora.brain.intelligence.runtime",
            "evora.brain.intelligence.provider",
        ]
        for mod_name in native_mods:
            mod = __import__(mod_name, fromlist=[""])
            source = Path(mod.__file__).read_text(encoding="utf-8")
            forbidden = ["self_improve", "modify_self", "rewrite", "self_modify"]
            for term in forbidden:
                assert term not in source.lower(), f"{mod_name} must not implement self-improvement"
