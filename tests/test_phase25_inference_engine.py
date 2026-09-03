"""
Phase 25 — Native Inference Engine tests.

Verifies:
1. InferenceRule has correct structure
2. InferenceResult has correct structure
3. InferenceType enum exists
4. ConfidenceLevel enum exists
5. NativeInferenceEngine initializes
6. NativeInferenceEngine adds rules
7. NativeInferenceEngine performs deductive inference
8. NativeInferenceEngine performs inductive inference
9. NativeInferenceEngine performs abductive inference
10. NativeInferenceEngine performs analogical inference
11. NativeInferenceEngine matches patterns
12. NativeInferenceEngine chains inference
13. NativeInferenceEngine optimizes decisions
14. NativeInferenceEngine returns stats
15. Inference integrates with KnowledgeGraph
16. No ModelManager dependency
17. No external dependencies
18. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.inference_engine import (
    ConfidenceLevel,
    InferenceResult,
    InferenceRule,
    InferenceType,
    NativeInferenceEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def inference_engine():
    engine = NativeInferenceEngine(logger=MagicMock())
    engine.add_rule(InferenceRule(
        name="Python is a language",
        condition="python",
        conclusion="Python is a programming language",
        confidence=0.9,
        inference_type=InferenceType.DEDUCTIVE,
    ))
    engine.add_rule(InferenceRule(
        name="Error indicates failure",
        condition="error",
        conclusion="Operation failed",
        confidence=0.8,
        inference_type=InferenceType.INDUCTIVE,
    ))
    return engine


@pytest.fixture
def inference_with_graph():
    kg = MagicMock()
    return NativeInferenceEngine(knowledge_graph=kg, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestInferenceRule
# ---------------------------------------------------------------------------

class TestInferenceRule:
    """Test InferenceRule."""

    def test_default_rule(self):
        rule = InferenceRule()
        assert rule.rule_id != ""
        assert rule.confidence == 0.5

    def test_rule_to_dict(self):
        rule = InferenceRule(name="Test", condition="if x", conclusion="then y", confidence=0.8)
        data = rule.to_dict()
        assert data["name"] == "Test"
        assert data["confidence"] == 0.8


# ---------------------------------------------------------------------------
# TestInferenceResult
# ---------------------------------------------------------------------------

class TestInferenceResult:
    """Test InferenceResult."""

    def test_default_result(self):
        result = InferenceResult()
        assert result.result_id != ""
        assert result.confidence == 0.0

    def test_result_to_dict(self):
        result = InferenceResult(conclusion="Test", confidence=0.8, evidence=["e1"])
        data = result.to_dict()
        assert data["conclusion"] == "Test"
        assert data["confidence"] == 0.8


# ---------------------------------------------------------------------------
# TestInferenceTypeEnum
# ---------------------------------------------------------------------------

class TestInferenceTypeEnum:
    """Test InferenceType enum."""

    def test_inference_types_exist(self):
        assert InferenceType.DEDUCTIVE is not None
        assert InferenceType.INDUCTIVE is not None
        assert InferenceType.ABDUCTIVE is not None
        assert InferenceType.ANALOGICAL is not None


# ---------------------------------------------------------------------------
# TestNativeInferenceEngine
# ---------------------------------------------------------------------------

class TestNativeInferenceEngine:
    """Test NativeInferenceEngine."""

    def test_inference_engine_initializes(self):
        engine = NativeInferenceEngine(logger=MagicMock())
        assert engine is not None

    def test_add_rule(self, inference_engine):
        assert len(inference_engine._rules) > 0

    def test_deductive_inference(self, inference_engine):
        result = inference_engine.infer(["python"], InferenceType.DEDUCTIVE)
        assert isinstance(result, InferenceResult)
        assert result.confidence > 0.0

    def test_inductive_inference(self, inference_engine):
        result = inference_engine.infer(["error"], InferenceType.INDUCTIVE)
        assert isinstance(result, InferenceResult)

    def test_abductive_inference(self, inference_engine):
        result = inference_engine.infer(["test"], InferenceType.ABDUCTIVE)
        assert isinstance(result, InferenceResult)
        assert result.confidence <= 0.1

    def test_analogical_inference(self, inference_engine):
        result = inference_engine.infer(["similar"], InferenceType.ANALOGICAL)
        assert isinstance(result, InferenceResult)

    def test_pattern_matching(self, inference_engine):
        result = inference_engine.match_pattern(r"\d+", "There are 123 items")
        assert result["match_count"] == 1

    def test_pattern_matching_no_match(self, inference_engine):
        result = inference_engine.match_pattern(r"\d+", "No numbers here")
        assert result["match_count"] == 0

    def test_chain_inference(self, inference_engine):
        results = inference_engine.chain_inference(["python"], max_depth=2)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_optimize_decision(self, inference_engine):
        options = [
            {"name": "A", "confidence": 0.9, "feasibility": 0.8, "impact": 0.7},
            {"name": "B", "confidence": 0.6, "feasibility": 0.9, "impact": 0.5},
        ]
        result = inference_engine.optimize_decision(options)
        assert "best_option" in result
        assert result["best_option"]["option"]["name"] == "A"

    def test_get_stats(self, inference_engine):
        inference_engine.infer(["python"])
        stats = inference_engine.get_inference_stats()
        assert "total_inferences" in stats
        assert stats["total_inferences"] >= 1


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 25 security boundaries."""

    def test_no_model_manager_in_inference(self):
        import evora.brain.intelligence.inference_engine as inf_mod
        source = Path(inf_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.inference_engine as inf_mod
        source = Path(inf_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 25 works offline."""

    def test_inference_works_offline(self):
        engine = NativeInferenceEngine(logger=MagicMock())
        result = engine.infer(["test"])
        assert isinstance(result, InferenceResult)

    def test_pattern_matching_offline(self):
        engine = NativeInferenceEngine(logger=MagicMock())
        result = engine.match_pattern(r"\w+", "hello world")
        assert result["match_count"] == 2


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 25 architecture readiness."""

    def test_native_inference_engine_exists(self):
        from evora.brain.intelligence.inference_engine import NativeInferenceEngine
        assert NativeInferenceEngine is not None

    def test_inference_rule_exists(self):
        from evora.brain.intelligence.inference_engine import InferenceRule
        assert InferenceRule is not None

    def test_inference_result_exists(self):
        from evora.brain.intelligence.inference_engine import InferenceResult
        assert InferenceResult is not None

    def test_inference_type_enum_exists(self):
        from evora.brain.intelligence.inference_engine import InferenceType
        assert InferenceType.DEDUCTIVE is not None
        assert InferenceType.INDUCTIVE is not None

    def test_inference_reuses_knowledge_graph(self, inference_with_graph):
        assert inference_with_graph.knowledge_graph is not None
