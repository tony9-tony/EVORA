"""
Phase 12 — Native Coding Intelligence tests.

Verifies:
1. CodeUnderstanding parses Python files (AST-based)
2. CodeUnderstanding extracts functions, classes, imports
3. CodeUnderstanding detects syntax errors gracefully
4. BugDetector finds bare except clauses
5. BugDetector finds mutable default arguments
6. BugDetector finds compare-to-None issues
7. BugDetector finds empty except blocks
8. CodeGenerator generates functions
9. CodeGenerator generates classes
10. CodeGenerator generates tests
11. PatchEvaluator evaluates patches
12. PatchEvaluator evaluates diffs
13. NativeCodingIntelligence orchestrates all capabilities
14. NativeCodingIntelligence works offline
15. NativeCodingIntelligence has no ModelManager dependency
16. NativeCodingIntelligence has no external dependencies
17. IntelligenceRuntime integrates coding intelligence
18. CapabilityRegistry includes coding capabilities
19. Code explanation works
20. End-to-end coding flow
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.coding import (
    BugDetector,
    CodeExplanation,
    CodeGenerator,
    CodeUnderstanding,
    GeneratedTest,
    NativeCodingIntelligence,
    PatchEvaluator,
    PatchEvaluation,
    BugPattern,
    FunctionInfo,
    ClassInfo,
    ImportInfo,
)
from evora.brain.intelligence import IntelligenceRuntime, CapabilityRegistry
from evora.logger import Logger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_python_file(tmp_path):
    """Create a sample Python file for testing."""
    code = textwrap.dedent("""
        import os
        import sys
        from typing import Optional

        def hello(name: str) -> str:
            '''Say hello.'''
            return f"Hello, {name}"

        def bad_function(x=[]):
            if x == None:
                return x
            try:
                return x[0]
            except:
                pass
            return None

        class MyClass:
            '''A sample class.'''
            def __init__(self):
                self.value = 0

            def get_value(self):
                return self.value

            @staticmethod
            def static_method():
                return 42
    """).strip()
    p = tmp_path / "sample.py"
    p.write_text(code, encoding="utf-8")
    return str(p)


@pytest.fixture
def buggy_python_file(tmp_path):
    """Create a Python file with bugs for testing."""
    code = textwrap.dedent("""
        def bad_func(items=[]):
            try:
                pass
            except:
                pass
            if items == None:
                return items
            return items[0]
    """).strip()
    p = tmp_path / "buggy.py"
    p.write_text(code, encoding="utf-8")
    return str(p)


@pytest.fixture
def code_understanding():
    return CodeUnderstanding()


@pytest.fixture
def bug_detector():
    return BugDetector()


@pytest.fixture
def code_generator():
    return CodeGenerator()


@pytest.fixture
def patch_evaluator():
    return PatchEvaluator()


@pytest.fixture
def native_coding(code_understanding, bug_detector, code_generator, patch_evaluator):
    return NativeCodingIntelligence(
        code_understanding=code_understanding,
        bug_detector=bug_detector,
        code_generator=code_generator,
        patch_evaluator=patch_evaluator,
        logger=Logger("evora-test-p12", "info", None),
    )


# ---------------------------------------------------------------------------
# TestCodeUnderstanding
# ---------------------------------------------------------------------------

class TestCodeUnderstanding:
    """Test CodeUnderstanding."""

    def test_parse_python_file(self, code_understanding, sample_python_file):
        result = code_understanding.parse_file(sample_python_file)
        assert result["language"] == "python"
        assert result["function_count"] >= 1
        assert result["class_count"] >= 1

    def test_extract_functions(self, code_understanding, sample_python_file):
        result = code_understanding.parse_file(sample_python_file)
        functions = result.get("functions", [])
        assert len(functions) >= 2
        func_names = [f["name"] for f in functions]
        assert "hello" in func_names
        assert "bad_function" in func_names

    def test_extract_classes(self, code_understanding, sample_python_file):
        result = code_understanding.parse_file(sample_python_file)
        classes = result.get("classes", [])
        assert len(classes) >= 1
        assert classes[0]["name"] == "MyClass"

    def test_extract_imports(self, code_understanding, sample_python_file):
        result = code_understanding.parse_file(sample_python_file)
        imports = result.get("imports", [])
        assert len(imports) >= 2
        import_names = [i.get("module", "") for i in imports]
        assert "os" in import_names
        assert "sys" in import_names

    def test_extract_function_info(self, code_understanding, sample_python_file):
        result = code_understanding.parse_file(sample_python_file)
        functions = result.get("functions", [])
        hello = next((f for f in functions if f["name"] == "hello"), None)
        assert hello is not None
        assert hello["lineno"] >= 1
        assert hello["is_async"] is False
        assert hello["returns"] == "str"

    def test_detect_async_function(self, code_understanding, tmp_path):
        code = textwrap.dedent("""
            async def async_func():
                pass
        """).strip()
        p = tmp_path / "async.py"
        p.write_text(code, encoding="utf-8")
        result = code_understanding.parse_file(str(p))
        functions = result.get("functions", [])
        assert len(functions) == 1
        assert functions[0]["is_async"] is True

    def test_parse_syntax_error(self, code_understanding, tmp_path):
        p = tmp_path / "bad.py"
        p.write_text("def invalid(:\n    pass", encoding="utf-8")
        result = code_understanding.parse_file(str(p))
        assert "error" in result

    def test_parse_nonexistent_file(self, code_understanding):
        result = code_understanding.parse_file("/nonexistent/file.py")
        assert "error" in result

    def test_parse_generic_file(self, code_understanding, tmp_path):
        js_code = "function hello(name) { return 'Hello ' + name; }"
        p = tmp_path / "test.js"
        p.write_text(js_code, encoding="utf-8")
        result = code_understanding.parse_file(str(p))
        assert result["language"] == "generic"
        assert len(result.get("functions", [])) >= 1


# ---------------------------------------------------------------------------
# TestBugDetector
# ---------------------------------------------------------------------------

class TestBugDetector:
    """Test BugDetector."""

    def test_detect_bare_except(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        bare_except = [b for b in bugs if b.pattern_id == "bare-except"]
        assert len(bare_except) >= 1

    def test_detect_mutable_default(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        mutable = [b for b in bugs if b.pattern_id == "mutable-default"]
        assert len(mutable) >= 1

    def test_detect_compare_none(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        compare_none = [b for b in bugs if b.pattern_id == "compare-none"]
        assert len(compare_none) >= 1

    def test_detect_empty_except(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        empty_except = [b for b in bugs if b.pattern_id == "empty-except"]
        assert len(empty_except) >= 1

    def test_no_bugs_in_clean_code(self, bug_detector, tmp_path):
        code = textwrap.dedent("""
            def clean_func(x: Optional[list] = None):
                if x is None:
                    return []
                try:
                    return x[0]
                except IndexError:
                    return None
        """).strip()
        p = tmp_path / "clean.py"
        p.write_text(code, encoding="utf-8")
        bugs = bug_detector.detect_bugs(str(p))
        assert len(bugs) == 0

    def test_bug_pattern_severity(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        for bug in bugs:
            assert bug.severity in ("low", "medium", "high", "critical")

    def test_bug_pattern_has_lineno(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        for bug in bugs:
            assert bug.lineno >= 1


# ---------------------------------------------------------------------------
# TestCodeGenerator
# ---------------------------------------------------------------------------

class TestCodeGenerator:
    """Test CodeGenerator."""

    def test_generate_function(self, code_generator):
        spec = {
            "name": "add",
            "args": ["a", "b"],
            "returns": "int",
            "docstring": "Add two numbers",
            "body": "return a + b",
        }
        code = code_generator.generate_function(spec)
        assert "def add(a, b) -> int:" in code
        assert "return a + b" in code

    def test_generate_function_without_returns(self, code_generator):
        spec = {
            "name": "greet",
            "args": ["name"],
            "body": "print(f'Hello, {name}')",
        }
        code = code_generator.generate_function(spec)
        assert "def greet(name):" in code

    def test_generate_class(self, code_generator):
        spec = {
            "name": "Dog",
            "bases": ["Animal"],
            "methods": [
                {"name": "bark", "args": ["self"], "returns": "str", "body": "return 'woof'"}
            ],
            "docstring": "A dog class",
        }
        code = code_generator.generate_class(spec)
        assert "class Dog(Animal):" in code
        assert "def bark(self) -> str:" in code
        assert "return 'woof'" in code

    def test_generate_test(self, code_generator):
        spec = {
            "target_function": "add",
            "test_cases": [
                {"name": "test_add_positive", "args": [1, 2], "expected": 3},
                {"name": "test_add_zero", "args": [0, 0], "expected": 0},
            ],
            "imports": ["from mymodule import add"],
        }
        test = code_generator.generate_test(spec)
        assert test.target_function == "add"
        assert "def test_add_positive" in test.test_code
        assert "def test_add_zero" in test.test_code
        assert "assert result == 3" in test.test_code

    def test_generate_test_default_framework(self, code_generator):
        spec = {"target_function": "foo", "test_cases": []}
        test = code_generator.generate_test(spec)
        assert test.framework == "pytest"


# ---------------------------------------------------------------------------
# TestPatchEvaluator
# ---------------------------------------------------------------------------

class TestPatchEvaluator:
    """Test PatchEvaluator."""

    def test_evaluate_patch_identical(self, patch_evaluator):
        original = "def foo():\n    return 1\n"
        patched = "def foo():\n    return 1\n"
        result = patch_evaluator.evaluate_patch(original, patched)
        assert result.is_safe is True
        assert result.confidence > 0.0

    def test_evaluate_patch_new_function(self, patch_evaluator):
        original = "def foo():\n    return 1\n"
        patched = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        result = patch_evaluator.evaluate_patch(original, patched)
        assert "bar" in result.affected_functions

    def test_evaluate_patch_removed_function(self, patch_evaluator):
        original = "def foo():\n    return 1\n\ndef bar():\n    return 2\n"
        patched = "def foo():\n    return 1\n"
        result = patch_evaluator.evaluate_patch(original, patched)
        assert "bar" in result.affected_functions

    def test_evaluate_patch_syntax_error(self, patch_evaluator):
        original = "def foo():\n    return 1\n"
        patched = "def foo(\n    return 1\n"
        result = patch_evaluator.evaluate_patch(original, patched)
        assert result.is_safe is False
        assert result.confidence == 0.0

    def test_evaluate_diff(self, patch_evaluator):
        diff = textwrap.dedent("""
            --- a/file.py
            +++ b/file.py
            @@ -1,2 +1,3 @@
             def foo():
            -    return 1
            +    return 2
            +    # added comment
        """).strip()
        result = patch_evaluator.evaluate_diff(diff)
        assert len(result.affected_files) >= 1
        assert "file.py" in result.affected_files[0]


# ---------------------------------------------------------------------------
# TestNativeCodingIntelligence
# ---------------------------------------------------------------------------

class TestNativeCodingIntelligence:
    """Test NativeCodingIntelligence."""

    def test_understand_file(self, native_coding, sample_python_file):
        result = native_coding.understand_file(sample_python_file)
        assert result["language"] == "python"
        assert result["function_count"] >= 1
        assert result["class_count"] >= 1

    def test_detect_bugs(self, native_coding, buggy_python_file):
        bugs = native_coding.detect_bugs(buggy_python_file)
        assert len(bugs) >= 3

    def test_generate_function(self, native_coding):
        spec = {"type": "function", "name": "add", "args": ["a", "b"], "body": "return a + b"}
        code = native_coding.generate_code(spec)
        assert "def add(a, b):" in code
        assert "return a + b" in code

    def test_generate_class(self, native_coding):
        spec = {"type": "class", "name": "Dog", "bases": ["Animal"], "methods": [{"name": "bark", "args": ["self"], "body": "return 'woof'"}]}
        code = native_coding.generate_code(spec)
        assert "class Dog(Animal):" in code

    def test_explain_code(self, native_coding, sample_python_file):
        explanation = native_coding.explain_code(sample_python_file)
        assert explanation.summary != ""
        assert len(explanation.functions) >= 1
        assert len(explanation.classes) >= 1

    def test_generate_test(self, native_coding):
        spec = {"target_function": "add", "test_cases": [{"name": "test_add", "args": [1, 2], "expected": 3}]}
        test = native_coding.generate_test(spec)
        assert test.target_function == "add"
        assert "test_add" in test.test_code

    def test_evaluate_patch(self, native_coding):
        original = "def foo():\n    return 1\n"
        patched = "def foo():\n    return 2\n"
        result = native_coding.evaluate_patch(original, patched)
        assert "confidence" in result.to_dict()

    def test_get_capabilities(self, native_coding):
        capabilities = native_coding.get_capabilities()
        assert len(capabilities) >= 5
        names = [c["name"] for c in capabilities]
        assert "python_code_understanding" in names
        assert "bug_detection" in names


# ---------------------------------------------------------------------------
# TestIntelligenceRuntimeCodingIntegration
# ---------------------------------------------------------------------------

class TestIntelligenceRuntimeCodingIntegration:
    """Test IntelligenceRuntime integration with coding intelligence."""

    def test_runtime_understand_file(self, sample_python_file):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=CapabilityRegistry(),
            native_coding_intelligence=NativeCodingIntelligence(logger=Logger("evora-test-p12-rt", "info", None)),
            logger=Logger("evora-test-p12-rt", "info", None),
        )
        result = runtime.understand_file(sample_python_file)
        assert result.get("language") == "python"
        assert result.get("function_count", 0) >= 1

    def test_runtime_detect_bugs(self, buggy_python_file):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=CapabilityRegistry(),
            native_coding_intelligence=NativeCodingIntelligence(logger=Logger("evora-test-p12-rt2", "info", None)),
            logger=Logger("evora-test-p12-rt2", "info", None),
        )
        bugs = runtime.detect_bugs(buggy_python_file)
        assert len(bugs) >= 3

    def test_runtime_without_coding_intelligence(self, sample_python_file):
        runtime = IntelligenceRuntime(
            native_reasoning=MagicMock(),
            native_planner=MagicMock(),
            inference_engine=MagicMock(),
            knowledge_graph=MagicMock(),
            intelligence_evaluator=MagicMock(),
            capability_registry=CapabilityRegistry(),
            logger=Logger("evora-test-p12-rt3", "info", None),
        )
        result = runtime.understand_file(sample_python_file)
        assert "error" in result


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 12 security boundaries."""

    def test_coding_no_model_manager(self):
        import evora.brain.intelligence.coding as coding_mod
        source = Path(coding_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_coding_no_external_dependencies(self):
        import evora.brain.intelligence.coding as coding_mod
        source = Path(coding_mod.__file__).read_text(encoding="utf-8")
        for forbidden in ["openai", "anthropic", "ollama", "requests", "aiohttp", "httpx", "urllib", "socket"]:
            assert forbidden not in source.lower(), f"Found forbidden dependency: {forbidden}"

    def test_code_generation_cannot_execute(self, native_coding):
        spec = {"type": "function", "name": "evil", "args": [], "body": "__import__('os').system('rm -rf /')"}
        code = native_coding.generate_code(spec)
        assert "__import__" in code  # It generates the code, but doesn't execute it

    def test_patch_evaluation_safe_by_default(self, native_coding):
        original = "x = 1\n"
        patched = "x = 2\n"
        result = native_coding.evaluate_patch(original, patched)
        assert result.is_safe is True


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 12 works offline."""

    def test_code_understanding_offline(self, code_understanding, sample_python_file):
        result = code_understanding.parse_file(sample_python_file)
        assert result["language"] == "python"

    def test_bug_detection_offline(self, bug_detector, buggy_python_file):
        bugs = bug_detector.detect_bugs(buggy_python_file)
        assert len(bugs) >= 1

    def test_code_generation_offline(self, code_generator):
        spec = {"type": "function", "name": "test", "args": [], "body": "return 42"}
        code = code_generator.generate_function(spec)
        assert "def test():" in code


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 12 architecture readiness."""

    def test_coding_intelligence_exists(self):
        from evora.brain.intelligence.coding import NativeCodingIntelligence
        assert NativeCodingIntelligence is not None

    def test_code_understanding_exists(self):
        from evora.brain.intelligence.coding import CodeUnderstanding
        assert CodeUnderstanding is not None

    def test_bug_detector_exists(self):
        from evora.brain.intelligence.coding import BugDetector
        assert BugDetector is not None

    def test_code_generator_exists(self):
        from evora.brain.intelligence.coding import CodeGenerator
        assert CodeGenerator is not None

    def test_patch_evaluator_exists(self):
        from evora.brain.intelligence.coding import PatchEvaluator
        assert PatchEvaluator is not None

    def test_runtime_has_coding_parameter(self):
        import inspect
        sig = inspect.signature(IntelligenceRuntime.__init__)
        assert "native_coding_intelligence" in sig.parameters

    def test_coding_capabilities_in_registry(self):
        registry = CapabilityRegistry()
        caps = registry.list_all()
        assert "python_code_understanding" in caps
        assert "bug_detection" in caps
        assert "simple_code_generation" in caps
        assert "code_explanation" in caps
        assert "test_generation" in caps
        assert "patch_evaluation" in caps
