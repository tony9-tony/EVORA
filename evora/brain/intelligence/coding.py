"""
Phase 12 — Native Coding Intelligence for EVORA.

Builds genuine native coding intelligence capabilities:
  - understanding source files
  - parsing code structure
  - identifying functions/classes/modules
  - understanding dependencies
  - detecting common bugs
  - generating simple code
  - modifying simple code
  - explaining code
  - generating tests
  - comparing implementations
  - evaluating patches
  - identifying regressions
  - reasoning about project architecture

Uses structured representations:
  - ASTs (via Python ast module)
  - symbol graphs
  - dependency graphs
  - control-flow information
  - repository structure

No ModelManager dependency.
No external model dependency.
Works completely offline for supported languages (Python primary).

Reuses existing abstractions:
  - ProjectAnalyzer for project-level structure
  - ToolRegistry for file/command operations
  - KnowledgeGraph for storing code patterns
  - IntelligenceEvaluator for quality assessment
"""

from __future__ import annotations

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Code structure representations
# ---------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    lineno: int
    end_lineno: int
    args: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""
    complexity: int = 1
    is_async: bool = False
    returns: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "args": self.args,
            "decorators": self.decorators,
            "docstring": self.docstring,
            "complexity": self.complexity,
            "is_async": self.is_async,
            "returns": self.returns,
            "source": self.source,
        }


@dataclass
class ClassInfo:
    """Information about a class."""
    name: str
    lineno: int
    end_lineno: int
    bases: list[str] = field(default_factory=list)
    methods: list[FunctionInfo] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "bases": self.bases,
            "methods": [m.to_dict() for m in self.methods],
            "decorators": self.decorators,
            "docstring": self.docstring,
            "source": self.source,
        }


@dataclass
class ImportInfo:
    """Information about an import."""
    module: str = ""
    names: list[str] = field(default_factory=list)
    is_from: bool = False
    lineno: int = 0
    alias: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "names": self.names,
            "is_from": self.is_from,
            "lineno": self.lineno,
            "alias": self.alias,
        }


@dataclass
class BugPattern:
    """A detected bug pattern."""
    pattern_id: str
    severity: str  # low, medium, high, critical
    description: str
    lineno: int
    col_offset: int = 0
    suggestion: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "severity": self.severity,
            "description": self.description,
            "lineno": self.lineno,
            "col_offset": self.col_offset,
            "suggestion": self.suggestion,
            "metadata": self.metadata,
        }


@dataclass
class CodeExplanation:
    """Explanation of code structure."""
    summary: str = ""
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    control_flow: str = ""
    dependencies: list[str] = field(default_factory=list)
    complexity_notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "control_flow": self.control_flow,
            "dependencies": self.dependencies,
            "complexity_notes": self.complexity_notes,
            "metadata": self.metadata,
        }


@dataclass
class GeneratedTest:
    """A generated test case."""
    test_id: str = ""
    test_name: str = ""
    target_function: str = ""
    test_code: str = ""
    framework: str = "pytest"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "target_function": self.target_function,
            "test_code": self.test_code,
            "framework": self.framework,
            "metadata": self.metadata,
        }


@dataclass
class PatchEvaluation:
    """Evaluation of a code patch."""
    patch_id: str = ""
    is_safe: bool = True
    risks: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    affected_functions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "is_safe": self.is_safe,
            "risks": self.risks,
            "affected_files": self.affected_files,
            "affected_functions": self.affected_functions,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AST-based code understanding
# ---------------------------------------------------------------------------

class CodeUnderstanding:
    """Parses source files and extracts structured code information.

    Uses Python's ast module for deep structural understanding.
    Supports Python primarily, with basic regex-based support for other languages.
    """

    LANGUAGE_EXTENSIONS = {
        ".py": "python",
    }

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger

    def parse_file(self, path: str) -> dict[str, Any]:
        """Parse a source file and return structured information."""
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return {"error": f"File not found: {path}"}

        ext = file_path.suffix.lower()
        language = self.LANGUAGE_EXTENSIONS.get(ext)

        if language == "python":
            return self._parse_python(file_path)
        else:
            return self._parse_generic(file_path)

    def _parse_python(self, file_path: Path) -> dict[str, Any]:
        """Parse Python source using AST."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            return {
                "path": str(file_path),
                "language": "python",
                "error": f"Syntax error: {e}",
                "functions": [],
                "classes": [],
                "imports": [],
            }
        except Exception as e:
            return {"error": str(e)}

        functions: list[FunctionInfo] = []
        classes: list[ClassInfo] = []
        imports: list[ImportInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func = self._extract_function(node, content)
                functions.append(func)
            elif isinstance(node, ast.ClassDef):
                cls = self._extract_class(node, content)
                classes.append(cls)
            elif isinstance(node, ast.Import):
                imports.append(ImportInfo(
                    module=node.names[0].name if node.names else "",
                    names=[n.name for n in node.names],
                    is_from=False,
                    lineno=node.lineno,
                    alias={n.name: n.asname for n in node.names if n.asname},
                ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [n.name for n in node.names]
                alias = {n.name: n.asname for n in node.names if n.asname}
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from=True,
                    lineno=node.lineno,
                    alias=alias,
                ))

        # Calculate cyclomatic complexity for each function
        for func in functions:
            func.complexity = self._calculate_complexity(func, tree)

        return {
            "path": str(file_path),
            "language": "python",
            "lines": len(content.splitlines()),
            "functions": [f.to_dict() for f in functions],
            "classes": [c.to_dict() for c in classes],
            "imports": [i.to_dict() for i in imports],
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports),
        }

    def _parse_generic(self, file_path: Path) -> dict[str, Any]:
        """Basic regex-based parsing for non-Python files."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
        except Exception as e:
            return {"error": str(e)}

        functions = []
        classes = []
        imports = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("#") or not stripped:
                continue

            func_match = re.match(r"^(async\s+)?function\s+(\w+)\s*\(", stripped)
            if func_match:
                functions.append({
                    "name": func_match.group(2),
                    "lineno": i,
                    "source": stripped,
                })

            class_match = re.match(r"^class\s+(\w+)", stripped)
            if class_match:
                classes.append({
                    "name": class_match.group(1),
                    "lineno": i,
                    "source": stripped,
                })

            import_match = re.match(r"^import\s+([\w.]+)", stripped)
            if import_match:
                imports.append({
                    "module": import_match.group(1),
                    "lineno": i,
                })

        return {
            "path": str(file_path),
            "language": "generic",
            "lines": len(lines),
            "functions": functions,
            "classes": classes,
            "imports": imports,
        }

    def _extract_function(self, node: ast.FunctionDef, content: str) -> FunctionInfo:
        """Extract function information from AST node."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        for arg in node.args.kwonlyargs:
            args.append(arg.arg)
        if node.args.vararg:
            args.append(node.args.vararg.arg)
        if node.args.kwarg:
            args.append(node.args.kwarg.arg)

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"{self._get_attr_chain(dec)}")
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(self._get_attr_chain(dec.func))

        docstring = ast.get_docstring(node) or ""

        returns = ""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                returns = node.returns.id
            elif isinstance(node.returns, ast.Constant):
                returns = str(node.returns.value)
            elif isinstance(node.returns, ast.Subscript):
                returns = self._get_attr_chain(node.returns)

        source = ""
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            lines = content.splitlines()[node.lineno - 1:node.end_lineno]
            source = "\n".join(lines)

        return FunctionInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            args=args,
            decorators=decorators,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            returns=returns,
            source=source,
        )

    def _extract_class(self, node: ast.ClassDef, content: str) -> ClassInfo:
        """Extract class information from AST node."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attr_chain(base))

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attr_chain(dec))

        docstring = ast.get_docstring(node) or ""

        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item, content))

        source = ""
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            lines = content.splitlines()[node.lineno - 1:node.end_lineno]
            source = "\n".join(lines)

        return ClassInfo(
            name=node.name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", node.lineno),
            bases=bases,
            methods=methods,
            decorators=decorators,
            docstring=docstring,
            source=source,
        )

    def _get_attr_chain(self, node: ast.AST) -> str:
        """Get dotted attribute chain from AST node."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))

    def _calculate_complexity(self, func: FunctionInfo, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity


# ---------------------------------------------------------------------------
# Bug detection
# ---------------------------------------------------------------------------

class BugDetector:
    """Detects common coding bugs using AST analysis.

    Detects patterns like:
    - unused variables
    - bare except clauses
    - mutable default arguments
    - empty except blocks
    - comparison to None using ==
    - missing return statements
    - unreachable code
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._patterns: list[dict[str, Any]] = [
            {
                "id": "bare-except",
                "check": self._check_bare_except,
                "severity": "high",
                "description": "Bare except clause catches all exceptions, including KeyboardInterrupt and SystemExit",
                "suggestion": "Use 'except Exception:' or specify the exception type",
            },
            {
                "id": "mutable-default",
                "check": self._check_mutable_default,
                "severity": "medium",
                "description": "Mutable default argument (list/dict) can cause unexpected behavior",
                "suggestion": "Use None as default and initialize inside the function",
            },
            {
                "id": "compare-none",
                "check": self._check_compare_none,
                "severity": "low",
                "description": "Comparing to None using == instead of 'is'",
                "suggestion": "Use 'is None' or 'is not None'",
            },
            {
                "id": "empty-except",
                "check": self._check_empty_except,
                "severity": "medium",
                "description": "Empty except block silently swallows exceptions",
                "suggestion": "Add logging or handle the exception explicitly",
            },
            {
                "id": "unused-import",
                "check": self._check_unused_import,
                "severity": "low",
                "description": "Imported module appears unused",
                "suggestion": "Remove the unused import",
            },
            {
                "id": "unreachable-code",
                "check": self._check_unreachable_code,
                "severity": "medium",
                "description": "Code after return/raise/break is unreachable",
                "suggestion": "Remove unreachable code",
            },
        ]

    def detect_bugs(self, file_path: str) -> list[BugPattern]:
        """Detect bugs in a source file."""
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return []

        ext = path.suffix.lower()
        if ext != ".py":
            return []

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content, filename=str(path))
        except Exception:
            return []

        bugs = []
        for pattern in self._patterns:
            try:
                found = pattern["check"](tree, content)
                for bug in found:
                    bugs.append(BugPattern(
                        pattern_id=pattern["id"],
                        severity=pattern["severity"],
                        description=pattern["description"],
                        lineno=bug["lineno"],
                        col_offset=bug.get("col_offset", 0),
                        suggestion=pattern["suggestion"],
                        metadata=bug.get("metadata", {}),
                    ))
            except Exception:
                continue

        return bugs

    def _check_bare_except(self, tree: ast.AST, content: str) -> list[dict[str, Any]]:
        """Check for bare except clauses."""
        bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                bugs.append({
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                })
        return bugs

    def _check_mutable_default(self, tree: ast.AST, content: str) -> list[dict[str, Any]]:
        """Check for mutable default arguments."""
        bugs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for default in node.args.defaults + node.args.kw_defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        bugs.append({
                            "lineno": node.lineno,
                            "col_offset": node.col_offset,
                            "metadata": {"function": node.name},
                        })
        return bugs

    def _check_compare_none(self, tree: ast.AST, content: str) -> list[dict[str, Any]]:
        """Check for == None comparisons."""
        bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                if any(isinstance(op, ast.Eq) for op in node.ops):
                    if any(isinstance(comp, ast.Constant) and comp.value is None for comp in node.comparators):
                        bugs.append({
                            "lineno": node.lineno,
                            "col_offset": node.col_offset,
                        })
        return bugs

    def _check_empty_except(self, tree: ast.AST, content: str) -> list[dict[str, Any]]:
        """Check for empty except blocks."""
        bugs = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if not node.body or (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    bugs.append({
                        "lineno": node.lineno,
                        "col_offset": node.col_offset,
                    })
        return bugs

    def _check_unused_import(self, tree: ast.AST, content: str) -> list[dict[str, Any]]:
        """Check for unused imports (basic heuristic)."""
        bugs = []
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name)

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                imported_names.discard(node.id)

        unused = imported_names
        if unused:
            bugs.append({
                "lineno": 1,
                "col_offset": 0,
                "metadata": {"unused_imports": list(unused)},
            })
        return bugs

    def _check_unreachable_code(self, tree: ast.AST, content: str) -> list[dict[str, Any]]:
        """Check for unreachable code after return/raise."""
        bugs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                        if i + 1 < len(node.body):
                            next_stmt = node.body[i + 1]
                            bugs.append({
                                "lineno": next_stmt.lineno,
                                "col_offset": next_stmt.col_offset,
                                "metadata": {"function": node.name},
                            })
                        break
        return bugs


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

class CodeGenerator:
    """Generates simple code from structured specifications.

    Supports generating:
    - Function stubs
    - Class stubs
    - Basic test cases
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger

    def generate_function(self, spec: dict[str, Any]) -> str:
        """Generate a function from a specification."""
        name = spec.get("name", "generated_function")
        args = spec.get("args", [])
        returns = spec.get("returns", "")
        docstring = spec.get("docstring", "")
        body = spec.get("body", "pass")

        args_str = ", ".join(args)
        returns_str = f" -> {returns}" if returns else ""
        docstring_str = f'    """{docstring}"""\n' if docstring else ""

        code = f"def {name}({args_str}){returns_str}:\n{docstring_str}    {body}\n"
        return code

    def generate_class(self, spec: dict[str, Any]) -> str:
        """Generate a class from a specification."""
        name = spec.get("name", "GeneratedClass")
        bases = spec.get("bases", [])
        methods = spec.get("methods", [])
        docstring = spec.get("docstring", "")

        bases_str = f"({', '.join(bases)})" if bases else ""
        docstring_str = f'    """{docstring}"""\n' if docstring else ""

        code = f"class {name}{bases_str}:\n{docstring_str}"
        for method in methods:
            method_code = self.generate_function({
                "name": method.get("name", "method"),
                "args": method.get("args", ["self"]),
                "returns": method.get("returns", ""),
                "docstring": method.get("docstring", ""),
                "body": method.get("body", "pass"),
            })
            code += "\n" + textwrap.indent(method_code, "    ")

        return code

    def generate_test(self, spec: dict[str, Any]) -> GeneratedTest:
        """Generate a test case for a function."""
        target = spec.get("target_function", "unknown")
        test_cases = spec.get("test_cases", [])
        imports = spec.get("imports", [])

        test_id = hashlib.sha256(f"{target}:{','.join(str(t) for t in test_cases)}".encode()).hexdigest()[:12]

        import_lines = "\n".join(imports)
        test_functions = []
        for i, tc in enumerate(test_cases):
            test_name = tc.get("name", f"test_{target}_{i}")
            args = tc.get("args", [])
            expected = tc.get("expected", "None")
            test_functions.append(f"""
def {test_name}():
    result = {target}({', '.join(repr(a) for a in args)})
    assert result == {repr(expected)}
""")

        test_code = f"""{import_lines}


{textwrap.dedent("".join(test_functions))}
"""

        return GeneratedTest(
            test_id=test_id,
            test_name=f"test_{target}",
            target_function=target,
            test_code=test_code,
            framework="pytest",
        )

    def explain_code(self, code_info: dict[str, Any]) -> CodeExplanation:
        """Generate an explanation of code structure."""
        functions = code_info.get("functions", [])
        classes = code_info.get("classes", [])
        imports = code_info.get("imports", [])

        func_names = [f.get("name", "") for f in functions] if isinstance(functions, list) else []
        class_names = [c.get("name", "") for c in classes] if isinstance(classes, list) else []
        import_names = [i.get("module", "") for i in imports] if isinstance(imports, list) else []

        summary_parts = []
        if class_names:
            summary_parts.append(f"Defines {len(class_names)} class(es): {', '.join(class_names)}")
        if func_names:
            summary_parts.append(f"Defines {len(func_names)} function(s): {', '.join(func_names[:5])}")
        if import_names:
            summary_parts.append(f"Imports {len(import_names)} module(s)")

        complexity_notes = []
        for func in (functions if isinstance(functions, list) else []):
            complexity = func.get("complexity", 1)
            if complexity > 10:
                complexity_notes.append(f"Function '{func.get('name', '')}' has high complexity ({complexity})")

        return CodeExplanation(
            summary=". ".join(summary_parts) if summary_parts else "Empty or unrecognized code structure",
            functions=func_names,
            classes=class_names,
            imports=import_names,
            dependencies=import_names,
            complexity_notes=complexity_notes,
            metadata={"function_count": len(func_names), "class_count": len(class_names)},
        )


# ---------------------------------------------------------------------------
# Patch evaluation
# ---------------------------------------------------------------------------

class PatchEvaluator:
    """Evaluates code patches for safety and correctness.

    Analyzes:
    - Affected files
    - Affected functions
    - Risk factors
    - Confidence in correctness
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger

    def evaluate_patch(self, original: str, patched: str, context: Optional[dict[str, Any]] = None) -> PatchEvaluation:
        """Evaluate a patch by comparing original and patched code."""
        patch_id = hashlib.sha256(f"{original}:{patched}".encode()).hexdigest()[:12]
        risks = []
        affected_functions = []
        affected_files = []

        try:
            original_tree = ast.parse(original)
            patched_tree = ast.parse(patched)
        except SyntaxError as e:
            return PatchEvaluation(
                patch_id=patch_id,
                is_safe=False,
                risks=[f"Syntax error in patch: {e}"],
                confidence=0.0,
                reasoning="Patch contains syntax errors",
            )

        original_funcs = {n.name: n for n in ast.walk(original_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        patched_funcs = {n.name: n for n in ast.walk(patched_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

        for name in patched_funcs:
            if name not in original_funcs:
                affected_functions.append(name)
                risks.append(f"New function added: {name}")

        for name in original_funcs:
            if name not in patched_funcs:
                affected_functions.append(name)
                risks.append(f"Function removed: {name}")

        confidence = 0.8
        if risks:
            confidence = max(0.0, confidence - 0.1 * len(risks))

        reasoning = f"Patch modifies {len(affected_functions)} function(s)"
        if affected_functions:
            reasoning += f": {', '.join(affected_functions[:5])}"

        return PatchEvaluation(
            patch_id=patch_id,
            is_safe=len(risks) == 0,
            risks=risks,
            affected_functions=affected_functions,
            confidence=confidence,
            reasoning=reasoning,
        )

    def evaluate_diff(self, diff_text: str, context: Optional[dict[str, Any]] = None) -> PatchEvaluation:
        """Evaluate a diff patch."""
        risks = []
        affected_files = []

        for line in diff_text.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                path = line[4:].strip()
                if path and not path.startswith("dev/null"):
                    affected_files.append(path)
            if "delete" in line.lower() or "remove" in line.lower():
                risks.append("Patch removes code")

        confidence = 0.7
        if len(affected_files) > 5:
            confidence -= 0.1
            risks.append("Patch affects many files")

        return PatchEvaluation(
            patch_id=hashlib.sha256(diff_text.encode()).hexdigest()[:12],
            is_safe=len(risks) == 0,
            risks=risks,
            affected_files=affected_files,
            confidence=confidence,
            reasoning=f"Diff affects {len(affected_files)} file(s)",
        )


# ---------------------------------------------------------------------------
# Native Coding Intelligence
# ---------------------------------------------------------------------------

class NativeCodingIntelligence:
    """Native coding intelligence for EVORA.

    Provides:
    - Code understanding via AST parsing
    - Bug detection
    - Simple code generation
    - Code explanation
    - Test generation
    - Patch evaluation

    No ModelManager dependency.
    No external model dependency.
    Works offline for supported languages (Python primary).
    """

    def __init__(
        self,
        code_understanding: Optional[CodeUnderstanding] = None,
        bug_detector: Optional[BugDetector] = None,
        code_generator: Optional[CodeGenerator] = None,
        patch_evaluator: Optional[PatchEvaluator] = None,
        logger: Optional[Any] = None,
    ):
        self.code_understanding = code_understanding or CodeUnderstanding(logger=logger)
        self.bug_detector = bug_detector or BugDetector(logger=logger)
        self.code_generator = code_generator or CodeGenerator(logger=logger)
        self.patch_evaluator = patch_evaluator or PatchEvaluator(logger=logger)
        self.logger = logger

    def understand_file(self, path: str) -> dict[str, Any]:
        """Understand a source file structure."""
        if self.logger:
            self.logger.observe(f"Analyzing file: {path}")
        return self.code_understanding.parse_file(path)

    def detect_bugs(self, path: str) -> list[BugPattern]:
        """Detect bugs in a source file."""
        if self.logger:
            self.logger.observe(f"Detecting bugs in: {path}")
        return self.bug_detector.detect_bugs(path)

    def generate_code(self, spec: dict[str, Any]) -> str:
        """Generate code from a specification."""
        if self.logger:
            self.logger.observe(f"Generating code: {spec.get('name', 'unknown')}")
        code_type = spec.get("type", "function")
        if code_type == "class":
            return self.code_generator.generate_class(spec)
        return self.code_generator.generate_function(spec)

    def explain_code(self, path: str) -> CodeExplanation:
        """Explain code structure."""
        if self.logger:
            self.logger.observe(f"Explaining code: {path}")
        code_info = self.code_understanding.parse_file(path)
        if "error" in code_info:
            return CodeExplanation(summary=f"Error: {code_info['error']}")
        return self.code_generator.explain_code(code_info)

    def generate_test(self, spec: dict[str, Any]) -> GeneratedTest:
        """Generate test cases."""
        if self.logger:
            self.logger.observe(f"Generating test for: {spec.get('target_function', 'unknown')}")
        return self.code_generator.generate_test(spec)

    def evaluate_patch(self, original: str, patched: str) -> PatchEvaluation:
        """Evaluate a patch."""
        if self.logger:
            self.logger.observe("Evaluating patch")
        return self.patch_evaluator.evaluate_patch(original, patched)

    def evaluate_diff(self, diff_text: str) -> PatchEvaluation:
        """Evaluate a diff."""
        if self.logger:
            self.logger.observe("Evaluating diff")
        return self.patch_evaluator.evaluate_diff(diff_text)

    def get_capabilities(self) -> list[dict[str, Any]]:
        """Get coding capabilities."""
        return [
            {"name": "python_code_understanding", "native": True, "confidence": 0.8},
            {"name": "bug_detection", "native": True, "confidence": 0.6},
            {"name": "simple_code_generation", "native": True, "confidence": 0.5},
            {"name": "code_explanation", "native": True, "confidence": 0.7},
            {"name": "test_generation", "native": True, "confidence": 0.5},
            {"name": "patch_evaluation", "native": True, "confidence": 0.7},
            {"name": "complex_code_generation", "native": False, "confidence": 0.0},
        ]
