"""
Project analyzer for EVORA.

Inspects a project directory to detect:
    - Language(s) and frameworks
    - Dependencies
    - Project structure
    - Entry points
    - Configuration files
    - Test setup
    - Git status
    - Build system
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evora.logger import Logger


@dataclass
class AnalysisResult:
    project_name: str
    workspace: str
    languages: dict[str, float] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    entry_points: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    has_git: bool = False
    git_branch: Optional[str] = None
    build_system: Optional[str] = None
    test_command: Optional[str] = None
    estimated_size: int = 0
    file_count: int = 0
    conventions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "workspace": self.workspace,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "dependencies": self.dependencies,
            "entry_points": self.entry_points,
            "config_files": self.config_files,
            "test_files": self.test_files,
            "has_git": self.has_git,
            "git_branch": self.git_branch,
            "build_system": self.build_system,
            "test_command": self.test_command,
            "estimated_size": self.estimated_size,
            "file_count": self.file_count,
            "conventions": self.conventions,
        }


class ProjectAnalyzer:
    """Analyzes project structure and conventions."""

    LANGUAGE_EXTENSIONS = {
        ".py": "Python",
        ".go": "Go",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".jsx": "JavaScript/JSX",
        ".tsx": "TypeScript/TSX",
        ".java": "Java",
        ".rs": "Rust",
        ".c": "C",
        ".cpp": "C++",
        ".h": "C/C++",
        ".cs": "C#",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".sh": "Shell",
        ".sql": "SQL",
        ".css": "CSS",
        ".html": "HTML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".xml": "XML",
        ".md": "Markdown",
    }

    ENTRY_POINTS = {
        "main.py": "Python",
        "app.py": "Python",
        "wsgi.py": "Python",
        "asgi.py": "Python",
        "main.go": "Go",
        "main.js": "JavaScript",
        "main.ts": "TypeScript",
        "index.js": "JavaScript",
        "index.ts": "TypeScript",
        "server.js": "JavaScript",
        "server.ts": "TypeScript",
        "__init__.py": "Python",
    }

    def __init__(self, workspace_dir: str, logger: Optional[Logger] = None):
        self.workspace = Path(workspace_dir).resolve()
        self.logger = logger

    def analyze(self) -> AnalysisResult:
        if self.logger:
            self.logger.plan(f"Analyzing project at {self.workspace}...")

        result = AnalysisResult(
            project_name=self.workspace.name,
            workspace=str(self.workspace),
        )

        self._detect_languages(result)
        self._detect_frameworks(result)
        self._detect_dependencies(result)
        self._detect_entry_points(result)
        self._detect_config_and_tests(result)
        self._detect_git(result)
        self._detect_build_system(result)
        self._calculate_stats(result)

        if self.logger:
            langs = ", ".join(f"{k} ({v:.0f}%)" for k, v in sorted(result.languages.items(), key=lambda x: -x[1]))
            self.logger.info(f"Languages: {langs}")
            if result.frameworks:
                self.logger.info(f"Frameworks: {', '.join(result.frameworks)}")
            if result.test_files:
                self.logger.info(f"Test files: {len(result.test_files)}")

        return result

    def _detect_languages(self, result: AnalysisResult):
        lang_counts = {}
        total_lines = 0
        skip_dirs = {"__pycache__", "node_modules", ".git", "venv", ".venv", "target", "dist", "build", ".evora"}

        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in self.LANGUAGE_EXTENSIONS:
                    filepath = Path(root) / f
                    lang = self.LANGUAGE_EXTENSIONS[ext]
                    try:
                        with open(filepath, "r", errors="ignore") as fh:
                            lines = len(fh.readlines())
                            lang_counts[lang] = lang_counts.get(lang, 0) + lines
                            total_lines += lines
                    except Exception:
                        pass

        if total_lines > 0:
            for lang, lines in lang_counts.items():
                result.languages[lang] = (lines / total_lines) * 100

    def _detect_frameworks(self, result: AnalysisResult):
        frameworks = set()

        if (self.workspace / "go.mod").exists():
            frameworks.add("Go")
            content = (self.workspace / "go.mod").read_text()
            if "gin-gonic/gin" in content:
                frameworks.add("Gin")
            if "echo-contrib" in content:
                frameworks.add("Echo")
            if "fiber" in content:
                frameworks.add("Fiber")

        if (self.workspace / "requirements.txt").exists() or (self.workspace / "pyproject.toml").exists():
            frameworks.add("Python")
            content = ""
            for f in ["requirements.txt", "pyproject.toml"]:
                path = self.workspace / f
                if path.exists():
                    content += path.read_text() + "\n"
            if "flask" in content.lower():
                frameworks.add("Flask")
            if "django" in content.lower():
                frameworks.add("Django")
            if "fastapi" in content.lower():
                frameworks.add("FastAPI")
            if "pytest" in content.lower():
                frameworks.add("pytest")

        if (self.workspace / "package.json").exists():
            frameworks.add("Node.js")
            content = (self.workspace / "package.json").read_text()
            if '"next"' in content:
                frameworks.add("Next.js")
            if '"react"' in content:
                frameworks.add("React")
            if '"vue"' in content:
                frameworks.add("Vue")
            if '"angular"' in content:
                frameworks.add("Angular")
            if '"express"' in content:
                frameworks.add("Express")

        if (self.workspace / "Cargo.toml").exists():
            frameworks.add("Rust")

        if (self.workspace / "Dockerfile").exists():
            frameworks.add("Docker")

        if (self.workspace / "docker-compose.yml").exists() or (self.workspace / "docker-compose.yaml").exists():
            frameworks.add("Docker Compose")

        result.frameworks = list(frameworks)

    def _detect_dependencies(self, result: AnalysisResult):
        deps = {}

        go_mod = self.workspace / "go.mod"
        if go_mod.exists():
            go_deps = []
            in_deps = False
            for line in go_mod.read_text().splitlines():
                if line.strip() == "require (":
                    in_deps = True
                    continue
                if in_deps:
                    if line.strip() == ")":
                        in_deps = False
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        go_deps.append(f"{parts[0]} {parts[1]}")
            deps["go"] = go_deps

        req_txt = self.workspace / "requirements.txt"
        if req_txt.exists():
            deps["python"] = [l.strip() for l in req_txt.read_text().splitlines() if l.strip() and not l.startswith("#")]

        pyproject = self.workspace / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib
                with open(pyproject, "rb") as f:
                    data = tomllib.load(f)
                if "project" in data and "dependencies" in data["project"]:
                    deps["python"] = data["project"]["dependencies"]
            except Exception:
                pass

        pkg_json = self.workspace / "package.json"
        if pkg_json.exists():
            try:
                import json
                data = json.loads(pkg_json.read_text())
                all_deps = {}
                all_deps.update(data.get("dependencies", {}))
                all_deps.update(data.get("devDependencies", {}))
                deps["node"] = [f"{k}: {v}" for k, v in all_deps.items()]
            except Exception:
                pass

        result.dependencies = deps

    def _detect_entry_points(self, result: AnalysisResult):
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"__pycache__", "node_modules", "venv", ".venv", "target", "dist", "build"}]
            for f in files:
                if f in self.ENTRY_POINTS:
                    result.entry_points.append(str(Path(root) / f))
                elif f in ("main.go", "main.py", "app.py", "wsgi.py", "asgi.py"):
                    result.entry_points.append(str(Path(root) / f))

    def _detect_config_and_tests(self, result: AnalysisResult):
        config_names = {
            ".env", ".env.local", ".env.production", "config.yaml", "config.yml",
            "config.json", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
            "Makefile", "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
            "nginx.conf", ".eslintrc", ".prettierrc", ".flake8", "tox.ini",
            "pytest.ini", "conftest.py", "tsconfig.json", "jsconfig.json",
        }

        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"__pycache__", "node_modules", "venv", ".venv", "target", "dist"}]
            for f in files:
                if f in config_names:
                    result.config_files.append(str(Path(root) / f))

        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in {"__pycache__", "node_modules", "venv", ".venv", "target", "dist"}]
            for f in files:
                if f.startswith("test_") and f.endswith(".py"):
                    result.test_files.append(str(Path(root) / f))
                elif f.endswith("_test.go"):
                    result.test_files.append(str(Path(root) / f))
                elif f.endswith(".test.js") or f.endswith(".test.ts") or (f.startswith("test.") and (f.endswith(".js") or f.endswith(".ts"))):
                    result.test_files.append(str(Path(root) / f))
                elif f == "conftest.py":
                    result.test_files.append(str(Path(root) / f))

    def _detect_git(self, result: AnalysisResult):
        result.has_git = (self.workspace / ".git").exists()
        if result.has_git:
            git_dir = self.workspace / ".git"
            head_file = git_dir / "HEAD"
            if head_file.exists():
                content = head_file.read_text().strip()
                if content.startswith("ref:"):
                    result.git_branch = content.replace("ref: refs/heads/", "")

    def _detect_build_system(self, result: AnalysisResult):
        if (self.workspace / "Makefile").exists():
            result.build_system = "Make"
        elif (self.workspace / "go.mod").exists():
            result.build_system = "Go"
        elif (self.workspace / "pyproject.toml").exists():
            result.build_system = "Python"
        elif (self.workspace / "package.json").exists():
            result.build_system = "Node.js"
        elif (self.workspace / "Cargo.toml").exists():
            result.build_system = "Cargo"
        elif (self.workspace / "build.gradle").exists():
            result.build_system = "Gradle"
        elif (self.workspace / "pom.xml").exists():
            result.build_system = "Maven"
        elif (self.workspace / "CMakeLists.txt").exists():
            result.build_system = "CMake"

        if (self.workspace / "go.mod").exists():
            result.test_command = "go test -v ./..."
        elif (self.workspace / "pytest.ini").exists() or any(self.workspace.rglob("test_*.py")):
            result.test_command = "python -m pytest"
        elif (self.workspace / "package.json").exists():
            result.test_command = "npm test"

    def _calculate_stats(self, result: AnalysisResult):
        total_size = 0
        file_count = 0
        skip_dirs = {"__pycache__", "node_modules", ".git", "venv", ".venv", "target", "dist"}

        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                fp = Path(root) / f
                try:
                    total_size += fp.stat().st_size
                    file_count += 1
                except Exception:
                    pass

        result.estimated_size = total_size
        result.file_count = file_count
