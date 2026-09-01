"""
Configuration management for EVORA.

Supports YAML config files, environment variables, and sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class ProviderConfig:
    name: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 30.0


@dataclass
class PermissionConfig:
    allow_file_write: bool = True
    allow_cmd_exec: bool = True
    allowed_cmds: list = field(default_factory=list)


@dataclass
class Config:
    api_key: str = ""
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    workspace_dir: str = "."
    log_level: str = "INFO"
    log_file: str = ""
    memory_dir: str = ""
    providers: dict = field(default_factory=dict)
    permissions: PermissionConfig = field(default_factory=PermissionConfig)


def _get_evora_dir() -> Path:
    home = Path.home()
    return home / ".evora"


def _default_config() -> str:
    return """# EVORA Configuration
model: "gpt-4o"
base_url: "https://api.openai.com/v1"
log_level: "INFO"
log_file: ""
workspace_dir: "."

providers:
  openai:
    model: "gpt-4o"
    base_url: "https://api.openai.com/v1"
  anthropic:
    model: "claude-3-5-sonnet-20241022"
    base_url: "https://api.anthropic.com"

permissions:
  allow_file_write: true
  allow_cmd_exec: true
  allowed_cmds: []
"""


def load_config(config_path: Optional[str] = None) -> Config:
    evora_dir = _get_evora_dir()
    evora_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(evora_dir / "config.yaml")
    candidates.append(evora_dir / "config.yml")
    candidates.append(Path("config.yaml"))
    candidates.append(Path("evora.yaml"))

    config_data = {}
    loaded_path = None
    for candidate in candidates:
        if candidate.exists() and candidate.suffix in (".yaml", ".yml"):
            loaded_path = candidate
            if yaml is not None:
                with open(candidate, "r") as f:
                    config_data = yaml.safe_load(f) or {}
            break

    if loaded_path is None:
        default_path = evora_dir / "config.yaml"
        if not default_path.exists():
            with open(default_path, "w") as f:
                f.write(_default_config())
        config_data = {}

    env_overrides = {
        "api_key": os.environ.get("EVORA_API_KEY", ""),
        "model": os.environ.get("EVORA_MODEL", ""),
        "base_url": os.environ.get("EVORA_BASE_URL", ""),
        "workspace_dir": os.environ.get("EVORA_WORKSPACE_DIR", ""),
        "log_level": os.environ.get("EVORA_LOG_LEVEL", ""),
        "log_file": os.environ.get("EVORA_LOG_FILE", ""),
        "memory_dir": os.environ.get("EVORA_MEMORY_DIR", ""),
    }
    for key, value in env_overrides.items():
        if value:
            config_data[key] = value

    providers_data = config_data.get("providers", {})
    providers = {}
    for name, pdata in providers_data.items():
        providers[name] = ProviderConfig(
            name=name,
            model=pdata.get("model", "gpt-4o"),
            api_key=os.environ.get(f"EVORA_{name.upper()}_API_KEY", pdata.get("api_key", "")),
            base_url=pdata.get("base_url", ""),
            timeout=pdata.get("timeout", 30.0),
        )
    if "openai" not in providers:
        providers["openai"] = ProviderConfig(
            name="openai",
            model=os.environ.get("EVORA_MODEL", "gpt-4o"),
            api_key=os.environ.get("EVORA_API_KEY", ""),
            base_url=os.environ.get("EVORA_BASE_URL", "https://api.openai.com/v1"),
            timeout=30.0,
        )

    perms_data = config_data.get("permissions", {})
    permissions = PermissionConfig(
        allow_file_write=perms_data.get("allow_file_write", True),
        allow_cmd_exec=perms_data.get("allow_cmd_exec", True),
        allowed_cmds=perms_data.get("allowed_cmds", []),
    )

    memory_dir = config_data.get("memory_dir", "") or os.environ.get("EVORA_MEMORY_DIR", "")
    if not memory_dir:
        memory_dir = str(evora_dir / "memory")
    Path(memory_dir).mkdir(parents=True, exist_ok=True)

    log_file = config_data.get("log_file", "") or os.environ.get("EVORA_LOG_FILE", "")
    if not log_file:
        log_file = str(evora_dir / "evora.log")

    workspace = config_data.get("workspace_dir", ".") or os.environ.get("EVORA_WORKSPACE_DIR", ".")
    if workspace == "." or not workspace:
        workspace = os.getcwd()
    workspace = os.path.abspath(workspace)

    return Config(
        api_key=config_data.get("api_key", os.environ.get("EVORA_API_KEY", "")),
        model=config_data.get("model", os.environ.get("EVORA_MODEL", "gpt-4o")),
        base_url=config_data.get("base_url", os.environ.get("EVORA_BASE_URL", "https://api.openai.com/v1")),
        workspace_dir=workspace,
        log_level=config_data.get("log_level", os.environ.get("EVORA_LOG_LEVEL", "INFO")),
        log_file=log_file,
        memory_dir=memory_dir,
        providers=providers,
        permissions=permissions,
    )
