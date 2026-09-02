"""
Phase 9 — Resource awareness for EVORA.

Provides safe, structured resource information about the EVORA runtime
environment. Resource data is intended for the Brain to consume, not
for unrestricted system probing.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class ResourceInfo:
    """Structured resource information."""

    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    cpu_count: int = 0
    memory_total_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    active_provider: str = ""
    active_model: str = ""
    available_providers: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    workspace_dir: str = ""
    execution_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_name": self.os_name,
            "os_version": self.os_version,
            "python_version": self.python_version,
            "cpu_count": self.cpu_count,
            "memory_total_mb": self.memory_total_mb,
            "memory_available_mb": self.memory_available_mb,
            "disk_total_gb": self.disk_total_gb,
            "disk_free_gb": self.disk_free_gb,
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "available_providers": self.available_providers,
            "available_tools": self.available_tools,
            "workspace_dir": self.workspace_dir,
            "execution_constraints": self.execution_constraints,
        }


class ResourceMonitor:
    """Safe resource inspection for the Brain.

    Avoids excessive system probing and respects existing security
    boundaries. All values are best-effort and may be zero/empty
    when inspection is not available or not permitted.
    """

    def __init__(self, workspace_dir: str = ".", logger: Optional[Logger] = None):
        self.workspace_dir = workspace_dir
        self.logger = logger

    def collect(self, *, model_manager: Any = None, tool_registry: Any = None) -> ResourceInfo:
        """Collect current resource information."""
        info = ResourceInfo(
            os_name=platform.system(),
            os_version=platform.version(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            workspace_dir=self.workspace_dir,
        )

        try:
            import os as _os
            if hasattr(_os, "cpu_count") and _os.cpu_count():
                info.cpu_count = _os.cpu_count()
        except Exception:
            pass

        try:
            import psutil
            vm = psutil.virtual_memory()
            info.memory_total_mb = vm.total / (1024 * 1024)
            info.memory_available_mb = vm.available / (1024 * 1024)
            disk = psutil.disk_usage(self.workspace_dir)
            info.disk_total_gb = disk.total / (1024 ** 3)
            info.disk_free_gb = disk.free / (1024 ** 3)
        except Exception:
            pass

        if model_manager is not None:
            try:
                active = model_manager.active
                if active is not None:
                    info.active_provider = active.name()
                    info.active_model = active.model()
                info.available_providers = model_manager.list_providers()
            except Exception:
                pass

        if tool_registry is not None:
            try:
                info.available_tools = list(tool_registry.list())
            except Exception:
                pass

        constraints: list[str] = []
        try:
            from evora.config import load_config
            config = load_config()
            if not config.api_key:
                constraints.append("no_cloud_api_key")
            if config.provider == "ollama":
                constraints.append("local_inference_only")
        except Exception:
            pass
        info.execution_constraints = constraints

        if self.logger:
            self.logger.observe(
                f"Collected resource info: cpu={info.cpu_count} mem={info.memory_available_mb:.0f}MB"
            )

        return info
