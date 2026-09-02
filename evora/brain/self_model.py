"""
Phase 9 — SelfModel for EVORA.

Provides a controlled, observable-based representation of EVORA's
understanding of itself. All fields are derived from observable system state.
No capability is invented by the model.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class Capabilities:
    """EVORA's known capabilities."""
    has_agent_loop: bool = True
    has_planning: bool = True
    has_reasoning: bool = True
    has_memory: bool = True
    has_learning: bool = True
    has_tools: bool = True
    has_identity: bool = True
    has_approval: bool = True
    has_security: bool = True
    has_self_improvement: bool = True
    has_web_search: bool = False
    has_code_analysis: bool = True
    has_git_integration: bool = True
    has_chat_ui: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_loop": self.has_agent_loop,
            "planning": self.has_planning,
            "reasoning": self.has_reasoning,
            "memory": self.has_memory,
            "learning": self.has_learning,
            "tools": self.has_tools,
            "identity": self.has_identity,
            "approval": self.has_approval,
            "security": self.has_security,
            "self_improvement": self.has_self_improvement,
            "web_search": self.has_web_search,
            "code_analysis": self.has_code_analysis,
            "git_integration": self.has_git_integration,
            "chat_ui": self.has_chat_ui,
        }


@dataclass
class Limitations:
    """Known limitations of the current EVORA instance."""
    requires_provider: bool = True
    requires_api_key_for_cloud: bool = True
    no_vector_db: bool = True
    no_persistent_agent_process: bool = True
    bounded_reasoning: bool = True
    no_hidden_chain_of_thought: bool = True
    cannot_bypass_security: bool = True
    cannot_self_modify_without_approval: bool = True
    no_continuous_learning: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requires_provider": self.requires_provider,
            "requires_api_key_for_cloud": self.requires_api_key_for_cloud,
            "no_vector_db": self.no_vector_db,
            "no_persistent_agent_process": self.no_persistent_agent_process,
            "bounded_reasoning": self.bounded_reasoning,
            "no_hidden_chain_of_thought": self.no_hidden_chain_of_thought,
            "cannot_bypass_security": self.cannot_bypass_security,
            "cannot_self_modify_without_approval": self.cannot_self_modify_without_approval,
            "no_continuous_learning": self.no_continuous_learning,
        }


@dataclass
class SelfModel:
    """Controlled representation of EVORA's understanding of itself.

    All fields are derived from observable system state.
    No capability is invented by the model.
    """

    version: str = "phase9"
    build: str = ""
    capabilities: Capabilities = field(default_factory=Capabilities)
    limitations: Limitations = field(default_factory=Limitations)
    active_provider: str = ""
    active_model: str = ""
    available_providers: list[str] = field(default_factory=list)
    available_tools: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    current_authority: str = ""
    system_status: str = "healthy"
    python_version: str = ""
    platform_info: str = ""

    def __post_init__(self):
        if not self.python_version:
            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if not self.platform_info:
            self.platform_info = platform.platform()

    @classmethod
    def from_observable_state(
        cls,
        *,
        version: str = "phase9",
        build: str = "",
        model_manager: Any = None,
        tool_registry: Any = None,
        identity_service: Any = None,
        logger: Optional[Logger] = None,
    ) -> "SelfModel":
        """Build SelfModel from observable system state."""
        available_providers = []
        active_provider = ""
        active_model = ""

        if model_manager is not None:
            try:
                available_providers = model_manager.list_providers()
                active = model_manager.active
                if active is not None:
                    active_provider = active.name()
                    active_model = active.model()
            except Exception:
                pass

        available_tools = []
        if tool_registry is not None:
            try:
                available_tools = list(tool_registry.list())
            except Exception:
                pass

        components = []
        try:
            import evora
            pkg_dir = evora.__path__[0]
            import os as _os
            for fname in _os.listdir(pkg_dir):
                if fname.endswith(".py") and fname != "__init__.py":
                    components.append(fname.replace(".py", ""))
        except Exception:
            pass

        current_authority = ""
        if identity_service is not None:
            try:
                identity = identity_service.current_identity()
                current_authority = identity.authority.value
            except Exception:
                pass

        return cls(
            version=version,
            build=build,
            active_provider=active_provider,
            active_model=active_model,
            available_providers=available_providers,
            available_tools=available_tools,
            components=sorted(components),
            current_authority=current_authority,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "build": self.build,
            "capabilities": self.capabilities.to_dict(),
            "limitations": self.limitations.to_dict(),
            "active_provider": self.active_provider,
            "active_model": self.active_model,
            "available_providers": self.available_providers,
            "available_tools": self.available_tools,
            "components": self.components,
            "current_authority": self.current_authority,
            "system_status": self.system_status,
            "python_version": self.python_version,
            "platform_info": self.platform_info,
        }

    def describe(self) -> str:
        """Human-readable self-description."""
        lines = [
            f"EVORA v{self.version}",
            f"Provider: {self.active_provider or 'none'} / {self.active_model or 'n/a'}",
            f"Authority: {self.current_authority or 'unknown'}",
            f"Tools: {len(self.available_tools)} available",
            f"Components: {', '.join(self.components) if self.components else 'n/a'}",
            f"Python: {self.python_version} on {self.platform_info}",
        ]
        return "\n".join(lines)
