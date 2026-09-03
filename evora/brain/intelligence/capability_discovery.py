"""
Phase 29 — Native Capability Discovery for EVORA.

Discovers and registers new capabilities dynamically.

Supports:
  - Capability scanning
  - Capability registration
  - Capability matching
  - Capability versioning
  - Integration with CapabilityRegistry
  - Integration with IntelligenceRuntime

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    """A capability definition."""
    capability_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    input_types: list[str] = field(default_factory=list)
    output_types: list[str] = field(default_factory=list)
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "input_types": self.input_types,
            "output_types": self.output_types,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at,
        }


@dataclass
class DiscoveryResult:
    """Result of a capability discovery operation."""
    success: bool = False
    discovered_count: int = 0
    registered_count: int = 0
    skipped_count: int = 0
    errors: list[str] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "discovered_count": self.discovered_count,
            "registered_count": self.registered_count,
            "skipped_count": self.skipped_count,
            "errors": self.errors,
            "capabilities": [c.to_dict() for c in self.capabilities],
        }


# ---------------------------------------------------------------------------
# Native Capability Discovery
# ---------------------------------------------------------------------------

class NativeCapabilityDiscovery:
    """Native capability discovery for EVORA.

    Discovers and registers capabilities.
    """

    def __init__(
        self,
        capability_registry: Any = None,
        intelligence_runtime: Any = None,
        logger: Optional[Any] = None,
    ):
        self.capability_registry = capability_registry
        self.intelligence_runtime = intelligence_runtime
        self.logger = logger
        self._discovered: dict[str, Capability] = {}

    def discover_from_registry(self) -> DiscoveryResult:
        """Discover capabilities from the capability registry."""
        result = DiscoveryResult()
        if self.capability_registry is None:
            result.errors.append("No capability registry available")
            return result
        try:
            if hasattr(self.capability_registry, "list_capabilities"):
                capabilities = self.capability_registry.list_capabilities()
                for cap in capabilities:
                    capability = self._normalize_capability(cap)
                    self._discovered[capability.capability_id] = capability
                    result.capabilities.append(capability)
                    result.registered_count += 1
                result.success = True
                result.discovered_count = len(result.capabilities)
        except Exception as e:
            result.errors.append(str(e))
        return result

    def discover_from_runtime(self) -> DiscoveryResult:
        """Discover capabilities from the intelligence runtime."""
        result = DiscoveryResult()
        if self.intelligence_runtime is None:
            result.errors.append("No intelligence runtime available")
            return result
        try:
            if hasattr(self.intelligence_runtime, "get_capabilities"):
                capabilities = self.intelligence_runtime.get_capabilities()
                for cap in capabilities:
                    capability = self._normalize_capability(cap)
                    if capability.capability_id not in self._discovered:
                        self._discovered[capability.capability_id] = capability
                        result.capabilities.append(capability)
                        result.registered_count += 1
                result.success = True
                result.discovered_count = len(result.capabilities)
        except Exception as e:
            result.errors.append(str(e))
        return result

    def register_capability(self, capability: Capability) -> bool:
        """Register a new capability."""
        self._discovered[capability.capability_id] = capability
        if self.capability_registry is not None and hasattr(self.capability_registry, "register"):
            try:
                self.capability_registry.register(capability.name, capability.metadata)
            except Exception:
                pass
        return True

    def match_capability(self, requirement: str) -> list[Capability]:
        """Match capabilities to a requirement."""
        matches = []
        requirement_lower = requirement.lower()
        for capability in self._discovered.values():
            if requirement_lower in capability.name.lower() or requirement_lower in capability.description.lower():
                matches.append(capability)
        matches.sort(key=lambda c: c.confidence, reverse=True)
        return matches

    def get_capability(self, capability_id: str) -> Optional[Capability]:
        """Get a capability by ID."""
        return self._discovered.get(capability_id)

    def get_all_capabilities(self) -> list[Capability]:
        """Get all discovered capabilities."""
        return list(self._discovered.values())

    def get_discovery_metrics(self) -> dict[str, Any]:
        """Get discovery metrics."""
        return {
            "total_capabilities": len(self._discovered),
            "capability_names": [c.name for c in self._discovered.values()],
        }

    def _normalize_capability(self, cap: Any) -> Capability:
        """Normalize a capability definition."""
        if isinstance(cap, Capability):
            return cap
        if isinstance(cap, dict):
            return Capability(
                name=cap.get("name", ""),
                description=cap.get("description", ""),
                version=cap.get("version", "1.0.0"),
                input_types=cap.get("input_types", []),
                output_types=cap.get("output_types", []),
                confidence=cap.get("confidence", 0.5),
                metadata=cap.get("metadata", {}),
            )
        return Capability(name=str(cap))
