"""
Phase 10 — Capability Registry for EVORA native intelligence.

Provides honest classification of EVORA's capabilities:
  - NATIVE: EVORA can do this without an external model
  - LOCAL_MODEL: Requires a local model provider
  - EXTERNAL_MODEL: Requires an external API provider
  - UNAVAILABLE: Not currently possible

The registry is metadata-only. It does NOT perform inference.
It does NOT call ModelManager.
It does NOT grant authority.
`requires_approval` is metadata only; existing IdentityService,
PermissionManager, and ApprovalSystem remain authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class CapabilityType(str, Enum):
    """Types of intelligence capabilities."""
    NATIVE = "native"
    LOCAL_MODEL = "local_model"
    EXTERNAL_MODEL = "external"
    UNAVAILABLE = "unavailable"


@dataclass
class IntelligenceCapability:
    """Description of an EVORA intelligence capability."""

    name: str
    description: str
    capability_type: CapabilityType
    native_confidence: float = 0.0
    requires_model: bool = False
    fallback_available: bool = False
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or not self.name.strip():
            raise ValueError("IntelligenceCapability name must be non-empty")
        if not self.description or not self.description.strip():
            raise ValueError("IntelligenceCapability description must be non-empty")
        if not isinstance(self.capability_type, CapabilityType):
            raise ValueError(f"capability_type must be a CapabilityType, got {type(self.capability_type)}")
        if self.native_confidence < 0.0:
            self.native_confidence = 0.0
        if self.native_confidence > 1.0:
            self.native_confidence = 1.0
        if not isinstance(self.requires_model, bool):
            raise ValueError("requires_model must be a bool")
        if not isinstance(self.fallback_available, bool):
            raise ValueError("fallback_available must be a bool")
        if not isinstance(self.requires_approval, bool):
            raise ValueError("requires_approval must be a bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capability_type": self.capability_type.value,
            "native_confidence": self.native_confidence,
            "requires_model": self.requires_model,
            "fallback_available": self.fallback_available,
            "requires_approval": self.requires_approval,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IntelligenceCapability":
        cap_type = data.get("capability_type", CapabilityType.UNAVAILABLE.value)
        if isinstance(cap_type, str):
            cap_type = CapabilityType(cap_type)
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            capability_type=cap_type,
            native_confidence=data.get("native_confidence", 0.0),
            requires_model=data.get("requires_model", False),
            fallback_available=data.get("fallback_available", False),
            requires_approval=data.get("requires_approval", False),
            metadata=data.get("metadata", {}),
        )


class CapabilityRegistry:
    """Registry of EVORA intelligence capabilities.

    Provides honest classification of what EVORA can do natively,
    what requires a model, and what is unavailable.

    The registry is metadata-only. It does NOT perform inference.
    It does NOT call ModelManager.
    It does NOT grant authority.
    """

    def __init__(self, logger: Any = None):
        self._capabilities: dict[str, IntelligenceCapability] = {}
        self._logger = logger
        self._max_capabilities = 200

    def register(self, capability: IntelligenceCapability) -> None:
        """Register a capability.

        Validates the capability metadata.
        Handles duplicates by keeping the higher-confidence entry.
        Bounds total number of capabilities.
        """
        if not isinstance(capability, IntelligenceCapability):
            raise TypeError(f"Expected IntelligenceCapability, got {type(capability)}")

        name = capability.name.strip()
        if not name:
            raise ValueError("Capability name must be non-empty")

        existing = self._capabilities.get(name)
        if existing is not None:
            if capability.native_confidence > existing.native_confidence:
                self._capabilities[name] = capability
            return

        if len(self._capabilities) >= self._max_capabilities:
            lowest_name = min(self._capabilities, key=lambda n: self._capabilities[n].native_confidence)
            del self._capabilities[lowest_name]

        self._capabilities[name] = capability
        if self._logger:
            self._logger.observe(
                f"Registered capability: {name} ({capability.capability_type.value}, confidence={capability.native_confidence:.2f})"
            )

    def get(self, name: str) -> Optional[IntelligenceCapability]:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def can_handle(self, name: str) -> bool:
        """Check if EVORA can handle a capability natively or with fallback."""
        cap = self._capabilities.get(name)
        if cap is None:
            return False
        return cap.capability_type != CapabilityType.UNAVAILABLE

    def requires_model(self, name: str) -> bool:
        """Check if a capability requires an external model."""
        cap = self._capabilities.get(name)
        if cap is None:
            return False
        return cap.requires_model

    def list_capabilities(
        self,
        capability_type: Optional[CapabilityType] = None,
        limit: int = 100,
    ) -> list[IntelligenceCapability]:
        """List capabilities, optionally filtered by type."""
        caps = list(self._capabilities.values())
        if capability_type is not None:
            caps = [c for c in caps if c.capability_type == capability_type]
        caps.sort(key=lambda c: c.native_confidence, reverse=True)
        return caps[:limit]

    def get_native_capabilities(self, limit: int = 100) -> list[IntelligenceCapability]:
        """Get capabilities that can be executed natively."""
        return self.list_capabilities(capability_type=CapabilityType.NATIVE, limit=limit)

    def get_model_enhanced_capabilities(self, limit: int = 100) -> list[IntelligenceCapability]:
        """Get capabilities that require or can be enhanced by a model."""
        caps = [
            c
            for c in self._capabilities.values()
            if c.capability_type in (CapabilityType.LOCAL_MODEL, CapabilityType.EXTERNAL_MODEL)
        ]
        caps.sort(key=lambda c: c.native_confidence, reverse=True)
        return caps[:limit]

    def get_unavailable_capabilities(self, limit: int = 100) -> list[IntelligenceCapability]:
        """Get capabilities that are currently unavailable."""
        caps = [c for c in self._capabilities.values() if c.capability_type == CapabilityType.UNAVAILABLE]
        caps.sort(key=lambda c: c.name)
        return caps[:limit]

    def summary(self) -> dict[str, Any]:
        """Return a summary of registered capabilities."""
        type_counts: dict[str, int] = {}
        for cap in self._capabilities.values():
            type_counts[cap.capability_type.value] = type_counts.get(cap.capability_type.value, 0) + 1

        native = self.get_native_capabilities()
        model_enhanced = self.get_model_enhanced_capabilities()
        unavailable = self.get_unavailable_capabilities()

        return {
            "total_capabilities": len(self._capabilities),
            "by_type": type_counts,
            "native_count": len(native),
            "model_enhanced_count": len(model_enhanced),
            "unavailable_count": len(unavailable),
            "native_capabilities": [c.name for c in native],
            "model_enhanced_capabilities": [c.name for c in model_enhanced],
            "unavailable_capabilities": [c.name for c in unavailable],
        }

    def seed_default_capabilities(self) -> None:
        """Seed the registry with sensible Phase 10 default capabilities."""
        native_caps = [
            IntelligenceCapability(
                name="simple reasoning",
                description="Reason about goals using deterministic rules, memory, and learned patterns",
                capability_type=CapabilityType.NATIVE,
                native_confidence=0.7,
                requires_model=False,
                fallback_available=True,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="known-pattern planning",
                description="Create plans using known patterns from knowledge graph and past experiences",
                capability_type=CapabilityType.NATIVE,
                native_confidence=0.8,
                requires_model=False,
                fallback_available=True,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="knowledge retrieval",
                description="Retrieve and apply relevant knowledge from memory and knowledge graph",
                capability_type=CapabilityType.NATIVE,
                native_confidence=1.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="tool suggestion",
                description="Suggest relevant tools based on goal and context",
                capability_type=CapabilityType.NATIVE,
                native_confidence=0.9,
                requires_model=False,
                fallback_available=True,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="known-fact inference",
                description="Infer answers from known facts and rules",
                capability_type=CapabilityType.NATIVE,
                native_confidence=1.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=False,
            ),
        ]

        model_enhanced_caps = [
            IntelligenceCapability(
                name="complex reasoning",
                description="Reason about novel or ambiguous goals requiring external model",
                capability_type=CapabilityType.EXTERNAL_MODEL,
                native_confidence=0.0,
                requires_model=True,
                fallback_available=False,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="novel planning",
                description="Create plans for novel tasks without known patterns",
                capability_type=CapabilityType.EXTERNAL_MODEL,
                native_confidence=0.0,
                requires_model=True,
                fallback_available=False,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="advanced code generation",
                description="Generate complex code requiring deep semantic understanding",
                capability_type=CapabilityType.EXTERNAL_MODEL,
                native_confidence=0.0,
                requires_model=True,
                fallback_available=False,
                requires_approval=False,
            ),
            IntelligenceCapability(
                name="creative problem solving",
                description="Solve problems requiring creative or abstract thinking",
                capability_type=CapabilityType.EXTERNAL_MODEL,
                native_confidence=0.0,
                requires_model=True,
                fallback_available=False,
                requires_approval=False,
            ),
        ]

        unavailable_caps = [
            IntelligenceCapability(
                name="autonomous self-modification",
                description="Modify EVORA's own code without creator approval",
                capability_type=CapabilityType.UNAVAILABLE,
                native_confidence=0.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=True,
            ),
            IntelligenceCapability(
                name="unrestricted computer control",
                description="Gain unrestricted control over the host system",
                capability_type=CapabilityType.UNAVAILABLE,
                native_confidence=0.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=True,
            ),
        ]

        web_caps = [
            IntelligenceCapability(
                name="web search",
                description="Search the web for information (via ToolRegistry)",
                capability_type=CapabilityType.LOCAL_MODEL,
                native_confidence=0.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=True,
            ),
            IntelligenceCapability(
                name="web browse",
                description="Browse and extract content from web pages (via ToolRegistry)",
                capability_type=CapabilityType.LOCAL_MODEL,
                native_confidence=0.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=True,
            ),
            IntelligenceCapability(
                name="web research",
                description="Perform multi-step web research (via ToolRegistry)",
                capability_type=CapabilityType.LOCAL_MODEL,
                native_confidence=0.0,
                requires_model=False,
                fallback_available=False,
                requires_approval=True,
            ),
        ]

        for cap in native_caps + model_enhanced_caps + unavailable_caps + web_caps:
            self.register(cap)
