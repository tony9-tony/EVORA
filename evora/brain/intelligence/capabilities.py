"""
Phase 10 — CapabilityRegistry for EVORA native intelligence.

Provides a deterministic, honest classification of EVORA's capabilities.
No capability claims native support merely because an external model can perform it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from evora.logger import Logger


class CapabilityType(str, Enum):
    """Types of intelligence capabilities."""
    NATIVE = "native"
    LOCAL_MODEL = "local_model"
    EXTERNAL_MODEL = "external_model"
    UNAVAILABLE = "unavailable"


@dataclass
class IntelligenceCapability:
    """A single intelligence capability with honest classification."""

    name: str
    description: str
    capability_type: CapabilityType
    native_confidence: float = 0.0
    requires_model: bool = False
    fallback_available: bool = False
    requires_approval: bool = False
    limitations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.capability_type == CapabilityType.NATIVE:
            if self.native_confidence < 0.0:
                self.native_confidence = 0.0
            elif self.native_confidence > 1.0:
                self.native_confidence = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "capability_type": self.capability_type.value,
            "native_confidence": self.native_confidence,
            "requires_model": self.requires_model,
            "fallback_available": self.fallback_available,
            "requires_approval": self.requires_approval,
            "limitations": self.limitations,
            "metadata": self.metadata,
        }


class CapabilityRegistry:
    """Deterministic registry of EVORA's intelligence capabilities.

    Honest about what EVORA can do natively vs. what requires models.
    No capability claims native support merely because an external model can perform it.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self._capabilities: dict[str, IntelligenceCapability] = {}
        self._logger = logger
        self._register_default_capabilities()

    def _register_default_capabilities(self) -> None:
        """Register default capability classifications."""
        # Native capabilities
        self.register(IntelligenceCapability(
            name="simple_reasoning",
            description="Basic reasoning over observations, constraints, and known facts",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
            fallback_available=True,
            limitations=["Limited to known patterns and facts", "No complex creative reasoning"],
        ))
        self.register(IntelligenceCapability(
            name="knowledge_retrieval",
            description="Retrieve and apply stored knowledge and lessons",
            capability_type=CapabilityType.NATIVE,
            native_confidence=1.0,
            fallback_available=False,
            limitations=["Depends on quality of stored knowledge"],
        ))
        self.register(IntelligenceCapability(
            name="tool_suggestion",
            description="Suggest relevant tools based on goal analysis",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.9,
            fallback_available=True,
            limitations=["Based on tool descriptions, not semantic understanding"],
        ))
        self.register(IntelligenceCapability(
            name="known_fact_inference",
            description="Infer conclusions from known facts and rules",
            capability_type=CapabilityType.NATIVE,
            native_confidence=1.0,
            fallback_available=False,
            limitations=["Only as good as the knowledge base"],
        ))
        self.register(IntelligenceCapability(
            name="planning_known_patterns",
            description="Plan tasks using known patterns and past experiences",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.8,
            fallback_available=True,
            limitations=["Works best with familiar task types", "Novel tasks may need external model"],
        ))
        self.register(IntelligenceCapability(
            name="simple_code_reasoning",
            description="Basic code structure analysis and pattern recognition",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.5,
            fallback_available=True,
            limitations=["No deep semantic understanding", "Limited to syntactic patterns"],
        ))
        self.register(IntelligenceCapability(
            name="python_code_understanding",
            description="Parse Python source files using AST for functions, classes, imports",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.8,
            fallback_available=False,
            limitations=["Python only", "No runtime analysis"],
        ))
        self.register(IntelligenceCapability(
            name="bug_detection",
            description="Detect common bugs using AST pattern matching",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.6,
            fallback_available=True,
            limitations=["Python only", "Limited to known patterns"],
        ))
        self.register(IntelligenceCapability(
            name="simple_code_generation",
            description="Generate simple code stubs from specifications",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.5,
            fallback_available=True,
            limitations=["Simple patterns only", "No complex algorithms"],
        ))
        self.register(IntelligenceCapability(
            name="code_explanation",
            description="Explain code structure and complexity",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
            fallback_available=True,
            limitations=["Structural only", "No semantic understanding"],
        ))
        self.register(IntelligenceCapability(
            name="test_generation",
            description="Generate basic test stubs from code structure",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.5,
            fallback_available=True,
            limitations=["Basic tests only", "No edge case generation"],
        ))
        self.register(IntelligenceCapability(
            name="patch_evaluation",
            description="Evaluate code patches for safety and affected scope",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
            fallback_available=True,
            limitations=["Structural analysis only", "No semantic correctness check"],
        ))
        self.register(IntelligenceCapability(
            name="intent_classification",
            description="Classify user intent from natural language",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
            fallback_available=True,
            limitations=["Keyword-based", "No deep semantic understanding"],
        ))
        self.register(IntelligenceCapability(
            name="entity_extraction",
            description="Extract entities (files, functions, paths) from text",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.6,
            fallback_available=True,
            limitations=["Regex-based", "Limited entity types"],
        ))
        self.register(IntelligenceCapability(
            name="context_building",
            description="Build conversation and project context",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
            fallback_available=True,
            limitations=["Depends on memory quality"],
        ))
        self.register(IntelligenceCapability(
            name="ambiguity_detection",
            description="Detect ambiguity in user inputs",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.6,
            fallback_available=True,
            limitations=["Heuristic-based"],
        ))

        # Model-enhanced capabilities
        self.register(IntelligenceCapability(
            name="complex_reasoning",
            description="Complex multi-step reasoning requiring language model inference",
            capability_type=CapabilityType.EXTERNAL_MODEL,
            requires_model=True,
            fallback_available=False,
            limitations=["Requires external LLM provider"],
        ))
        self.register(IntelligenceCapability(
            name="complex_code_generation",
            description="Generate complex code requiring deep semantic understanding",
            capability_type=CapabilityType.EXTERNAL_MODEL,
            requires_model=True,
            fallback_available=False,
            limitations=["Requires external LLM provider"],
        ))
        self.register(IntelligenceCapability(
            name="novel_planning",
            description="Plan entirely novel tasks without known patterns",
            capability_type=CapabilityType.EXTERNAL_MODEL,
            requires_model=True,
            fallback_available=False,
            limitations=["Requires external LLM provider"],
        ))

        # Unavailable capabilities
        self.register(IntelligenceCapability(
            name="unknown_capability",
            description="Capability not yet defined or available",
            capability_type=CapabilityType.UNAVAILABLE,
            native_confidence=0.0,
            fallback_available=False,
            limitations=["Not yet implemented"],
        ))

    def register(self, capability: IntelligenceCapability) -> None:
        """Register a new capability."""
        self._capabilities[capability.name] = capability
        if self._logger:
            self._logger.observe(
                f"Registered capability: {capability.name} ({capability.capability_type.value})"
            )

    def get(self, name: str) -> Optional[IntelligenceCapability]:
        """Get a capability by name."""
        return self._capabilities.get(name)

    def can_handle(self, task_type: str) -> IntelligenceCapability:
        """Check if EVORA can handle a task type.

        Returns the capability classification.
        If not found, returns UNAVAILABLE.
        """
        capability = self._capabilities.get(task_type)
        if capability is not None:
            return capability
        return IntelligenceCapability(
            name=task_type,
            description=f"Unknown capability: {task_type}",
            capability_type=CapabilityType.UNAVAILABLE,
            native_confidence=0.0,
            requires_model=False,
            fallback_available=False,
            limitations=["Not registered"],
        )

    def get_native_capabilities(self) -> list[IntelligenceCapability]:
        """Get all capabilities that can be executed natively."""
        return [c for c in self._capabilities.values() if c.capability_type == CapabilityType.NATIVE]

    def get_model_enhanced_capabilities(self) -> list[IntelligenceCapability]:
        """Get capabilities that can be enhanced by external models."""
        return [c for c in self._capabilities.values() if c.capability_type in
                (CapabilityType.LOCAL_MODEL, CapabilityType.EXTERNAL_MODEL)]

    def get_unavailable_capabilities(self) -> list[IntelligenceCapability]:
        """Get capabilities that are currently unavailable."""
        return [c for c in self._capabilities.values() if c.capability_type == CapabilityType.UNAVAILABLE]

    def get_capabilities_requiring_approval(self) -> list[IntelligenceCapability]:
        """Get capabilities that require creator approval."""
        return [c for c in self._capabilities.values() if c.requires_approval]

    def list_all(self) -> list[str]:
        """List all registered capability names."""
        return list(self._capabilities.keys())

    def summary(self) -> dict[str, Any]:
        """Return a summary of capabilities."""
        by_type: dict[str, int] = {}
        for c in self._capabilities.values():
            by_type[c.capability_type.value] = by_type.get(c.capability_type.value, 0) + 1
        return {
            "total": len(self._capabilities),
            "by_type": by_type,
            "native": [c.name for c in self.get_native_capabilities()],
            "model_enhanced": [c.name for c in self.get_model_enhanced_capabilities()],
            "unavailable": [c.name for c in self.get_unavailable_capabilities()],
        }
