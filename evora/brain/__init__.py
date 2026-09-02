"""
Phase 9 — EVORA Brain package.

Provides the standalone Brain architecture:
  - BrainController: central orchestration (thin layer)
  - BrainState: persistent internal state
  - ContextBuilder: bounded context construction
  - SelfModel: observable-based self-understanding
  - ResourceMonitor: safe resource awareness
"""

from evora.brain.brain import BrainController, BrainResponse
from evora.brain.context import ContextBuilder, BrainContext
from evora.brain.self_model import SelfModel, Capabilities, Limitations
from evora.brain.state import BrainState, ResourceState, DevelopmentState, SystemStatus
from evora.brain.resources import ResourceMonitor, ResourceInfo

__all__ = [
    "BrainController",
    "BrainResponse",
    "ContextBuilder",
    "BrainContext",
    "SelfModel",
    "Capabilities",
    "Limitations",
    "BrainState",
    "ResourceState",
    "DevelopmentState",
    "SystemStatus",
    "ResourceMonitor",
    "ResourceInfo",
]
