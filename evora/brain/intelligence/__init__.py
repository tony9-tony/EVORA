"""
Phase 10 — Native Intelligence package.

Provides EVORA's own cognitive capabilities independent of external models.
"""

from evora.brain.intelligence.knowledge import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeType,
    RelationType,
)
from evora.brain.intelligence.capabilities import (
    CapabilityRegistry,
    IntelligenceCapability,
    CapabilityType,
)
from evora.brain.intelligence.evaluation import (
    IntelligenceEvaluator,
    EvaluationResult,
    EvaluationGrade,
)
from evora.brain.intelligence.reasoning import (
    NativeReasoning,
    ReasoningFacts,
    ReasoningResult,
)
from evora.brain.intelligence.planner import (
    NativePlanner,
    NativePlan,
    PlanStep,
)
from evora.brain.intelligence.inference import (
    InferenceEngine,
    InferenceRule,
    InferenceResult,
)
from evora.brain.intelligence.runtime import IntelligenceRuntime
from evora.brain.intelligence.provider import NativeIntelligenceProvider
from evora.brain.intelligence.training import (
    TrainingPipeline,
    ConfidenceCalibrator,
    ContradictionDetector,
    NativeIntelligenceMetrics,
    TrainingExample,
    OutcomeType,
    TrainingExampleStatus,
    ProvenanceTracker,
    ProvenanceRecord,
)

__all__ = [
    "KnowledgeGraph",
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeType",
    "RelationType",
    "CapabilityRegistry",
    "IntelligenceCapability",
    "CapabilityType",
    "IntelligenceEvaluator",
    "EvaluationResult",
    "EvaluationGrade",
    "NativeReasoning",
    "ReasoningFacts",
    "ReasoningResult",
    "NativePlanner",
    "NativePlan",
    "PlanStep",
    "InferenceEngine",
    "InferenceRule",
    "InferenceResult",
    "IntelligenceRuntime",
    "NativeIntelligenceProvider",
    "TrainingPipeline",
    "ConfidenceCalibrator",
    "ContradictionDetector",
    "NativeIntelligenceMetrics",
    "TrainingExample",
    "OutcomeType",
    "TrainingExampleStatus",
    "ProvenanceTracker",
    "ProvenanceRecord",
]
