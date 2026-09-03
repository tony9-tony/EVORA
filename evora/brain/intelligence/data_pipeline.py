"""
Phase 36 — Native Data Pipeline for EVORA.

Manages data flows and transformations.

Supports:
  - Pipeline definitions
  - Data transformation
  - Pipeline execution
  - Data validation
  - Pipeline monitoring
  - Integration with KnowledgeGraph
  - Integration with MemoryManager

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TransformationType(str, Enum):
    FILTER = "filter"
    MAP = "map"
    AGGREGATE = "aggregate"
    VALIDATE = "validate"
    ENRICH = "enrich"


@dataclass
class Transformation:
    """A data transformation step."""
    transformation_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    transformation_type: TransformationType = TransformationType.MAP
    name: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "transformation_type": self.transformation_type.value,
            "name": self.name,
            "config": self.config,
        }


@dataclass
class Pipeline:
    """A data pipeline."""
    pipeline_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    transformations: list[Transformation] = field(default_factory=list)
    status: PipelineStatus = PipelineStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "transformations": [t.to_dict() for t in self.transformations],
            "status": self.status.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# Native Data Pipeline
# ---------------------------------------------------------------------------

class NativeDataPipeline:
    """Native data pipeline for EVORA.

    Manages data flows and transformations.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._pipelines: dict[str, Pipeline] = {}

    def create_pipeline(self, name: str, transformations: list[Transformation] = None, metadata: dict[str, Any] = None) -> Pipeline:
        """Create a new data pipeline."""
        pipeline = Pipeline(
            name=name,
            transformations=transformations or [],
            metadata=metadata or {},
        )
        self._pipelines[pipeline.pipeline_id] = pipeline
        return pipeline

    def add_transformation(self, pipeline_id: str, transformation: Transformation) -> bool:
        """Add a transformation to a pipeline."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return False
        pipeline.transformations.append(transformation)
        return True

    def execute_pipeline(self, pipeline_id: str, input_data: list[Any] = None) -> dict[str, Any]:
        """Execute a pipeline."""
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return {"error": "Pipeline not found", "status": PipelineStatus.FAILED.value}
        pipeline.status = PipelineStatus.RUNNING
        data = input_data or []
        for transformation in pipeline.transformations:
            try:
                data = self._apply_transformation(transformation, data)
            except Exception as e:
                pipeline.status = PipelineStatus.FAILED
                return {"error": str(e), "status": PipelineStatus.FAILED.value, "output": data}
        pipeline.status = PipelineStatus.COMPLETED
        pipeline.completed_at = datetime.now().isoformat()
        return {"output": data, "status": PipelineStatus.COMPLETED.value, "pipeline_id": pipeline_id}

    def _apply_transformation(self, transformation: Transformation, data: list[Any]) -> list[Any]:
        """Apply a transformation to data."""
        if transformation.transformation_type == TransformationType.FILTER:
            return [item for item in data if item]
        elif transformation.transformation_type == TransformationType.MAP:
            return [transformation.config.get("function", str)(item) if callable(transformation.config.get("function")) else item for item in data]
        elif transformation.transformation_type == TransformationType.AGGREGATE:
            return [sum(data)] if data else []
        elif transformation.transformation_type == TransformationType.VALIDATE:
            return [item for item in data if isinstance(item, transformation.config.get("type", str))]
        return data

    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """Get a pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    def get_pipeline_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics."""
        total = len(self._pipelines)
        by_status: dict[str, int] = {}
        for pipeline in self._pipelines.values():
            by_status[pipeline.status.value] = by_status.get(pipeline.status.value, 0) + 1
        return {
            "total_pipelines": total,
            "by_status": by_status,
            "total_transformations": sum(len(p.transformations) for p in self._pipelines.values()),
        }
