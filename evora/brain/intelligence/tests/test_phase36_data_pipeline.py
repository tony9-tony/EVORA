"""
Phase 36 — Tests for Native Data Pipeline.

Tests data flows and transformations.
"""

import unittest
from evora.brain.intelligence.data_pipeline import (
    NativeDataPipeline,
    Pipeline,
    Transformation,
    PipelineStatus,
    TransformationType,
)


class TestNativeDataPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = NativeDataPipeline()

    def test_create_pipeline(self):
        pipeline = self.pipeline.create_pipeline(name="test_pipeline")
        self.assertEqual(pipeline.name, "test_pipeline")
        self.assertEqual(pipeline.status, PipelineStatus.PENDING)

    def test_add_transformation(self):
        pipeline = self.pipeline.create_pipeline(name="test_pipeline")
        transformation = Transformation(name="test_transform")
        result = self.pipeline.add_transformation(pipeline.pipeline_id, transformation)
        self.assertTrue(result)
        self.assertEqual(len(pipeline.transformations), 1)

    def test_add_transformation_invalid_pipeline(self):
        transformation = Transformation(name="test_transform")
        result = self.pipeline.add_transformation("nonexistent", transformation)
        self.assertFalse(result)

    def test_execute_pipeline_filter(self):
        pipeline = self.pipeline.create_pipeline(name="filter_test")
        self.pipeline.add_transformation(
            pipeline.pipeline_id,
            Transformation(transformation_type=TransformationType.FILTER),
        )
        result = self.pipeline.execute_pipeline(pipeline.pipeline_id, [1, 0, 2, None, 3])
        self.assertEqual(result["output"], [1, 2, 3])
        self.assertEqual(result["status"], PipelineStatus.COMPLETED.value)

    def test_execute_pipeline_map(self):
        pipeline = self.pipeline.create_pipeline(name="map_test")
        self.pipeline.add_transformation(
            pipeline.pipeline_id,
            Transformation(
                transformation_type=TransformationType.MAP,
                config={"function": lambda x: x * 2},
            ),
        )
        result = self.pipeline.execute_pipeline(pipeline.pipeline_id, [1, 2, 3])
        self.assertEqual(result["output"], [2, 4, 6])

    def test_execute_pipeline_aggregate(self):
        pipeline = self.pipeline.create_pipeline(name="aggregate_test")
        self.pipeline.add_transformation(
            pipeline.pipeline_id,
            Transformation(transformation_type=TransformationType.AGGREGATE),
        )
        result = self.pipeline.execute_pipeline(pipeline.pipeline_id, [1, 2, 3, 4])
        self.assertEqual(result["output"], [10])

    def test_execute_pipeline_invalid(self):
        result = self.pipeline.execute_pipeline("nonexistent", [])
        self.assertEqual(result["status"], PipelineStatus.FAILED.value)

    def test_get_pipeline(self):
        pipeline = self.pipeline.create_pipeline(name="test_pipeline")
        retrieved = self.pipeline.get_pipeline(pipeline.pipeline_id)
        self.assertEqual(retrieved, pipeline)

    def test_get_pipeline_invalid(self):
        retrieved = self.pipeline.get_pipeline("nonexistent")
        self.assertIsNone(retrieved)

    def test_get_pipeline_metrics(self):
        for _ in range(5):
            self.pipeline.create_pipeline(name="test_pipeline")
        metrics = self.pipeline.get_pipeline_metrics()
        self.assertEqual(metrics["total_pipelines"], 5)
        self.assertEqual(metrics["by_status"][PipelineStatus.PENDING.value], 5)

    def test_to_dict(self):
        pipeline = Pipeline(name="test")
        pipeline_dict = pipeline.to_dict()
        self.assertEqual(pipeline_dict["name"], "test")
        self.assertEqual(pipeline_dict["status"], PipelineStatus.PENDING.value)

    def test_transformation_to_dict(self):
        transformation = Transformation(
            name="test",
            transformation_type=TransformationType.MAP,
            config={"key": "value"},
        )
        result = transformation.to_dict()
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["transformation_type"], TransformationType.MAP.value)
        self.assertEqual(result["config"], {"key": "value"})


if __name__ == "__main__":
    unittest.main()
