"""
Phase 39 — Tests for Native Streaming Data Processor.

Tests real-time stream processing capabilities.
"""

import unittest
from evora.brain.intelligence.streaming_processor import (
    NativeStreamingProcessor,
    StreamEvent,
    StreamWindow,
    WindowType,
    ProcessorStatus,
)


class TestNativeStreamingProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = NativeStreamingProcessor()

    def test_register_stream(self):
        result = self.processor.register_stream("test_stream")
        self.assertEqual(result, "test_stream")

    def test_create_event(self):
        event = self.processor.create_event("source1", {"key": "value"})
        self.assertEqual(event.source, "source1")
        self.assertEqual(event.data, {"key": "value"})

    def test_process_event(self):
        self.processor.register_stream("test_stream")
        event = StreamEvent(source="test", data={"value": 10})
        result = self.processor.process_event("test_stream", event)
        self.assertIn("status", result)

    def test_process_event_new_stream(self):
        event = StreamEvent(source="test", data={"value": 10})
        result = self.processor.process_event("new_stream", event)
        self.assertIn("status", result)

    def test_stream_window_aggregate(self):
        window = StreamWindow(events=[
            StreamEvent(data={"value": 10}),
            StreamEvent(data={"value": 20}),
            StreamEvent(data={"value": 30}),
        ])
        self.assertEqual(window.aggregate("value", "count"), 3)
        self.assertEqual(window.aggregate("value", "sum"), 60)
        self.assertEqual(window.aggregate("value", "avg"), 20)
        self.assertEqual(window.aggregate("value", "min"), 10)
        self.assertEqual(window.aggregate("value", "max"), 30)

    def test_stream_window_add_event_after_close(self):
        window = StreamWindow(closed=True)
        event = StreamEvent(data={"value": 10})
        window.add_event(event)
        self.assertEqual(len(window.events), 0)

    def test_filter_events(self):
        self.processor.register_stream("test")
        events = [StreamEvent(data={"value": i}) for i in range(5)]
        for e in events:
            self.processor.process_event("test", e)
        filtered = self.processor.filter_events("test", lambda e: e.data["value"] > 2)
        self.assertEqual(len(filtered), 2)

    def test_filter_events_empty_stream(self):
        filtered = self.processor.filter_events("nonexistent", lambda e: True)
        self.assertEqual(len(filtered), 0)

    def test_transform_events(self):
        self.processor.register_stream("test")
        event = StreamEvent(data={"value": 1})
        self.processor.process_event("test", event)
        transformed = self.processor.transform_events("test", lambda e: StreamEvent(data={"doubled": e.data["value"] * 2}))
        self.assertEqual(transformed[0].data["doubled"], 2)

    def test_aggregate_stream(self):
        self.processor.register_stream("test")
        for val in [10, 20, 30, 40]:
            self.processor.process_event("test", StreamEvent(data={"value": val}))
        result = self.processor.aggregate_stream("test", "value", ["count", "sum", "avg", "min", "max"])
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["sum"], 100)
        self.assertEqual(result["avg"], 25)
        self.assertEqual(result["min"], 10)
        self.assertEqual(result["max"], 40)

    def test_get_stream_events(self):
        self.processor.register_stream("test")
        event = StreamEvent(data={"value": 10})
        self.processor.process_event("test", event)
        events = self.processor.get_stream_events("test")
        self.assertEqual(len(events), 1)

    def test_get_stream_events_invalid(self):
        events = self.processor.get_stream_events("nonexistent")
        self.assertEqual(len(events), 0)

    def test_clear_stream(self):
        self.processor.register_stream("test")
        self.processor.process_event("test", StreamEvent(data={"value": 1}))
        self.processor.clear_stream("test")
        self.assertEqual(len(self.processor.get_stream_events("test")), 0)

    def test_get_metrics(self):
        self.processor.register_stream("stream1")
        self.processor.register_stream("stream2")
        self.processor.process_event("stream1", StreamEvent(data={"value": 1}))
        metrics = self.processor.get_metrics()
        self.assertEqual(metrics["total_streams"], 2)
        self.assertEqual(metrics["total_events"], 1)
        self.assertIn("stream1", metrics["stream_names"])

    def test_get_event_count(self):
        self.processor.register_stream("test")
        self.processor.process_event("test", StreamEvent(data={"value": 1}))
        self.processor.process_event("test", StreamEvent(data={"value": 2}))
        self.assertEqual(self.processor.get_event_count("test"), 2)

    def test_get_event_count_invalid(self):
        self.assertEqual(self.processor.get_event_count("nonexistent"), 0)

    def test_stream_event_to_dict(self):
        event = StreamEvent(source="test", data={"key": "value"})
        result = event.to_dict()
        self.assertEqual(result["source"], "test")
        self.assertEqual(result["data"], {"key": "value"})

    def test_stream_window_to_dict(self):
        window = StreamWindow(window_type=WindowType.TUMBLING)
        result = window.to_dict()
        self.assertEqual(result["window_type"], WindowType.TUMBLING.value)
        self.assertFalse(result["closed"])


if __name__ == "__main__":
    unittest.main()
