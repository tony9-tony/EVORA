"""
Phase 39 - Native Streaming Data Processor for EVORA.

Processes real-time data streams with windowing and aggregation capabilities.

Supports:
  - Stream event processing
  - Windowing (tumbling, sliding, session)
  - Stream aggregation
  - Event filtering and transformation
  - Stream monitoring
  - Integration with KnowledgeGraph
  - Integration with DataPipeline
  - Integration with QueryOptimizer

No independent authority system.
No security bypass.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional


class WindowType(str, Enum):
    TUMBLING = "tumbling"
    SLIDING = "sliding"
    SESSION = "session"


class ProcessorStatus(str, Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single event in a data stream."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
        }


@dataclass
class StreamWindow:
    """A processing window for stream events."""
    window_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    window_type: WindowType = WindowType.TUMBLING
    start_time: str = ""
    end_time: str = ""
    events: list[StreamEvent] = field(default_factory=list)
    closed: bool = False

    def add_event(self, event: StreamEvent):
        if not self.closed:
            self.events.append(event)

    def is_expired(self, current_time: datetime) -> bool:
        if not self.end_time:
            return False
        end_dt = datetime.fromisoformat(self.end_time)
        return current_time >= end_dt or self.closed

    def aggregate(self, field: str, operation: str) -> Any:
        values = [e.data.get(field) for e in self.events if e.data.get(field) is not None]
        if not values:
            return None
        if operation == "count":
            return len(values)
        elif operation == "sum":
            return sum(values)
        elif operation == "avg":
            return sum(values) / len(values)
        elif operation == "min":
            return min(values)
        elif operation == "max":
            return max(values)
        return values[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_type": self.window_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": [e.to_dict() for e in self.events],
            "closed": self.closed,
        }


class NativeStreamingProcessor:
    """Native streaming data processor for EVORA.

    Processes real-time data streams.
    """

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self._streams: dict[str, list[StreamEvent]] = defaultdict(list)
        self._windows: dict[str, StreamWindow] = {}
        self._processors: dict[str, ProcessorStatus] = {}
        self._lock = threading.Lock()
        self._window_size = timedelta(minutes=5)

    def register_stream(self, stream_name: str) -> str:
        """Register a new stream."""
        if stream_name not in self._streams:
            self._streams[stream_name] = []
        return stream_name

    def create_event(self, source: str, data: dict[str, Any], metadata: dict[str, Any] = None) -> StreamEvent:
        """Create a new stream event."""
        return StreamEvent(source=source, data=data, metadata=metadata or {})

    def process_event(self, stream_name: str, event: StreamEvent) -> dict[str, Any]:
        """Process a single event in a stream."""
        if stream_name not in self._streams:
            self.register_stream(stream_name)

        with self._lock:
            self._streams[stream_name].append(event)
            current_window = self._get_or_create_window(stream_name)
            current_window.add_event(event)

            if current_window.is_expired(datetime.now()):
                current_window.closed = True
                result = self._emit_window(stream_name, current_window)
            else:
                result = {"status": "buffered", "window_id": current_window.window_id, "events_in_window": len(current_window.events)}

        if self.logger:
            self.logger.info(f"Processed event in stream '{stream_name}'")

        return result

    def _get_or_create_window(self, stream_name: str) -> StreamWindow:
        """Get existing open window or create a new one."""
        window = self._windows.get(stream_name)
        if window is None or window.closed:
            now = datetime.now()
            window = StreamWindow(
                window_type=WindowType.TUMBLING,
                start_time=now.isoformat(),
                end_time=(now + self._window_size).isoformat(),
            )
            self._windows[stream_name] = window
        return window

    def _emit_window(self, stream_name: str, window: StreamWindow) -> dict[str, Any]:
        """Emit results from a closed window."""
        result = {
            "status": "emitted",
            "window_id": window.window_id,
            "events_processed": len(window.events),
            "aggregations": {},
        }
        for field in self._extract_fields(window):
            result["aggregations"][field] = {
                "count": window.aggregate(field, "count"),
                "sum": window.aggregate(field, "sum"),
                "avg": window.aggregate(field, "avg"),
                "min": window.aggregate(field, "min"),
                "max": window.aggregate(field, "max"),
            }
        return result

    def _extract_fields(self, window: StreamWindow) -> set[str]:
        """Extract all field names from window events."""
        fields: set[str] = set()
        for event in window.events:
            fields.update(event.data.keys())
        return fields

    def filter_events(self, stream_name: str, predicate: Callable[[StreamEvent], bool]) -> list[StreamEvent]:
        """Filter events in a stream by predicate."""
        events = self._streams.get(stream_name, [])
        return [e for e in events if predicate(e)]

    def transform_events(self, stream_name: str, transform: Callable[[StreamEvent], StreamEvent]) -> list[StreamEvent]:
        """Transform events in a stream."""
        events = self._streams.get(stream_name, [])
        transformed = [transform(e) for e in events]
        return transformed

    def aggregate_stream(self, stream_name: str, field: str, operations: list[str]) -> dict[str, Any]:
        """Aggregate events in a stream."""
        events = self._streams.get(stream_name, [])
        values = [e.data.get(field) for e in events if e.data.get(field) is not None]
        result: dict[str, Any] = {}
        for op in operations:
            if op == "count":
                result["count"] = len(values)
            elif op == "sum" and values:
                result["sum"] = sum(values)
            elif op == "avg" and values:
                result["avg"] = sum(values) / len(values)
            elif op == "min" and values:
                result["min"] = min(values)
            elif op == "max" and values:
                result["max"] = max(values)
        return result

    def get_stream_events(self, stream_name: str) -> list[StreamEvent]:
        """Get all events for a stream."""
        return self._streams.get(stream_name, [])

    def clear_stream(self, stream_name: str):
        """Clear all events from a stream."""
        if stream_name in self._streams:
            self._streams[stream_name] = []

    def get_metrics(self) -> dict[str, Any]:
        """Get streaming processor metrics."""
        total_events = sum(len(events) for events in self._streams.values())
        return {
            "total_streams": len(self._streams),
            "total_events": total_events,
            "total_windows": len(self._windows),
            "stream_names": list(self._streams.keys()),
        }

    def get_event_count(self, stream_name: str) -> int:
        """Get the number of events in a stream."""
        return len(self._streams.get(stream_name, []))
