"""Thread-safe state management for evaluation monitoring."""
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MonitorState:
    """Thread-safe state container for metrics, logs, and results."""

    metrics: dict = field(default_factory=dict)
    history: dict = field(lambda: defaultdict(list))
    logs: list = field(default_factory=list)
    results: list = field(default_factory=list)
    current: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, event: dict) -> None:
        """Process a single event and update state accordingly."""
        with self._lock:
            t = event.get("type")

            if t == "metric":
                for k, v in event.get("data", {}).items():
                    self.metrics[k] = v
                    self.history[k].append(v)

            elif t == "log":
                msg = event.get("message", "")
                self.logs.append(msg)
                self.logs = self.logs[-200:]

            elif t == "result":
                self.results.append(event.get("data"))

            elif t == "stage":
                self.current = event.get("data", {})

    def snapshot(self) -> tuple[dict, dict, list, list, dict]:
        """Return a consistent snapshot of all state (thread-safe)."""
        with self._lock:
            return (
                dict(self.metrics),
                dict(self.history),
                list(self.logs),
                list(self.results),
                dict(self.current),
            )

    def clear(self) -> None:
        """Reset all state."""
        with self._lock:
            self.metrics.clear()
            self.history.clear()
            self.logs.clear()
            self.results.clear()
            self.current.clear()
