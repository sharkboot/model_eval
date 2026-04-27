"""Event queue and consumer for evaluation monitoring."""
import queue
import threading
from typing import Callable

from .state import MonitorState

_event_queue: queue.Queue = queue.Queue()
_state = MonitorState()
_consumer_thread: threading.Thread | None = None


def get_state() -> MonitorState:
    """Get the global MonitorState instance."""
    return _state


def put_event(event: dict) -> None:
    """Add an event to the processing queue."""
    _event_queue.put(event)


def _default_consumer() -> None:
    """Default event consumer loop - processes events from queue."""
    while True:
        try:
            event = _event_queue.get(timeout=0.1)
            _state.update(event)
        except queue.Empty:
            continue


def start_consumer(consumer_fn: Callable[[], None] | None = None) -> threading.Thread:
    """Start the event consumer in a daemon thread.

    Args:
        consumer_fn: Custom consumer function. If None, uses default.

    Returns:
        The consumer thread.
    """
    global _consumer_thread
    if _consumer_thread and _consumer_thread.is_alive():
        return _consumer_thread

    fn = consumer_fn or _default_consumer
    _consumer_thread = threading.Thread(target=fn, daemon=True)
    _consumer_thread.start()
    return _consumer_thread


def stop_consumer() -> None:
    """Stop the consumer thread gracefully."""
    global _consumer_thread
    if _consumer_thread:
        _consumer_thread = None
