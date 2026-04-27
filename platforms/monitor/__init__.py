"""Evaluation monitoring dashboard - a lightweight WandB alternative.

Usage:
    from platforms.monitor import launch

    # Use with demo evaluator
    launch()

    # Or with your own evaluator
    def my_eval():
        from platforms.monitor import put_event
        for i in range(100):
            put_event({"type": "metric", "data": {"accuracy": i / 100}})
            time.sleep(1)

    launch(evaluator_fn=my_eval, title="My Eval")
"""
from .events import get_state, put_event, start_consumer
from .state import MonitorState
from .ui import build_app, launch

__all__ = ["MonitorState", "get_state", "put_event", "start_consumer", "build_app", "launch"]
