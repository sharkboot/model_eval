from tasks.standard_runner import StandardTaskRunner
from core.registry import Registry

Registry.register("standard", "tasks", allow_override=True)

__all__ = ["StandardTaskRunner"]
