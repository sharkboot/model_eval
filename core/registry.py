from typing import Any, Callable, Dict, List, Type


class Registry:
    _registry: Dict[str, Dict[str, Type]] = {}

    @classmethod
    def register(cls, name: str, group: str):
        def wrapper(obj):
            cls._registry.setdefault(group, {})[name] = obj
            return obj
        return wrapper

    @classmethod
    def get(cls, name: str, group: str):
        return cls._registry[group][name]

    @classmethod
    def create(cls, name: str, group: str, **kwargs):
        return cls.get(name, group)(kwargs)

    @classmethod
    def list_registered(cls, group: str) -> List[str]:
        """List all registered names in a group."""
        return list(cls._registry.get(group, {}).keys())
