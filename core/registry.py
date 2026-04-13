from typing import Any

class Registry:
    _store = {}
    
    @classmethod
    def register(cls, name: str, obj: Any):
        cls._store[name] = obj
        
    @classmethod
    def get(cls, name: str) -> Any:
        return cls._store.get(name)

# 具体的注册中心实例
DatasetRegistry = Registry()
ModelRegistry = Registry()
EvaluatorRegistry = Registry()
