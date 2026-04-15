from typing import Any, Callable

class Registry:
    _store = {}
    
    @classmethod
    def register(cls, name: str, obj: Any = None) -> Callable:
        """注册组件的装饰器
        
        可以作为装饰器使用：
        @Registry.register('name')
        class MyClass:
            pass
        
        也可以作为方法调用：
        Registry.register('name', MyClass)
        """
        if obj is not None:
            # 作为方法调用
            cls._store[name] = obj
            return obj
        else:
            # 作为装饰器使用
            def decorator(cls_obj: Any) -> Any:
                cls._store[name] = cls_obj
                return cls_obj
            return decorator
    
    @classmethod
    def get(cls, name: str) -> Any:
        return cls._store.get(name)

