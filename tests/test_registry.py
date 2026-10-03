"""Tests for core.registry.Registry."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import Registry


def setup_function():
    """Clear registry between tests."""
    Registry._registry.clear()


def test_register_and_get():
    @Registry.register("MyClass", "group_a")
    class MyClass:
        pass

    cls = Registry.get("MyClass", "group_a")
    assert cls is MyClass


def test_register_returns_original():
    @Registry.register("Func", "group_b")
    def func():
        return 42

    assert func() == 42


def test_list_registered():
    @Registry.register("A", "group_c")
    class A:
        pass

    @Registry.register("B", "group_c")
    class B:
        pass

    names = Registry.list_registered("group_c")
    assert sorted(names) == ["A", "B"]


def test_list_registered_empty():
    assert Registry.list_registered("nonexistent") == []


def test_create():
    """Registry.create passes config dict as first arg — classes must accept dict."""
    @Registry.register("Creator", "group_d")
    class Creator:
        def __init__(self, config):
            self.x = config.get("x")

    obj = Registry.create("Creator", "group_d", x=10)
    assert obj.x == 10


def test_duplicate_registration():
    @Registry.register("Dup", "group_e")
    class Dup1:
        pass

    @Registry.register("Dup", "group_e")
    class Dup2:
        pass

    # Later registration overwrites earlier one
    assert Registry.get("Dup", "group_e") is Dup2


def test_get_missing_raises():
    try:
        Registry.get("Missing", "group_f")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
