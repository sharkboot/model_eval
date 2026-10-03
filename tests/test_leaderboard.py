"""Tests for core.leaderboard.Leaderboard."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.leaderboard import Leaderboard


def test_add_and_summary():
    lb = Leaderboard()
    lb.add("task_a", {"accuracy": 0.85})
    lb.add("task_b", {"f1": 0.92})
    assert lb.summary() == {"task_a": {"accuracy": 0.85}, "task_b": {"f1": 0.92}}


def test_pretty_print_numeric():
    lb = Leaderboard()
    lb.add("task", {"accuracy": 0.8567, "f1": 0.9234})
    # Should not raise
    lb.pretty_print()


def test_pretty_print_mixed_types():
    """Issue #6 regression: non-numeric values should not crash."""
    lb = Leaderboard()
    lb.add("task", {"accuracy": 0.85, "note": "manual review needed"})
    # Should not raise TypeError
    lb.pretty_print()


def test_pretty_print_string_only():
    lb = Leaderboard()
    lb.add("task", {"message": "all good"})
    lb.pretty_print()


def test_empty_leaderboard():
    lb = Leaderboard()
    assert lb.summary() == {}
    lb.pretty_print()  # should not raise


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
