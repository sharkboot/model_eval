"""Tests for core.data_filter.DataFilter."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_filter import DataFilter
from core.base import DataItem


def _item(i, cats=None, diff=None):
    return DataItem(
        id=f"item_{i}",
        prompt=f"question {i}",
        reference=f"answer {i}",
        category=cats or [],
        difficulty=diff,
    )


# ── categories_include ──────────────────────────────────────────────
def test_include_single():
    f = DataFilter(categories_include=["math"])
    items = [_item(1, ["math"]), _item(2, ["physics"]), _item(3, ["math", "chem"])]
    assert [it.id for it in f.apply(items)] == ["item_1", "item_3"]


def test_include_multiple_all_match():
    f = DataFilter(categories_include=["math", "physics"])
    items = [_item(1, ["math"]), _item(2, ["physics"]), _item(3, ["chem"])]
    assert [it.id for it in f.apply(items)] == ["item_1", "item_2"]


def test_include_none_match():
    f = DataFilter(categories_include=["astro"])
    items = [_item(1, ["math"]), _item(2, ["physics"])]
    assert f.apply(items) == []


# ── categories_exclude ──────────────────────────────────────────────
def test_exclude():
    f = DataFilter(categories_exclude=["physics"])
    items = [_item(1, ["math"]), _item(2, ["physics"]), _item(3, ["math", "physics"])]
    assert [it.id for it in f.apply(items)] == ["item_1"]


def test_exclude_none():
    f = DataFilter(categories_exclude=["astro"])
    items = [_item(1, ["math"]), _item(2, ["physics"])]
    assert len(f.apply(items)) == 2


# ── custom_filter ───────────────────────────────────────────────────
def test_custom_filter():
    f = DataFilter(custom_filter=lambda it: it.difficulty == "hard")
    items = [_item(1, diff="easy"), _item(2, diff="hard"), _item(3, diff="hard")]
    assert [it.id for it in f.apply(items)] == ["item_2", "item_3"]


# ── combined ────────────────────────────────────────────────────────
def test_include_and_exclude():
    f = DataFilter(categories_include=["math"], categories_exclude=["chem"])
    items = [
        _item(1, ["math"]),       # keep
        _item(2, ["math", "chem"]),  # excluded
        _item(3, ["physics"]),    # not included
    ]
    assert [it.id for it in f.apply(items)] == ["item_1"]


def test_include_exclude_and_custom():
    f = DataFilter(
        categories_include=["math"],
        custom_filter=lambda it: int(it.id.split("_")[1]) <= 2,
    )
    items = [_item(1, ["math"]), _item(2, ["math"]), _item(3, ["math"])]
    assert [it.id for it in f.apply(items)] == ["item_1", "item_2"]


# ── edge cases ──────────────────────────────────────────────────────
def test_empty_filter_no_change():
    f = DataFilter()
    items = [_item(1), _item(2)]
    assert [it.id for it in f.apply(items)] == ["item_1", "item_2"]


def test_empty_input():
    f = DataFilter(categories_include=["math"])
    assert f.apply([]) == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
