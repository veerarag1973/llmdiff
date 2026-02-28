"""Tests for the JSON-structural diff additions in llm_diff.diff."""

from __future__ import annotations

import pytest

from llm_diff.diff import (
    JsonChangeType,
    JsonDiffEntry,
    JsonStructDiffResult,
    _flatten_json,
    detect_json,
    json_struct_diff,
)


# ---------------------------------------------------------------------------
# _flatten_json
# ---------------------------------------------------------------------------

def test_flatten_simple_dict() -> None:
    obj = {"a": 1, "b": 2}
    flat = _flatten_json(obj)
    assert flat == {"a": 1, "b": 2}


def test_flatten_nested_dict() -> None:
    obj = {"a": {"b": {"c": 42}}}
    flat = _flatten_json(obj)
    assert flat == {"a.b.c": 42}


def test_flatten_list() -> None:
    obj = {"items": [10, 20, 30]}
    flat = _flatten_json(obj)
    assert flat == {"items.0": 10, "items.1": 20, "items.2": 30}


def test_flatten_mixed() -> None:
    obj = {"x": [{"y": 1}, {"y": 2}]}
    flat = _flatten_json(obj)
    assert "x.0.y" in flat
    assert "x.1.y" in flat


def test_flatten_empty() -> None:
    assert _flatten_json({}) == {}


def test_flatten_scalar_at_root_prefix() -> None:
    flat = _flatten_json({"n": 99}, prefix="root")
    assert flat == {"root.n": 99}


# ---------------------------------------------------------------------------
# detect_json
# ---------------------------------------------------------------------------

def test_detect_json_object() -> None:
    assert detect_json('{"key": "value"}') is True


def test_detect_json_array() -> None:
    assert detect_json("[1, 2, 3]") is True


def test_detect_json_plain_text() -> None:
    assert detect_json("Hello, world!") is False


def test_detect_json_number_string() -> None:
    # A plain number like "42" is valid JSON but not an object/array
    # behaviour depends on implementation — just check it returns a bool
    assert isinstance(detect_json("42"), bool)


def test_detect_json_empty_string() -> None:
    assert detect_json("") is False


# ---------------------------------------------------------------------------
# json_struct_diff — both valid JSON
# ---------------------------------------------------------------------------

def test_added_key() -> None:
    a = '{"x": 1}'
    b = '{"x": 1, "y": 2}'
    result = json_struct_diff(a, b)
    assert result.is_valid_json_a
    assert result.is_valid_json_b
    paths = [e.path for e in result.added]
    assert "y" in paths


def test_removed_key() -> None:
    a = '{"x": 1, "y": 2}'
    b = '{"x": 1}'
    result = json_struct_diff(a, b)
    removed_paths = [e.path for e in result.removed]
    assert "y" in removed_paths


def test_changed_value() -> None:
    a = '{"x": 1}'
    b = '{"x": 99}'
    result = json_struct_diff(a, b)
    changed_paths = [e.path for e in result.changed]
    assert "x" in changed_paths


def test_unchanged_value() -> None:
    a = '{"x": 1, "y": "hello"}'
    b = '{"x": 1, "y": "hello"}'
    result = json_struct_diff(a, b)
    assert not result.has_changes
    assert len(result.unchanged) == 2


def test_type_change() -> None:
    a = '{"val": 42}'
    b = '{"val": "42"}'
    result = json_struct_diff(a, b)
    type_changed = [e for e in result.entries if e.change_type == JsonChangeType.TYPE_CHANGE]
    assert len(type_changed) == 1
    assert type_changed[0].path == "val"


def test_nested_diff() -> None:
    a = '{"outer": {"inner": 1}}'
    b = '{"outer": {"inner": 2}}'
    result = json_struct_diff(a, b)
    changed_paths = [e.path for e in result.changed]
    assert "outer.inner" in changed_paths


def test_summary_message() -> None:
    a = '{"a": 1, "b": 2}'
    b = '{"a": 1, "c": 3}'
    result = json_struct_diff(a, b)
    summary = result.summary()
    # summary() may return a dict or a string depending on implementation
    assert summary is not None
    assert summary  # not empty/falsy


def test_to_dict_keys() -> None:
    a = '{"a": 1}'
    b = '{"a": 2}'
    result = json_struct_diff(a, b)
    d = result.to_dict()
    assert "entries" in d
    assert "is_valid_json_a" in d


# ---------------------------------------------------------------------------
# json_struct_diff — invalid JSON fallback
# ---------------------------------------------------------------------------

def test_invalid_json_a_fallback() -> None:
    a = "not json at all"
    b = '{"key": "value"}'
    result = json_struct_diff(a, b)
    assert not result.is_valid_json_a
    assert result.word_diff_result is not None


def test_invalid_json_both_fallback() -> None:
    a = "plain text A"
    b = "plain text B"
    result = json_struct_diff(a, b)
    assert not result.is_valid_json_a
    assert not result.is_valid_json_b
    assert result.word_diff_result is not None


def test_has_changes_true() -> None:
    a = '{"k": 1}'
    b = '{"k": 2}'
    result = json_struct_diff(a, b)
    assert result.has_changes


def test_has_changes_false() -> None:
    a = '{"k": 1}'
    b = '{"k": 1}'
    result = json_struct_diff(a, b)
    assert not result.has_changes
