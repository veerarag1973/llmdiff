"""Tests for llm_diff.pricing."""

from __future__ import annotations

import pytest

from llm_diff.pricing import CostEstimate, PRICING, estimate_cost, format_cost_table


# ---------------------------------------------------------------------------
# estimate_cost — known models
# ---------------------------------------------------------------------------

def test_gpt4o_known() -> None:
    ce = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    assert ce.known_model is True
    assert ce.prompt_usd == pytest.approx(1000 / 1_000_000 * PRICING["gpt-4o"]["prompt"])
    assert ce.completion_usd == pytest.approx(500 / 1_000_000 * PRICING["gpt-4o"]["completion"])
    assert ce.total_usd == pytest.approx(ce.prompt_usd + ce.completion_usd)


def test_gpt4o_mini_known() -> None:
    ce = estimate_cost("gpt-4o-mini", prompt_tokens=2000, completion_tokens=1000)
    assert ce.known_model is True
    assert ce.total_usd > 0


def test_claude_sonnet_known() -> None:
    ce = estimate_cost("claude-3-5-sonnet-20241022", prompt_tokens=500, completion_tokens=200)
    assert ce.known_model is True


# ---------------------------------------------------------------------------
# estimate_cost — prefix alias matching
# ---------------------------------------------------------------------------

def test_prefix_alias_gpt4o_variant() -> None:
    """gpt-4o-2024-xxxx should match gpt-4o pricing."""
    ce = estimate_cost("gpt-4o-2024-11-20", prompt_tokens=100, completion_tokens=50)
    assert ce.known_model is True
    assert ce.model == "gpt-4o-2024-11-20"


def test_prefix_alias_claude_prefix() -> None:
    ce = estimate_cost("claude-3-5-sonnet-99999", prompt_tokens=100, completion_tokens=50)
    assert ce.known_model is True


# ---------------------------------------------------------------------------
# estimate_cost — unknown model
# ---------------------------------------------------------------------------

def test_unknown_model_zero_cost() -> None:
    ce = estimate_cost("some-new-model-xyz", prompt_tokens=1000, completion_tokens=500)
    assert ce.known_model is False
    assert ce.total_usd == 0.0


def test_unknown_model_returns_estimate() -> None:
    ce = estimate_cost("mystery-model", prompt_tokens=0, completion_tokens=0)
    assert isinstance(ce, CostEstimate)


# ---------------------------------------------------------------------------
# estimate_cost — overrides
# ---------------------------------------------------------------------------

def test_override_pricing() -> None:
    ce = estimate_cost(
        "my-custom-model",
        prompt_tokens=1000,
        completion_tokens=1000,
        overrides={"my-custom-model": {"prompt": 10.0, "completion": 10.0}},
    )
    assert ce.known_model is True
    assert ce.total_usd == pytest.approx(20.0 / 1000)


# ---------------------------------------------------------------------------
# CostEstimate helpers
# ---------------------------------------------------------------------------

def test_total_usd_str_nonzero() -> None:
    ce = estimate_cost("gpt-4o", prompt_tokens=10000, completion_tokens=5000)
    s = ce.total_usd_str
    assert s.startswith("$")
    assert "." in s


def test_total_usd_str_zero() -> None:
    ce = estimate_cost("unknown-xyz", prompt_tokens=0, completion_tokens=0)
    # Zero cost for unknown model — just verify it starts with $ and is a valid string
    assert ce.total_usd_str.startswith("$")
    assert ce.total_usd == 0.0


def test_to_dict_keys() -> None:
    ce = estimate_cost("gpt-4o-mini", prompt_tokens=500, completion_tokens=200)
    d = ce.to_dict()
    assert "model" in d
    assert "total_usd" in d
    assert "prompt_tokens" in d
    assert "completion_tokens" in d
    assert "known_model" in d


# ---------------------------------------------------------------------------
# format_cost_table
# ---------------------------------------------------------------------------

def test_format_cost_table_returns_rows() -> None:
    ca = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    cb = estimate_cost("gpt-4o-mini", prompt_tokens=1000, completion_tokens=500)
    rows = format_cost_table(ca, cb)
    assert isinstance(rows, list)
    assert len(rows) >= 1
    # Each row should be a 3-tuple (label, value_a, value_b)
    for row in rows:
        assert len(row) == 3
