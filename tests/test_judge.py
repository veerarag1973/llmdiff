"""Tests for llm_diff.judge."""

from __future__ import annotations

import json

import pytest

from llm_diff.judge import JudgeResult, _normalise_winner, _parse_judge_response

# ---------------------------------------------------------------------------
# _normalise_winner
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        ("B", "B"),
        ("tie", "tie"),
        ("Tie", "tie"),
        ("TIE", "tie"),
        ("a", "A"),
        ("b", "B"),
        ("neither", "tie"),
        ("equal", "tie"),
        ("draw", "tie"),
        ("unknown_value", "tie"),
    ],
)
def test_normalise_winner(raw: str, expected: str) -> None:
    assert _normalise_winner(raw) == expected


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------

def test_parse_direct_json() -> None:
    raw = json.dumps(
        {"winner": "A", "score_a": 8, "score_b": 6, "reasoning": "A was better"}
    )
    result = _parse_judge_response(raw)
    assert result["winner"] == "A"
    assert result["score_a"] == 8


def test_parse_code_fence_json() -> None:
    raw = '```json\n{"winner": "B", "score_a": 5, "score_b": 9, "reasoning": "B"}\n```'
    result = _parse_judge_response(raw)
    assert result["winner"] == "B"
    assert result["score_b"] == 9


def test_parse_embedded_json_braces() -> None:
    raw = 'Here is my answer: {"winner": "tie", "score_a": 7, "score_b": 7, "reasoning": "Equal"} — done.'
    result = _parse_judge_response(raw)
    assert result["winner"] == "tie"


def test_parse_raises_on_completely_unparseable() -> None:
    raw = "Totally not JSON at all."
    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_judge_response(raw)


# ---------------------------------------------------------------------------
# JudgeResult
# ---------------------------------------------------------------------------

def test_judge_result_to_dict() -> None:
    jr = JudgeResult(
        winner="A",
        reasoning="Model A was more accurate",
        score_a=8.0,
        score_b=5.5,
        judge_model="gpt-4o",
        raw_response='{"winner":"A"}',
    )
    d = jr.to_dict()
    assert d["winner"] == "A"
    assert d["score_a"] == 8.0
    assert d["reasoning"] == "Model A was more accurate"
    assert d["judge_model"] == "gpt-4o"


def test_judge_result_defaults() -> None:
    jr = JudgeResult(
        winner="tie",
        reasoning="",
        score_a=0.0,
        score_b=0.0,
        judge_model="gpt-4o-mini",
        raw_response="",
    )
    assert jr.winner == "tie"


def test_judge_result_to_dict_round_trip() -> None:
    jr = JudgeResult(
        winner="B",
        reasoning="shorter and cleaner",
        score_a=4.0,
        score_b=9.0,
        judge_model="claude-3-5-sonnet-20241022",
        raw_response="{}",
    )
    d = jr.to_dict()
    assert json.dumps(d)  # serialisable
    assert d["winner"] == "B"
