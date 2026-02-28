"""Regression tests — snapshot-style checks for terminal and JSON rendering.

These tests ensure that the structure and content of all rendered output
(terminal via Rich Console, JSON via ``--json``) do not regress silently
across refactors or dependency updates.

Strategy
--------
* Terminal output is captured with ``Console(record=True, no_color=True)``
  and checked for mandatory structural markers.
* JSON output is checked against a fixed schema (required keys, value ranges,
  and valid chunk types).
* Tests are intentionally broad — they document *what must always be present*,
  not pixel-perfect layout, so they stay green across minor formatting tweaks
  while catching real regressions.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from llm_diff.cli import main
from llm_diff.config import LLMDiffConfig
from llm_diff.diff import word_diff
from llm_diff.providers import ComparisonResult, ModelResponse
from llm_diff.renderer import render_diff

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    model: str = "gpt-4o",
    text: str = "Hello world",
    provider: str = "openai",
    latency_ms: float = 100.0,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ModelResponse:
    return ModelResponse(
        model=model,
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
        provider=provider,
    )


def _make_comparison(
    text_a: str = "The quick brown fox jumps over the lazy dog.",
    text_b: str = "The fast brown fox leaps over the lazy cat.",
    model_a: str = "gpt-4o",
    model_b: str = "claude-3-5-sonnet",
) -> ComparisonResult:
    return ComparisonResult(
        response_a=_make_response(model=model_a, text=text_a, latency_ms=100.0),
        response_b=_make_response(
            model=model_b, text=text_b, provider="anthropic", latency_ms=200.0
        ),
    )


def _render_to_str(
    *,
    text_a: str = "The quick brown fox.",
    text_b: str = "The fast green fox.",
    model_a: str = "gpt-4o",
    model_b: str = "claude-3-5-sonnet",
    semantic_score: float | None = None,
    paragraph_scores: list | None = None,
    prompt: str = "Test prompt",
) -> str:
    """Render to a plain-text string (no ANSI codes)."""
    console = Console(record=True, no_color=True, width=120)
    cmp = _make_comparison(text_a, text_b, model_a, model_b)
    dr = word_diff(text_a, text_b)
    render_diff(
        prompt=prompt,
        result=cmp,
        diff_result=dr,
        console=console,
        semantic_score=semantic_score,
        paragraph_scores=paragraph_scores,
    )
    return console.export_text()


def _make_test_cfg() -> LLMDiffConfig:
    cfg = LLMDiffConfig()
    cfg.openai.api_key = "sk-test"
    cfg.anthropic.api_key = "sk-ant-test"
    return cfg


def _invoke_json(text_a: str, text_b: str, extra_args: list[str] | None = None) -> dict:
    """Invoke CLI in JSON mode with pre-built model responses; return parsed dict."""
    from unittest.mock import AsyncMock

    ra = _make_response(model="gpt-4o", text=text_a)
    rb = _make_response(model="claude-3-5-sonnet", text=text_b, provider="anthropic")
    cmp = ComparisonResult(response_a=ra, response_b=rb)
    runner = CliRunner()
    args = ["Test prompt", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--json"]
    if extra_args:
        args += extra_args
    with (
        patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=cmp)),
        patch("llm_diff.cli.load_config", return_value=_make_test_cfg()),
    ):
        result = runner.invoke(main, args, catch_exceptions=False)
    return json.loads(result.output)


# ---------------------------------------------------------------------------
# Header structural regression
# ---------------------------------------------------------------------------


class TestHeaderRegression:
    def test_header_contains_llm_diff_brand(self) -> None:
        output = _render_to_str()
        assert "llm-diff" in output

    def test_header_contains_model_a_name(self) -> None:
        output = _render_to_str(model_a="gpt-4o")
        assert "gpt-4o" in output

    def test_header_contains_model_b_name(self) -> None:
        output = _render_to_str(model_b="claude-3-5-sonnet")
        assert "claude-3-5-sonnet" in output

    def test_header_contains_vs_separator(self) -> None:
        output = _render_to_str()
        assert "vs" in output

    def test_header_shows_prompt_text(self) -> None:
        output = _render_to_str(prompt="Explain recursion please")
        assert "recursion" in output.lower()

    def test_long_prompt_is_truncated_with_ellipsis(self) -> None:
        long_prompt = "word " * 100  # 100 words, >> 80 chars
        output = _render_to_str(prompt=long_prompt)
        # Renderer truncates prompt to 80 chars and appends "..."
        full_prompt = "word " * 100
        assert full_prompt not in output
        assert "..." in output

    def test_model_names_appear_in_correct_order(self) -> None:
        """Model A must appear before Model B in the header."""
        output = _render_to_str(model_a="alpha-model", model_b="beta-model")
        pos_a = output.find("alpha-model")
        pos_b = output.find("beta-model")
        assert pos_a != -1
        assert pos_b != -1
        assert pos_a < pos_b, "Model A should appear before Model B in the header"

    def test_prompt_is_shown_as_quoted_string(self) -> None:
        output = _render_to_str(prompt="Show this prompt")
        assert "'Show this prompt'" in output or '"Show this prompt"' in output


# ---------------------------------------------------------------------------
# Similarity score rendering regression
# ---------------------------------------------------------------------------


class TestSimilarityScoreRegression:
    def test_output_contains_word_similarity_label(self) -> None:
        output = _render_to_str(
            text_a="The quick fox.", text_b="The slow fox."
        )
        assert "Similarity" in output or "similarity" in output

    def test_identical_texts_show_100_percent(self) -> None:
        same = "Exact same sentence here."
        output = _render_to_str(text_a=same, text_b=same)
        assert "100%" in output

    def test_completely_different_texts_show_0_percent(self) -> None:
        output = _render_to_str(
            text_a="alpha beta gamma delta",
            text_b="omega theta kappa sigma",
        )
        assert "0%" in output

    def test_semantic_score_line_present_when_provided(self) -> None:
        output = _render_to_str(semantic_score=0.87)
        assert "Semantic" in output or "87" in output

    def test_semantic_score_absent_when_none(self) -> None:
        output = _render_to_str(semantic_score=None)
        assert "Semantic" not in output

    def test_similarity_percentage_format(self) -> None:
        """Similarity must be shown as an integer percentage (e.g. '75%')."""
        output = _render_to_str(
            text_a="The quick brown fox.", text_b="The quick brown cat."
        )
        import re
        assert re.search(r"\d+%", output), "No percentage pattern found in output"


# ---------------------------------------------------------------------------
# Word count / token badge regression
# ---------------------------------------------------------------------------


class TestWordCountRegression:
    def test_word_count_a_shown_in_output(self) -> None:
        output = _render_to_str(
            text_a="one two three four five",
            text_b="one two three four six",
        )
        assert "5" in output  # 5 words in text_a

    def test_word_count_b_shown_in_output(self) -> None:
        output = _render_to_str(
            text_a="one two three",
            text_b="one two three four five six",
        )
        assert "6" in output  # 6 words in text_b

    def test_token_count_shown_in_output(self) -> None:
        """Mock responses have total_tokens=30; that value should appear."""
        output = _render_to_str()
        assert "30" in output


# ---------------------------------------------------------------------------
# Diff content regression
# ---------------------------------------------------------------------------


class TestDiffContentRegression:
    def test_inserted_words_appear_in_output(self) -> None:
        output = _render_to_str(
            text_a="The fox jumped.",
            text_b="The fox jumped quickly.",
        )
        assert "quickly" in output

    def test_deleted_words_appear_in_output(self) -> None:
        output = _render_to_str(
            text_a="The quick fox jumped.",
            text_b="The fox jumped.",
        )
        assert "quick" in output

    def test_equal_words_appear_in_output(self) -> None:
        output = _render_to_str(
            text_a="The fox jumped.",
            text_b="The fox leaped.",
        )
        assert "fox" in output

    def test_both_response_texts_represented(self) -> None:
        """Unique words from both sides must appear in the rendered output."""
        output = _render_to_str(
            text_a="The cat sat on the mat.",
            text_b="The dog sat on the rug.",
        )
        # "cat" and "mat" are deletes; "dog" and "rug" are inserts
        assert "cat" in output
        assert "dog" in output


# ---------------------------------------------------------------------------
# JSON output schema regression
# ---------------------------------------------------------------------------


class TestJsonSchemaRegression:
    def test_required_top_level_keys_present(self) -> None:
        data = _invoke_json("Hello world.", "Hello there.")
        required = ("model_a", "model_b", "similarity_score", "diff", "tokens", "latency_ms")
        for key in required:
            assert key in data, f"Required key missing from JSON output: {key!r}"

    def test_similarity_1_for_identical_texts(self) -> None:
        data = _invoke_json("Identical text.", "Identical text.")
        assert data["similarity_score"] == pytest.approx(1.0)

    def test_similarity_0_for_disjoint_texts(self) -> None:
        data = _invoke_json("alpha beta gamma", "omega theta sigma")
        assert data["similarity_score"] == pytest.approx(0.0)

    def test_diff_chunk_types_are_valid(self) -> None:
        data = _invoke_json("Hello world.", "Hello everyone.")
        valid = {"equal", "insert", "delete"}
        for chunk in data["diff"]:
            assert chunk["type"] in valid, f"Invalid chunk type: {chunk['type']!r}"

    def test_diff_has_at_least_one_chunk_for_different_texts(self) -> None:
        data = _invoke_json("Hello world.", "Hello everyone.")
        assert len(data["diff"]) >= 1

    def test_diff_is_empty_for_identical_texts(self) -> None:
        """Identical texts produce only EQUAL chunks — no inserts or deletes."""
        data = _invoke_json("Same text.", "Same text.")
        non_equal = [c for c in data["diff"] if c["type"] != "equal"]
        assert non_equal == []

    def test_tokens_a_and_b_are_integers(self) -> None:
        data = _invoke_json("Test A.", "Test B.")
        assert isinstance(data["tokens"]["a"], int)
        assert isinstance(data["tokens"]["b"], int)

    def test_latency_a_and_b_are_numbers(self) -> None:
        data = _invoke_json("Test A.", "Test B.")
        assert isinstance(data["latency_ms"]["a"], int | float)
        assert isinstance(data["latency_ms"]["b"], int | float)

    def test_model_a_and_b_are_strings(self) -> None:
        data = _invoke_json("Test A.", "Test B.")
        assert isinstance(data["model_a"], str)
        assert isinstance(data["model_b"], str)

    def test_prompt_key_present(self) -> None:
        data = _invoke_json("Test A.", "Test B.")
        assert "prompt" in data

    def test_semantic_score_present_when_requested(self) -> None:
        from unittest.mock import AsyncMock

        ra = _make_response(model="gpt-4o", text="Hello world.")
        rb = _make_response(model="claude-3-5-sonnet", text="Hello world.", provider="anthropic")
        cmp = ComparisonResult(response_a=ra, response_b=rb)
        runner = CliRunner()
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=cmp)),
            patch("llm_diff.cli.load_config", return_value=_make_test_cfg()),
            patch(
                "llm_diff.semantic.compute_semantic_similarity",
                return_value=0.95,
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "Test prompt", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--json", "--semantic",
                ],
                catch_exceptions=False,
            )
        data = json.loads(result.output)
        assert "semantic_score" in data

    def test_diff_text_chunks_reconstruct_correctly(self) -> None:
        """Concatenating all chunk texts must rebuild both response texts."""
        text_a = "The quick brown fox."
        text_b = "The fast green fox."
        data = _invoke_json(text_a, text_b)

        reconstructed_a = "".join(
            c["text"] for c in data["diff"] if c["type"] in ("equal", "delete")
        )
        reconstructed_b = "".join(
            c["text"] for c in data["diff"] if c["type"] in ("equal", "insert")
        )
        assert reconstructed_a == text_a
        assert reconstructed_b == text_b


# ---------------------------------------------------------------------------
# No-color mode regression
# ---------------------------------------------------------------------------


class TestNoColorModeRegression:
    def test_no_color_flag_removes_ansi_escape_codes(self) -> None:
        from unittest.mock import AsyncMock

        ra = _make_response(model="gpt-4o", text="Hello world.")
        rb = _make_response(model="claude-3-5-sonnet", text="Hello there.", provider="anthropic")
        cmp = ComparisonResult(response_a=ra, response_b=rb)
        runner = CliRunner()
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=cmp)),
            patch("llm_diff.cli.load_config", return_value=_make_test_cfg()),
        ):
            result = runner.invoke(
                main,
                [
                    "Test prompt", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--no-color",
                ],
                catch_exceptions=False,
            )
        assert "\x1b[" not in result.output

    def test_no_color_output_still_contains_model_names(self) -> None:
        from unittest.mock import AsyncMock

        ra = _make_response(model="gpt-4o", text="Hello world.")
        rb = _make_response(model="claude-3-5-sonnet", text="Hello there.", provider="anthropic")
        cmp = ComparisonResult(response_a=ra, response_b=rb)
        runner = CliRunner()
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=cmp)),
            patch("llm_diff.cli.load_config", return_value=_make_test_cfg()),
        ):
            result = runner.invoke(
                main,
                [
                    "Test prompt", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--no-color",
                ],
                catch_exceptions=False,
            )
        assert "gpt-4o" in result.output
        assert "claude-3-5-sonnet" in result.output
