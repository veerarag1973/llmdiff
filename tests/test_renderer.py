"""Tests for llm_diff.renderer — terminal output rendering."""

from __future__ import annotations

from rich.console import Console

from llm_diff.diff import DiffChunk, DiffType, word_diff
from llm_diff.providers import ComparisonResult, ModelResponse
from llm_diff.renderer import (
    _build_diff_text,
    _latency_badge,
    _score_colour,
    _token_badge,
    render_diff,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_response(
    model: str = "gpt-4o",
    text: str = "Hello world",
    provider: str = "openai",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    latency_ms: float = 123.0,
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
    text_a: str = "Hello world",
    text_b: str = "Hello there",
    model_a: str = "gpt-4o",
    model_b: str = "claude-3-5-sonnet",
) -> ComparisonResult:
    return ComparisonResult(
        response_a=_make_response(model=model_a, text=text_a, latency_ms=100.0),
        response_b=_make_response(
            model=model_b, text=text_b, provider="anthropic", latency_ms=200.0
        ),
    )


def _recording_console() -> Console:
    """Return a Rich Console that records all output, with colour disabled."""
    return Console(record=True, no_color=True, width=120)


# ---------------------------------------------------------------------------
# _score_colour
# ---------------------------------------------------------------------------


class TestScoreColour:
    def test_high_similarity_returns_green(self) -> None:
        assert _score_colour(0.8) == "bold green"

    def test_exactly_0_8_returns_green(self) -> None:
        assert _score_colour(0.8) == "bold green"

    def test_above_0_8_returns_green(self) -> None:
        assert _score_colour(0.95) == "bold green"

    def test_mid_similarity_returns_yellow(self) -> None:
        assert _score_colour(0.5) == "bold yellow"

    def test_just_below_0_8_returns_yellow(self) -> None:
        assert _score_colour(0.79) == "bold yellow"

    def test_exactly_0_5_returns_yellow(self) -> None:
        assert _score_colour(0.5) == "bold yellow"

    def test_low_similarity_returns_red(self) -> None:
        assert _score_colour(0.49) == "bold red"

    def test_zero_returns_red(self) -> None:
        assert _score_colour(0.0) == "bold red"


# ---------------------------------------------------------------------------
# _build_diff_text
# ---------------------------------------------------------------------------


class TestBuildDiffText:
    def test_empty_chunks_produces_empty_text(self) -> None:
        result = _build_diff_text([])
        assert str(result) == ""

    def test_equal_chunk_appended(self) -> None:
        chunks = [DiffChunk(type=DiffType.EQUAL, text="hello ")]
        text = _build_diff_text(chunks)
        assert "hello" in text.plain

    def test_insert_chunk_appended(self) -> None:
        chunks = [DiffChunk(type=DiffType.INSERT, text="world")]
        text = _build_diff_text(chunks)
        assert "world" in text.plain

    def test_delete_chunk_appended(self) -> None:
        chunks = [DiffChunk(type=DiffType.DELETE, text="gone")]
        text = _build_diff_text(chunks)
        assert "gone" in text.plain

    def test_mixed_chunks_all_present(self) -> None:
        chunks = [
            DiffChunk(type=DiffType.EQUAL, text="same "),
            DiffChunk(type=DiffType.DELETE, text="old "),
            DiffChunk(type=DiffType.INSERT, text="new"),
        ]
        text = _build_diff_text(chunks)
        assert "same" in text.plain
        assert "old" in text.plain
        assert "new" in text.plain

    def test_insert_has_green_style(self) -> None:
        chunks = [DiffChunk(type=DiffType.INSERT, text="added")]
        text = _build_diff_text(chunks)
        # Rich Text stores spans; at least one span should reference green
        styles = [str(span.style) for span in text._spans]
        assert any("green" in s for s in styles)

    def test_delete_has_red_style(self) -> None:
        chunks = [DiffChunk(type=DiffType.DELETE, text="removed")]
        text = _build_diff_text(chunks)
        styles = [str(span.style) for span in text._spans]
        assert any("red" in s for s in styles)


# ---------------------------------------------------------------------------
# _token_badge / _latency_badge
# ---------------------------------------------------------------------------


class TestBadges:
    def test_token_badge_contains_count(self) -> None:
        badge = _token_badge("Tokens: ", 42)
        assert "42" in badge.plain

    def test_token_badge_contains_label(self) -> None:
        badge = _token_badge("Tok: ", 100)
        assert "Tok:" in badge.plain

    def test_latency_badge_contains_ms(self) -> None:
        badge = _latency_badge("Latency: ", 456.7)
        assert "457ms" in badge.plain

    def test_latency_badge_rounds_to_integer(self) -> None:
        badge = _latency_badge("L: ", 99.4)
        assert "99ms" in badge.plain

    def test_latency_badge_contains_label(self) -> None:
        badge = _latency_badge("Lat: ", 50.0)
        assert "Lat:" in badge.plain


# ---------------------------------------------------------------------------
# render_diff — integration
# ---------------------------------------------------------------------------


class TestRenderDiff:
    def _render(
        self,
        text_a: str = "The quick brown fox",
        text_b: str = "The slow brown wolf",
        prompt: str = "Test prompt",
        model_a: str = "gpt-4o",
        model_b: str = "claude-3-5-sonnet",
    ) -> str:
        """Helper: render diff and return captured plain text."""
        comparison = _make_comparison(
            text_a=text_a, text_b=text_b, model_a=model_a, model_b=model_b
        )
        diff_result = word_diff(text_a, text_b)
        console = _recording_console()
        render_diff(
            prompt=prompt,
            result=comparison,
            diff_result=diff_result,
            console=console,
        )
        return console.export_text()

    def test_model_names_appear_in_output(self) -> None:
        output = self._render()
        assert "gpt-4o" in output
        assert "claude-3-5-sonnet" in output

    def test_prompt_appears_in_output(self) -> None:
        output = self._render(prompt="Explain recursion")
        assert "Explain recursion" in output

    def test_similarity_line_appears(self) -> None:
        output = self._render()
        assert "Similarity" in output

    def test_tokens_line_appears(self) -> None:
        output = self._render()
        assert "Tokens" in output

    def test_latency_line_appears(self) -> None:
        output = self._render()
        assert "ms" in output

    def test_diff_body_contains_changed_words(self) -> None:
        output = self._render(
            text_a="The quick brown fox",
            text_b="The slow brown wolf",
        )
        # Changed words should appear in the rendered diff body
        assert "quick" in output or "slow" in output

    def test_identical_texts_show_100_percent(self) -> None:
        output = self._render(text_a="Same text", text_b="Same text")
        # similarity_pct may render as "100%" or "100.00%" depending on rounding
        assert "100" in output and "%" in output

    def test_long_prompt_truncated_to_83_chars(self) -> None:
        long_prompt = "A" * 100
        output = self._render(prompt=long_prompt)
        # Prompt display is capped at 80 chars (77 + "...")
        assert "..." in output

    def test_short_prompt_not_truncated(self) -> None:
        short = "Short"
        output = self._render(prompt=short)
        assert "..." not in output

    def test_renders_without_raising(self) -> None:
        """Smoke test: no exception for a typical diff."""
        self._render()

    def test_fully_different_texts(self) -> None:
        output = self._render(text_a="alpha beta", text_b="gamma delta")
        assert "Similarity" in output

    def test_empty_responses_handled_gracefully(self) -> None:
        output = self._render(text_a="", text_b="")
        assert "Similarity" in output
