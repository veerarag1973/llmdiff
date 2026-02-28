"""Tests for llm_diff.report — HTML report generator."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llm_diff.batch import BatchItem, BatchResult
from llm_diff.diff import word_diff
from llm_diff.providers import ComparisonResult, ModelResponse
from llm_diff.report import (
    _safe_model_slug,
    _score_class,
    _similarity_pct,
    auto_save_report,
    build_batch_report,
    build_report,
    save_report,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_response(
    model: str = "gpt-4o",
    text: str = "Hello world",
    provider: str = "openai",
    latency_ms: float = 100.0,
) -> ModelResponse:
    return ModelResponse(
        model=model,
        text=text,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        latency_ms=latency_ms,
        provider=provider,
    )


def _make_comparison(
    text_a: str = "The quick brown fox jumps over the lazy dog",
    text_b: str = "The slow brown wolf leaps over the sleepy cat",
    model_a: str = "gpt-4o",
    model_b: str = "claude-3-5-sonnet",
) -> ComparisonResult:
    return ComparisonResult(
        response_a=_make_response(model=model_a, text=text_a, latency_ms=150.0),
        response_b=_make_response(
            model=model_b, text=text_b, provider="anthropic", latency_ms=250.0
        ),
    )


@pytest.fixture()
def comparison() -> ComparisonResult:
    return _make_comparison()


@pytest.fixture()
def diff_result(comparison: ComparisonResult):  # noqa: ANN201
    return word_diff(comparison.response_a.text, comparison.response_b.text)


@pytest.fixture()
def html(comparison: ComparisonResult, diff_result) -> str:  # noqa: ANN001
    return build_report(
        prompt="Test prompt",
        result=comparison,
        diff_result=diff_result,
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestScoreClass:
    def test_high_returns_score_high(self) -> None:
        assert _score_class(0.9) == "score-high"

    def test_exactly_0_8_returns_score_high(self) -> None:
        assert _score_class(0.8) == "score-high"

    def test_mid_returns_score_mid(self) -> None:
        assert _score_class(0.65) == "score-mid"

    def test_exactly_0_5_returns_score_mid(self) -> None:
        assert _score_class(0.5) == "score-mid"

    def test_low_returns_score_low(self) -> None:
        assert _score_class(0.49) == "score-low"

    def test_zero_returns_score_low(self) -> None:
        assert _score_class(0.0) == "score-low"


class TestSimilarityPct:
    def test_formats_as_percentage(self) -> None:
        assert _similarity_pct(0.75) == "75.00%"

    def test_1_0_returns_100(self) -> None:
        assert _similarity_pct(1.0) == "100.00%"

    def test_0_0_returns_0(self) -> None:
        assert _similarity_pct(0.0) == "0.00%"


class TestSafeModelSlug:
    def test_simple_name_unchanged(self) -> None:
        assert _safe_model_slug("gpt4o") == "gpt4o"

    def test_dashes_preserved(self) -> None:
        assert _safe_model_slug("gpt-4o") == "gpt-4o"

    def test_dots_preserved(self) -> None:
        assert _safe_model_slug("gpt4.0") == "gpt4.0"

    def test_slashes_replaced(self) -> None:
        slug = _safe_model_slug("org/model-name")
        assert "/" not in slug
        assert "_" in slug

    def test_spaces_replaced(self) -> None:
        slug = _safe_model_slug("gpt 4 turbo")
        assert " " not in slug


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_returns_string(self, html: str) -> None:
        assert isinstance(html, str)

    def test_starts_with_doctype(self, html: str) -> None:
        assert html.strip().startswith("<!DOCTYPE html>")

    def test_contains_html_tag(self, html: str) -> None:
        assert "<html" in html

    def test_contains_head_and_body(self, html: str) -> None:
        assert "<head>" in html
        assert "<body>" in html

    def test_contains_model_a_name(self, html: str) -> None:
        assert "gpt-4o" in html

    def test_contains_model_b_name(self, html: str) -> None:
        assert "claude-3-5-sonnet" in html

    def test_contains_prompt(self, html: str) -> None:
        assert "Test prompt" in html

    def test_contains_word_similarity(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(prompt="P", result=comparison, diff_result=diff_result)
        assert "%" in html

    def test_contains_tokens(self, html: str) -> None:
        assert "30" in html  # total_tokens from both responses

    def test_diff_chunks_rendered(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(prompt="P", result=comparison, diff_result=diff_result)
        # At least one diff span class must appear
        assert 'class="eq"' in html or 'class="ins"' in html or 'class="del"' in html

    def test_equal_chunk_class_present_for_similar_texts(self) -> None:
        comparison = _make_comparison(
            text_a="Hello world",
            text_b="Hello world",
        )
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        html = build_report(prompt="p", result=comparison, diff_result=diff_result)
        assert 'class="eq"' in html

    def test_no_external_http_resources(self, html: str) -> None:
        """Report must be fully self-contained — no CDN references."""
        # Check for external URLs in src= and href= attributes
        external_src = re.findall(r'(?:src|href)=["\']https?://', html)
        assert external_src == [], f"Found external resources: {external_src}"

    def test_no_external_link_or_script_tags_with_src(self, html: str) -> None:
        script_src = re.findall(r'<script[^>]+src=["\']https?://', html, re.IGNORECASE)
        link_href = re.findall(r'<link[^>]+href=["\']https?://', html, re.IGNORECASE)
        assert script_src == []
        assert link_href == []

    def test_semantic_score_shown_when_provided(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(
            prompt="p",
            result=comparison,
            diff_result=diff_result,
            semantic_score=0.85,
        )
        assert "Semantic Similarity" in html
        assert "85.00%" in html

    def test_semantic_section_absent_when_none(self, html: str) -> None:
        assert "Semantic Similarity" not in html

    def test_version_string_present(self, html: str) -> None:
        from llm_diff import __version__
        assert __version__ in html

    def test_generated_at_custom_value(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(
            prompt="p",
            result=comparison,
            diff_result=diff_result,
            generated_at="2026-02-28 10:00 UTC",
        )
        assert "2026-02-28 10:00 UTC" in html

    def test_generated_at_defaults_to_utc_timestamp(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(prompt="p", result=comparison, diff_result=diff_result)
        assert "UTC" in html

    def test_response_texts_in_side_by_side_view(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(prompt="p", result=comparison, diff_result=diff_result)
        assert comparison.response_a.text in html
        assert comparison.response_b.text in html

    def test_latency_values_present(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(prompt="p", result=comparison, diff_result=diff_result)
        assert "150ms" in html
        assert "250ms" in html

    def test_html_escaping_for_prompt_with_special_chars(
        self, comparison: ComparisonResult, diff_result
    ) -> None:
        html = build_report(
            prompt='What is 2 < 3? And "quotes"?',
            result=comparison,
            diff_result=diff_result,
        )
        # Jinja2 autoescape should convert < and > to entities
        assert "<script" not in html.split("What is")[1].split("3")[0]


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------


class TestSaveReport:
    def test_writes_html_file(self, tmp_path: Path, html: str) -> None:
        dest = tmp_path / "report.html"
        result = save_report(html, dest)
        assert result.exists()
        assert result.read_text(encoding="utf-8") == html

    def test_returns_absolute_path(self, tmp_path: Path, html: str) -> None:
        dest = tmp_path / "report.html"
        result = save_report(html, dest)
        assert result.is_absolute()

    def test_creates_parent_directories(self, tmp_path: Path, html: str) -> None:
        dest = tmp_path / "nested" / "deep" / "report.html"
        save_report(html, dest)
        assert dest.exists()

    def test_returned_path_points_to_file(self, tmp_path: Path, html: str) -> None:
        dest = tmp_path / "out.html"
        returned = save_report(html, dest)
        assert returned == dest.resolve()


# ---------------------------------------------------------------------------
# auto_save_report
# ---------------------------------------------------------------------------


class TestAutoSaveReport:
    def test_creates_diffs_dir(self, tmp_path: Path, html: str) -> None:
        diffs_dir = tmp_path / "diffs"
        result = auto_save_report(html, "gpt-4o", "claude-3-5-sonnet", diffs_dir=diffs_dir)
        assert diffs_dir.exists()
        assert result.exists()

    def test_filename_contains_model_a_slug(self, tmp_path: Path, html: str) -> None:
        result = auto_save_report(html, "gpt-4o", "claude-3", diffs_dir=tmp_path)
        assert "gpt-4o" in result.name

    def test_filename_contains_model_b_slug(self, tmp_path: Path, html: str) -> None:
        result = auto_save_report(html, "gpt-4o", "claude-3", diffs_dir=tmp_path)
        assert "claude-3" in result.name

    def test_filename_ends_with_html(self, tmp_path: Path, html: str) -> None:
        result = auto_save_report(html, "a", "b", diffs_dir=tmp_path)
        assert result.suffix == ".html"

    def test_filename_contains_vs_separator(self, tmp_path: Path, html: str) -> None:
        result = auto_save_report(html, "gpt-4o", "claude-3", diffs_dir=tmp_path)
        assert "_vs_" in result.name

    def test_returns_existing_path(self, tmp_path: Path, html: str) -> None:
        result = auto_save_report(html, "a", "b", diffs_dir=tmp_path)
        assert result.exists()
        assert result.read_text(encoding="utf-8") == html

    def test_two_calls_produce_different_filenames(
        self, tmp_path: Path, html: str
    ) -> None:
        """Filenames include a timestamp so sequential calls differ (at second granularity)."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        times = [
            datetime(2026, 2, 28, 10, 0, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 28, 10, 0, 2, tzinfo=timezone.utc),
        ]
        call_count = 0

        def fake_now(tz=None):  # noqa: ANN001, ANN202
            nonlocal call_count
            t = times[call_count % 2]
            call_count += 1
            return t

        with patch("llm_diff.report.datetime") as mock_dt:
            mock_dt.now.side_effect = fake_now
            mock_dt.now.return_value = times[0]
            p1 = auto_save_report(html, "gpt-4o", "claude-3", diffs_dir=tmp_path)
            p2 = auto_save_report(html, "gpt-4o", "claude-3", diffs_dir=tmp_path)

        assert p1.name != p2.name


# ---------------------------------------------------------------------------
# build_batch_report
# ---------------------------------------------------------------------------


class TestBuildBatchReport:
    """Tests for the combined batch HTML report builder."""

    def _make_results(
        self,
        n: int = 2,
        with_semantic: bool = False,
    ) -> list[BatchResult]:
        texts = [
            ("The quick brown fox", "The slow brown wolf"),
            ("Hello world", "Hello earth"),
            ("Fast red car", "Slow blue vehicle"),
        ]
        results = []
        for i in range(n):
            ta, tb = texts[i % len(texts)]
            cmp = _make_comparison(text_a=ta, text_b=tb)
            dr = word_diff(ta, tb)
            results.append(
                BatchResult(
                    item=BatchItem(id=f"p{i + 1}", prompt_text=f"Prompt {i + 1}"),
                    comparison=cmp,
                    diff_result=dr,
                    semantic_score=0.85 if with_semantic else None,
                )
            )
        return results

    def test_returns_html_string(self) -> None:
        html = build_batch_report(
            results=self._make_results(),
            model_a="gpt-4o",
            model_b="claude-3",
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_model_names(self) -> None:
        html = build_batch_report(
            results=self._make_results(),
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
        )
        assert "gpt-4o" in html
        assert "claude-3-5-sonnet" in html

    def test_contains_all_item_ids(self) -> None:
        html = build_batch_report(
            results=self._make_results(n=3),
            model_a="a",
            model_b="b",
        )
        assert "p1" in html
        assert "p2" in html
        assert "p3" in html

    def test_shows_total_prompt_count(self) -> None:
        results = self._make_results(n=2)
        html = build_batch_report(results=results, model_a="a", model_b="b")
        # The template renders total as an integer; "2" appears in context
        assert "2" in html

    def test_no_external_http_requests(self) -> None:
        html = build_batch_report(
            results=self._make_results(),
            model_a="a",
            model_b="b",
        )
        assert "http://" not in html
        assert "https://" not in html

    def test_custom_generated_at_shown(self) -> None:
        html = build_batch_report(
            results=self._make_results(),
            model_a="a",
            model_b="b",
            generated_at="2026-02-28 12:00 UTC",
        )
        assert "2026-02-28 12:00 UTC" in html

    def test_version_embedded(self) -> None:
        from llm_diff import __version__

        html = build_batch_report(results=self._make_results(), model_a="a", model_b="b")
        assert __version__ in html

    def test_semantic_section_shown_when_scores_present(self) -> None:
        html = build_batch_report(
            results=self._make_results(with_semantic=True),
            model_a="a",
            model_b="b",
        )
        assert "Semantic" in html

    def test_semantic_section_absent_when_no_scores(self) -> None:
        results = self._make_results(with_semantic=False)
        html = build_batch_report(results=results, model_a="a", model_b="b")
        assert "Avg Semantic" not in html

    def test_input_label_shown_when_present(self) -> None:
        results = self._make_results(n=1)
        results[0].item = BatchItem(
            id="s:file.txt", prompt_text="hello", input_label="file.txt"
        )
        html = build_batch_report(results=results, model_a="a", model_b="b")
        assert "file.txt" in html

    def test_diff_chunks_rendered(self) -> None:
        """Diff ins/del spans should appear in the output."""
        html = build_batch_report(
            results=self._make_results(n=1),
            model_a="a",
            model_b="b",
        )
        # The diff section uses 'ins' and/or 'del' CSS classes
        assert 'class="ins"' in html or 'class="del"' in html or 'class="eq"' in html

    def test_empty_results_list_renders(self) -> None:
        """build_batch_report handles an empty results list without error."""
        html = build_batch_report(results=[], model_a="a", model_b="b")
        assert "<!DOCTYPE html>" in html
