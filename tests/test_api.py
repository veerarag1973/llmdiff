"""Tests for llm_diff.api — programmatic Python API."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_diff.api import (
    ComparisonReport,
    _compute_similarity,
    _resolve_config,
    compare,
    compare_batch,
    compare_prompts,
)
from llm_diff.config import LLMDiffConfig
from llm_diff.diff import DiffResult
from llm_diff.providers import ComparisonResult, ModelResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(text: str = "Hello world", model: str = "gpt-4o") -> ModelResponse:
    return ModelResponse(
        model=model,
        text=text,
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        latency_ms=100.0,
        provider="openai",
    )


def _make_comparison(
    text_a: str = "Hello world",
    text_b: str = "Hello earth",
) -> ComparisonResult:
    return ComparisonResult(
        response_a=_make_response(text=text_a, model="gpt-4o"),
        response_b=_make_response(text=text_b, model="claude-3"),
    )


def _make_diff_mock(similarity: float = 0.80) -> MagicMock:
    dr = MagicMock(spec=DiffResult)
    dr.similarity = similarity
    dr.chunks = []
    return dr


# ---------------------------------------------------------------------------
# ComparisonReport
# ---------------------------------------------------------------------------


class TestComparisonReport:
    def _make(
        self,
        semantic_score: float | None = None,
        paragraph_scores=None,
        html_report: str | None = None,
        similarity: float = 0.75,
    ) -> ComparisonReport:
        comparison = _make_comparison()
        diff_result = _make_diff_mock(similarity=similarity)
        return ComparisonReport(
            prompt_a="prompt",
            prompt_b="prompt",
            comparison=comparison,
            diff_result=diff_result,
            semantic_score=semantic_score,
            paragraph_scores=paragraph_scores,
            html_report=html_report,
        )

    def test_word_similarity_delegates_to_diff_result(self) -> None:
        report = self._make(similarity=0.75)
        assert report.word_similarity == pytest.approx(0.75)

    def test_primary_score_returns_semantic_when_set(self) -> None:
        report = self._make(semantic_score=0.9, similarity=0.50)
        assert report.primary_score == pytest.approx(0.9)

    def test_primary_score_falls_back_to_word_when_semantic_none(self) -> None:
        report = self._make(semantic_score=None, similarity=0.65)
        assert report.primary_score == pytest.approx(0.65)

    def test_semantic_score_defaults_to_none(self) -> None:
        report = self._make()
        assert report.semantic_score is None

    def test_paragraph_scores_defaults_to_none(self) -> None:
        report = self._make()
        assert report.paragraph_scores is None

    def test_html_report_defaults_to_none(self) -> None:
        report = self._make()
        assert report.html_report is None

    def test_report_has_expected_fields(self) -> None:
        field_names = {f.name for f in fields(ComparisonReport)}
        assert field_names == {
            "prompt_a",
            "prompt_b",
            "comparison",
            "diff_result",
            "semantic_score",
            "paragraph_scores",
            "bleu_score",
            "rouge_l_score",
            "html_report",
        }

    def test_prompt_a_and_b_stored(self) -> None:
        comparison = _make_comparison()
        diff_result = _make_diff_mock()
        report = ComparisonReport(
            prompt_a="alpha",
            prompt_b="beta",
            comparison=comparison,
            diff_result=diff_result,
        )
        assert report.prompt_a == "alpha"
        assert report.prompt_b == "beta"

    def test_primary_score_zero_semantic_uses_semantic(self) -> None:
        """semantic_score=0.0 is still used (not treated as falsy None)."""
        report = self._make(semantic_score=0.0, similarity=0.8)
        assert report.primary_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _resolve_config
# ---------------------------------------------------------------------------


class TestResolveConfig:
    def test_none_config_loads_from_environment(self) -> None:
        sentinel = LLMDiffConfig()
        with patch("llm_diff.api.load_config", return_value=sentinel) as mock_load:
            result = _resolve_config(None, temperature=None, max_tokens=None, timeout=None)
        mock_load.assert_called_once()
        assert result is sentinel

    def test_provided_config_not_reloaded(self) -> None:
        cfg = LLMDiffConfig()
        with patch("llm_diff.api.load_config") as mock_load:
            result = _resolve_config(cfg, temperature=None, max_tokens=None, timeout=None)
        mock_load.assert_not_called()
        assert result is cfg

    def test_temperature_override_applied(self) -> None:
        cfg = LLMDiffConfig()
        cfg.temperature = 0.5
        result = _resolve_config(cfg, temperature=0.9, max_tokens=None, timeout=None)
        assert result.temperature == pytest.approx(0.9)

    def test_max_tokens_override_applied(self) -> None:
        cfg = LLMDiffConfig()
        result = _resolve_config(cfg, temperature=None, max_tokens=512, timeout=None)
        assert result.max_tokens == 512

    def test_timeout_override_applied(self) -> None:
        cfg = LLMDiffConfig()
        result = _resolve_config(cfg, temperature=None, max_tokens=None, timeout=30)
        assert result.timeout == 30

    def test_none_overrides_do_not_mutate_temperature(self) -> None:
        cfg = LLMDiffConfig()
        cfg.temperature = 0.7
        _resolve_config(cfg, temperature=None, max_tokens=None, timeout=None)
        assert cfg.temperature == pytest.approx(0.7)

    def test_returns_same_config_object_when_provided(self) -> None:
        cfg = LLMDiffConfig()
        result = _resolve_config(cfg, temperature=None, max_tokens=None, timeout=None)
        assert result is cfg


# ---------------------------------------------------------------------------
# _compute_similarity
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    def _run(self, **kwargs):
        return asyncio.run(_compute_similarity(**kwargs))

    def test_both_false_returns_none_none(self) -> None:
        comparison = _make_comparison()
        sem, para = self._run(comparison=comparison, semantic=False, paragraph=False)
        assert sem is None
        assert para is None

    def test_semantic_true_calls_compute_semantic(self) -> None:
        comparison = _make_comparison()
        with patch(
            "llm_diff.semantic.compute_semantic_similarity", return_value=0.85
        ) as mock_sem:
            sem, para = self._run(comparison=comparison, semantic=True, paragraph=False)
        mock_sem.assert_called_once()
        assert sem == pytest.approx(0.85)
        assert para is None

    def test_semantic_true_paragraph_scores_remain_none(self) -> None:
        comparison = _make_comparison()
        with patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.80):
            _, para = self._run(comparison=comparison, semantic=True, paragraph=False)
        assert para is None

    def test_paragraph_true_calls_both_functions(self) -> None:
        comparison = _make_comparison()
        para_scores = [MagicMock()]
        with (
            patch(
                "llm_diff.semantic.compute_paragraph_similarity",
                return_value=para_scores,
            ) as mock_para,
            patch(
                "llm_diff.semantic.compute_semantic_similarity",
                return_value=0.80,
            ) as mock_sem,
        ):
            sem, para = self._run(comparison=comparison, semantic=False, paragraph=True)
        mock_para.assert_called_once()
        mock_sem.assert_called_once()
        assert sem == pytest.approx(0.80)
        assert para is para_scores

    def test_paragraph_true_semantic_score_set(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.semantic.compute_paragraph_similarity", return_value=[]),
            patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.72),
        ):
            sem, _ = self._run(comparison=comparison, semantic=False, paragraph=True)
        assert sem == pytest.approx(0.72)

    def test_paragraph_flag_calls_semantic_exactly_once(self) -> None:
        """paragraph=True calls semantic similarity exactly once (not twice)."""
        comparison = _make_comparison()
        with (
            patch("llm_diff.semantic.compute_paragraph_similarity", return_value=[]),
            patch(
                "llm_diff.semantic.compute_semantic_similarity", return_value=0.70
            ) as mock_sem,
        ):
            self._run(comparison=comparison, semantic=True, paragraph=True)
        mock_sem.assert_called_once()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompare:
    def _run(self, **kwargs):
        return asyncio.run(compare(**kwargs))

    def _defaults(self, **overrides):
        base: dict = {
            "prompt": "Test prompt",
            "model_a": "gpt-4o",
            "model_b": "claude-3",
        }
        base.update(overrides)
        return base

    def test_returns_comparison_report(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults())
        assert isinstance(report, ComparisonReport)

    def test_prompt_stored_as_prompt_a_and_b(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults(prompt="My question"))
        assert report.prompt_a == "My question"
        assert report.prompt_b == "My question"

    def test_compare_models_called_with_correct_models(self) -> None:
        comparison = _make_comparison()
        with (
            patch(
                "llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)
            ) as mock_cmp,
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            self._run(**self._defaults())
        call_kwargs = mock_cmp.call_args.kwargs
        assert call_kwargs["model_a"] == "gpt-4o"
        assert call_kwargs["model_b"] == "claude-3"

    def test_compare_models_receives_prompt(self) -> None:
        comparison = _make_comparison()
        with (
            patch(
                "llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)
            ) as mock_cmp,
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            self._run(**self._defaults(prompt="Test prompt"))
        call_kwargs = mock_cmp.call_args.kwargs
        assert call_kwargs["prompt_a"] == "Test prompt"

    def test_no_semantic_by_default(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults())
        assert report.semantic_score is None

    def test_no_paragraph_scores_by_default(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults())
        assert report.paragraph_scores is None

    def test_semantic_true_sets_semantic_score(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
            patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.88),
        ):
            report = self._run(**self._defaults(semantic=True))
        assert report.semantic_score == pytest.approx(0.88)

    def test_paragraph_true_sets_paragraph_scores(self) -> None:
        comparison = _make_comparison()
        para_scores = [MagicMock(), MagicMock()]
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
            patch(
                "llm_diff.semantic.compute_paragraph_similarity",
                return_value=para_scores,
            ),
            patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.75),
        ):
            report = self._run(**self._defaults(paragraph=True))
        assert report.paragraph_scores is para_scores

    def test_paragraph_true_also_sets_semantic_score(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
            patch("llm_diff.semantic.compute_paragraph_similarity", return_value=[]),
            patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.81),
        ):
            report = self._run(**self._defaults(paragraph=True))
        assert report.semantic_score == pytest.approx(0.81)

    def test_html_report_none_by_default(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults())
        assert report.html_report is None

    def test_build_html_true_populates_html_report(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults(build_html=True))
        assert report.html_report is not None
        assert "<!DOCTYPE html>" in report.html_report

    def test_config_kwarg_skips_load_config(self) -> None:
        comparison = _make_comparison()
        custom_cfg = LLMDiffConfig()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config") as mock_load,
        ):
            self._run(**self._defaults(config=custom_cfg))
        mock_load.assert_not_called()

    def test_diff_result_has_similarity(self) -> None:
        comparison = _make_comparison(text_a="foo bar", text_b="foo baz")
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults())
        assert hasattr(report.diff_result, "similarity")

    def test_comparison_stored_in_report(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(**self._defaults())
        assert report.comparison is comparison


# ---------------------------------------------------------------------------
# compare_prompts
# ---------------------------------------------------------------------------


class TestComparePrompts:
    def _run(self, **kwargs):
        return asyncio.run(compare_prompts(**kwargs))

    def test_returns_comparison_report(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(
                prompt_a="First prompt",
                prompt_b="Second prompt",
                model="gpt-4o",
            )
        assert isinstance(report, ComparisonReport)

    def test_prompt_a_and_b_stored_separately(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(prompt_a="First", prompt_b="Second", model="gpt-4o")
        assert report.prompt_a == "First"
        assert report.prompt_b == "Second"

    def test_same_model_used_for_both_sides(self) -> None:
        comparison = _make_comparison()
        with (
            patch(
                "llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)
            ) as mock_cmp,
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            self._run(prompt_a="A", prompt_b="B", model="gpt-4o")
        call_kwargs = mock_cmp.call_args.kwargs
        assert call_kwargs["model_a"] == "gpt-4o"
        assert call_kwargs["model_b"] == "gpt-4o"

    def test_prompts_passed_correctly(self) -> None:
        comparison = _make_comparison()
        with (
            patch(
                "llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)
            ) as mock_cmp,
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            self._run(prompt_a="Alpha", prompt_b="Beta", model="gpt-4o")
        call_kwargs = mock_cmp.call_args.kwargs
        assert call_kwargs["prompt_a"] == "Alpha"
        assert call_kwargs["prompt_b"] == "Beta"

    def test_semantic_flag_works(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
            patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.72),
        ):
            report = self._run(
                prompt_a="A", prompt_b="B", model="gpt-4o", semantic=True
            )
        assert report.semantic_score == pytest.approx(0.72)

    def test_no_html_by_default(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(prompt_a="A", prompt_b="B", model="gpt-4o")
        assert report.html_report is None

    def test_build_html_true_populates_html_report(self) -> None:
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(
                prompt_a="A", prompt_b="B", model="gpt-4o", build_html=True
            )
        assert report.html_report is not None

    def test_identical_prompts_stored_as_prompt_a(self) -> None:
        """When both prompts are the same, prompt_a and prompt_b equal the input."""
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            report = self._run(prompt_a="Same", prompt_b="Same", model="gpt-4o")
        assert report.prompt_a == "Same"
        assert report.prompt_b == "Same"


# ---------------------------------------------------------------------------
# compare_batch
# ---------------------------------------------------------------------------


class TestCompareBatch:
    def _run(self, **kwargs):
        return asyncio.run(compare_batch(**kwargs))

    def test_returns_list(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text(
            "prompts:\n"
            "  - id: p1\n"
            "    text: 'First prompt'\n"
            "  - id: p2\n"
            "    text: 'Second prompt'\n",
            encoding="utf-8",
        )
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            reports = self._run(
                batch_path=yaml_path, model_a="gpt-4o", model_b="claude-3"
            )
        assert isinstance(reports, list)

    def test_returns_one_report_per_item(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text(
            "prompts:\n"
            "  - id: p1\n"
            "    text: 'First'\n"
            "  - id: p2\n"
            "    text: 'Second'\n",
            encoding="utf-8",
        )
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            reports = self._run(
                batch_path=yaml_path, model_a="gpt-4o", model_b="claude-3"
            )
        assert len(reports) == 2

    def test_each_element_is_comparison_report(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text(
            "prompts:\n  - id: p1\n    text: 'Hello'\n",
            encoding="utf-8",
        )
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            reports = self._run(
                batch_path=yaml_path, model_a="gpt-4o", model_b="claude-3"
            )
        assert all(isinstance(r, ComparisonReport) for r in reports)

    def test_compare_models_called_once_per_item(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text(
            "prompts:\n"
            "  - id: p1\n"
            "    text: 'First'\n"
            "  - id: p2\n"
            "    text: 'Second'\n"
            "  - id: p3\n"
            "    text: 'Third'\n",
            encoding="utf-8",
        )
        comparison = _make_comparison()
        with (
            patch(
                "llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)
            ) as mock_cmp,
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            self._run(batch_path=yaml_path, model_a="gpt-4o", model_b="claude-3")
        assert mock_cmp.call_count == 3

    def test_accepts_string_path(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text(
            "prompts:\n  - id: p1\n    text: 'Hello'\n",
            encoding="utf-8",
        )
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
        ):
            reports = self._run(
                batch_path=str(yaml_path), model_a="gpt-4o", model_b="claude-3"
            )
        assert len(reports) == 1

    def test_semantic_flag_propagated(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text(
            "prompts:\n  - id: p1\n    text: 'Hello'\n",
            encoding="utf-8",
        )
        comparison = _make_comparison()
        with (
            patch("llm_diff.api.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.api.load_config", return_value=LLMDiffConfig()),
            patch(
                "llm_diff.semantic.compute_semantic_similarity", return_value=0.9
            ),
        ):
            reports = self._run(
                batch_path=yaml_path,
                model_a="gpt-4o",
                model_b="claude-3",
                semantic=True,
            )
        assert reports[0].semantic_score == pytest.approx(0.9)

    def test_empty_batch_raises_value_error(self, tmp_path) -> None:
        yaml_path = tmp_path / "prompts.yml"
        yaml_path.write_text("prompts: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            self._run(batch_path=yaml_path, model_a="gpt-4o", model_b="claude-3")
