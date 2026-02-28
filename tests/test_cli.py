"""Tests for llm_diff.cli — Click entry point and orchestration logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from llm_diff.cli import _die, _read_file_arg, _render_json, _render_verbose, _resolve_inputs, main
from llm_diff.config import LLMDiffConfig
from llm_diff.diff import word_diff
from llm_diff.providers import ComparisonResult, ModelResponse

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_response(
    model: str = "gpt-4o",
    text: str = "The quick brown fox",
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
    text_a: str = "The quick brown fox",
    text_b: str = "The slow brown wolf",
    model_a: str = "gpt-4o",
    model_b: str = "claude-3-5-sonnet",
) -> ComparisonResult:
    return ComparisonResult(
        response_a=_make_response(model=model_a, text=text_a),
        response_b=_make_response(
            model=model_b, text=text_b, provider="anthropic", latency_ms=200.0
        ),
    )


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def mock_cfg() -> LLMDiffConfig:
    cfg = LLMDiffConfig()
    cfg.openai.api_key = "sk-test"
    cfg.anthropic.api_key = "sk-ant-test"
    return cfg


@pytest.fixture()
def mock_comparison() -> ComparisonResult:
    return _make_comparison()


# ---------------------------------------------------------------------------
# _resolve_inputs
# ---------------------------------------------------------------------------


class TestResolveInputs:
    def _call(self, **kwargs) -> tuple[str, str, str, str]:
        defaults = {
            "prompt": None,
            "prompt_a": None,
            "prompt_b": None,
            "model_a": None,
            "model_b": None,
            "model": None,
        }
        defaults.update(kwargs)
        return _resolve_inputs(**defaults)  # type: ignore[arg-type]

    def test_positional_prompt_with_two_models(self) -> None:
        pa, pb, ma, mb = self._call(prompt="Hello", model_a="gpt-4o", model_b="claude-3")
        assert pa == pb == "Hello"
        assert ma == "gpt-4o"
        assert mb == "claude-3"

    def test_no_prompt_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="Provide a prompt"):
            self._call(model_a="gpt-4o", model_b="claude-3")

    def test_only_prompt_a_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="--prompt-a requires --prompt-b"):
            self._call(prompt_a="file.txt", model_a="gpt-4o", model_b="claude-3")

    def test_only_prompt_b_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="--prompt-b requires --prompt-a"):
            self._call(prompt_b="file.txt", model_a="gpt-4o", model_b="claude-3")

    def test_only_model_a_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="--model-a"):
            self._call(prompt="Hello", model_a="gpt-4o")

    def test_only_model_b_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="--model-b"):
            self._call(prompt="Hello", model_b="claude-3")

    def test_no_model_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="Specify models"):
            self._call(prompt="Hello")

    def test_model_flag_without_prompt_files_raises(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="--model requires --prompt-a"):
            self._call(prompt="Hello", model="gpt-4o")

    def test_model_flag_with_prompt_files(self, tmp_path: Path) -> None:
        pa = tmp_path / "a.txt"
        pb = tmp_path / "b.txt"
        pa.write_text("Prompt A")
        pb.write_text("Prompt B")
        rpa, rpb, rma, rmb = self._call(
            prompt_a=str(pa),
            prompt_b=str(pb),
            model="gpt-4o",
        )
        assert rpa == "Prompt A"
        assert rpb == "Prompt B"
        assert rma == rmb == "gpt-4o"


# ---------------------------------------------------------------------------
# _read_file_arg
# ---------------------------------------------------------------------------


class TestReadFileArg:
    def test_missing_file_raises_usage_error(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="file not found"):
            _read_file_arg("/no/such/file.txt", "--prompt-a")

    def test_existing_file_returns_stripped_content(self, tmp_path: Path) -> None:
        f = tmp_path / "prompt.txt"
        f.write_text("  hello world  \n")
        assert _read_file_arg(str(f), "--prompt-a") == "hello world"


# ---------------------------------------------------------------------------
# _die
# ---------------------------------------------------------------------------


class TestDie:
    def test_die_exits_with_code_1_by_default(self) -> None:
        console = Console(no_color=True)
        with pytest.raises(SystemExit) as exc_info:
            _die(console, "something went wrong")
        assert exc_info.value.code == 1

    def test_die_exits_with_custom_code(self) -> None:
        console = Console(no_color=True)
        with pytest.raises(SystemExit) as exc_info:
            _die(console, "oops", exit_code=42)
        assert exc_info.value.code == 42


# ---------------------------------------------------------------------------
# _render_json
# ---------------------------------------------------------------------------


class TestRenderJson:
    def test_outputs_valid_json(self, capsys: pytest.CaptureFixture) -> None:
        comparison = _make_comparison()
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        console = Console(no_color=True)
        _render_json(comparison, diff_result, console)
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["model_a"] == "gpt-4o"
        assert payload["model_b"] == "claude-3-5-sonnet"
        assert "similarity_score" in payload
        assert "diff" in payload

    def test_json_keys_structure(self, capsys: pytest.CaptureFixture) -> None:
        comparison = _make_comparison()
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        console = Console(no_color=True)
        _render_json(comparison, diff_result, console)
        payload = json.loads(capsys.readouterr().out)
        expected_keys = {
            "prompt", "model_a", "model_b", "similarity_score",
            "tokens", "latency_ms", "diff",
        }
        assert set(payload.keys()) == expected_keys
        assert "a" in payload["tokens"] and "b" in payload["tokens"]
        assert "a" in payload["latency_ms"] and "b" in payload["latency_ms"]


# ---------------------------------------------------------------------------
# _render_verbose
# ---------------------------------------------------------------------------


class TestRenderVerbose:
    def test_renders_table_with_model_names(self) -> None:
        comparison = _make_comparison()
        console = Console(record=True, no_color=True, width=120)
        _render_verbose(comparison, console)
        output = console.export_text()
        assert "gpt-4o" in output
        assert "claude-3-5-sonnet" in output

    def test_renders_token_rows(self) -> None:
        comparison = _make_comparison()
        console = Console(record=True, no_color=True, width=120)
        _render_verbose(comparison, console)
        output = console.export_text()
        assert "Total tokens" in output


# ---------------------------------------------------------------------------
# main() via CliRunner
# ---------------------------------------------------------------------------


class TestMainCli:
    """Integration tests for the Click CLI using CliRunner."""

    def _invoke_with_mocks(
        self,
        runner: CliRunner,
        args: list[str],
        cfg: LLMDiffConfig,
        comparison: ComparisonResult,
    ):
        """Invoke main() with compare_models and load_config patched."""
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.cli.load_config", return_value=cfg),
        ):
            return runner.invoke(main, args, catch_exceptions=False)

    def test_version_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.3.0" in result.output

    def test_help_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_no_prompt_exits_1(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["-a", "gpt-4o", "-b", "claude-3"])
        assert result.exit_code == 1

    def test_missing_both_models_exits_1(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["Hello world"])
        assert result.exit_code == 1

    def test_successful_word_diff(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_json_mode_outputs_json(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--json"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "model_a" in payload
        assert "diff" in payload

    def test_json_shorthand_flag(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "-j"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "similarity_score" in payload

    def test_semantic_flag_accepted(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        # --semantic flag adds similarity scoring in phase 2
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
            patch("llm_diff.semantic.compute_semantic_similarity", return_value=0.75),
        ):
            result = runner.invoke(
                main,
                ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--semantic"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_verbose_flag_shows_metadata(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--verbose"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0
        assert "Total tokens" in result.output

    def test_value_error_from_compare_models_exits_1(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig
    ) -> None:
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(side_effect=ValueError("bad key"))),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1

    def test_timeout_error_from_compare_models_exits_1(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig
    ) -> None:
        with (
            patch(
                "llm_diff.cli.compare_models",
                new=AsyncMock(side_effect=TimeoutError("timed out")),
            ),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert result.exit_code == 1

    def test_no_color_flag_accepted(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--no-color"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_temperature_flag_overrides_config(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--temperature", "0.2"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_max_tokens_flag_overrides_config(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--max-tokens", "512"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_timeout_flag_overrides_config(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--timeout", "60"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_prompt_file_flags(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        tmp_path: Path,
    ) -> None:
        pa = tmp_path / "a.txt"
        pb = tmp_path / "b.txt"
        pa.write_text("Prompt A text")
        pb.write_text("Prompt B text")
        result = self._invoke_with_mocks(
            runner,
            ["--prompt-a", str(pa), "--prompt-b", str(pb), "-a", "gpt-4o", "-b", "claude-3"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_prompt_file_not_found_exits_1(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            ["--prompt-a", "/no/such/file.txt", "--prompt-b", "/no/other.txt",
             "-a", "gpt-4o", "-b", "claude-3"],
        )
        assert result.exit_code != 0

    def test_batch_missing_file_exits_1(self, runner: CliRunner) -> None:
        """--batch with a non-existent file raises FileNotFoundError → exit 1."""
        result = runner.invoke(
            main,
            ["--batch", "no_such_file.yml", "-a", "gpt-4o", "-b", "claude-3"],
        )
        assert result.exit_code == 1

    def test_display_prompt_truncated_when_prompts_differ(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult,
        tmp_path: Path,
    ) -> None:
        """When prompt_a != prompt_b the display prompt is truncated with ellipsis."""
        pa = tmp_path / "a.txt"
        pb = tmp_path / "b.txt"
        pa.write_text("A" * 50)
        pb.write_text("B" * 50)
        result = self._invoke_with_mocks(
            runner,
            ["--prompt-a", str(pa), "--prompt-b", str(pb), "-a", "gpt-4o", "-b", "claude-3"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0

    def test_save_flag_sets_config_save(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3", "--save"],
                catch_exceptions=False,
            )
        assert mock_cfg.save is True

    def test_keyboard_interrupt_exits_130(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig
    ) -> None:
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(side_effect=KeyboardInterrupt)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3"],
                catch_exceptions=False,
            )
        assert result.exit_code == 130


# ---------------------------------------------------------------------------
# Phase 2: _render_json — prompt field and semantic_score
# ---------------------------------------------------------------------------


class TestRenderJsonPhase2:
    def test_json_includes_prompt_field(self, capsys: pytest.CaptureFixture) -> None:
        comparison = _make_comparison()
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        console = Console(no_color=True)
        _render_json(comparison, diff_result, console, prompt="Explain recursion")
        payload = json.loads(capsys.readouterr().out)
        assert payload["prompt"] == "Explain recursion"

    def test_json_includes_semantic_score_when_provided(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        comparison = _make_comparison()
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        console = Console(no_color=True)
        _render_json(comparison, diff_result, console, prompt="p", semantic_score=0.7654)
        payload = json.loads(capsys.readouterr().out)
        assert "semantic_score" in payload
        assert payload["semantic_score"] == pytest.approx(0.7654, abs=0.001)

    def test_json_omits_semantic_score_when_none(self, capsys: pytest.CaptureFixture) -> None:
        comparison = _make_comparison()
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        console = Console(no_color=True)
        _render_json(comparison, diff_result, console, prompt="p", semantic_score=None)
        payload = json.loads(capsys.readouterr().out)
        assert "semantic_score" not in payload

    def test_json_prompt_default_is_empty_string(self, capsys: pytest.CaptureFixture) -> None:
        comparison = _make_comparison()
        diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)
        console = Console(no_color=True)
        _render_json(comparison, diff_result, console)
        payload = json.loads(capsys.readouterr().out)
        assert payload["prompt"] == ""


# ---------------------------------------------------------------------------
# Phase 2: CLI integration — semantic mode, --out, --save with report
# ---------------------------------------------------------------------------


class TestMainCliPhase2:
    def _invoke_with_mocks(
        self,
        runner: CliRunner,
        args: list,
        cfg: LLMDiffConfig,
        comparison: ComparisonResult,
        extra_patches: dict | None = None,
    ):
        patches = {
            "llm_diff.cli.compare_models": AsyncMock(return_value=comparison),
            "llm_diff.cli.load_config": cfg,
        }
        if extra_patches:
            patches.update(extra_patches)
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=comparison)),
            patch("llm_diff.cli.load_config", return_value=cfg),
        ):
            return runner.invoke(main, args, catch_exceptions=False)

    def test_semantic_mode_computes_similarity(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
            patch(
                "llm_diff.semantic.compute_semantic_similarity",
                return_value=0.82,
            ) as mock_sem,
        ):
            result = runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3", "--semantic"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        mock_sem.assert_called_once()

    def test_json_mode_includes_prompt_in_output(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        result = self._invoke_with_mocks(
            runner,
            ["My prompt", "-a", "gpt-4o", "-b", "claude-3", "--json"],
            mock_cfg,
            mock_comparison,
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["prompt"] == "My prompt"

    def test_semantic_json_mode_includes_semantic_score(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
            patch(
                "llm_diff.semantic.compute_semantic_similarity",
                return_value=0.91,
            ),
        ):
            result = runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3", "--semantic", "--json"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert "semantic_score" in payload
        assert payload["semantic_score"] == pytest.approx(0.91, abs=0.001)

    def test_out_flag_saves_html_report(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        tmp_path: Path,
    ) -> None:
        report_path = tmp_path / "report.html"
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3", "--out", str(report_path)],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert report_path.exists()
        assert "<!DOCTYPE html>" in report_path.read_text(encoding="utf-8")

    def test_save_flag_creates_file_in_diffs_dir(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        tmp_path: Path,
    ) -> None:
        mock_cfg.save = False  # reset; --save flag will flip it
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
            patch(
                "llm_diff.report.auto_save_report",
                return_value=tmp_path / "report.html",
            ) as mock_auto_save,
        ):
            result = runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3", "--save"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        mock_auto_save.assert_called_once()

    def test_semantic_import_error_exits_1(
        self, runner: CliRunner, mock_cfg: LLMDiffConfig, mock_comparison: ComparisonResult
    ) -> None:
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
            patch(
                "llm_diff.semantic.compute_semantic_similarity",
                side_effect=ImportError("sentence-transformers not installed"),
            ),
        ):
            result = runner.invoke(
                main,
                ["Hello", "-a", "gpt-4o", "-b", "claude-3", "--semantic"],
            )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Phase 3: CLI integration — batch mode
# ---------------------------------------------------------------------------


class TestBatchCli:
    """Integration tests for --batch mode."""

    @pytest.fixture()
    def batch_yaml(self, tmp_path: Path) -> Path:
        """Minimal batch YAML with two plain prompts (no inputs)."""
        f = tmp_path / "prompts.yml"
        f.write_text(
            "prompts:\n"
            "  - id: p1\n"
            "    text: 'First prompt'\n"
            "  - id: p2\n"
            "    text: 'Second prompt'\n",
            encoding="utf-8",
        )
        return f

    def test_batch_processes_all_items(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        batch_yaml: Path,
    ) -> None:
        """compare_models is called once per batch item."""
        with (
            patch(
                "llm_diff.cli.compare_models",
                new=AsyncMock(return_value=mock_comparison),
            ) as mock_cmp,
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["--batch", str(batch_yaml), "-a", "gpt-4o", "-b", "claude-3"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert mock_cmp.call_count == 2

    def test_batch_missing_models_exits_1(
        self,
        runner: CliRunner,
        batch_yaml: Path,
    ) -> None:
        """--batch without model flags exits with code 1."""
        result = runner.invoke(
            main,
            ["--batch", str(batch_yaml)],
        )
        assert result.exit_code == 1

    def test_batch_with_out_saves_html_report(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        batch_yaml: Path,
        tmp_path: Path,
    ) -> None:
        """--out writes a combined batch HTML report."""
        report_path = tmp_path / "batch_report.html"
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                [
                    "--batch", str(batch_yaml),
                    "-a", "gpt-4o",
                    "-b", "claude-3",
                    "--out", str(report_path),
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert report_path.exists()
        assert "<!DOCTYPE html>" in report_path.read_text(encoding="utf-8")

    def test_batch_invalid_yaml_exits_1(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        tmp_path: Path,
    ) -> None:
        """A file with unparseable YAML exits with code 1."""
        bad = tmp_path / "bad.yml"
        bad.write_text("[unclosed bracket", encoding="utf-8")
        with patch("llm_diff.cli.load_config", return_value=mock_cfg):
            result = runner.invoke(
                main,
                ["--batch", str(bad), "-a", "gpt-4o", "-b", "claude-3"],
            )
        assert result.exit_code == 1

    def test_batch_with_input_expansion(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        tmp_path: Path,
    ) -> None:
        """{input} placeholder is expanded with input file content."""
        inp = tmp_path / "data.txt"
        inp.write_text("sample data", encoding="utf-8")
        batch_yaml = tmp_path / "prompts.yml"
        batch_yaml.write_text(
            "prompts:\n"
            "  - id: summarize\n"
            "    text: 'Summarize: {input}'\n"
            f"    inputs: [{inp.name}]\n",
            encoding="utf-8",
        )
        with (
            patch(
                "llm_diff.cli.compare_models",
                new=AsyncMock(return_value=mock_comparison),
            ) as mock_cmp,
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["--batch", str(batch_yaml), "-a", "gpt-4o", "-b", "claude-3"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        call_kwargs = mock_cmp.call_args.kwargs
        assert "sample data" in call_kwargs["prompt_a"]

    def test_batch_semantic_flag_calls_scorer_per_item(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        batch_yaml: Path,
    ) -> None:
        """compute_semantic_similarity is called once per batch item with --semantic."""
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
            patch(
                "llm_diff.semantic.compute_semantic_similarity",
                return_value=0.75,
            ) as mock_sem,
        ):
            result = runner.invoke(
                main,
                ["--batch", str(batch_yaml), "-a", "gpt-4o", "-b", "claude-3", "--semantic"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert mock_sem.call_count == 2

    def test_batch_model_flag_accepted(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        batch_yaml: Path,
    ) -> None:
        """--model is accepted as shorthand for same-model batch comparison."""
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["--batch", str(batch_yaml), "--model", "gpt-4o"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_batch_shows_completion_summary(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        batch_yaml: Path,
    ) -> None:
        """Terminal output contains total item count after batch completes."""
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["--batch", str(batch_yaml), "-a", "gpt-4o", "-b", "claude-3"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "2" in result.output

    def test_batch_verbose_flag_shows_metadata(
        self,
        runner: CliRunner,
        mock_cfg: LLMDiffConfig,
        mock_comparison: ComparisonResult,
        batch_yaml: Path,
    ) -> None:
        """--verbose shows API metadata table for each batch item."""
        with (
            patch("llm_diff.cli.compare_models", new=AsyncMock(return_value=mock_comparison)),
            patch("llm_diff.cli.load_config", return_value=mock_cfg),
        ):
            result = runner.invoke(
                main,
                ["--batch", str(batch_yaml), "-a", "gpt-4o", "-b", "claude-3", "--verbose"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "Total tokens" in result.output
