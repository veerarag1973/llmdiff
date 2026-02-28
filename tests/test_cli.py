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
            "batch": None,
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

    def test_batch_mode_raises_not_supported(self) -> None:
        import click
        with pytest.raises(click.UsageError, match="batch"):
            self._call(batch="prompts.yml", model_a="a", model_b="b")

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
        expected_keys = {"model_a", "model_b", "similarity_score", "tokens", "latency_ms", "diff"}
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
        assert "0.1.0" in result.output

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
        # --semantic flag sets mode but word diff is used as fallback in phase 1
        result = self._invoke_with_mocks(
            runner,
            ["Hello world", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--semantic"],
            mock_cfg,
            mock_comparison,
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

    def test_batch_flag_exits_1_not_supported(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main,
            ["--batch", "prompts.yml", "-a", "gpt-4o", "-b", "claude-3"],
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
