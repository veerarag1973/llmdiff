"""Integration tests — full pipeline end-to-end with mocked HTTP layer.

These tests exercise the complete CLI pipeline:
    CLI → config → providers (mocked AsyncOpenAI) → diff → render / report

All I/O is mocked at the AsyncOpenAI class level so no real API calls are
made, but every other component (diff engine, renderer, report builder)
runs with production code paths.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from llm_diff.cli import main
from llm_diff.config import LLMDiffConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESPONSE_A = "Recursion is when a function calls itself to solve smaller sub-problems."
_RESPONSE_B = "Recursion occurs when a function invokes itself repeatedly until a base case is met."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    openai_key: str = "sk-test-openai",
    anthropic_key: str = "sk-ant-test",
) -> LLMDiffConfig:
    cfg = LLMDiffConfig()
    cfg.openai.api_key = openai_key
    cfg.anthropic.api_key = anthropic_key
    return cfg


def _make_mock_client(responses: list[str]) -> tuple[MagicMock, list[dict]]:
    """Return (client_instance, call_log).

    *call_log* is appended to on each *chat.completions.create* call,
    recording the model name and returned text.
    """
    call_log: list[dict] = []
    counter = [0]

    async def fake_create(**kwargs) -> MagicMock:  # noqa: ANN003
        text = responses[counter[0] % len(responses)]
        counter[0] += 1
        call_log.append({"model": kwargs.get("model"), "text": text})

        choice = MagicMock()
        choice.message.content = text

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 20
        usage.total_tokens = 30

        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = usage
        return resp

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=fake_create)
    client.close = AsyncMock()
    return client, call_log


@contextmanager
def _mock_pipeline(
    responses: list[str] | None = None,
    cfg: LLMDiffConfig | None = None,
) -> Generator[list[dict], None, None]:
    """Context manager: patch AsyncOpenAI + load_config for integration tests.

    Yields the *call_log* list so tests can inspect which models were called.
    The ResultCache is disabled so cached responses from previous runs never
    interfere with the mocked API responses.
    """
    from llm_diff.cache import ResultCache as _ResultCache  # noqa: PLC0415

    if responses is None:
        responses = [_RESPONSE_A, _RESPONSE_B]
    if cfg is None:
        cfg = _make_cfg()

    client, call_log = _make_mock_client(responses)
    with (
        patch("llm_diff.providers.AsyncOpenAI", return_value=client),
        patch("llm_diff.cli.load_config", return_value=cfg),
        patch("llm_diff.cli.ResultCache", return_value=_ResultCache(enabled=False)),
    ):
        yield call_log


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Word mode
# ---------------------------------------------------------------------------


class TestIntegrationWordMode:
    def test_exits_zero_on_success(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_output_contains_model_a_name(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert "gpt-4o" in result.output

    def test_output_contains_model_b_name(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert "claude-3-5-sonnet" in result.output

    def test_output_contains_similarity_percentage(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert "%" in result.output

    def test_output_contains_prompt_text(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert "recursion" in result.output.lower()

    def test_identical_responses_show_100_percent(self, runner: CliRunner) -> None:
        same = "Exactly the same response from both models."
        with _mock_pipeline(responses=[same, same]):
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert "100%" in result.output

    def test_exactly_two_api_calls_are_made(self, runner: CliRunner) -> None:
        with _mock_pipeline() as call_log:
            runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert len(call_log) == 2

    def test_correct_model_names_sent_to_api(self, runner: CliRunner) -> None:
        with _mock_pipeline() as call_log:
            runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        models = {entry["model"] for entry in call_log}
        assert "gpt-4o" in models
        assert "claude-3-5-sonnet" in models

    def test_no_color_flag_suppresses_ansi_codes(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--no-color"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        # ANSI escape sequences start with ESC = \x1b
        assert "\x1b[" not in result.output

    def test_temperature_override_accepted(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--temperature", "0.5",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_max_tokens_override_accepted(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--max-tokens", "512",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_prompt_file_mode_word_diff(self, runner: CliRunner, tmp_path: Path) -> None:
        pa = tmp_path / "prompt_a.txt"
        pb = tmp_path / "prompt_b.txt"
        pa.write_text("Explain recursion simply.", encoding="utf-8")
        pb.write_text("Explain recursion in depth.", encoding="utf-8")
        with _mock_pipeline():
            result = runner.invoke(
                main,
                [
                    "--prompt-a", str(pa),
                    "--prompt-b", str(pb),
                    "--model", "gpt-4o",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# JSON mode
# ---------------------------------------------------------------------------


class TestIntegrationJsonMode:
    def _get_json_output(
        self,
        runner: CliRunner,
        extra_args: list[str] | None = None,
        responses: list[str] | None = None,
    ) -> tuple[dict, int]:
        args = ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--json"]
        if extra_args:
            args += extra_args
        with _mock_pipeline(responses=responses):
            result = runner.invoke(main, args, catch_exceptions=False)
        return json.loads(result.output), result.exit_code

    def test_output_is_valid_json(self, runner: CliRunner) -> None:
        data, code = self._get_json_output(runner)
        assert isinstance(data, dict)
        assert code == 0

    def test_json_has_required_keys(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        for key in ("model_a", "model_b", "similarity_score", "diff", "tokens", "latency_ms"):
            assert key in data, f"Missing key: {key!r}"

    def test_json_model_a_name_correct(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        assert data["model_a"] == "gpt-4o"

    def test_json_model_b_name_correct(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        assert data["model_b"] == "claude-3-5-sonnet"

    def test_json_similarity_is_float_in_range(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        assert isinstance(data["similarity_score"], float)
        assert 0.0 <= data["similarity_score"] <= 1.0

    def test_json_diff_is_list(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        assert isinstance(data["diff"], list)

    def test_json_diff_chunks_have_type_and_text(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        for chunk in data["diff"]:
            assert "type" in chunk
            assert "text" in chunk
            assert chunk["type"] in ("equal", "insert", "delete")

    def test_json_identical_responses_similarity_is_1(self, runner: CliRunner) -> None:
        same = "Identical response from both models."
        data, _ = self._get_json_output(runner, responses=[same, same])
        assert data["similarity_score"] == pytest.approx(1.0)

    def test_json_tokens_keys_present(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        assert "a" in data["tokens"]
        assert "b" in data["tokens"]

    def test_json_latency_keys_present(self, runner: CliRunner) -> None:
        data, _ = self._get_json_output(runner)
        assert "a" in data["latency_ms"]
        assert "b" in data["latency_ms"]

    def test_json_only_mode_flag_works(self, runner: CliRunner) -> None:
        """-j / --json flag equivalence."""
        with _mock_pipeline():
            r1 = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "-j"],
                catch_exceptions=False,
            )
        data = json.loads(r1.output)
        assert "model_a" in data


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


class TestIntegrationHtmlReport:
    def test_saves_html_report_when_out_specified(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "report.html"
        with _mock_pipeline():
            result = runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--out", str(out_file),
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert out_file.exists()

    def test_html_report_is_valid_markup(self, runner: CliRunner, tmp_path: Path) -> None:
        out_file = tmp_path / "report.html"
        with _mock_pipeline():
            runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--out", str(out_file),
                ],
                catch_exceptions=False,
            )
        html = out_file.read_text(encoding="utf-8")
        assert "<html" in html or "<!DOCTYPE html" in html

    def test_html_report_contains_model_names(self, runner: CliRunner, tmp_path: Path) -> None:
        out_file = tmp_path / "report.html"
        with _mock_pipeline():
            runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--out", str(out_file),
                ],
                catch_exceptions=False,
            )
        html = out_file.read_text(encoding="utf-8")
        assert "gpt-4o" in html
        assert "claude-3-5-sonnet" in html

    def test_html_report_terminal_output_mentions_saved(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        out_file = tmp_path / "report.html"
        with _mock_pipeline():
            result = runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--out", str(out_file),
                ],
                catch_exceptions=False,
            )
        assert "saved" in result.output.lower() or str(out_file.name) in result.output


# ---------------------------------------------------------------------------
# Verbose mode
# ---------------------------------------------------------------------------


class TestIntegrationVerboseMode:
    def test_verbose_exits_zero(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--verbose"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_verbose_shows_latency_metadata(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "-v"],
                catch_exceptions=False,
            )
        # Verbose output shows an API Metadata table with latency rows
        assert "Latency" in result.output or "latency" in result.output.lower()

    def test_verbose_shows_token_counts(self, runner: CliRunner) -> None:
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet", "--verbose"],
                catch_exceptions=False,
            )
        assert "token" in result.output.lower()


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


class TestIntegrationBatchMode:
    def _make_batch_file(self, tmp_path: Path, n_prompts: int = 2) -> Path:
        lines = ["prompts:"]
        prompts = [
            ("recursion", "Explain recursion"),
            ("sorting", "Explain bubble sort"),
            ("async", "Explain async/await"),
        ]
        for name, text in prompts[:n_prompts]:
            lines += [f"  - id: {name}", f"    text: '{text}'"]
        f = tmp_path / "prompts.yml"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f

    def test_batch_mode_exits_zero(self, runner: CliRunner, tmp_path: Path) -> None:
        batch_file = self._make_batch_file(tmp_path)
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["--batch", str(batch_file), "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_batch_mode_calls_api_twice_per_prompt(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        batch_file = self._make_batch_file(tmp_path, n_prompts=2)
        with _mock_pipeline() as call_log:
            runner.invoke(
                main,
                ["--batch", str(batch_file), "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        # 2 prompts × 2 models = 4 total API calls
        assert len(call_log) == 4

    def test_batch_mode_shows_batch_complete(self, runner: CliRunner, tmp_path: Path) -> None:
        batch_file = self._make_batch_file(tmp_path)
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["--batch", str(batch_file), "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert "Batch complete" in result.output or "prompt" in result.output.lower()

    def test_batch_mode_output_contains_prompt_ids(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        batch_file = self._make_batch_file(tmp_path)
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["--batch", str(batch_file), "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert "recursion" in result.output.lower()
        assert "sorting" in result.output.lower()

    def test_batch_mode_saves_html_report(self, runner: CliRunner, tmp_path: Path) -> None:
        batch_file = self._make_batch_file(tmp_path)
        out_file = tmp_path / "batch_report.html"
        with _mock_pipeline():
            result = runner.invoke(
                main,
                [
                    "--batch", str(batch_file),
                    "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--out", str(out_file),
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert out_file.exists()

    def test_batch_mode_requires_model_flags(self, runner: CliRunner, tmp_path: Path) -> None:
        batch_file = self._make_batch_file(tmp_path)
        with patch("llm_diff.cli.load_config", return_value=_make_cfg()):
            result = runner.invoke(main, ["--batch", str(batch_file)])
        assert result.exit_code != 0

    def test_batch_mode_with_input_templates(self, runner: CliRunner, tmp_path: Path) -> None:
        (tmp_path / "doc.txt").write_text("Machine learning content.", encoding="utf-8")
        yaml_content = (
            "prompts:\n"
            "  - id: summarise\n"
            "    text: 'Summarise: {input}'\n"
            "    inputs: [doc.txt]\n"
        )
        f = tmp_path / "prompts.yml"
        f.write_text(yaml_content, encoding="utf-8")
        with _mock_pipeline() as call_log:
            result = runner.invoke(
                main,
                ["--batch", str(f), "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        assert len(call_log) == 2  # 1 template × 1 input × 2 models

    def test_batch_mode_with_same_model_flag(self, runner: CliRunner, tmp_path: Path) -> None:
        batch_file = self._make_batch_file(tmp_path, n_prompts=1)
        with _mock_pipeline():
            result = runner.invoke(
                main,
                ["--batch", str(batch_file), "--model", "gpt-4o"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# --fail-under integration
# ---------------------------------------------------------------------------


class TestIntegrationFailUnder:
    def test_fail_under_exits_1_when_similarity_below_threshold(
        self, runner: CliRunner
    ) -> None:
        # Maximally different responses → similarity near 0
        with _mock_pipeline(
            responses=[
                "alpha beta gamma delta epsilon zeta.",
                "omega sigma theta kappa lambda upsilon.",
            ]
        ):
            result = runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--fail-under", "0.99",
                ],
            )
        assert result.exit_code == 1

    def test_fail_under_exits_0_when_similarity_at_or_above_threshold(
        self, runner: CliRunner
    ) -> None:
        same = "The exact same response from both models."
        with _mock_pipeline(responses=[same, same]):
            result = runner.invoke(
                main,
                [
                    "Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--fail-under", "0.5",
                ],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

    def test_batch_fail_under_exits_1_when_any_item_below(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        batch_file = tmp_path / "prompts.yml"
        batch_file.write_text(
            "prompts:\n  - id: test\n    text: Explain recursion\n",
            encoding="utf-8",
        )
        with _mock_pipeline(
            responses=[
                "alpha beta gamma.",
                "omega sigma theta kappa lambda upsilon.",
            ]
        ):
            result = runner.invoke(
                main,
                [
                    "--batch", str(batch_file),
                    "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                    "--fail-under", "0.99",
                ],
            )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestIntegrationErrorHandling:
    def test_missing_model_flags_exits_nonzero(self, runner: CliRunner) -> None:
        with patch("llm_diff.cli.load_config", return_value=_make_cfg()):
            result = runner.invoke(main, ["Explain recursion"])
        assert result.exit_code != 0

    def test_missing_api_key_exits_nonzero(self, runner: CliRunner) -> None:
        bad_cfg = LLMDiffConfig()  # no API keys
        client, _ = _make_mock_client([_RESPONSE_A, _RESPONSE_B])
        with (
            patch("llm_diff.providers.AsyncOpenAI", return_value=client),
            patch("llm_diff.cli.load_config", return_value=bad_cfg),
        ):
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
            )
        assert result.exit_code == 1

    def test_provider_runtime_error_exits_nonzero(self, runner: CliRunner) -> None:
        cfg = _make_cfg()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("All attempts failed")
        )
        client.close = AsyncMock()
        with (
            patch("llm_diff.providers.AsyncOpenAI", return_value=client),
            patch("llm_diff.cli.load_config", return_value=cfg),
        ):
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                 "--no-cache"],
            )
        assert result.exit_code == 1

    def test_nonexistent_prompt_file_exits_nonzero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        with patch("llm_diff.cli.load_config", return_value=_make_cfg()):
            result = runner.invoke(
                main,
                [
                    "--prompt-a", str(tmp_path / "missing_a.txt"),
                    "--prompt-b", str(tmp_path / "missing_b.txt"),
                    "--model", "gpt-4o",
                ],
            )
        assert result.exit_code != 0

    def test_timeout_error_shown_as_clean_message(self, runner: CliRunner) -> None:
        cfg = _make_cfg()
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            side_effect=TimeoutError("Request timed out after 30s.")
        )
        client.close = AsyncMock()
        with (
            patch("llm_diff.providers.AsyncOpenAI", return_value=client),
            patch("llm_diff.cli.load_config", return_value=cfg),
        ):
            result = runner.invoke(
                main,
                ["Explain recursion", "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                 "--no-cache"],
            )
        assert result.exit_code == 1

    def test_nonexistent_batch_file_exits_nonzero(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        with patch("llm_diff.cli.load_config", return_value=_make_cfg()):
            result = runner.invoke(
                main,
                [
                    "--batch", str(tmp_path / "missing.yml"),
                    "-a", "gpt-4o", "-b", "claude-3-5-sonnet",
                ],
            )
        assert result.exit_code == 1
