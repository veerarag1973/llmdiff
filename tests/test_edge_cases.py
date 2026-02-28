"""Edge-case tests — boundary conditions, unicode, performance, and timeouts.

These tests complement the unit tests in test_providers, test_diff, and
test_batch by focusing on unusual inputs that could cause crashes or poor UX:

  * Empty / whitespace-only model responses (new warning added in v0.8)
  * Unicode (CJK, Arabic, emoji, RTL) in LLM responses
  * Very long responses (10 000-word performance regression guard)
  * Network timeout message clarity
  * Malformed batch YAML surfaced through the CLI path
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_diff.config import LLMDiffConfig, ProviderConfig
from llm_diff.diff import DiffType, word_diff
from llm_diff.providers import _call_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider_cfg() -> ProviderConfig:
    return ProviderConfig(api_key="sk-test-key")


def _make_provider_call(
    content: str | None,
    prompt_tokens: int = 5,
    completion_tokens: int = 10,
) -> MagicMock:
    """Build a mock OpenAI response that returns *content* as message text."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Empty and whitespace-only model responses
# ---------------------------------------------------------------------------


class TestEmptyModelResponse:
    async def test_empty_string_response_returns_model_response(
        self, provider_cfg: ProviderConfig
    ) -> None:
        """Model returning '' must not crash — returns ModelResponse with text=''."""
        from llm_diff.providers import ModelResponse

        resp = _make_provider_call("")
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=resp)
            client.close = AsyncMock()
            mock_make.return_value = client

            result = await _call_model(
                model="gpt-4o",
                prompt="Test",
                provider_cfg=provider_cfg,
                provider_name="openai",
                temperature=0.7,
                max_tokens=100,
                timeout=30,
            )

        assert isinstance(result, ModelResponse)
        assert result.text == ""

    async def test_whitespace_only_response_is_stripped_to_empty(
        self, provider_cfg: ProviderConfig
    ) -> None:
        """Model returning '\\n  \\t  ' should result in text=''."""
        resp = _make_provider_call("\n\n  \t  ")
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=resp)
            client.close = AsyncMock()
            mock_make.return_value = client

            result = await _call_model(
                model="gpt-4o",
                prompt="Test",
                provider_cfg=provider_cfg,
                provider_name="openai",
                temperature=0.7,
                max_tokens=100,
                timeout=30,
            )

        assert result.text == ""

    async def test_empty_response_emits_warning_log(
        self,
        provider_cfg: ProviderConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An empty model response must emit a WARNING-level log entry."""
        import logging

        resp = _make_provider_call("")
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=resp)
            client.close = AsyncMock()
            mock_make.return_value = client

            with caplog.at_level(logging.WARNING, logger="llm_diff.providers"):
                await _call_model(
                    model="gpt-4o",
                    prompt="Test",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=30,
                )

        assert any(
            "empty" in record.message.lower() for record in caplog.records
        ), f"No warning about empty response found in logs: {caplog.records}"

    async def test_whitespace_response_emits_warning_log(
        self,
        provider_cfg: ProviderConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A pure-whitespace response must also trigger the warning."""
        import logging

        resp = _make_provider_call("   ")
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=resp)
            client.close = AsyncMock()
            mock_make.return_value = client

            with caplog.at_level(logging.WARNING, logger="llm_diff.providers"):
                await _call_model(
                    model="my-model",
                    prompt="Test",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=30,
                )

        assert any(
            "empty" in record.message.lower() for record in caplog.records
        )

    async def test_non_empty_response_does_not_warn(
        self,
        provider_cfg: ProviderConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A normal (non-empty) response must NOT trigger the empty-response warning."""
        import logging

        resp = _make_provider_call("A perfectly normal response.")
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=resp)
            client.close = AsyncMock()
            mock_make.return_value = client

            with caplog.at_level(logging.WARNING, logger="llm_diff.providers"):
                await _call_model(
                    model="gpt-4o",
                    prompt="Test",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=30,
                )

        warning_records = [
            r for r in caplog.records
            if "empty" in r.message.lower()
        ]
        assert len(warning_records) == 0

    def test_diff_of_two_empty_strings(self) -> None:
        """word_diff on two empty strings should return a valid DiffResult."""
        result = word_diff("", "")
        assert result.similarity == pytest.approx(1.0)
        assert result.word_count_a == 0
        assert result.word_count_b == 0
        assert result.chunks == []

    def test_diff_of_empty_a_vs_nonempty_b(self) -> None:
        result = word_diff("", "This is a response.")
        assert result.similarity == pytest.approx(0.0)
        assert result.word_count_a == 0
        assert result.word_count_b > 0

    def test_diff_of_nonempty_a_vs_empty_b(self) -> None:
        result = word_diff("This is a response.", "")
        assert result.similarity == pytest.approx(0.0)
        assert result.word_count_a > 0
        assert result.word_count_b == 0

    def test_diff_chunks_for_empty_a_are_all_inserts(self) -> None:
        result = word_diff("", "hello world")
        chunk_types = {c.type for c in result.chunks}
        assert DiffType.DELETE not in chunk_types
        assert DiffType.EQUAL not in chunk_types

    def test_diff_chunks_for_empty_b_are_all_deletes(self) -> None:
        result = word_diff("hello world", "")
        chunk_types = {c.type for c in result.chunks}
        assert DiffType.INSERT not in chunk_types
        assert DiffType.EQUAL not in chunk_types


# ---------------------------------------------------------------------------
# Unicode and emoji
# ---------------------------------------------------------------------------


class TestUnicodeAndEmoji:
    def test_diff_with_emoji_characters(self) -> None:
        """Emoji in responses must not crash the diff engine."""
        result = word_diff(
            "Great answer! 🎉 The model performs well. 🚀",
            "Excellent answer! 🎉 The model works great. ✅",
        )
        assert 0.0 <= result.similarity <= 1.0
        assert result.word_count_a > 0

    def test_diff_with_cjk_characters(self) -> None:
        """CJK (Chinese/Japanese/Korean) characters must be handled."""
        result = word_diff(
            "这是一个关于递归的解释。",
            "这是一个关于递归的详细解释。",
        )
        assert 0.0 <= result.similarity <= 1.0

    def test_diff_with_arabic_rtl_text(self) -> None:
        """Right-to-left Arabic text must not crash the diff engine."""
        result = word_diff(
            "التكرار هو عملية يستدعي فيها البرنامج نفسه.",
            "التكرار في البرمجة يعني استدعاء الدالة لنفسها.",
        )
        assert 0.0 <= result.similarity <= 1.0

    def test_diff_with_mixed_unicode_and_ascii(self) -> None:
        result = word_diff(
            "Python is great 👍 for machine learning (ML).",
            "Python is excellent 🌟 for deep learning (DL).",
        )
        assert 0.0 <= result.similarity <= 1.0

    def test_diff_with_newlines_and_tabs(self) -> None:
        result = word_diff(
            "First line.\nSecond line.\tTabbed.",
            "First line.\nSecond line.\tTabbed differently.",
        )
        # Most tokens match → similarity should be well above 0
        assert result.similarity > 0.5

    def test_diff_with_special_punctuation(self) -> None:
        result = word_diff(
            "Hello — world… (test) [bracket] {brace}",
            "Hello — world… (test) [bracket] {different}",
        )
        assert 0.0 <= result.similarity <= 1.0

    def test_diff_with_mathematical_symbols(self) -> None:
        result = word_diff(
            "The formula is O(n²) with σ = 0.5.",
            "The formula is O(n log n) with σ = 0.3.",
        )
        assert 0.0 <= result.similarity <= 1.0

    def test_diff_with_accented_european_characters(self) -> None:
        result = word_diff(
            "Ça va bien, merci. Über alles.",
            "Ça va très bien, merci. Über alles.",
        )
        assert result.similarity > 0.5

    def test_diff_reconstructs_text_with_emoji(self) -> None:
        """Joining diff chunks must exactly reproduce both originals."""
        text_a = "Recursion uses a 🔄 loop-like mechanism."
        text_b = "Iteration uses a ➿ loop-like mechanism."
        result = word_diff(text_a, text_b)
        reconstructed_a = "".join(
            c.text for c in result.chunks if c.type.value in ("equal", "delete")
        )
        reconstructed_b = "".join(
            c.text for c in result.chunks if c.type.value in ("equal", "insert")
        )
        assert reconstructed_a == text_a
        assert reconstructed_b == text_b

    def test_diff_with_null_byte_does_not_crash(self) -> None:
        """Embedded null bytes must not crash the diff engine."""
        result = word_diff("hello\x00world", "hello world")
        assert 0.0 <= result.similarity <= 1.0

    def test_diff_with_lone_emoji_tokens(self) -> None:
        """Responses consisting entirely of emoji should diff without error."""
        result = word_diff("🐍 🤖 🎯 🔥", "🐍 🤖 🎯 ✨")
        assert 0.0 <= result.similarity <= 1.0


# ---------------------------------------------------------------------------
# Very long responses (performance regression guard)
# ---------------------------------------------------------------------------


class TestVeryLongResponses:
    @staticmethod
    def _generate_long_text(word_count: int, seed: str = "word") -> str:
        return " ".join(f"{seed}{i}" for i in range(word_count))

    def test_10k_word_diff_completes_under_one_second(self) -> None:
        """word_diff on 10 000-word texts must finish in < 1 second."""
        text_a = self._generate_long_text(10_000, "alpha")
        text_b = self._generate_long_text(10_000, "beta")
        start = time.monotonic()
        result = word_diff(text_a, text_b)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"word_diff took {elapsed:.3f}s on 10k-word input"
        assert result.word_count_a == 10_000

    def test_5k_word_diff_produces_correct_word_counts(self) -> None:
        text_a = self._generate_long_text(5_000, "word")
        text_b = self._generate_long_text(5_000, "word")
        result = word_diff(text_a, text_b)
        assert result.word_count_a == 5_000
        assert result.word_count_b == 5_000

    def test_identical_long_responses_have_similarity_1(self) -> None:
        text = self._generate_long_text(5_000, "token")
        result = word_diff(text, text)
        assert result.similarity == pytest.approx(1.0)
        assert all(c.type == DiffType.EQUAL for c in result.chunks)

    def test_completely_different_long_responses_have_similarity_0(self) -> None:
        text_a = self._generate_long_text(1_000, "alpha")
        text_b = self._generate_long_text(1_000, "beta")
        result = word_diff(text_a, text_b)
        # Every token is unique → no equal chunks
        assert result.similarity == pytest.approx(0.0)

    def test_long_response_text_reconstruction_is_exact(self) -> None:
        """Joining diff chunks must exactly reproduce both 1000-word texts."""
        text_a = self._generate_long_text(1_000, "aaaa")
        text_b = self._generate_long_text(1_000, "bbbb")
        result = word_diff(text_a, text_b)
        reconstructed_a = "".join(
            c.text for c in result.chunks if c.type.value in ("equal", "delete")
        )
        reconstructed_b = "".join(
            c.text for c in result.chunks if c.type.value in ("equal", "insert")
        )
        assert reconstructed_a == text_a
        assert reconstructed_b == text_b

    def test_long_response_word_count_b_correct(self) -> None:
        text_a = self._generate_long_text(3_000, "a")
        text_b = self._generate_long_text(4_500, "b")
        result = word_diff(text_a, text_b)
        assert result.word_count_a == 3_000
        assert result.word_count_b == 4_500


# ---------------------------------------------------------------------------
# Network timeout error messages
# ---------------------------------------------------------------------------


class TestNetworkTimeoutMessages:
    async def test_timeout_raises_timeout_error(
        self, provider_cfg: ProviderConfig
    ) -> None:
        """asyncio.TimeoutError from wait_for must surface as TimeoutError."""
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.close = AsyncMock()

            async def slow_create(**kwargs):  # noqa: ANN003
                await asyncio.sleep(60)

            client.chat.completions.create = slow_create
            mock_make.return_value = client

            with pytest.raises(TimeoutError, match="timed out"):
                await _call_model(
                    model="gpt-4o",
                    prompt="Hello",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=1,
                )

    async def test_timeout_message_contains_model_name(
        self, provider_cfg: ProviderConfig
    ) -> None:
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.close = AsyncMock()

            async def slow_create(**kwargs):  # noqa: ANN003
                await asyncio.sleep(60)

            client.chat.completions.create = slow_create
            mock_make.return_value = client

            with pytest.raises(TimeoutError, match="gpt-4-turbo"):
                await _call_model(
                    model="gpt-4-turbo",
                    prompt="Hello",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=1,
                )

    async def test_timeout_message_contains_timeout_seconds(
        self, provider_cfg: ProviderConfig
    ) -> None:
        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.close = AsyncMock()

            async def slow_create(**kwargs):  # noqa: ANN003
                await asyncio.sleep(60)

            client.chat.completions.create = slow_create
            mock_make.return_value = client

            with pytest.raises(TimeoutError, match="5s"):
                await _call_model(
                    model="gpt-4o",
                    prompt="Hello",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=5,
                )

    async def test_timeout_is_not_retried(self, provider_cfg: ProviderConfig) -> None:
        """TimeoutError must not be retried (it is raised immediately)."""
        call_count = [0]

        with patch("llm_diff.providers._make_client") as mock_make:
            client = AsyncMock()
            client.close = AsyncMock()

            async def slow_create(**kwargs):  # noqa: ANN003
                call_count[0] += 1
                await asyncio.sleep(60)

            client.chat.completions.create = slow_create
            mock_make.return_value = client

            with pytest.raises(TimeoutError):
                await _call_model(
                    model="gpt-4o",
                    prompt="Hello",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=1,
                )

        # Only one attempt was made — TimeoutError is NOT retried
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Malformed YAML through the CLI path
# ---------------------------------------------------------------------------


class TestMalformedBatchYamlCli:
    """Test that malformed YAML batch files produce clean CLI error messages."""

    def _invoke_batch(self, runner, batch_path: str | Path) -> object:

        from llm_diff.cli import main

        cfg = LLMDiffConfig()
        cfg.openai.api_key = "sk-test"
        cfg.anthropic.api_key = "sk-ant-test"
        with patch("llm_diff.cli.load_config", return_value=cfg):
            return runner.invoke(
                main,
                ["--batch", str(batch_path), "-a", "gpt-4o", "-b", "claude-3-5-sonnet"],
            )

    def test_invalid_yaml_syntax_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        f = tmp_path / "bad.yml"
        f.write_text("[unclosed bracket", encoding="utf-8")
        result = self._invoke_batch(CliRunner(), f)
        assert result.exit_code == 1

    def test_missing_prompts_key_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        f = tmp_path / "bad.yml"
        f.write_text("something_else: []\n", encoding="utf-8")
        result = self._invoke_batch(CliRunner(), f)
        assert result.exit_code == 1

    def test_empty_prompts_list_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        f = tmp_path / "bad.yml"
        f.write_text("prompts: []\n", encoding="utf-8")
        result = self._invoke_batch(CliRunner(), f)
        assert result.exit_code == 1

    def test_batch_file_not_found_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        result = self._invoke_batch(CliRunner(), tmp_path / "nonexistent.yml")
        assert result.exit_code == 1

    def test_input_file_missing_exits_nonzero(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        f = tmp_path / "prompts.yml"
        f.write_text(
            "prompts:\n  - id: t\n    text: '{input}'\n    inputs: [missing.txt]\n",
            encoding="utf-8",
        )
        result = self._invoke_batch(CliRunner(), f)
        assert result.exit_code == 1
