"""Tests for llm_diff.providers — API abstraction layer."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_diff.config import LLMDiffConfig, ProviderConfig
from llm_diff.providers import (
    ComparisonResult,
    ModelResponse,
    _call_model,
    _make_client,
    _should_retry,
    compare_models,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def base_config() -> LLMDiffConfig:
    cfg = LLMDiffConfig()
    cfg.openai.api_key = "sk-test-openai-key"
    cfg.anthropic.api_key = "sk-ant-test-key"
    return cfg


@pytest.fixture()
def mock_response() -> MagicMock:
    """A minimal mock of an openai chat completion response."""
    choice = MagicMock()
    choice.message.content = "This is a test response."

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# _make_client
# ---------------------------------------------------------------------------


class TestMakeClient:
    def test_returns_async_openai_instance(self) -> None:
        from openai import AsyncOpenAI

        cfg = ProviderConfig(api_key="sk-test")
        client = _make_client(cfg)
        assert isinstance(client, AsyncOpenAI)

    def test_uses_base_url_when_set(self) -> None:
        cfg = ProviderConfig(api_key="sk-test", base_url="http://localhost:11434/v1")
        client = _make_client(cfg)
        assert str(client.base_url).startswith("http://localhost:11434")

    def test_no_key_uses_placeholder(self) -> None:
        """A missing key should not raise during client construction."""
        cfg = ProviderConfig(api_key="")
        client = _make_client(cfg)
        assert client is not None


# ---------------------------------------------------------------------------
# _should_retry
# ---------------------------------------------------------------------------


class TestShouldRetry:
    def test_rate_limit_error_retried(self) -> None:
        import openai

        exc = openai.RateLimitError("rate limit", response=MagicMock(), body={})
        assert _should_retry(exc) is True

    def test_value_error_not_retried(self) -> None:
        assert _should_retry(ValueError("bad input")) is False

    def test_connection_error_retried(self) -> None:
        import openai

        exc = openai.APIConnectionError(request=MagicMock())
        assert _should_retry(exc) is True

    def test_500_status_retried(self) -> None:
        import openai

        exc = openai.APIStatusError(
            "server error",
            response=MagicMock(status_code=500),
            body={},
        )
        assert _should_retry(exc) is True

    def test_400_status_not_retried(self) -> None:
        import openai

        exc = openai.APIStatusError(
            "bad request",
            response=MagicMock(status_code=400),
            body={},
        )
        assert _should_retry(exc) is False


# ---------------------------------------------------------------------------
# _call_model
# ---------------------------------------------------------------------------


class TestCallModel:
    @pytest.fixture()
    def provider_cfg(self) -> ProviderConfig:
        return ProviderConfig(api_key="sk-test")

    async def test_successful_call_returns_model_response(
        self, mock_response: MagicMock, provider_cfg: ProviderConfig
    ) -> None:
        with patch("llm_diff.providers._make_client") as mock_make_client:
            client_mock = AsyncMock()
            client_mock.chat.completions.create = AsyncMock(return_value=mock_response)
            client_mock.close = AsyncMock()
            mock_make_client.return_value = client_mock

            result = await _call_model(
                model="gpt-4o",
                prompt="Hello",
                provider_cfg=provider_cfg,
                provider_name="openai",
                temperature=0.7,
                max_tokens=100,
                timeout=30,
            )

        assert isinstance(result, ModelResponse)
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert result.text == "This is a test response."
        assert result.total_tokens == 30
        assert result.latency_ms >= 0

    async def test_timeout_raises_timeout_error(
        self, provider_cfg: ProviderConfig
    ) -> None:
        with patch("llm_diff.providers._make_client") as mock_make_client:
            client_mock = AsyncMock()
            client_mock.close = AsyncMock()

            async def slow(*args, **kwargs):  # noqa: ANN002, ANN003
                await asyncio.sleep(10)

            client_mock.chat.completions.create = slow
            mock_make_client.return_value = client_mock

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

    async def test_non_retryable_error_raised_immediately(
        self, provider_cfg: ProviderConfig
    ) -> None:
        import openai

        with patch("llm_diff.providers._make_client") as mock_make_client:
            client_mock = AsyncMock()
            client_mock.close = AsyncMock()
            client_mock.chat.completions.create = AsyncMock(
                side_effect=openai.AuthenticationError(
                    "invalid key", response=MagicMock(), body={}
                )
            )
            mock_make_client.return_value = client_mock

            with pytest.raises(openai.AuthenticationError):
                await _call_model(
                    model="gpt-4o",
                    prompt="Hello",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=30,
                )


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    async def test_returns_comparison_result(
        self,
        mock_response: MagicMock,
        base_config: LLMDiffConfig,
    ) -> None:
        with patch("llm_diff.providers._call_model") as mock_call:
            mock_resp_a = ModelResponse(
                model="gpt-4o",
                text="Response A",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=100.0,
                provider="openai",
            )
            mock_resp_b = ModelResponse(
                model="claude-3-5-sonnet",
                text="Response B",
                prompt_tokens=10,
                completion_tokens=6,
                total_tokens=16,
                latency_ms=120.0,
                provider="anthropic",
            )
            mock_call.side_effect = [mock_resp_a, mock_resp_b]

            result = await compare_models(
                prompt_a="Test prompt",
                prompt_b="Test prompt",
                model_a="gpt-4o",
                model_b="claude-3-5-sonnet",
                config=base_config,
            )

        assert isinstance(result, ComparisonResult)
        assert result.response_a.model == "gpt-4o"
        assert result.response_b.model == "claude-3-5-sonnet"

    async def test_calls_are_concurrent(
        self,
        base_config: LLMDiffConfig,
    ) -> None:
        """Both model calls must fire concurrently (asyncio.gather)."""
        call_order: list[str] = []

        async def fake_call(**kwargs):  # noqa: ANN003
            model = kwargs["model"]
            call_order.append(f"start:{model}")
            await asyncio.sleep(0.05)
            call_order.append(f"end:{model}")
            return ModelResponse(
                model=model,
                text="test",
                prompt_tokens=5,
                completion_tokens=5,
                total_tokens=10,
                latency_ms=50.0,
                provider="openai",
            )

        with patch("llm_diff.providers._call_model", side_effect=fake_call):
            await compare_models(
                prompt_a="Test",
                prompt_b="Test",
                model_a="gpt-4o",
                model_b="gpt-4-turbo",
                config=base_config,
            )

        # Both "start" events must appear before both "end" events,
        # proving they ran concurrently.
        start_indices = [i for i, e in enumerate(call_order) if e.startswith("start")]
        end_indices = [i for i, e in enumerate(call_order) if e.startswith("end")]
        assert max(start_indices) < min(end_indices), (
            f"Calls were not concurrent: {call_order}"
        )

    async def test_missing_api_key_raises_value_error(self) -> None:
        cfg = LLMDiffConfig()  # no keys set
        with pytest.raises(ValueError, match="No API key"):
            await compare_models(
                prompt_a="test",
                prompt_b="test",
                model_a="gpt-4o",  # requires openai key
                model_b="claude-3-5-sonnet",
                config=cfg,
            )


# ---------------------------------------------------------------------------
# ModelResponse.tokens property
# ---------------------------------------------------------------------------


class TestModelResponse:
    def test_tokens_property_equals_total_tokens(self) -> None:
        resp = ModelResponse(
            model="gpt-4o",
            text="hello",
            prompt_tokens=5,
            completion_tokens=15,
            total_tokens=20,
            latency_ms=50.0,
            provider="openai",
        )
        assert resp.tokens == 20
        assert resp.tokens == resp.total_tokens


# ---------------------------------------------------------------------------
# _call_model — retry loop and exhaustion
# ---------------------------------------------------------------------------


class TestCallModelRetry:
    @pytest.fixture()
    def provider_cfg(self) -> ProviderConfig:
        return ProviderConfig(api_key="sk-test")

    async def test_retryable_error_retried_then_succeeds(
        self,
        mock_response: MagicMock,
        provider_cfg: ProviderConfig,
    ) -> None:
        """First attempt raises RateLimitError; second attempt succeeds."""
        import openai

        call_count = 0

        async def flaky_create(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise openai.RateLimitError("rate limit", response=MagicMock(), body={})
            return mock_response

        with (
            patch("llm_diff.providers._make_client") as mock_make_client,
            patch("llm_diff.providers.asyncio.sleep", new=AsyncMock()),
        ):
            client_mock = AsyncMock()
            client_mock.chat.completions.create = flaky_create
            client_mock.close = AsyncMock()
            mock_make_client.return_value = client_mock

            result = await _call_model(
                model="gpt-4o",
                prompt="Hello",
                provider_cfg=provider_cfg,
                provider_name="openai",
                temperature=0.7,
                max_tokens=100,
                timeout=30,
            )

        assert isinstance(result, ModelResponse)
        assert call_count == 2

    async def test_all_retries_exhausted_raises_runtime_error(
        self, provider_cfg: ProviderConfig
    ) -> None:
        """Three consecutive retryable errors should raise RuntimeError."""
        import openai

        with (
            patch("llm_diff.providers._make_client") as mock_make_client,
            patch("llm_diff.providers.asyncio.sleep", new=AsyncMock()),
        ):
            client_mock = AsyncMock()
            client_mock.chat.completions.create = AsyncMock(
                side_effect=openai.RateLimitError("rate limit", response=MagicMock(), body={})
            )
            client_mock.close = AsyncMock()
            mock_make_client.return_value = client_mock

            with pytest.raises(RuntimeError, match="All .* attempts failed"):
                await _call_model(
                    model="gpt-4o",
                    prompt="Hello",
                    provider_cfg=provider_cfg,
                    provider_name="openai",
                    temperature=0.7,
                    max_tokens=100,
                    timeout=30,
                )
