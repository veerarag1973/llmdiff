"""Provider abstraction layer for llm-diff.

Handles all LLM API communication:
  - OpenAI-compatible endpoint calls (supports any provider via base_url)
  - Concurrent model-A / model-B requests (never sequential)
  - Exponential-backoff retry on rate-limit and transient errors
  - Timeout enforcement
  - Typed response objects
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

from llm_diff.config import LLMDiffConfig, ProviderConfig, get_provider_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0   # seconds — first retry delay
_RETRY_MAX_DELAY = 30.0   # seconds — cap on back-off
_RETRY_EXP_BASE = 2.0     # exponential multiplier


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelResponse:
    """The result of a single model call."""

    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    provider: str

    @property
    def tokens(self) -> int:
        return self.total_tokens


@dataclass
class ComparisonResult:
    """Paired responses from both models."""

    response_a: ModelResponse
    response_b: ModelResponse


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_client(provider_cfg: ProviderConfig) -> AsyncOpenAI:
    """Construct an :class:`AsyncOpenAI` client, never logging the API key."""
    kwargs: dict = {"api_key": provider_cfg.api_key or "sk-no-key"}
    if provider_cfg.base_url:
        kwargs["base_url"] = provider_cfg.base_url
    return AsyncOpenAI(**kwargs)


def _should_retry(exc: Exception) -> bool:
    """Return True if the error is transient and worth retrying."""
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code in {429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return False


async def _call_model(
    *,
    model: str,
    prompt: str,
    provider_cfg: ProviderConfig,
    provider_name: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> ModelResponse:
    """Send a single chat-completion request with retry logic.

    A fresh :class:`AsyncOpenAI` client is created for each attempt so
    that a closed connection from a previous (failed) attempt is never
    reused.

    Raises
    ------
    TimeoutError
        When the request exceeds *timeout* seconds (not retried).
    openai.APIError
        When all retry attempts are exhausted or the error is not retryable.
    """
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        client = _make_client(provider_cfg)
        try:
            start = time.monotonic()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=float(timeout),
            )
            elapsed_ms = (time.monotonic() - start) * 1_000

            if not response.choices:
                raise RuntimeError(
                    f"Model '{model}' returned an empty choices list. "
                    "Check that the model name is correct and the provider is responding."
                )
            choice = response.choices[0]
            if choice.message.content is None:
                raise RuntimeError(
                    f"Model '{model}' returned a null message content. "
                    "The provider may have refused the request or returned an incomplete response."
                )
            usage = response.usage

            return ModelResponse(
                model=model,
                text=(choice.message.content or "").strip(),
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                latency_ms=round(elapsed_ms, 1),
                provider=provider_name,
            )

        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                "Model '%s' timed out after %ss (attempt %d/%d)",
                model,
                timeout,
                attempt + 1,
                _MAX_RETRIES,
            )
            # Timeout is not retried — surface immediately as a clear error.
            raise TimeoutError(
                f"Request to model '{model}' timed out after {timeout}s."
            ) from exc

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if not _should_retry(exc):
                raise

            delay = min(_RETRY_BASE_DELAY * (_RETRY_EXP_BASE**attempt), _RETRY_MAX_DELAY)
            logger.warning(
                "Retryable error for model '%s' (attempt %d/%d): %s — retrying in %.1fs",
                model,
                attempt + 1,
                _MAX_RETRIES,
                type(exc).__name__,
                delay,
            )
            await asyncio.sleep(delay)

        finally:
            await client.close()

    # All retries exhausted
    raise RuntimeError(
        f"All {_MAX_RETRIES} attempts failed for model '{model}'."
    ) from last_exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compare_models(
    *,
    prompt_a: str,
    prompt_b: str,
    model_a: str,
    model_b: str,
    config: LLMDiffConfig,
) -> ComparisonResult:
    """Call both models **concurrently** and return a :class:`ComparisonResult`.

    *prompt_a* is sent to *model_a* and *prompt_b* is sent to *model_b*.
    Pass the same value for both prompts when diffing model behaviour
    rather than prompt variations.

    Both API calls are fired at the same time using :func:`asyncio.gather`;
    the total wall-clock time equals the *slower* of the two calls, not
    their sum.

    Raises
    ------
    RuntimeError
        If either call fails after all retry attempts.
    ValueError
        If a required API key is missing for either model.
    """
    # Validate and resolve provider details in one call each.
    provider_a_name, provider_a_cfg = _validate_provider(config, model_a)
    provider_b_name, provider_b_cfg = _validate_provider(config, model_b)

    task_a = _call_model(
        model=model_a,
        prompt=prompt_a,
        provider_cfg=provider_a_cfg,
        provider_name=provider_a_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )
    task_b = _call_model(
        model=model_b,
        prompt=prompt_b,
        provider_cfg=provider_b_cfg,
        provider_name=provider_b_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    response_a, response_b = await asyncio.gather(task_a, task_b)
    return ComparisonResult(response_a=response_a, response_b=response_b)


def _validate_provider(
    config: LLMDiffConfig, model: str
) -> tuple[str, ProviderConfig]:
    """Return (provider_name, cfg) and raise :class:`ValueError` if the API
    key for the inferred provider is missing."""
    provider_name, provider_cfg = get_provider_config(config, model)

    # Local / custom endpoints (e.g. Ollama) often require no key.
    needs_key = provider_name != "custom" or (
        provider_cfg.base_url is None or "localhost" not in provider_cfg.base_url
    )

    if needs_key and not provider_cfg.api_key:
        from llm_diff.config import _ENV_KEY_MAP  # local import — avoid circular

        env_var = _ENV_KEY_MAP.get(provider_name, "the appropriate API key env var")
        raise ValueError(
            f"No API key found for provider '{provider_name}' (model: '{model}'). "
            f"Set the {env_var} environment variable or add it to your .llmdiff config."
        )

    return provider_name, provider_cfg
