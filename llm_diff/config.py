"""Configuration loading for llm-diff.

Priority order (highest → lowest):
  1. Explicit CLI flags (injected at runtime)
  2. Project-level  .llmdiff  TOML in the current working directory
  3. User-level     ~/.llmdiff TOML in the home directory
  4. .env file in the current working directory
  5. Process environment variables
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# tomllib is stdlib in Python 3.11+; fall back to the third-party tomli package.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    try:  # pragma: no cover
        import tomli as tomllib  # type: ignore[no-redef]  # pragma: no cover
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Python < 3.11 requires the 'tomli' package: pip install tomli"
        ) from exc

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = ".llmdiff"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    """Credentials and endpoint for a single LLM provider."""

    api_key: str = ""
    base_url: str | None = None  # None → use the SDK default

    def __repr__(self) -> str:  # never leak the real key
        masked = f"{self.api_key[:6]}…" if len(self.api_key) > 6 else "***"
        return f"ProviderConfig(api_key='{masked}', base_url={self.base_url!r})"


@dataclass
class LLMDiffConfig:
    """Resolved, validated configuration for a single llm-diff invocation."""

    # --- per-provider credentials ---
    openai: ProviderConfig = field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = field(default_factory=ProviderConfig)
    groq: ProviderConfig = field(default_factory=ProviderConfig)
    mistral: ProviderConfig = field(default_factory=ProviderConfig)
    custom: ProviderConfig = field(default_factory=ProviderConfig)

    # --- request defaults ---
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30  # seconds

    # --- output defaults ---
    no_color: bool = False
    save: bool = False  # auto-save HTML to ./diffs/

    # --- telemetry (opt-in only) ---
    telemetry: bool = False

    def __repr__(self) -> str:
        return (
            f"LLMDiffConfig(temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}, timeout={self.timeout})"
        )


# ---------------------------------------------------------------------------
# Provider → env-var mapping
# ---------------------------------------------------------------------------

_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

_ENV_BASE_URL_MAP: dict[str, str] = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "groq": "GROQ_BASE_URL",
    "mistral": "MISTRAL_BASE_URL",
    "custom": "LLM_DIFF_CUSTOM_BASE_URL",
}


def _auto_detect_provider(model: str) -> str:
    """Infer the provider from a model identifier string.

    Returns one of: ``"openai"``, ``"anthropic"``, ``"groq"``,
    ``"mistral"``, ``"custom"``.
    """
    model_lower = model.lower()
    if model_lower.startswith(("gpt-", "o1", "o3", "text-davinci")):
        return "openai"
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith(("llama", "mixtral", "gemma")) and "groq" not in model_lower:
        # Groq hosts Llama / Mixtral / Gemma — but so does Together AI.
        # Return "groq" as the most common free option; callers can override.
        return "groq"
    if model_lower.startswith(("mistral", "mixtral")):
        return "mistral"
    return "custom"


# ---------------------------------------------------------------------------
# TOML file helpers
# ---------------------------------------------------------------------------


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Return the parsed TOML document, or ``{}`` if the file doesn't exist."""
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse %s: %s", path, exc)
        return {}


def _merge_toml(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts; *override* wins on scalar conflicts.

    Nested dicts are merged recursively so that, for example, setting a
    ``base_url`` in the project config does not wipe out the ``api_key``
    already loaded from the home config.
    """
    merged = {**base}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_toml(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(
    *,
    cwd: Path | None = None,
    home: Path | None = None,
    dotenv_path: Path | None = None,
) -> LLMDiffConfig:
    """Build a :class:`LLMDiffConfig` from all configuration sources.

    Parameters
    ----------
    cwd:
        Working directory to search for a project-level ``.llmdiff`` file.
        Defaults to :func:`pathlib.Path.cwd`.
    home:
        Home directory to search for a user-level ``.llmdiff`` file.
        Defaults to :func:`pathlib.Path.home`.
    dotenv_path:
        Path to the ``.env`` file to load.  Defaults to ``<cwd>/.env``.
    """
    cwd = cwd or Path.cwd()
    home = home or Path.home()

    # 1. Load .env (lowest priority; sets env vars before we read them)
    load_dotenv(dotenv_path=dotenv_path or (cwd / ".env"), override=False)

    # 2. Read TOML files — home first, then project (project wins)
    home_toml = _load_toml_file(home / _CONFIG_FILENAME)
    project_toml = _load_toml_file(cwd / _CONFIG_FILENAME)
    toml_data: dict[str, Any] = _merge_toml(home_toml, project_toml)

    # 3. Build config object
    cfg = LLMDiffConfig()

    # Request defaults from TOML
    if "defaults" in toml_data:
        defaults = toml_data["defaults"]
        cfg.temperature = float(defaults.get("temperature", cfg.temperature))
        cfg.max_tokens = int(defaults.get("max_tokens", cfg.max_tokens))
        cfg.timeout = int(defaults.get("timeout", cfg.timeout))
        cfg.no_color = bool(defaults.get("no_color", cfg.no_color))
        cfg.save = bool(defaults.get("save", cfg.save))
        cfg.telemetry = bool(defaults.get("telemetry", cfg.telemetry))

    # Provider credentials from TOML, then overridden by env vars
    for provider_name in ("openai", "anthropic", "groq", "mistral", "custom"):
        provider_cfg: ProviderConfig = getattr(cfg, provider_name)

        # TOML values
        toml_provider = toml_data.get("providers", {}).get(provider_name, {})
        if toml_provider.get("api_key"):
            provider_cfg.api_key = toml_provider["api_key"]
        if toml_provider.get("base_url"):
            provider_cfg.base_url = toml_provider["base_url"]

        # Env-var override (env vars always win over TOML)
        env_key_name = _ENV_KEY_MAP.get(provider_name)
        if env_key_name:
            env_key_value = os.environ.get(env_key_name, "")
            if env_key_value:
                provider_cfg.api_key = env_key_value

        env_url_name = _ENV_BASE_URL_MAP.get(provider_name)
        if env_url_name:
            env_url_value = os.environ.get(env_url_name, "")
            if env_url_value:
                provider_cfg.base_url = env_url_value

    return cfg


def get_provider_config(config: LLMDiffConfig, model: str) -> tuple[str, ProviderConfig]:
    """Return ``(provider_name, ProviderConfig)`` for the given model identifier.

    Auto-detects the provider from the model name.  Falls back to the
    ``custom`` provider if the model is unrecognised.
    """
    provider_name = _auto_detect_provider(model)
    provider_cfg = getattr(config, provider_name, config.custom)

    if not provider_cfg.api_key and provider_name != "custom":
        logger.debug(
            "No API key found for provider '%s'. "
            "Set the %s environment variable.",
            provider_name,
            _ENV_KEY_MAP.get(provider_name, "the appropriate API key env var"),
        )

    return provider_name, provider_cfg
