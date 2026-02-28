"""Tests for llm_diff.config — configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_diff.config import (
    LLMDiffConfig,
    ProviderConfig,
    _auto_detect_provider,
    _merge_toml,
    get_provider_config,
    load_config,
)

# ---------------------------------------------------------------------------
# _auto_detect_provider
# ---------------------------------------------------------------------------


class TestAutoDetectProvider:
    @pytest.mark.parametrize(
        "model, expected",
        [
            ("gpt-4o", "openai"),
            ("gpt-4-turbo", "openai"),
            ("o1-preview", "openai"),
            ("o3-mini", "openai"),
            ("claude-3-5-sonnet", "anthropic"),
            ("claude-3-opus", "anthropic"),
            ("mistral-large", "mistral"),
            ("mixtral-8x7b", "groq"),
            ("llama-3-70b", "groq"),
            ("totally-unknown-model", "custom"),
        ],
    )
    def test_detection(self, model: str, expected: str) -> None:
        assert _auto_detect_provider(model) == expected


# ---------------------------------------------------------------------------
# _merge_toml
# ---------------------------------------------------------------------------


class TestMergeToml:
    def test_override_wins(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _merge_toml(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_dict_merged(self) -> None:
        base = {"providers": {"openai": {"api_key": "base-key"}}}
        override = {"providers": {"openai": {"base_url": "http://localhost"}}}
        result = _merge_toml(base, override)
        assert result["providers"]["openai"]["api_key"] == "base-key"
        assert result["providers"]["openai"]["base_url"] == "http://localhost"

    def test_base_unchanged(self) -> None:
        base = {"x": 1}
        _merge_toml(base, {"y": 2})
        assert "y" not in base


# ---------------------------------------------------------------------------
# ProviderConfig repr — key masking
# ---------------------------------------------------------------------------


class TestProviderConfigRepr:
    def test_key_is_masked_in_repr(self) -> None:
        cfg = ProviderConfig(api_key="sk-supersecretkey123")
        r = repr(cfg)
        assert "supersecretkey123" not in r
        assert "sk-sup" in r  # first 6 chars present

    def test_short_key_fully_masked(self) -> None:
        cfg = ProviderConfig(api_key="abc")
        assert "abc" not in repr(cfg)  # short key → fully masked as "***"


# ---------------------------------------------------------------------------
# load_config — env vars
# ---------------------------------------------------------------------------


class TestLoadConfigEnvVars:
    def test_openai_key_loaded_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-openai")
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.openai.api_key == "sk-env-openai"

    def test_anthropic_key_loaded_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.anthropic.api_key == "sk-ant-env"

    def test_env_var_overrides_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml_content = b'[providers.openai]\napi_key = "sk-from-toml"\n'
        (tmp_path / ".llmdiff").write_bytes(toml_content)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.openai.api_key == "sk-from-env"


# ---------------------------------------------------------------------------
# load_config — TOML file
# ---------------------------------------------------------------------------


class TestLoadConfigToml:
    def test_project_toml_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        content = b'[providers.openai]\napi_key = "sk-toml-key"\n'
        (tmp_path / ".llmdiff").write_bytes(content)

        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.openai.api_key == "sk-toml-key"

    def test_defaults_section_applied(self, tmp_path: Path) -> None:
        content = b"[defaults]\ntemperature = 0.3\nmax_tokens = 512\ntimeout = 15\n"
        (tmp_path / ".llmdiff").write_bytes(content)

        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.temperature == pytest.approx(0.3)
        assert cfg.max_tokens == 512
        assert cfg.timeout == 15

    def test_project_toml_overrides_home_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        (home_dir / ".llmdiff").write_bytes(
            b'[providers.openai]\napi_key = "sk-home"\n'
        )
        (project_dir / ".llmdiff").write_bytes(
            b'[providers.openai]\napi_key = "sk-project"\n'
        )

        cfg = load_config(cwd=project_dir, home=home_dir)
        assert cfg.openai.api_key == "sk-project"

    def test_missing_toml_is_graceful(self, tmp_path: Path) -> None:
        """No .llmdiff file → should not raise."""
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert isinstance(cfg, LLMDiffConfig)

    def test_malformed_toml_is_graceful(self, tmp_path: Path) -> None:
        (tmp_path / ".llmdiff").write_bytes(b"not valid toml ][[\n")
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert isinstance(cfg, LLMDiffConfig)


# ---------------------------------------------------------------------------
# load_config — .env file
# ---------------------------------------------------------------------------


class TestLoadConfigDotEnv:
    def test_dotenv_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")

        cfg = load_config(cwd=tmp_path, home=tmp_path, dotenv_path=dotenv_file)
        assert cfg.openai.api_key == "sk-from-dotenv"

    def test_existing_env_var_not_overridden_by_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-already-set")
        dotenv_file = tmp_path / ".env"
        dotenv_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n", encoding="utf-8")

        cfg = load_config(cwd=tmp_path, home=tmp_path, dotenv_path=dotenv_file)
        # dotenv loads with override=False, so env var wins
        assert cfg.openai.api_key == "sk-already-set"


# ---------------------------------------------------------------------------
# get_provider_config
# ---------------------------------------------------------------------------


class TestGetProviderConfig:
    def test_gpt4_returns_openai(self) -> None:
        cfg = LLMDiffConfig()
        cfg.openai.api_key = "sk-openai"
        name, provider = get_provider_config(cfg, "gpt-4o")
        assert name == "openai"
        assert provider.api_key == "sk-openai"

    def test_claude_returns_anthropic(self) -> None:
        cfg = LLMDiffConfig()
        cfg.anthropic.api_key = "sk-anthropic"
        name, provider = get_provider_config(cfg, "claude-3-5-sonnet")
        assert name == "anthropic"
        assert provider.api_key == "sk-anthropic"

    def test_unknown_model_returns_custom(self) -> None:
        cfg = LLMDiffConfig()
        name, _ = get_provider_config(cfg, "some-random-model")
        assert name == "custom"


# ---------------------------------------------------------------------------
# LLMDiffConfig.__repr__
# ---------------------------------------------------------------------------


class TestLLMDiffConfigRepr:
    def test_repr_contains_temperature(self) -> None:
        cfg = LLMDiffConfig()
        r = repr(cfg)
        assert "temperature" in r

    def test_repr_contains_max_tokens(self) -> None:
        cfg = LLMDiffConfig()
        r = repr(cfg)
        assert "max_tokens" in r

    def test_repr_contains_timeout(self) -> None:
        cfg = LLMDiffConfig()
        r = repr(cfg)
        assert "timeout" in r


# ---------------------------------------------------------------------------
# load_config — base_url from TOML and env var
# ---------------------------------------------------------------------------


class TestLoadConfigBaseUrl:
    def test_base_url_loaded_from_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        toml_content = (
            '[providers.openai]\n'
            'api_key = "sk-toml"\n'
            'base_url = "https://my-openai-proxy.example.com/v1"\n'
        )
        (tmp_path / ".llmdiff").write_text(toml_content, encoding="utf-8")
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.openai.base_url == "https://my-openai-proxy.example.com/v1"

    def test_base_url_loaded_from_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env-proxy.example.com/v1")
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.openai.base_url == "https://env-proxy.example.com/v1"

    def test_env_var_base_url_overrides_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env-wins.example.com/v1")
        toml_content = (
            '[providers.openai]\n'
            'base_url = "https://toml-loses.example.com/v1"\n'
        )
        (tmp_path / ".llmdiff").write_text(toml_content, encoding="utf-8")
        cfg = load_config(cwd=tmp_path, home=tmp_path)
        assert cfg.openai.base_url == "https://env-wins.example.com/v1"
