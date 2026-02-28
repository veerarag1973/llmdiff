"""Tests for llm_diff.cache.ResultCache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_diff.cache import ResultCache
from llm_diff.providers import ModelResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> ResultCache:
    """Return a ResultCache backed by a temporary directory."""
    return ResultCache(cache_dir=tmp_path)


@pytest.fixture()
def sample_response() -> ModelResponse:
    return ModelResponse(
        model="gpt-4o",
        text="Hello, world!",
        prompt_tokens=5,
        completion_tokens=3,
        total_tokens=8,
        latency_ms=120.0,
        provider="openai",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_cache_dir(self) -> None:
        """Default cache_dir is ~/.cache/llm-diff/."""
        cache = ResultCache()
        expected = Path.home() / ".cache" / "llm-diff"
        assert cache.cache_dir == expected

    def test_custom_cache_dir(self, tmp_path: Path) -> None:
        custom = tmp_path / "my-cache"
        cache = ResultCache(cache_dir=custom)
        assert cache.cache_dir == custom

    def test_enabled_default(self) -> None:
        cache = ResultCache()
        assert cache.enabled is True

    def test_disabled(self, tmp_path: Path) -> None:
        cache = ResultCache(cache_dir=tmp_path, enabled=False)
        assert cache.enabled is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_enabled_property_true(self, tmp_cache: ResultCache) -> None:
        assert tmp_cache.enabled is True

    def test_cache_dir_property(self, tmp_path: Path) -> None:
        cache = ResultCache(cache_dir=tmp_path)
        assert cache.cache_dir is tmp_path


# ---------------------------------------------------------------------------
# make_key
# ---------------------------------------------------------------------------


class TestMakeKey:
    def test_returns_64_char_hex(self, tmp_cache: ResultCache) -> None:
        key = tmp_cache.make_key("gpt-4o", "prompt", 0.7, 1024)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_deterministic(self, tmp_cache: ResultCache) -> None:
        k1 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 512)
        k2 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 512)
        assert k1 == k2

    def test_different_models_produce_different_keys(self, tmp_cache: ResultCache) -> None:
        k1 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 512)
        k2 = tmp_cache.make_key("claude-3-5-sonnet", "foo", 0.5, 512)
        assert k1 != k2

    def test_different_prompts_produce_different_keys(self, tmp_cache: ResultCache) -> None:
        k1 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 512)
        k2 = tmp_cache.make_key("gpt-4o", "bar", 0.5, 512)
        assert k1 != k2

    def test_different_temperatures_produce_different_keys(self, tmp_cache: ResultCache) -> None:
        k1 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 512)
        k2 = tmp_cache.make_key("gpt-4o", "foo", 0.9, 512)
        assert k1 != k2

    def test_different_max_tokens_produce_different_keys(self, tmp_cache: ResultCache) -> None:
        k1 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 512)
        k2 = tmp_cache.make_key("gpt-4o", "foo", 0.5, 1024)
        assert k1 != k2


# ---------------------------------------------------------------------------
# get — disabled cache
# ---------------------------------------------------------------------------


class TestGetDisabled:
    def test_get_disabled_returns_none(
        self, tmp_path: Path, sample_response: ModelResponse
    ) -> None:
        cache = ResultCache(cache_dir=tmp_path, enabled=False)
        key = cache.make_key("gpt-4o", "prompt", 0.7, 1024)
        assert cache.get(key) is None

    def test_get_disabled_ignores_existing_file(
        self, tmp_path: Path, sample_response: ModelResponse
    ) -> None:
        """Even if a cache file exists, disabled cache should return None."""
        cache = ResultCache(cache_dir=tmp_path, enabled=False)
        key = cache.make_key("gpt-4o", "prompt", 0.7, 1024)
        # Manually write a cache file
        entry = tmp_path / key[:2] / f"{key}.json"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text(json.dumps({"model": "gpt-4o", "text": "hi",
                                      "prompt_tokens": 1, "completion_tokens": 1,
                                      "latency_ms": 10.0}), encoding="utf-8")
        assert cache.get(key) is None


# ---------------------------------------------------------------------------
# get — miss / hit
# ---------------------------------------------------------------------------


class TestGetMiss:
    def test_miss_returns_none(self, tmp_cache: ResultCache) -> None:
        key = tmp_cache.make_key("gpt-4o", "prompt", 0.7, 1024)
        assert tmp_cache.get(key) is None

    def test_nonexistent_key_returns_none(self, tmp_cache: ResultCache) -> None:
        assert tmp_cache.get("a" * 64) is None


class TestGetHit:
    def test_hit_returns_model_response(
        self, tmp_cache: ResultCache, sample_response: ModelResponse
    ) -> None:
        key = tmp_cache.make_key(
            sample_response.model, "Hi", 0.7, 1024
        )
        tmp_cache.put(key, sample_response)
        result = tmp_cache.get(key)
        assert result is not None
        assert isinstance(result, ModelResponse)
        assert result.text == sample_response.text
        assert result.model == sample_response.model

    def test_round_trip_preserves_all_fields(
        self, tmp_cache: ResultCache
    ) -> None:
        response = ModelResponse(
            model="claude-3-5-sonnet",
            text="Deep thought answer.",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            latency_ms=250.5,
            provider="anthropic",
        )
        key = tmp_cache.make_key("claude-3-5-sonnet", "What is 6×7?", 0.0, 256)
        tmp_cache.put(key, response)
        got = tmp_cache.get(key)
        assert got is not None
        assert got.model == response.model
        assert got.text == response.text
        assert got.prompt_tokens == response.prompt_tokens
        assert got.completion_tokens == response.completion_tokens
        assert abs(got.latency_ms - response.latency_ms) < 0.001


# ---------------------------------------------------------------------------
# get — corrupt / unreadable entries
# ---------------------------------------------------------------------------


class TestGetCorrupt:
    def test_corrupt_json_returns_none(self, tmp_cache: ResultCache) -> None:
        key = tmp_cache.make_key("gpt-4o", "p", 0.7, 100)
        entry = tmp_cache._entry_path(key)
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("NOT VALID JSON {{{", encoding="utf-8")
        assert tmp_cache.get(key) is None

    def test_corrupt_json_logs_warning(
        self, tmp_cache: ResultCache, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        key = tmp_cache.make_key("gpt-4o", "p", 0.7, 100)
        entry = tmp_cache._entry_path(key)
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("{}", encoding="utf-8")  # valid JSON but wrong shape
        with caplog.at_level(logging.WARNING, logger="llm_diff.cache"):
            result = tmp_cache.get(key)
        assert result is None
        assert any("corrupt" in rec.message.lower() for rec in caplog.records)

    def test_truncated_file_returns_none(self, tmp_cache: ResultCache) -> None:
        key = tmp_cache.make_key("gpt-4o", "q", 0.5, 50)
        entry = tmp_cache._entry_path(key)
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_bytes(b'{"model": "gpt-4o", "text"')  # truncated
        assert tmp_cache.get(key) is None


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


class TestPut:
    def test_put_creates_file(
        self, tmp_cache: ResultCache, sample_response: ModelResponse
    ) -> None:
        key = tmp_cache.make_key("gpt-4o", "test", 0.7, 512)
        tmp_cache.put(key, sample_response)
        assert tmp_cache._entry_path(key).is_file()

    def test_put_creates_subdirectory(
        self, tmp_cache: ResultCache, sample_response: ModelResponse
    ) -> None:
        key = tmp_cache.make_key("gpt-4o", "test", 0.7, 512)
        tmp_cache.put(key, sample_response)
        subdir = tmp_cache.cache_dir / key[:2]
        assert subdir.is_dir()

    def test_put_disabled_is_noop(
        self, tmp_path: Path, sample_response: ModelResponse
    ) -> None:
        cache = ResultCache(cache_dir=tmp_path, enabled=False)
        key = cache.make_key("gpt-4o", "test", 0.7, 512)
        cache.put(key, sample_response)
        assert not cache._entry_path(key).exists()

    def test_put_disabled_does_not_create_dir(
        self, tmp_path: Path, sample_response: ModelResponse
    ) -> None:
        cache = ResultCache(cache_dir=tmp_path, enabled=False)
        key = cache.make_key("gpt-4o", "test", 0.7, 512)
        cache.put(key, sample_response)
        assert not (tmp_path / key[:2]).exists()

    def test_put_overwrites_existing_entry(
        self, tmp_cache: ResultCache
    ) -> None:
        key = tmp_cache.make_key("gpt-4o", "x", 0.7, 512)
        r1 = ModelResponse(model="gpt-4o", text="v1", prompt_tokens=1,
                           completion_tokens=1, total_tokens=2,
                           latency_ms=10.0, provider="openai")
        r2 = ModelResponse(model="gpt-4o", text="v2", prompt_tokens=2,
                           completion_tokens=2, total_tokens=4,
                           latency_ms=20.0, provider="openai")
        tmp_cache.put(key, r1)
        tmp_cache.put(key, r2)
        got = tmp_cache.get(key)
        assert got is not None
        assert got.text == "v2"

    def test_put_silently_warns_on_write_error(
        self,
        tmp_cache: ResultCache,
        sample_response: ModelResponse,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """put() swallows write errors and logs a warning."""
        import logging
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        key = tmp_cache.make_key("gpt-4o", "test", 0.7, 512)

        fake_parent = MagicMock()
        fake_path = MagicMock(spec=Path)
        fake_path.parent = fake_parent
        fake_path.write_text.side_effect = OSError("disk full")

        with patch.object(tmp_cache, "_entry_path", return_value=fake_path):
            with caplog.at_level(logging.WARNING, logger="llm_diff.cache"):
                tmp_cache.put(key, sample_response)

        assert any("Failed to write" in r.message for r in caplog.records)

    def test_put_writes_valid_json(
        self, tmp_cache: ResultCache, sample_response: ModelResponse
    ) -> None:
        key = tmp_cache.make_key("gpt-4o", "test", 0.7, 512)
        tmp_cache.put(key, sample_response)
        raw = tmp_cache._entry_path(key).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert data["model"] == "gpt-4o"
        assert data["text"] == "Hello, world!"


# ---------------------------------------------------------------------------
# _entry_path (internal helper)
# ---------------------------------------------------------------------------


class TestEntryPath:
    def test_path_uses_2char_shard(self, tmp_path: Path) -> None:
        cache = ResultCache(cache_dir=tmp_path)
        key = "abcdef1234567890" + "0" * 48  # 64-char key
        expected = tmp_path / "ab" / f"{key}.json"
        assert cache._entry_path(key) == expected

    def test_path_under_cache_dir(self, tmp_path: Path) -> None:
        cache = ResultCache(cache_dir=tmp_path)
        key = "ff" + "0" * 62
        path = cache._entry_path(key)
        assert str(path).startswith(str(tmp_path))

    def test_different_keys_have_different_paths(self, tmp_cache: ResultCache) -> None:
        k1 = tmp_cache.make_key("m1", "p", 0.0, 1)
        k2 = tmp_cache.make_key("m2", "p", 0.0, 1)
        assert tmp_cache._entry_path(k1) != tmp_cache._entry_path(k2)


# ---------------------------------------------------------------------------
# Full round-trip integration
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_miss_then_put_then_hit(
        self, tmp_cache: ResultCache, sample_response: ModelResponse
    ) -> None:
        key = tmp_cache.make_key("gpt-4o", "round-trip", 0.7, 512)
        assert tmp_cache.get(key) is None  # miss
        tmp_cache.put(key, sample_response)  # store
        result = tmp_cache.get(key)  # hit
        assert result is not None
        assert result.text == sample_response.text

    def test_multiple_keys_independent(self, tmp_cache: ResultCache) -> None:
        r1 = ModelResponse(model="m1", text="a", prompt_tokens=1,
                           completion_tokens=1, total_tokens=2,
                           latency_ms=1.0, provider="openai")
        r2 = ModelResponse(model="m2", text="b", prompt_tokens=2,
                           completion_tokens=2, total_tokens=4,
                           latency_ms=2.0, provider="anthropic")
        k1 = tmp_cache.make_key("m1", "prompt", 0.0, 1)
        k2 = tmp_cache.make_key("m2", "prompt", 0.0, 1)
        tmp_cache.put(k1, r1)
        tmp_cache.put(k2, r2)
        assert tmp_cache.get(k1).text == "a"  # type: ignore[union-attr]
        assert tmp_cache.get(k2).text == "b"  # type: ignore[union-attr]

    def test_cache_persists_across_instances(
        self, tmp_path: Path, sample_response: ModelResponse
    ) -> None:
        """Data written by one ResultCache instance is readable by another."""
        c1 = ResultCache(cache_dir=tmp_path)
        key = c1.make_key("gpt-4o", "persist", 0.7, 512)
        c1.put(key, sample_response)

        c2 = ResultCache(cache_dir=tmp_path)
        result = c2.get(key)
        assert result is not None
        assert result.text == sample_response.text
