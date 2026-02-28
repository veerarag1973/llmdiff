"""Response cache for llm-diff.

Caches :class:`~llm_diff.providers.ModelResponse` objects on disk, keyed by a
SHA-256 hash of ``(model, prompt, temperature, max_tokens)``.  Subsequent runs
with the same parameters return the cached response without making an API call
— useful for iterating on report formatting or diffing prompts repeatedly.

Cache layout::

    ~/.cache/llm-diff/
        {key[:2]}/
            {key}.json          ← ModelResponse serialised as JSON

Usage
-----
The cache is wired in automatically when the CLI runs.  Pass ``--no-cache``
to bypass both reads and writes.

Programmatic use::

    from llm_diff.cache import ResultCache

    cache = ResultCache()
    key = cache.make_key("gpt-4o", "Explain recursion", 0.7, 1024)
    cached = cache.get(key)  # None on miss
    if cached is None:
        response = await _call_model(...)
        cache.put(key, response)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ResultCache:
    """Disk-backed LLM response cache.

    Parameters
    ----------
    cache_dir:
        Override the default cache directory
        (``~/.cache/llm-diff/``).  Useful in tests.
    enabled:
        When *False* (e.g. ``--no-cache`` CLI flag), all :meth:`get` calls
        return ``None`` and all :meth:`put` calls are no-ops.  The attribute
        is exposed as :attr:`enabled` for introspection.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._dir: Path = (
            cache_dir if cache_dir is not None else Path.home() / ".cache" / "llm-diff"
        )
        self._enabled: bool = enabled

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """``True`` when cache reads and writes are active."""
        return self._enabled

    @property
    def cache_dir(self) -> Path:
        """Root directory where cached responses are stored."""
        return self._dir

    # ------------------------------------------------------------------
    # Cache key
    # ------------------------------------------------------------------

    def make_key(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return the SHA-256 hex digest for the given call parameters.

        The key is deterministic — the same ``(model, prompt, temperature,
        max_tokens)`` tuple always produces the same key.

        Parameters
        ----------
        model:
            Model identifier string, e.g. ``"gpt-4o"``.
        prompt:
            Full prompt text sent to the model.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens requested.

        Returns
        -------
        str
            64-character lowercase hexadecimal SHA-256 digest.
        """
        raw = f"{model}\x00{prompt}\x00{temperature}\x00{max_tokens}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str) -> object | None:
        """Return a cached :class:`~llm_diff.providers.ModelResponse` or ``None``.

        Returns ``None`` when:

        * The cache is disabled (``enabled=False``).
        * No cache file exists for *key*.
        * The cache file cannot be parsed (corrupt/truncated); in this case a
          warning is logged and the entry is treated as a miss.

        Parameters
        ----------
        key:
            Cache key produced by :meth:`make_key`.

        Returns
        -------
        ModelResponse | None
            The cached response, or ``None`` on any miss/error.
        """
        if not self._enabled:
            return None

        path = self._entry_path(key)
        if not path.is_file():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            from llm_diff.providers import ModelResponse  # noqa: PLC0415

            return ModelResponse(**data)
        except Exception:  # noqa: BLE001
            logger.warning("Cache entry for key %s is corrupt — ignoring.", key[:8])
            return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, key: str, response: object) -> None:
        """Serialise *response* to disk under *key*.

        Silently does nothing when the cache is disabled.

        Parameters
        ----------
        key:
            Cache key produced by :meth:`make_key`.
        response:
            A :class:`~llm_diff.providers.ModelResponse` instance to persist.
        """
        if not self._enabled:
            return

        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps(asdict(response)),  # type: ignore[call-overload]
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to write cache entry for key %s.", key[:8])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry_path(self, key: str) -> Path:
        """Return the filesystem path for cache entry *key*.

        Uses 2-character prefix sharding to avoid very large flat directories::

            {cache_dir}/{key[:2]}/{key}.json
        """
        return self._dir / key[:2] / f"{key}.json"
