"""Embedding-based semantic similarity scoring for llm-diff.

Computes the cosine similarity between two text passages using a local
sentence-transformer model (``all-MiniLM-L6-v2``).  The model is downloaded
on first use (~80 MB) and cached by the ``sentence-transformers`` library.

The whole ``sentence-transformers`` package is **optional**.  Install it with::

    pip install "llm-diff[semantic]"

If the package is absent and semantic mode is invoked, a clear
:class:`ImportError` is raised rather than crashing silently.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# Module-level cache so the ~80 MB model is only loaded once per process.
_cached_model: object | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_model() -> object:
    """Return (or lazily load) the cached sentence-transformer model.

    Raises
    ------
    ImportError
        When ``sentence-transformers`` is not installed.
    """
    global _cached_model  # noqa: PLW0603
    if _cached_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Semantic diff requires the 'sentence-transformers' package.\n"
                "Install it with:  pip install 'llm-diff[semantic]'"
            ) from exc

        logger.debug("Loading sentence-transformer model '%s'", _SEMANTIC_MODEL_NAME)
        _cached_model = SentenceTransformer(_SEMANTIC_MODEL_NAME)
        logger.debug("Model loaded and cached.")

    return _cached_model


def _cosine_similarity(a, b) -> float:  # noqa: ANN001
    """Compute cosine similarity between two 1-D numpy arrays."""
    import numpy as np  # noqa: PLC0415

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Return the cosine similarity between *text_a* and *text_b* (0.0–1.0).

    The score is computed by encoding both texts with a local
    ``SentenceTransformer`` model and taking their cosine similarity.

    Parameters
    ----------
    text_a, text_b:
        The two response texts to compare.

    Returns
    -------
    float
        Similarity in the range ``[0.0, 1.0]``.  Clamped so that
        floating-point drift never produces values outside this range.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    """
    model = _get_model()
    # encode() is imported via the cached SentenceTransformer instance.
    embeddings = model.encode([text_a, text_b], convert_to_numpy=True)  # type: ignore[union-attr]
    similarity = _cosine_similarity(embeddings[0], embeddings[1])
    # Clamp to [0.0, 1.0] to guard against floating-point rounding (-1e-7 etc.)
    return max(0.0, min(1.0, similarity))


def reset_model_cache() -> None:
    """Clear the in-process model cache.

    Intended for tests that need to verify the lazy-load path without
    keeping state between test cases.
    """
    global _cached_model  # noqa: PLW0603
    _cached_model = None
