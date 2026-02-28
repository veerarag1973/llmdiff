"""Embedding-based semantic similarity scoring for llm-diff.

Computes the cosine similarity between two text passages using a local
sentence-transformer model (``all-MiniLM-L6-v2``).  The model is downloaded
on first use (~80 MB) and cached by the ``sentence-transformers`` library.

The whole ``sentence-transformers`` package is **optional**.  Install it with::

    pip install "llm-diff[semantic]"

If the package is absent and semantic mode is invoked, a clear
:class:`ImportError` is raised rather than crashing silently.

Public API
----------
- :class:`ParagraphScore`              — similarity score for a single paragraph pair.
- :func:`compute_semantic_similarity`  — whole-text cosine similarity (0.0–1.0).
- :func:`compute_paragraph_similarity` — paragraph-level cosine similarity list.
- :func:`reset_model_cache`            — clear in-process model cache (tests only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParagraphScore:
    """Semantic similarity score for a single aligned paragraph pair.

    Attributes
    ----------
    text_a:
        Paragraph text from response A (may be empty if A has fewer paragraphs).
    text_b:
        Paragraph text from response B (may be empty if B has fewer paragraphs).
    score:
        Cosine similarity in ``[0.0, 1.0]``.  ``0.0`` when either paragraph is
        absent (one side has fewer paragraphs than the other).
    index:
        Zero-based paragraph index.
    """

    text_a: str
    text_b: str
    score: float
    index: int


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


def compute_paragraph_similarity(text_a: str, text_b: str) -> list[ParagraphScore]:
    """Return per-paragraph semantic similarity scores.

    Splits *text_a* and *text_b* on double-newlines (``\\n\\n``) to produce
    paragraph lists, then aligns each pair by index.  When one side has fewer
    paragraphs, the missing paragraphs are treated as empty strings and the
    corresponding score is ``0.0``.

    If either text contains no double-newline separators, the whole text is
    treated as a single paragraph and a one-element list is returned.

    Parameters
    ----------
    text_a, text_b:
        The two response texts to compare paragraph-by-paragraph.

    Returns
    -------
    list[ParagraphScore]
        One entry per aligned paragraph pair, ordered by index.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is not installed.
    """
    paras_a = [p.strip() for p in text_a.split("\n\n") if p.strip()]
    paras_b = [p.strip() for p in text_b.split("\n\n") if p.strip()]

    # Fall back to a zero score when neither side has parseable paragraphs
    # (which only occurs when both texts are empty/all-whitespace).
    if not paras_a or not paras_b:
        return [
            ParagraphScore(
                text_a=text_a.strip(),
                text_b=text_b.strip(),
                score=0.0,
                index=0,
            )
        ]

    # Pad the shorter side.
    max_len = max(len(paras_a), len(paras_b))
    paras_a += [""] * (max_len - len(paras_a))
    paras_b += [""] * (max_len - len(paras_b))

    scores: list[ParagraphScore] = []
    for i, (pa, pb) in enumerate(zip(paras_a, paras_b)):
        score = compute_semantic_similarity(pa, pb) if pa and pb else 0.0
        scores.append(ParagraphScore(text_a=pa, text_b=pb, score=score, index=i))

    return scores


def reset_model_cache() -> None:
    """Clear the in-process model cache.

    Intended for tests that need to verify the lazy-load path without
    keeping state between test cases.
    """
    global _cached_model  # noqa: PLW0603
    _cached_model = None
