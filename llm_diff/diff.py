"""Core diff engine for llm-diff.

Provides word-level diffing of two text strings using Python's stdlib
``difflib.SequenceMatcher``.  All output is expressed as a list of
:class:`DiffChunk` objects so the same data can be rendered to the
terminal (via ``rich``) or serialised to JSON without re-computation.

Performance target: < 100ms on 2,000-word inputs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DiffType(str, Enum):
    """The kind of change represented by a :class:`DiffChunk`."""

    EQUAL = "equal"
    INSERT = "insert"   # present in B, absent from A
    DELETE = "delete"   # present in A, absent from B


@dataclass(frozen=True)
class DiffChunk:
    """An atomic unit of a word-level diff."""

    type: DiffType
    text: str  # reconstructed text with original whitespace

    def to_dict(self) -> dict:
        return {"type": self.type.value, "text": self.text}


@dataclass
class DiffResult:
    """The full result of comparing two texts."""

    chunks: list[DiffChunk]
    similarity: float        # 0.0 – 1.0
    word_count_a: int
    word_count_b: int

    @property
    def similarity_pct(self) -> str:
        return f"{self.similarity * 100:.0f}%"

    def to_dict(self) -> dict:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "similarity_score": round(self.similarity, 4),
            "word_count_a": self.word_count_a,
            "word_count_b": self.word_count_b,
        }


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

# Splits on whitespace boundaries while keeping each token paired with the
# trailing whitespace that follows it, so we can reconstruct the original
# text exactly when joining.
_TOKEN_RE = re.compile(r"(\S+)(\s*)")


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Return ``[(word, trailing_space), ...]`` for every non-whitespace run.

    Trailing space includes the newline / spaces between this token and the
    next one, so joining ``word + trailing_space`` for all tokens faithfully
    reconstructs the original string.
    """
    return _TOKEN_RE.findall(text)


def _tokens_to_text(tokens: Sequence[tuple[str, str]]) -> str:
    """Reconstruct text from a sequence of ``(word, trailing_space)`` pairs."""
    return "".join(word + space for word, space in tokens)


# ---------------------------------------------------------------------------
# Diff computation
# ---------------------------------------------------------------------------


def word_diff(text_a: str, text_b: str) -> DiffResult:
    """Compute a word-level diff between *text_a* and *text_b*.

    Parameters
    ----------
    text_a:
        The "before" / model-A text.
    text_b:
        The "after" / model-B text.

    Returns
    -------
    DiffResult
        Structured diff output ready for terminal rendering or JSON export.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    words_a = [w for w, _ in tokens_a]
    words_b = [w for w, _ in tokens_b]

    matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)

    chunks: list[DiffChunk] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            chunks.append(
                DiffChunk(
                    type=DiffType.EQUAL,
                    text=_tokens_to_text(tokens_a[i1:i2]),
                )
            )
        elif tag == "insert":
            chunks.append(
                DiffChunk(
                    type=DiffType.INSERT,
                    text=_tokens_to_text(tokens_b[j1:j2]),
                )
            )
        elif tag == "delete":
            chunks.append(
                DiffChunk(
                    type=DiffType.DELETE,
                    text=_tokens_to_text(tokens_a[i1:i2]),
                )
            )
        elif tag == "replace":
            # Treat as delete-then-insert so callers always deal with
            # exactly three chunk types.
            chunks.append(
                DiffChunk(
                    type=DiffType.DELETE,
                    text=_tokens_to_text(tokens_a[i1:i2]),
                )
            )
            chunks.append(
                DiffChunk(
                    type=DiffType.INSERT,
                    text=_tokens_to_text(tokens_b[j1:j2]),
                )
            )

    # SequenceMatcher.ratio() is already computed internally; reuse it.
    similarity = matcher.ratio()

    return DiffResult(
        chunks=chunks,
        similarity=similarity,
        word_count_a=len(words_a),
        word_count_b=len(words_b),
    )


def compute_similarity(text_a: str, text_b: str) -> float:
    """Return the SequenceMatcher similarity ratio (0.0 – 1.0) without
    building the full diff.  Faster when you only need the score."""
    tokens_a = [w for w, _ in _tokenize(text_a)]
    tokens_b = [w for w, _ in _tokenize(text_b)]
    return SequenceMatcher(None, tokens_a, tokens_b, autojunk=False).ratio()
