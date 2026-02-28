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


# ---------------------------------------------------------------------------
# JSON structural diff
# ---------------------------------------------------------------------------

import json as _json  # noqa: E402  — stdlib, safe to import here
from enum import Enum as _Enum


class JsonChangeType(str, _Enum):
    """The kind of structural change at a JSON key."""

    ADDED = "added"        # key present in B, absent from A
    REMOVED = "removed"    # key present in A, absent from B
    CHANGED = "changed"    # key present in both, value differs
    TYPE_CHANGE = "type_change"   # key present in both, types differ
    UNCHANGED = "unchanged"       # key present in both, value identical


@dataclass(frozen=True)
class JsonDiffEntry:
    """A single key-level change in a JSON structural diff.

    Attributes
    ----------
    path:
        Dot-separated key path (e.g. ``"user.address.city"``).
    change_type:
        The kind of change (:class:`JsonChangeType`).
    value_a:
        The value from document A (``None`` if the key was absent in A).
    value_b:
        The value from document B (``None`` if the key was absent in B).
    type_a:
        Python type name of *value_a*.
    type_b:
        Python type name of *value_b*.
    """

    path: str
    change_type: JsonChangeType
    value_a: object = None
    value_b: object = None
    type_a: str = ""
    type_b: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "change": self.change_type.value,
            "type_a": self.type_a,
            "type_b": self.type_b,
            "value_a": self.value_a,
            "value_b": self.value_b,
        }


@dataclass
class JsonStructDiffResult:
    """Full result of a structural JSON diff between two documents.

    Attributes
    ----------
    entries:
        All key-level diff entries, in depth-first traversal order.
    is_valid_json_a, is_valid_json_b:
        Whether each input was valid JSON.  When either is ``False``, the
        diff falls back to a word-level diff via :attr:`word_diff_result`.
    word_diff_result:
        Word-level diff, populated as a fallback only when JSON parsing fails.
    """

    entries: list[JsonDiffEntry]
    is_valid_json_a: bool = True
    is_valid_json_b: bool = True
    word_diff_result: DiffResult | None = None

    @property
    def added(self) -> list[JsonDiffEntry]:
        return [e for e in self.entries if e.change_type == JsonChangeType.ADDED]

    @property
    def removed(self) -> list[JsonDiffEntry]:
        return [e for e in self.entries if e.change_type == JsonChangeType.REMOVED]

    @property
    def changed(self) -> list[JsonDiffEntry]:
        return [e for e in self.entries if e.change_type in {
            JsonChangeType.CHANGED, JsonChangeType.TYPE_CHANGE
        }]

    @property
    def unchanged(self) -> list[JsonDiffEntry]:
        return [e for e in self.entries if e.change_type == JsonChangeType.UNCHANGED]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def summary(self) -> dict:
        return {
            "added": len(self.added),
            "removed": len(self.removed),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
        }

    def to_dict(self) -> dict:
        return {
            "is_valid_json_a": self.is_valid_json_a,
            "is_valid_json_b": self.is_valid_json_b,
            "summary": self.summary(),
            "entries": [e.to_dict() for e in self.entries],
        }


def _flatten_json(obj: object, prefix: str = "") -> dict[str, object]:
    """Recursively flatten a parsed JSON object into ``{dotted.path: value}`` pairs.

    Lists are handled order-sensitively: each element becomes a numeric key
    (e.g. ``"items.0"``, ``"items.1"``).
    """
    result: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                result.update(_flatten_json(value, full_key))
            else:
                result[full_key] = value
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            full_key = f"{prefix}.{idx}" if prefix else str(idx)
            if isinstance(value, (dict, list)):
                result.update(_flatten_json(value, full_key))
            else:
                result[full_key] = value
    else:
        # Scalar at the root — unusual but handle it
        if prefix:
            result[prefix] = obj
    return result


def json_struct_diff(text_a: str, text_b: str) -> JsonStructDiffResult:
    """Diff two JSON strings at the key/value level.

    Both inputs are parsed as JSON; if parsing fails for either, a fallback
    word-level diff is used instead (see :attr:`JsonStructDiffResult.word_diff_result`).

    The comparison is performed on a *flattened* key space where nested
    objects are represented with dot-separated paths (e.g. ``"user.name"``).
    Arrays are indexed numerically (``"items.0"``, ``"items.1"``).

    Parameters
    ----------
    text_a:
        JSON string from model/document A.
    text_b:
        JSON string from model/document B.

    Returns
    -------
    JsonStructDiffResult
        Structured diff result with per-key change entries.
    """
    valid_a, valid_b = True, True
    parsed_a: object = None
    parsed_b: object = None

    try:
        parsed_a = _json.loads(text_a)
    except (_json.JSONDecodeError, ValueError):
        valid_a = False

    try:
        parsed_b = _json.loads(text_b)
    except (_json.JSONDecodeError, ValueError):
        valid_b = False

    if not valid_a or not valid_b:
        # Fallback: word-level diff
        return JsonStructDiffResult(
            entries=[],
            is_valid_json_a=valid_a,
            is_valid_json_b=valid_b,
            word_diff_result=word_diff(text_a, text_b),
        )

    flat_a = _flatten_json(parsed_a)
    flat_b = _flatten_json(parsed_b)

    all_keys = sorted(set(flat_a.keys()) | set(flat_b.keys()))
    entries: list[JsonDiffEntry] = []

    for path in all_keys:
        in_a = path in flat_a
        in_b = path in flat_b
        val_a = flat_a.get(path)
        val_b = flat_b.get(path)
        type_a = type(val_a).__name__ if val_a is not None else ""
        type_b = type(val_b).__name__ if val_b is not None else ""

        if in_a and not in_b:
            change = JsonChangeType.REMOVED
        elif in_b and not in_a:
            change = JsonChangeType.ADDED
        elif type(val_a) is not type(val_b):
            change = JsonChangeType.TYPE_CHANGE
        elif val_a == val_b:
            change = JsonChangeType.UNCHANGED
        else:
            change = JsonChangeType.CHANGED

        entries.append(
            JsonDiffEntry(
                path=path,
                change_type=change,
                value_a=val_a if in_a else None,
                value_b=val_b if in_b else None,
                type_a=type_a,
                type_b=type_b,
            )
        )

    return JsonStructDiffResult(
        entries=entries,
        is_valid_json_a=True,
        is_valid_json_b=True,
    )


def detect_json(text: str) -> bool:
    """Return ``True`` if *text* parses as a JSON object or array."""
    stripped = text.strip()
    if not stripped or stripped[0] not in ("{", "["):
        return False
    try:
        parsed = _json.loads(stripped)
        return isinstance(parsed, (dict, list))
    except (_json.JSONDecodeError, ValueError):
        return False
