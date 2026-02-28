"""Tests for llm_diff.diff — word-level diff engine."""

from __future__ import annotations

import time

import pytest

from llm_diff.diff import (
    DiffChunk,
    DiffResult,
    DiffType,
    _tokenize,
    _tokens_to_text,
    compute_similarity,
    word_diff,
)

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_sentence(self) -> None:
        tokens = _tokenize("hello world foo")
        assert [w for w, _ in tokens] == ["hello", "world", "foo"]

    def test_preserves_trailing_whitespace(self) -> None:
        tokens = _tokenize("a  b\tc")
        assert tokens[0] == ("a", "  ")
        assert tokens[1] == ("b", "\t")
        assert tokens[2] == ("c", "")

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_single_word(self) -> None:
        assert _tokenize("hello") == [("hello", "")]

    def test_punctuation_attached_to_word(self) -> None:
        tokens = _tokenize("hello, world!")
        words = [w for w, _ in tokens]
        assert "hello," in words
        assert "world!" in words

    def test_roundtrip(self) -> None:
        text = "The quick brown  fox\njumps over the lazy dog."
        tokens = _tokenize(text)
        assert _tokens_to_text(tokens) == text


# ---------------------------------------------------------------------------
# word_diff — return type
# ---------------------------------------------------------------------------


class TestWordDiffReturnType:
    def test_returns_diff_result(self) -> None:
        r = word_diff("hello world", "hello world")
        assert isinstance(r, DiffResult)

    def test_chunks_are_diff_chunks(self) -> None:
        r = word_diff("a b c", "a x c")
        assert all(isinstance(c, DiffChunk) for c in r.chunks)

    def test_chunk_types_are_enum(self) -> None:
        r = word_diff("foo bar", "foo baz")
        types = {c.type for c in r.chunks}
        assert types <= {DiffType.EQUAL, DiffType.INSERT, DiffType.DELETE}


# ---------------------------------------------------------------------------
# word_diff — identical texts
# ---------------------------------------------------------------------------


class TestIdenticalTexts:
    def test_all_chunks_equal(self) -> None:
        r = word_diff("same same same", "same same same")
        assert all(c.type == DiffType.EQUAL for c in r.chunks)

    def test_similarity_is_one(self) -> None:
        r = word_diff("same same same", "same same same")
        assert r.similarity == pytest.approx(1.0)

    def test_word_counts_match(self) -> None:
        r = word_diff("one two three", "one two three")
        assert r.word_count_a == 3
        assert r.word_count_b == 3


# ---------------------------------------------------------------------------
# word_diff — completely different texts
# ---------------------------------------------------------------------------


class TestCompletelyDifferentTexts:
    def test_no_equal_chunks(self) -> None:
        r = word_diff("alpha beta gamma", "delta epsilon zeta")
        equal_chunks = [c for c in r.chunks if c.type == DiffType.EQUAL]
        assert len(equal_chunks) == 0

    def test_similarity_is_zero(self) -> None:
        r = word_diff("alpha beta gamma", "delta epsilon zeta")
        assert r.similarity == pytest.approx(0.0)

    def test_has_delete_and_insert(self) -> None:
        r = word_diff("alpha beta gamma", "delta epsilon zeta")
        types = {c.type for c in r.chunks}
        assert DiffType.DELETE in types
        assert DiffType.INSERT in types


# ---------------------------------------------------------------------------
# word_diff — partial overlap
# ---------------------------------------------------------------------------


class TestPartialOverlap:
    def test_similarity_between_zero_and_one(self) -> None:
        text_a = "Recursion is when a function calls itself to solve a smaller version."
        text_b = (
            "Recursion is when a function calls itself to solve a smaller piece, "
            "until it reaches a base case that stops the repetition."
        )
        r = word_diff(text_a, text_b)
        assert 0.0 < r.similarity < 1.0

    def test_changed_words_are_delete_then_insert(self) -> None:
        # "version" → "piece"
        r = word_diff("solve a version now", "solve a piece now")
        deletes = [c.text.strip() for c in r.chunks if c.type == DiffType.DELETE]
        inserts = [c.text.strip() for c in r.chunks if c.type == DiffType.INSERT]
        assert "version" in deletes
        assert "piece" in inserts


# ---------------------------------------------------------------------------
# word_diff — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_a(self) -> None:
        r = word_diff("", "hello world")
        insert_text = " ".join(c.text.strip() for c in r.chunks if c.type == DiffType.INSERT)
        assert "hello" in insert_text

    def test_empty_b(self) -> None:
        r = word_diff("hello world", "")
        delete_text = " ".join(c.text.strip() for c in r.chunks if c.type == DiffType.DELETE)
        assert "hello" in delete_text

    def test_both_empty(self) -> None:
        r = word_diff("", "")
        assert r.chunks == []
        assert r.similarity == pytest.approx(1.0)

    def test_single_word_change(self) -> None:
        r = word_diff("cat", "dog")
        assert any(c.type == DiffType.DELETE for c in r.chunks)
        assert any(c.type == DiffType.INSERT for c in r.chunks)

    def test_newlines_preserved_in_equal_chunks(self) -> None:
        text = "line one\nline two\nline three"
        r = word_diff(text, text)
        reconstructed = "".join(c.text for c in r.chunks)
        assert reconstructed == text


# ---------------------------------------------------------------------------
# word_diff — serialisation
# ---------------------------------------------------------------------------


class TestDiffResultSerialization:
    def test_to_dict_has_required_keys(self) -> None:
        r = word_diff("a b", "a c")
        d = r.to_dict()
        assert "chunks" in d
        assert "similarity_score" in d
        assert "word_count_a" in d
        assert "word_count_b" in d

    def test_chunk_dicts_have_type_and_text(self) -> None:
        r = word_diff("a b", "a c")
        for chunk in r.to_dict()["chunks"]:
            assert "type" in chunk
            assert "text" in chunk
            assert chunk["type"] in ("equal", "insert", "delete")

    def test_similarity_pct_format(self) -> None:
        r = word_diff("a b c d", "a b c d")
        assert r.similarity_pct == "100%"


# ---------------------------------------------------------------------------
# compute_similarity
# ---------------------------------------------------------------------------


class TestComputeSimilarity:
    def test_identical(self) -> None:
        assert compute_similarity("foo bar baz", "foo bar baz") == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert compute_similarity("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_consistent_with_word_diff(self) -> None:
        a = "The quick brown fox jumps over the lazy dog"
        b = "The quick brown cat jumps over the lazy dog"
        assert compute_similarity(a, b) == pytest.approx(word_diff(a, b).similarity)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_two_thousand_word_diff_under_100ms(self) -> None:
        """Word diff on 2,000-word outputs must complete in < 100ms."""
        words = " ".join([f"word{i}" for i in range(1000)])
        words_b = " ".join([f"word{i}" if i % 5 != 0 else f"changed{i}" for i in range(1000)])

        start = time.perf_counter()
        word_diff(words, words_b)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert elapsed_ms < 100, f"Performance regression: {elapsed_ms:.1f}ms (limit 100ms)"
