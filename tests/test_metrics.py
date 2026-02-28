"""Tests for llm_diff.metrics — BLEU and ROUGE-L scoring."""

from __future__ import annotations

import pytest

from llm_diff.metrics import (
    _count_ngrams,
    _lcs_length,
    _tokenize,
    compute_bleu,
    compute_rouge_l,
)


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_sentence(self) -> None:
        assert _tokenize("Hello world") == ["hello", "world"]

    def test_punctuation_stripped(self) -> None:
        assert _tokenize("Hello, world!") == ["hello", "world"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_unicode(self) -> None:
        tokens = _tokenize("café résumé")
        assert tokens == ["café", "résumé"]

    def test_mixed_whitespace(self) -> None:
        assert _tokenize("  hello\t\nworld  ") == ["hello", "world"]


# ---------------------------------------------------------------------------
# N-gram counting
# ---------------------------------------------------------------------------


class TestCountNgrams:
    def test_unigrams(self) -> None:
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        ngrams = _count_ngrams(tokens, 1)
        assert ngrams[("the",)] == 2
        assert ngrams[("cat",)] == 1

    def test_bigrams(self) -> None:
        tokens = ["the", "cat", "sat"]
        ngrams = _count_ngrams(tokens, 2)
        assert ngrams[("the", "cat")] == 1
        assert ngrams[("cat", "sat")] == 1
        assert len(ngrams) == 2

    def test_empty_tokens(self) -> None:
        assert _count_ngrams([], 1) == {}

    def test_n_greater_than_length(self) -> None:
        assert _count_ngrams(["hello"], 2) == {}


# ---------------------------------------------------------------------------
# LCS length
# ---------------------------------------------------------------------------


class TestLcsLength:
    def test_identical(self) -> None:
        seq = ["a", "b", "c"]
        assert _lcs_length(seq, seq) == 3

    def test_no_overlap(self) -> None:
        assert _lcs_length(["a", "b"], ["c", "d"]) == 0

    def test_partial_overlap(self) -> None:
        assert _lcs_length(["a", "b", "c", "d"], ["b", "d"]) == 2

    def test_empty_inputs(self) -> None:
        assert _lcs_length([], ["a"]) == 0
        assert _lcs_length(["a"], []) == 0
        assert _lcs_length([], []) == 0

    def test_known_case(self) -> None:
        x = ["the", "cat", "sat", "on", "the", "mat"]
        y = ["the", "cat", "on", "the", "mat"]
        assert _lcs_length(x, y) == 5


# ---------------------------------------------------------------------------
# compute_bleu
# ---------------------------------------------------------------------------


class TestComputeBleu:
    def test_identical_texts_return_near_1(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        score = compute_bleu(text, text)
        assert pytest.approx(score, abs=1e-6) == 1.0

    def test_completely_different_returns_0(self) -> None:
        score = compute_bleu("alpha beta gamma", "delta epsilon zeta")
        assert score == 0.0

    def test_empty_reference_returns_0(self) -> None:
        assert compute_bleu("", "hello world") == 0.0

    def test_empty_hypothesis_returns_0(self) -> None:
        assert compute_bleu("hello world", "") == 0.0

    def test_both_empty_returns_0(self) -> None:
        assert compute_bleu("", "") == 0.0

    def test_partial_overlap(self) -> None:
        ref = "the cat sat on the mat"
        hyp = "the cat on the mat"
        # Default max_n=4 returns 0 because the 4-gram overlap is empty
        # on these short texts. With max_n=2 we get a meaningful score.
        score = compute_bleu(ref, hyp, max_n=2)
        assert 0.0 < score < 1.0

    def test_brevity_penalty_applied(self) -> None:
        ref = "the quick brown fox jumps over the lazy dog near the river bank"
        hyp = "the fox"
        score = compute_bleu(ref, hyp)
        # Very short hypothesis should get heavily penalised
        assert score < 0.3

    def test_score_in_0_1_range(self) -> None:
        ref = "Recursion is a technique where a function calls itself"
        hyp = "Recursion is when a function calls itself repeatedly"
        score = compute_bleu(ref, hyp)
        assert 0.0 <= score <= 1.0

    def test_custom_max_n(self) -> None:
        ref = "hello world foo bar"
        hyp = "hello world foo bar"
        score = compute_bleu(ref, hyp, max_n=2)
        assert pytest.approx(score, abs=1e-6) == 1.0

    def test_single_word_identical(self) -> None:
        score = compute_bleu("hello", "hello")
        # max_n=4 but only 1 token → 2-gram, 3-gram, 4-gram are empty
        # returns 0 because higher-order n-grams have 0 precision
        assert score == 0.0

    def test_single_word_with_max_n_1(self) -> None:
        score = compute_bleu("hello", "hello", max_n=1)
        assert pytest.approx(score, abs=1e-6) == 1.0


# ---------------------------------------------------------------------------
# compute_rouge_l
# ---------------------------------------------------------------------------


class TestComputeRougeL:
    def test_identical_texts_return_1(self) -> None:
        text = "The quick brown fox jumps over the lazy dog"
        score = compute_rouge_l(text, text)
        assert pytest.approx(score, abs=1e-6) == 1.0

    def test_completely_different_returns_0(self) -> None:
        score = compute_rouge_l("alpha beta gamma", "delta epsilon zeta")
        assert score == 0.0

    def test_empty_reference_returns_0(self) -> None:
        assert compute_rouge_l("", "hello world") == 0.0

    def test_empty_hypothesis_returns_0(self) -> None:
        assert compute_rouge_l("hello world", "") == 0.0

    def test_both_empty_returns_0(self) -> None:
        assert compute_rouge_l("", "") == 0.0

    def test_partial_overlap(self) -> None:
        ref = "the cat sat on the mat"
        hyp = "the cat on the mat"
        score = compute_rouge_l(ref, hyp)
        assert 0.0 < score < 1.0

    def test_score_in_0_1_range(self) -> None:
        ref = "Recursion is a technique where a function calls itself"
        hyp = "Recursion is when a function calls itself repeatedly"
        score = compute_rouge_l(ref, hyp)
        assert 0.0 <= score <= 1.0

    def test_subset_text(self) -> None:
        ref = "the cat sat on the mat near the door"
        hyp = "the cat sat on the mat"
        score = compute_rouge_l(ref, hyp)
        # hyp is a subset → precision 1.0 but recall < 1.0
        assert 0.5 < score < 1.0

    def test_superset_text(self) -> None:
        ref = "the cat on the mat"
        hyp = "the cat sat on the mat near the door"
        score = compute_rouge_l(ref, hyp)
        # recall = 1.0 but precision < 1.0
        assert 0.5 < score < 1.0

    def test_known_lcs_case(self) -> None:
        """Verify F1 against a hand-computed example.

        ref = [the, cat, sat, on, the, mat]  (6 tokens)
        hyp = [the, cat, on, the, mat]        (5 tokens)
        LCS = [the, cat, on, the, mat]        (length 5)
        precision = 5/5 = 1.0
        recall    = 5/6 ≈ 0.833
        F1 = 2 * 1.0 * 0.833 / (1.0 + 0.833) ≈ 0.909
        """
        ref = "the cat sat on the mat"
        hyp = "the cat on the mat"
        score = compute_rouge_l(ref, hyp)
        assert pytest.approx(score, abs=0.01) == 0.909


# ---------------------------------------------------------------------------
# Integration: real-ish LLM output comparison
# ---------------------------------------------------------------------------


class TestMetricsIntegration:
    """Smoke tests on longer, LLM-style text."""

    REF = (
        "Recursion is a technique where a function calls itself with a "
        "simpler version of the problem until a base case is reached."
    )
    HYP = (
        "Recursion is when a function calls itself repeatedly until a "
        "base condition is met, solving the problem step by step."
    )

    def test_bleu_realistic(self) -> None:
        score = compute_bleu(self.REF, self.HYP)
        assert 0.1 < score < 0.8

    def test_rouge_l_realistic(self) -> None:
        score = compute_rouge_l(self.REF, self.HYP)
        assert 0.3 < score < 0.9
