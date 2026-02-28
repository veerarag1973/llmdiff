"""BLEU and ROUGE-L metrics for llm-diff.

Provides lightweight, zero-dependency implementations of two standard NLP
evaluation metrics for comparing LLM outputs:

- **BLEU** — N-gram precision with brevity penalty (1–4-gram by default).
- **ROUGE-L** — F1 score based on the Longest Common Subsequence (LCS).

Both metrics are pure-Python, require no extra packages, and run in under
10 ms on typical LLM responses (~500 words).

Public API
----------
- :func:`compute_bleu`    — corpus-level BLEU score (0.0–1.0).
- :func:`compute_rouge_l` — ROUGE-L F1 score (0.0–1.0).
"""

from __future__ import annotations

import math
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Tokenisation (shared)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation suitable for BLEU / ROUGE-L."""
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------


def _count_ngrams(tokens: list[str], n: int) -> Counter:
    """Return a :class:`Counter` of all *n*-grams in *tokens*."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    reference: str,
    hypothesis: str,
    *,
    max_n: int = 4,
    weights: tuple[float, ...] | None = None,
) -> float:
    """Compute a sentence-level BLEU score.

    Uses clipped n-gram precision for 1-gram through *max_n*-gram with a
    brevity penalty, following the original Papineni et al. (2002) formulation.

    Parameters
    ----------
    reference:
        The reference text (model A response).
    hypothesis:
        The hypothesis text (model B response).
    max_n:
        Maximum n-gram order (default 4).
    weights:
        Tuple of *max_n* floats that sum to 1.0, used for the weighted
        geometric mean.  Defaults to uniform weights ``(1/max_n, …)``.

    Returns
    -------
    float
        BLEU score in ``[0.0, 1.0]``.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    if weights is None:
        weights = tuple(1.0 / max_n for _ in range(max_n))

    # Clipped n-gram precisions
    log_precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = _count_ngrams(ref_tokens, n)
        hyp_ngrams = _count_ngrams(hyp_tokens, n)

        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            # No n-grams of this order in hypothesis (very short text).
            # Standard BLEU returns 0 when any precision is 0.
            return 0.0

        precision = clipped / total
        if precision == 0:
            return 0.0

        log_precisions.append(math.log(precision))

    # Brevity penalty
    bp = 1.0
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1.0 - len(ref_tokens) / len(hyp_tokens))

    # Weighted geometric mean of precisions × brevity penalty
    weighted_log = sum(w * lp for w, lp in zip(weights, log_precisions))
    return bp * math.exp(weighted_log)


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Return the length of the Longest Common Subsequence of *x* and *y*.

    Uses the standard dynamic-programming table in O(m × n) time.
    """
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Space-optimised: only keep two rows.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute the ROUGE-L F1 score between *reference* and *hypothesis*.

    ROUGE-L uses the Longest Common Subsequence (LCS) to measure sentence-
    level structural similarity:

    - **Precision** = LCS length / hypothesis length
    - **Recall**    = LCS length / reference length
    - **F1**        = harmonic mean of precision and recall

    Parameters
    ----------
    reference:
        The reference text (model A response).
    hypothesis:
        The hypothesis text (model B response).

    Returns
    -------
    float
        ROUGE-L F1 score in ``[0.0, 1.0]``.
    """
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1
