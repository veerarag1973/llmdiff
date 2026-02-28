"""Programmatic Python API for llm-diff.

This module exposes high-level async functions that mirror the CLI behaviour
without any Click dependency, making llm-diff usable as a library.

Quick Start
-----------
.. code-block:: python

    import asyncio
    from llm_diff import compare

    report = asyncio.run(
        compare("Explain recursion", model_a="gpt-4o", model_b="gpt-3.5-turbo")
    )
    print(f"Word similarity:    {report.word_similarity:.2%}")
    print(f"Response A tokens:  {report.comparison.response_a.total_tokens}")

Semantic Scoring
----------------
.. code-block:: python

    report = asyncio.run(
        compare(
            "Explain recursion",
            model_a="gpt-4o",
            model_b="gpt-3.5-turbo",
            semantic=True,
        )
    )
    print(f"Semantic similarity: {report.semantic_score:.2%}")

HTML Report
-----------
.. code-block:: python

    report = asyncio.run(
        compare("Explain recursion", model_a="gpt-4o", model_b="gpt-3.5-turbo", build_html=True)
    )
    with open("report.html", "w") as f:
        f.write(report.html_report)

Batch Mode
----------
.. code-block:: python

    reports = asyncio.run(
        compare_batch("prompts.yml", model_a="gpt-4o", model_b="gpt-3.5-turbo")
    )
    for r in reports:
        print(f"{r.comparison.response_a.model}: {r.word_similarity:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from llm_diff.config import LLMDiffConfig, load_config
from llm_diff.diff import DiffResult, word_diff
from llm_diff.providers import ComparisonResult, compare_models

if TYPE_CHECKING:
    from llm_diff.semantic import ParagraphScore


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    """The full result of a single llm-diff comparison.

    Attributes
    ----------
    prompt_a:
        The prompt sent to model A.
    prompt_b:
        The prompt sent to model B (often identical to *prompt_a*).
    comparison:
        Raw paired model responses from
        :func:`~llm_diff.providers.compare_models`.
    diff_result:
        Word-level diff result from :func:`~llm_diff.diff.word_diff`.
    semantic_score:
        Whole-text cosine similarity (0.0–1.0), or ``None`` if not requested.
    paragraph_scores:
        Per-paragraph similarity scores, or ``None`` if not requested.
    html_report:
        Rendered HTML string (fully self-contained), or ``None`` if report
        generation was not requested via ``build_html=True``.
    """

    prompt_a: str
    prompt_b: str
    comparison: ComparisonResult
    diff_result: DiffResult
    semantic_score: float | None = field(default=None)
    paragraph_scores: list[ParagraphScore] | None = field(default=None)
    bleu_score: float | None = field(default=None)
    rouge_l_score: float | None = field(default=None)
    html_report: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def word_similarity(self) -> float:
        """Word-level similarity score (0.0–1.0)."""
        return self.diff_result.similarity

    @property
    def primary_score(self) -> float:
        """Return ``semantic_score`` if available, otherwise ``word_similarity``.

        This is the score used by ``--fail-under`` threshold checks in the CLI.
        """
        return self.semantic_score if self.semantic_score is not None else self.word_similarity


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compare(
    prompt: str,
    *,
    model_a: str,
    model_b: str,
    semantic: bool = False,
    paragraph: bool = False,
    bleu: bool = False,
    rouge: bool = False,
    build_html: bool = False,
    config: LLMDiffConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> ComparisonReport:
    """Compare two models on the same prompt and return a :class:`ComparisonReport`.

    This is the primary programmatic entry point for llm-diff.  It runs the
    full pipeline — concurrent API calls, word-level diff computation, optional
    semantic scoring, and optional HTML report — and returns a structured result
    with no Click dependency.

    Parameters
    ----------
    prompt:
        The prompt text sent to both models.
    model_a:
        Model identifier for side A (e.g. ``"gpt-4o"``).
    model_b:
        Model identifier for side B (e.g. ``"claude-3-5-sonnet"``).
    semantic:
        When ``True``, compute whole-text cosine similarity using
        ``sentence-transformers``.  Requires the ``[semantic]`` extra.
    paragraph:
        When ``True``, compute paragraph-level similarity (implies *semantic*).
    bleu:
        When ``True``, compute a sentence-level BLEU score (n-gram precision).
    rouge:
        When ``True``, compute a ROUGE-L F1 score (longest common subsequence).
    build_html:
        When ``True``, render and attach a fully self-contained HTML report.
    config:
        Optional pre-built :class:`~llm_diff.config.LLMDiffConfig`.  When
        ``None``, config is loaded from the environment / ``.llmdiff`` TOML.
    temperature:
        Override the temperature in *config*.
    max_tokens:
        Override ``max_tokens`` in *config*.
    timeout:
        Override the request timeout (seconds) in *config*.

    Returns
    -------
    ComparisonReport
        Fully populated result object.
    """
    cfg = _resolve_config(config, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    comparison = await compare_models(
        prompt_a=prompt,
        prompt_b=prompt,
        model_a=model_a,
        model_b=model_b,
        config=cfg,
    )
    diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)

    semantic_score, paragraph_scores = await _compute_similarity(
        comparison, semantic=semantic, paragraph=paragraph
    )

    bleu_score, rouge_l_score = _compute_metrics(
        comparison, bleu=bleu, rouge=rouge
    )

    html_report: str | None = None
    if build_html:
        from llm_diff.report import build_report  # noqa: PLC0415

        html_report = build_report(
            prompt=prompt,
            result=comparison,
            diff_result=diff_result,
            semantic_score=semantic_score,
            paragraph_scores=paragraph_scores,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
        )

    return ComparisonReport(
        prompt_a=prompt,
        prompt_b=prompt,
        comparison=comparison,
        diff_result=diff_result,
        semantic_score=semantic_score,
        paragraph_scores=paragraph_scores,
        bleu_score=bleu_score,
        rouge_l_score=rouge_l_score,
        html_report=html_report,
    )


async def compare_prompts(
    prompt_a: str,
    prompt_b: str,
    *,
    model: str,
    semantic: bool = False,
    paragraph: bool = False,
    bleu: bool = False,
    rouge: bool = False,
    build_html: bool = False,
    config: LLMDiffConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> ComparisonReport:
    """Compare two *different* prompts against the *same* model.

    This is the prompt-diff variant of :func:`compare`.  Both prompts are sent
    to the same model and the responses are diffed.

    Parameters
    ----------
    prompt_a:
        Prompt sent to the model on side A.
    prompt_b:
        Prompt sent to the model on side B.
    model:
        Model identifier used for both API calls.
    semantic, paragraph, bleu, rouge, build_html, config, temperature, max_tokens, timeout:
        See :func:`compare`.

    Returns
    -------
    ComparisonReport
        Fully populated result object.
    """
    cfg = _resolve_config(config, temperature=temperature, max_tokens=max_tokens, timeout=timeout)

    comparison = await compare_models(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        model_a=model,
        model_b=model,
        config=cfg,
    )
    diff_result = word_diff(comparison.response_a.text, comparison.response_b.text)

    semantic_score, paragraph_scores = await _compute_similarity(
        comparison, semantic=semantic, paragraph=paragraph
    )

    bleu_score, rouge_l_score = _compute_metrics(
        comparison, bleu=bleu, rouge=rouge
    )

    display_prompt = (
        prompt_a if prompt_a == prompt_b else f"{prompt_a[:40]}…"
    )

    html_report: str | None = None
    if build_html:
        from llm_diff.report import build_report  # noqa: PLC0415

        html_report = build_report(
            prompt=display_prompt,
            result=comparison,
            diff_result=diff_result,
            semantic_score=semantic_score,
            paragraph_scores=paragraph_scores,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
        )

    return ComparisonReport(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        comparison=comparison,
        diff_result=diff_result,
        semantic_score=semantic_score,
        paragraph_scores=paragraph_scores,
        bleu_score=bleu_score,
        rouge_l_score=rouge_l_score,
        html_report=html_report,
    )


async def compare_batch(
    batch_path: str | Path,
    *,
    model_a: str,
    model_b: str,
    semantic: bool = False,
    paragraph: bool = False,
    bleu: bool = False,
    rouge: bool = False,
    build_html: bool = False,
    config: LLMDiffConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> list[ComparisonReport]:
    """Run a full batch diff from a ``prompts.yml`` file.

    Loads all batch items from the YAML file, runs :func:`compare` for each
    item sequentially, and returns a list of results in the same order as the
    YAML file.

    Parameters
    ----------
    batch_path:
        Path to a YAML batch file (see :func:`~llm_diff.batch.load_batch`).
    model_a:
        Model identifier for side A.
    model_b:
        Model identifier for side B.
    semantic, paragraph, bleu, rouge, build_html, config, temperature, max_tokens, timeout:
        See :func:`compare`.

    Returns
    -------
    list[ComparisonReport]
        One report per batch item, in the same order as the YAML file.
    """
    from llm_diff.batch import load_batch  # noqa: PLC0415

    cfg = _resolve_config(config, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    items = load_batch(str(batch_path))

    reports: list[ComparisonReport] = []
    for item in items:
        report = await compare(
            item.prompt_text,
            model_a=model_a,
            model_b=model_b,
            semantic=semantic,
            paragraph=paragraph,
            bleu=bleu,
            rouge=rouge,
            build_html=build_html,
            config=cfg,
        )
        reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_config(
    config: LLMDiffConfig | None,
    *,
    temperature: float | None,
    max_tokens: int | None,
    timeout: int | None,
) -> LLMDiffConfig:
    """Return the effective config with any kwarg overrides applied."""
    cfg = config if config is not None else load_config()
    if temperature is not None:
        cfg.temperature = temperature
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    if timeout is not None:
        cfg.timeout = timeout
    return cfg


async def _compute_similarity(
    comparison: ComparisonResult,
    *,
    semantic: bool,
    paragraph: bool,
) -> tuple[float | None, list | None]:
    """Compute semantic / paragraph similarity for *comparison*.

    Returns ``(semantic_score, paragraph_scores)`` where either may be
    ``None`` when the corresponding flag is ``False``.
    """
    semantic_score: float | None = None
    paragraph_scores = None

    if paragraph:
        from llm_diff.semantic import (  # noqa: PLC0415
            compute_paragraph_similarity,
            compute_semantic_similarity,
        )

        paragraph_scores = compute_paragraph_similarity(
            comparison.response_a.text,
            comparison.response_b.text,
        )
        semantic_score = compute_semantic_similarity(
            comparison.response_a.text,
            comparison.response_b.text,
        )
    elif semantic:
        from llm_diff.semantic import compute_semantic_similarity  # noqa: PLC0415

        semantic_score = compute_semantic_similarity(
            comparison.response_a.text,
            comparison.response_b.text,
        )

    return semantic_score, paragraph_scores


def _compute_metrics(
    comparison: ComparisonResult,
    *,
    bleu: bool,
    rouge: bool,
) -> tuple[float | None, float | None]:
    """Compute BLEU / ROUGE-L for *comparison*.

    Returns ``(bleu_score, rouge_l_score)`` where either may be ``None``
    when the corresponding flag is ``False``.
    """
    bleu_score: float | None = None
    rouge_l_score: float | None = None

    text_a = comparison.response_a.text
    text_b = comparison.response_b.text

    if bleu:
        from llm_diff.metrics import compute_bleu  # noqa: PLC0415

        bleu_score = compute_bleu(text_a, text_b)

    if rouge:
        from llm_diff.metrics import compute_rouge_l  # noqa: PLC0415

        rouge_l_score = compute_rouge_l(text_a, text_b)

    return bleu_score, rouge_l_score
