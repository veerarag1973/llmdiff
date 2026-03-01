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

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from llm_diff.config import LLMDiffConfig, load_config
from llm_diff.diff import DiffResult, word_diff
from llm_diff.providers import ComparisonResult, compare_models

if TYPE_CHECKING:
    from llm_diff.judge import JudgeResult
    from llm_diff.pricing import CostEstimate
    from llm_diff.semantic import ParagraphScore

logger = logging.getLogger(__name__)


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
    judge_result: JudgeResult | None = field(default=None)
    cost_a: CostEstimate | None = field(default=None)
    cost_b: CostEstimate | None = field(default=None)
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
    judge: str | None = None,
    show_cost: bool = False,
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
        When ``True``, compute a ROUGE-L F1 score (longest common subsequence).    judge:
        When set to a model identifier (e.g. ``"gpt-4o"``), the judge model
        is called with both responses and returns a winner + reasoning.
    show_cost:
        When ``True``, estimate the USD cost of each model call and attach
        :class:`~llm_diff.pricing.CostEstimate` objects to the report.    build_html:
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

    # Emit comparison started event (best-effort)
    _started_event_id: str = ""
    try:
        from llm_diff.schema_events import (  # noqa: PLC0415
            emit as schema_emit,
        )
        from llm_diff.schema_events import (
            make_comparison_started_event,
        )

        started_evt = make_comparison_started_event(
            model_a=model_a,
            model_b=model_b,
            prompt=prompt,
        )
        schema_emit(started_evt)
        _started_event_id = started_evt.event_id
    except Exception:  # noqa: BLE001
        logger.debug("Schema event emission failed", exc_info=True)

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

    judge_result = await _run_judge(
        prompt=prompt,
        comparison=comparison,
        judge_model=judge,
        config=cfg,
    )

    cost_a, cost_b = _compute_cost(comparison, show_cost=show_cost)

    # Emit cost recorded events for each model call (best-effort)
    if cost_a is not None:
        try:
            from llm_diff.schema_events import (  # noqa: PLC0415
                emit as schema_emit,
            )
            from llm_diff.schema_events import (
                make_cost_recorded_event,
            )

            schema_emit(
                make_cost_recorded_event(
                    input_cost=cost_a.prompt_usd,
                    output_cost=cost_a.completion_usd,
                    total_cost=cost_a.total_usd,
                    model=cost_a.model,
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Schema event emission failed", exc_info=True)
    if cost_b is not None:
        try:
            from llm_diff.schema_events import (  # noqa: PLC0415
                emit as schema_emit,
            )
            from llm_diff.schema_events import (
                make_cost_recorded_event,
            )

            schema_emit(
                make_cost_recorded_event(
                    input_cost=cost_b.prompt_usd,
                    output_cost=cost_b.completion_usd,
                    total_cost=cost_b.total_usd,
                    model=cost_b.model,
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("Schema event emission failed", exc_info=True)

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
            judge_result=judge_result,
            cost_a=cost_a,
            cost_b=cost_b,
        )

    # Emit comparison completed event (best-effort)
    try:
        from llm_diff.schema_events import (  # noqa: PLC0415
            emit as schema_emit,
        )
        from llm_diff.schema_events import (
            make_comparison_completed_event,
        )

        schema_emit(
            make_comparison_completed_event(
                model_a=model_a,
                model_b=model_b,
                diff_type="completion",
                completion_diff=diff_result.as_unified_diff() or None,
                similarity_score=diff_result.similarity,
                base_event_id=_started_event_id,
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("Schema event emission failed", exc_info=True)

    return ComparisonReport(
        prompt_a=prompt,
        prompt_b=prompt,
        comparison=comparison,
        diff_result=diff_result,
        semantic_score=semantic_score,
        paragraph_scores=paragraph_scores,
        bleu_score=bleu_score,
        rouge_l_score=rouge_l_score,
        judge_result=judge_result,
        cost_a=cost_a,
        cost_b=cost_b,
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
    judge: str | None = None,
    show_cost: bool = False,
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

    # Emit comparison started event (best-effort) — diff_type is "prompt"
    _started_event_id_p: str = ""
    try:
        from llm_diff.schema_events import (  # noqa: PLC0415
            emit as schema_emit,
        )
        from llm_diff.schema_events import (
            make_comparison_started_event,
        )

        started_evt = make_comparison_started_event(
            model_a=model,
            model_b=model,
            prompt=prompt_a,
        )
        schema_emit(started_evt)
        _started_event_id_p = started_evt.event_id
    except Exception:  # noqa: BLE001
        logger.debug("Schema event emission failed", exc_info=True)

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

    judge_result = await _run_judge(
        prompt=display_prompt,
        comparison=comparison,
        judge_model=judge,
        config=cfg,
    )

    cost_a, cost_b = _compute_cost(comparison, show_cost=show_cost)

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
            judge_result=judge_result,
            cost_a=cost_a,
            cost_b=cost_b,
        )

    # Emit comparison completed event (prompt diff, best-effort)
    try:
        from llm_diff.schema_events import (  # noqa: PLC0415
            emit as schema_emit,
        )
        from llm_diff.schema_events import (
            make_comparison_completed_event,
        )

        schema_emit(
            make_comparison_completed_event(
                model_a=model,
                model_b=model,
                diff_type="prompt",
                completion_diff=diff_result.as_unified_diff() or None,
                similarity_score=diff_result.similarity,
                base_event_id=_started_event_id_p,
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("Schema event emission failed", exc_info=True)

    return ComparisonReport(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        comparison=comparison,
        diff_result=diff_result,
        semantic_score=semantic_score,
        paragraph_scores=paragraph_scores,
        bleu_score=bleu_score,
        rouge_l_score=rouge_l_score,
        judge_result=judge_result,
        cost_a=cost_a,
        cost_b=cost_b,
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
    judge: str | None = None,
    show_cost: bool = False,
    build_html: bool = False,
    config: LLMDiffConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    concurrency: int = 4,
) -> list[ComparisonReport]:
    """Run a full batch diff from a ``prompts.yml`` file.

    Loads all batch items from the YAML file and runs :func:`compare` for
    every item **concurrently**, bounded by *concurrency* simultaneous
    coroutines.  Results are returned in the same order as the YAML file
    regardless of completion order.

    Parameters
    ----------
    batch_path:
        Path to a YAML batch file (see :func:`~llm_diff.batch.load_batch`).
    model_a:
        Model identifier for side A.
    model_b:
        Model identifier for side B.
    concurrency:
        Maximum number of :func:`compare` calls that may be in-flight at the
        same time.  Defaults to ``4``.  Increase for large batches with
        high-throughput API keys; reduce when rate-limits are tight.
    semantic, paragraph, bleu, rouge, judge, show_cost, build_html, config, temperature, max_tokens, timeout:
        See :func:`compare`.

    Returns
    -------
    list[ComparisonReport]
        One report per batch item, in the same order as the YAML file.
    """
    from llm_diff.batch import load_batch  # noqa: PLC0415

    cfg = _resolve_config(config, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    items = load_batch(str(batch_path))

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(batch_item: object) -> ComparisonReport:
        async with sem:
            return await compare(
                batch_item.prompt_text,  # type: ignore[attr-defined]
                model_a=model_a,
                model_b=model_b,
                semantic=semantic,
                paragraph=paragraph,
                bleu=bleu,
                rouge=rouge,
                judge=judge,
                show_cost=show_cost,
                build_html=build_html,
                config=cfg,
            )

    return list(await asyncio.gather(*(_run_one(item) for item in items)))


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


async def _run_judge(
    *,
    prompt: str,
    comparison: ComparisonResult,
    judge_model: str | None,
    config: LLMDiffConfig,
) -> JudgeResult | None:
    """Run the LLM-as-a-Judge if *judge_model* is set; else return ``None``."""
    if not judge_model:
        return None

    from llm_diff.judge import run_judge  # noqa: PLC0415

    return await run_judge(
        prompt=prompt,
        response_a=comparison.response_a.text,
        response_b=comparison.response_b.text,
        judge_model=judge_model,
        config=config,
    )


def _compute_cost(
    comparison: ComparisonResult,
    *,
    show_cost: bool,
) -> tuple[CostEstimate | None, CostEstimate | None]:
    """Estimate cost for both model calls if *show_cost* is ``True``."""
    if not show_cost:
        return None, None

    from llm_diff.pricing import estimate_cost  # noqa: PLC0415

    ra = comparison.response_a
    rb = comparison.response_b
    cost_a = estimate_cost(
        ra.model,
        prompt_tokens=ra.prompt_tokens,
        completion_tokens=ra.completion_tokens,
    )
    cost_b = estimate_cost(
        rb.model,
        prompt_tokens=rb.prompt_tokens,
        completion_tokens=rb.completion_tokens,
    )
    return cost_a, cost_b
