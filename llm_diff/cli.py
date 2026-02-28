"""CLI entry point for llm-diff.

Usage examples:
    llm-diff "Explain recursion" --a gpt-4o --b claude-3-5-sonnet
    llm-diff "Explain recursion" --a gpt-4o --b claude-3-5-sonnet --semantic
    llm-diff "Explain recursion" --a gpt-4o --b claude-3-5-sonnet --json
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import NoReturn

import click
from rich.console import Console
from rich.text import Text

from llm_diff import __version__
from llm_diff.cache import ResultCache
from llm_diff.config import LLMDiffConfig, load_config
from llm_diff.diff import DiffResult, word_diff
from llm_diff.providers import ComparisonResult, compare_models
from llm_diff.renderer import render_diff

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format=_LOG_FORMAT, level=level)
    # Suppress third-party loggers that may emit auth headers or noisy output
    # even when verbose mode is enabled.  httpx debug logs include Authorization
    # headers, so keeping it at WARNING prevents accidental key leakage.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("prompt", required=False, default=None)
@click.option(
    "--prompt-a", "prompt_a",
    metavar="PATH", default=None,
    help="Path to a text file used as prompt for model A.",
)
@click.option(
    "--prompt-b", "prompt_b",
    metavar="PATH", default=None,
    help="Path to a text file used as prompt for model B.",
)
@click.option(
    "--model-a", "-a", "model_a",
    metavar="MODEL", default=None,
    help="Model identifier for side A (e.g. gpt-4o).",
)
@click.option(
    "--model-b", "-b", "model_b",
    metavar="MODEL", default=None,
    help="Model identifier for side B (e.g. claude-3-5-sonnet).",
)
@click.option(
    "--model", "-m", "model",
    metavar="MODEL", default=None,
    help="Use the same model for both sides (prompt-diff mode).",
)
@click.option(
    "--mode",
    default="word",
    type=click.Choice(["word", "json", "json-struct"], case_sensitive=False),
    show_default=True,
    help="Diff mode: 'word' (default), 'json' (JSON stdout output), 'json-struct' (key-level JSON diff).",
)
@click.option(
    "--semantic", "-s",
    is_flag=True, default=False,
    help="Compute and show embedding-based similarity score.",
)
@click.option(
    "--paragraph", "-p",
    is_flag=True, default=False,
    help="Compute paragraph-level semantic similarity (implies --semantic).",
)
@click.option(
    "--bleu",
    is_flag=True, default=False,
    help="Compute BLEU score (n-gram precision, no extra dependencies).",
)
@click.option(
    "--rouge",
    is_flag=True, default=False,
    help="Compute ROUGE-L F1 score (longest common subsequence, no extra dependencies).",
)
@click.option(
    "--json", "-j", "mode",
    flag_value="json",
    help="Output raw JSON diff to stdout.",
)
@click.option(
    "--batch",
    metavar="PATH", default=None,
    help="Path to a prompts.yml file for batch comparison.",
)
@click.option(
    "--judge",
    metavar="MODEL", default=None,
    help="Model to use as LLM-as-a-Judge (e.g. gpt-4o). Rates both responses and picks a winner.",
)
@click.option(
    "--show-cost",
    is_flag=True, default=False,
    help="Estimate and display the USD cost of each model call.",
)
@click.option(
    "--model-c", "model_c",
    metavar="MODEL", default=None,
    help="Add a third model to the comparison (multi-model mode).",
)
@click.option(
    "--model-d", "model_d",
    metavar="MODEL", default=None,
    help="Add a fourth model to the comparison (multi-model mode, requires --model-c).",
)
@click.option(
    "--out", "-o",
    metavar="PATH", default=None,
    help="Save HTML report to this path.",
)
@click.option(
    "--save",
    is_flag=True, default=False,
    help="Auto-save HTML report to ./diffs/ directory.",
)
@click.option(
    "--temperature", "-t",
    default=None, type=float, metavar="FLOAT",
    help="Temperature passed to both models (default 0.7).",
)
@click.option(
    "--max-tokens",
    default=None, type=int, metavar="INT",
    help="Max tokens for each model's response (default 1024).",
)
@click.option(
    "--timeout",
    default=None, type=int, metavar="SECS",
    help="Request timeout in seconds (default 30).",
)
@click.option(
    "--no-color",
    is_flag=True, default=False,
    help="Disable terminal colour output.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Show full API request/response metadata.",
)
@click.option(
    "--fail-under",
    default=None, type=click.FloatRange(0.0, 1.0), metavar="FLOAT",
    help=(
        "Exit with code 1 if the primary similarity score is below this threshold "
        "(0.0–1.0). Uses semantic score when --semantic / --paragraph is set, "
        "otherwise uses word similarity. Useful in CI pipelines."
    ),
)
@click.option(
    "--concurrency",
    default=4, type=int, metavar="INT",
    help="Max concurrent API calls in batch mode (default: 4).",
)
@click.option(
    "--no-cache",
    is_flag=True, default=False,
    help="Disable LLM response caching (always call the API).",
)
@click.version_option(version=__version__, prog_name="llm-diff")
def main(  # noqa: PLR0913 — many CLI parameters is unavoidable
    prompt: str | None,
    prompt_a: str | None,
    prompt_b: str | None,
    model_a: str | None,
    model_b: str | None,
    model: str | None,
    mode: str,
    semantic: bool,
    paragraph: bool,
    bleu: bool,
    rouge: bool,
    batch: str | None,
    out: str | None,
    save: bool,
    temperature: float | None,
    max_tokens: int | None,
    timeout: int | None,
    no_color: bool,
    verbose: bool,
    fail_under: float | None,
    concurrency: int,
    no_cache: bool,
    judge: str | None,
    show_cost: bool,
    model_c: str | None,
    model_d: str | None,
) -> None:
    """Compare two LLM responses — semantically, visually, and at scale.

    \b
    Examples:
      llm-diff "Explain recursion" -a gpt-4o -b claude-3-5-sonnet
      llm-diff "Explain recursion" -a gpt-4o -b claude-3-5-sonnet --semantic
      llm-diff "Explain recursion" -a gpt-4o -b claude-3-5-sonnet --paragraph
      llm-diff --prompt-a v1.txt --prompt-b v2.txt --model gpt-4o
      llm-diff "Explain recursion" -a gpt-4o -b gpt-3.5-turbo --semantic --fail-under 0.8
      llm-diff "Explain recursion" -a gpt-4o -b gpt-4o-mini --judge gpt-4o
      llm-diff "Explain recursion" -a gpt-4o -b gpt-4o-mini --show-cost
      llm-diff "Explain recursion" -a gpt-4o -b gpt-4o-mini --model-c claude-3-5-sonnet
    """
    _configure_logging(verbose)
    console = Console(no_color=no_color, stderr=False)

    # ── Load configuration ───────────────────────────────────────────────────
    cfg = load_config()

    # CLI flags override config-file defaults
    if temperature is not None:
        cfg.temperature = temperature
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    if timeout is not None:
        cfg.timeout = timeout
    if no_color:
        cfg.no_color = True
    if save:
        cfg.save = True

    _cache: ResultCache = ResultCache(enabled=not no_cache)

    # ── Dispatch ─────────────────────────────────────────────────────────────
    try:
        if batch:
            # Resolve models for batch mode
            resolved_ma: str = model or model_a or ""
            resolved_mb: str = model or model_b or ""
            if not resolved_ma or not resolved_mb:
                _die(
                    console,
                    "--batch requires --model-a / -a and --model-b / -b "
                    "(or --model for same-model comparison).",
                )
            asyncio.run(
                _run_batch(
                    batch=batch,
                    model_a=resolved_ma,
                    model_b=resolved_mb,
                    mode=mode,
                    semantic=semantic,
                    paragraph=paragraph,
                    bleu=bleu,
                    rouge=rouge,
                    judge=judge,
                    show_cost=show_cost,
                    fail_under=fail_under,
                    out=out,
                    config=cfg,
                    console=console,
                    verbose=verbose,
                    concurrency=concurrency,
                    cache=_cache,
                )
            )
        elif model_c or model_d:
            # ── Multi-model comparison ───────────────────────────────────────
            resolved_ma_mm: str = model or model_a or ""
            resolved_mb_mm: str = model or model_b or ""
            if not resolved_ma_mm or not resolved_mb_mm:
                _die(console, "Multi-model mode requires --model-a / -a and --model-b / -b.")
            resolved_prompt = prompt or ""
            if not resolved_prompt:
                _die(console, "Multi-model mode requires a prompt as a positional argument.")
            all_models: list[str] = [resolved_ma_mm, resolved_mb_mm]
            if model_c:
                all_models.append(model_c)
            if model_d:
                all_models.append(model_d)
            asyncio.run(
                _run_multi(
                    prompt=resolved_prompt,
                    models=all_models,
                    semantic=semantic,
                    config=cfg,
                    console=console,
                    concurrency=concurrency,
                    cache=_cache,
                )
            )
        else:
            # ── Validate inputs (non-batch) ──────────────────────────────────
            try:
                resolved_prompt_a, resolved_prompt_b, resolved_model_a, resolved_model_b = (
                    _resolve_inputs(
                        prompt=prompt,
                        prompt_a=prompt_a,
                        prompt_b=prompt_b,
                        model_a=model_a,
                        model_b=model_b,
                        model=model,
                    )
                )
            except click.UsageError as exc:
                _die(console, str(exc))
            asyncio.run(
                _run_diff(
                    prompt_a=resolved_prompt_a,
                    prompt_b=resolved_prompt_b,
                    model_a=resolved_model_a,
                    model_b=resolved_model_b,
                    mode=mode,
                    semantic=semantic,
                    paragraph=paragraph,
                    bleu=bleu,
                    rouge=rouge,
                    judge=judge,
                    show_cost=show_cost,
                    fail_under=fail_under,
                    out=out,
                    config=cfg,
                    console=console,
                    verbose=verbose,
                    cache=_cache,
                )
            )
    except (ValueError, RuntimeError, TimeoutError, ImportError, FileNotFoundError) as exc:
        _die(console, str(exc))
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        sys.exit(130)


# ---------------------------------------------------------------------------
# Input resolution
# ---------------------------------------------------------------------------


def _resolve_inputs(
    *,
    prompt: str | None,
    prompt_a: str | None,
    prompt_b: str | None,
    model_a: str | None,
    model_b: str | None,
    model: str | None,
) -> tuple[str, str, str, str]:
    """Return ``(prompt_a_text, prompt_b_text, model_a, model_b)``.

    Raises :class:`click.UsageError` with a human-readable message if the
    flag combination is invalid.
    """
    # --- prompt resolution ---
    if prompt_a and prompt_b:
        resolved_pa = _read_file_arg(prompt_a, "--prompt-a")
        resolved_pb = _read_file_arg(prompt_b, "--prompt-b")
    elif prompt:
        resolved_pa = prompt
        resolved_pb = prompt
    elif prompt_a:
        raise click.UsageError(
            "--prompt-a requires --prompt-b "
            "(or use --model for same-model comparison)."
        )
    elif prompt_b:
        raise click.UsageError("--prompt-b requires --prompt-a.")
    else:
        raise click.UsageError(
            "Provide a prompt as a positional argument or via --prompt-a / --prompt-b.\n"
            "Run llm-diff --help for usage examples."
        )

    # --- model resolution ---
    if model:
        # Same model for both sides — must use --prompt-a/--prompt-b
        if not (prompt_a and prompt_b):
            raise click.UsageError(
                "--model requires --prompt-a and --prompt-b to provide two different prompts."
            )
        resolved_ma = model
        resolved_mb = model
    elif model_a and model_b:
        resolved_ma = model_a
        resolved_mb = model_b
    elif model_a:
        raise click.UsageError("--model-a / -a requires --model-b / -b.")
    elif model_b:
        raise click.UsageError("--model-b / -b requires --model-a / -a.")
    else:
        raise click.UsageError(
            "Specify models with -a MODEL -b MODEL, or use --model for same-model comparison."
        )

    return resolved_pa, resolved_pb, resolved_ma, resolved_mb


def _read_file_arg(path: str, flag: str) -> str:
    """Read a prompt file, raising :class:`click.UsageError` if missing."""
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        raise click.UsageError(f"{flag}: file not found: {path}")
    return p.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Async batch runner
# ---------------------------------------------------------------------------


async def _run_batch(
    *,
    batch: str,
    model_a: str,
    model_b: str,
    mode: str,
    semantic: bool,
    paragraph: bool,
    bleu: bool,
    rouge: bool,
    judge: str | None = None,
    show_cost: bool = False,
    fail_under: float | None,
    out: str | None,
    config: LLMDiffConfig,
    console: Console,
    verbose: bool,
    concurrency: int = 4,
    cache: object | None = None,
) -> None:
    """Orchestrate the batch diff pipeline asynchronously.

    Loads all :class:`~llm_diff.batch.BatchItem` objects from *batch*, fetches
    model responses concurrently (up to *concurrency* simultaneous API calls),
    renders terminal output for each, and optionally writes a combined HTML
    report to *out*.  When *fail_under* is set, exits with code 1 if any
    item's primary score is below the threshold.
    """
    from llm_diff.batch import BatchItem, BatchResult, load_batch  # noqa: PLC0415

    items = load_batch(batch)
    n = len(items)
    sem = asyncio.Semaphore(concurrency)

    async def _fetch(item: BatchItem) -> ComparisonResult:
        async with sem:
            return await compare_models(
                prompt_a=item.prompt_text,
                prompt_b=item.prompt_text,
                model_a=model_a,
                model_b=model_b,
                config=config,
                cache=cache,
            )

    console.print(
        f"[dim]Fetching responses for {n} prompt(s) "
        f"(concurrency={concurrency})…[/dim]"
    )
    comparisons: list[ComparisonResult] = list(
        await asyncio.gather(*[_fetch(item) for item in items])
    )

    batch_results: list[BatchResult] = []

    for i, (item, comparison) in enumerate(zip(items, comparisons), 1):
        console.rule(f"[dim][{i}/{n}] {item.id}[/dim]")

        diff_result: DiffResult = word_diff(
            comparison.response_a.text,
            comparison.response_b.text,
        )

        semantic_score: float | None = None
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
            paragraph_scores = None
        else:
            paragraph_scores = None

        bleu_score: float | None = None
        rouge_l_score: float | None = None
        if bleu or rouge:
            from llm_diff.metrics import compute_bleu, compute_rouge_l  # noqa: PLC0415

            text_a = comparison.response_a.text
            text_b = comparison.response_b.text
            if bleu:
                bleu_score = compute_bleu(text_a, text_b)
            if rouge:
                rouge_l_score = compute_rouge_l(text_a, text_b)

        judge_result = None
        if judge:
            from llm_diff.judge import run_judge  # noqa: PLC0415
            judge_result = await run_judge(
                prompt=item.prompt_text[:200],
                response_a=comparison.response_a.text,
                response_b=comparison.response_b.text,
                judge_model=judge,
                config=config,
            )

        cost_a = cost_b = None
        if show_cost:
            from llm_diff.pricing import estimate_cost  # noqa: PLC0415
            cost_a = estimate_cost(
                comparison.response_a.model,
                prompt_tokens=comparison.response_a.prompt_tokens,
                completion_tokens=comparison.response_a.completion_tokens,
            )
            cost_b = estimate_cost(
                comparison.response_b.model,
                prompt_tokens=comparison.response_b.prompt_tokens,
                completion_tokens=comparison.response_b.completion_tokens,
            )

        render_diff(
            prompt=item.prompt_text[:60],
            result=comparison,
            diff_result=diff_result,
            console=console,
            semantic_score=semantic_score,
            paragraph_scores=paragraph_scores,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
            judge_result=judge_result,
            cost_a=cost_a,
            cost_b=cost_b,
        )

        if verbose:
            _render_verbose(comparison, console)

        batch_results.append(
            BatchResult(
                item=item,
                comparison=comparison,
                diff_result=diff_result,
                semantic_score=semantic_score,
                paragraph_scores=paragraph_scores,
                bleu_score=bleu_score,
                rouge_l_score=rouge_l_score,
            )
        )

    console.rule("[dim]Batch complete[/dim]")
    console.print(f"[dim]Processed {n} prompt(s).[/dim]")

    # ── Fail-under check ─────────────────────────────────────────
    if fail_under is not None:
        failing = [
            r for r in batch_results
            if (r.semantic_score if r.semantic_score is not None else r.diff_result.similarity)
            < fail_under
        ]
        if failing:
            err_console = Console(stderr=True)
            err_console.print(
                f"[bold red]--fail-under {fail_under:.2f}: "
                f"{len(failing)}/{len(batch_results)} item(s) below threshold.[/bold red]"
            )
            sys.exit(1)

    if out:
        from llm_diff.report import build_batch_report, save_report  # noqa: PLC0415

        html = build_batch_report(
            results=batch_results,
            model_a=model_a,
            model_b=model_b,
        )
        saved = save_report(html, out)
        console.print(f"[dim]Batch report saved → {saved}[/dim]")


# ---------------------------------------------------------------------------
# Async diff runner
# ---------------------------------------------------------------------------


async def _run_diff(
    *,
    prompt_a: str,
    prompt_b: str,
    model_a: str,
    model_b: str,
    mode: str,
    semantic: bool,
    paragraph: bool,
    bleu: bool,
    rouge: bool,
    judge: str | None = None,
    show_cost: bool = False,
    fail_under: float | None,
    out: str | None,
    config: LLMDiffConfig,
    console: Console,
    verbose: bool,
    cache: object | None = None,
) -> None:
    """Orchestrate the full diff pipeline asynchronously."""
    # Prompt shown in the header is the shared prompt when both are identical.
    display_prompt = prompt_a if prompt_a == prompt_b else f"{prompt_a[:40]}…"

    # ── Fetch model responses (always concurrent, one call each) ─────────────
    with console.status("[dim]Calling models…[/dim]", spinner="dots"):
        comparison: ComparisonResult = await compare_models(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            model_a=model_a,
            model_b=model_b,
            config=config,
            cache=cache,
        )

    # ── Compute diff ─────────────────────────────────────────────────────────
    diff_result: DiffResult = word_diff(
        comparison.response_a.text,
        comparison.response_b.text,
    )

    # ── Semantic similarity ──────────────────────────────────────────────────
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
        # Overall semantic score is the whole-text similarity (not the para avg).
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

    # ── BLEU / ROUGE-L ──────────────────────────────────────────────────────
    bleu_score: float | None = None
    rouge_l_score: float | None = None
    if bleu or rouge:
        from llm_diff.metrics import compute_bleu, compute_rouge_l  # noqa: PLC0415

        text_a = comparison.response_a.text
        text_b = comparison.response_b.text
        if bleu:
            bleu_score = compute_bleu(text_a, text_b)
        if rouge:
            rouge_l_score = compute_rouge_l(text_a, text_b)

    # ── LLM-as-a-Judge ───────────────────────────────────────────────────────
    judge_result = None
    if judge:
        from llm_diff.judge import run_judge  # noqa: PLC0415
        with console.status(f"[dim]Calling judge model ({judge})…[/dim]", spinner="dots"):
            judge_result = await run_judge(
                prompt=display_prompt,
                response_a=comparison.response_a.text,
                response_b=comparison.response_b.text,
                judge_model=judge,
                config=config,
            )

    # ── Cost estimation ───────────────────────────────────────────────────────
    cost_a = cost_b = None
    if show_cost:
        from llm_diff.pricing import estimate_cost  # noqa: PLC0415
        cost_a = estimate_cost(
            comparison.response_a.model,
            prompt_tokens=comparison.response_a.prompt_tokens,
            completion_tokens=comparison.response_a.completion_tokens,
        )
        cost_b = estimate_cost(
            comparison.response_b.model,
            prompt_tokens=comparison.response_b.prompt_tokens,
            completion_tokens=comparison.response_b.completion_tokens,
        )

    # ── JSON structural diff ─────────────────────────────────────────────────
    json_struct_result = None
    if mode == "json-struct":
        from llm_diff.diff import json_struct_diff  # noqa: PLC0415
        json_struct_result = json_struct_diff(
            comparison.response_a.text,
            comparison.response_b.text,
        )

    # ── Render ───────────────────────────────────────────────────────────────
    if mode == "json":
        _render_json(
            comparison, diff_result, console,
            prompt=display_prompt,
            semantic_score=semantic_score,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
            judge_result=judge_result,
            cost_a=cost_a,
            cost_b=cost_b,
        )
    elif mode == "json-struct":
        from llm_diff.renderer import render_json_struct_diff  # noqa: PLC0415
        render_json_struct_diff(
            prompt=display_prompt,
            result=comparison,
            json_struct_result=json_struct_result,
            diff_result=diff_result,
            console=console,
            judge_result=judge_result,
            cost_a=cost_a,
            cost_b=cost_b,
        )
    else:
        render_diff(
            prompt=display_prompt,
            result=comparison,
            diff_result=diff_result,
            console=console,
            semantic_score=semantic_score,
            paragraph_scores=paragraph_scores,
            bleu_score=bleu_score,
            rouge_l_score=rouge_l_score,
            judge_result=judge_result,
            cost_a=cost_a,
            cost_b=cost_b,
        )

    if verbose:
        _render_verbose(comparison, console)

    # ── Fail-under check ─────────────────────────────────────────────────────
    if fail_under is not None:
        primary = (
            semantic_score if semantic_score is not None else diff_result.similarity
        )
        if primary < fail_under:
            err_console = Console(stderr=True)
            err_console.print(
                f"[bold red]--fail-under {fail_under:.2f}: "
                f"score {primary:.4f} is below threshold.[/bold red]"
            )
            sys.exit(1)

    # ── Save HTML report ─────────────────────────────────────────────────────
    if out or config.save:
        from llm_diff.report import auto_save_report, build_report, save_report  # noqa: PLC0415

        html = build_report(
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
        if out:
            saved = save_report(html, out)
            console.print(f"[dim]Report saved → {saved}[/dim]")
        if config.save:
            saved = auto_save_report(html, model_a, model_b)
            console.print(f"[dim]Report saved → {saved}[/dim]")


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _render_json(
    comparison: ComparisonResult,
    diff_result: DiffResult,
    console: Console,
    *,
    prompt: str = "",
    semantic_score: float | None = None,
    bleu_score: float | None = None,
    rouge_l_score: float | None = None,
    judge_result: object = None,
    cost_a: object = None,
    cost_b: object = None,
) -> None:
    ra = comparison.response_a
    rb = comparison.response_b
    payload: dict = {
        "prompt": prompt,
        "model_a": ra.model,
        "model_b": rb.model,
        "similarity_score": round(diff_result.similarity, 4),
        "tokens": {"a": ra.total_tokens, "b": rb.total_tokens},
        "latency_ms": {"a": ra.latency_ms, "b": rb.latency_ms},
        "diff": diff_result.to_dict()["chunks"],
    }
    if semantic_score is not None:
        payload["semantic_score"] = round(semantic_score, 4)
    if bleu_score is not None:
        payload["bleu_score"] = round(bleu_score, 4)
    if rouge_l_score is not None:
        payload["rouge_l_score"] = round(rouge_l_score, 4)
    if judge_result is not None:
        payload["judge"] = judge_result.to_dict()  # type: ignore[union-attr]
    if cost_a is not None and cost_b is not None:
        payload["cost"] = {
            "model_a": cost_a.to_dict(),  # type: ignore[union-attr]
            "model_b": cost_b.to_dict(),  # type: ignore[union-attr]
        }
    # Use click.echo not console so output is clean, unprettified JSON.
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


def _render_verbose(comparison: ComparisonResult, console: Console) -> None:
    ra = comparison.response_a
    rb = comparison.response_b
    from rich.table import Table

    table = Table(title="API Metadata", show_header=True, header_style="bold cyan")
    table.add_column("", style="dim")
    table.add_column(f"[yellow]{ra.model}[/yellow]", justify="right")
    table.add_column(f"[magenta]{rb.model}[/magenta]", justify="right")
    table.add_row("Provider", ra.provider, rb.provider)
    table.add_row("Prompt tokens", str(ra.prompt_tokens), str(rb.prompt_tokens))
    table.add_row("Completion tokens", str(ra.completion_tokens), str(rb.completion_tokens))
    table.add_row("Total tokens", str(ra.total_tokens), str(rb.total_tokens))
    table.add_row("Latency (ms)", f"{ra.latency_ms:.0f}", f"{rb.latency_ms:.0f}")
    console.print(table)


# ---------------------------------------------------------------------------
# Multi-model runner
# ---------------------------------------------------------------------------


async def _run_multi(
    *,
    prompt: str,
    models: list[str],
    semantic: bool,
    config: LLMDiffConfig,
    console: Console,
    concurrency: int,
    cache: object | None,
) -> None:
    """Run *prompt* against all *models* and render a pairwise similarity matrix."""
    from llm_diff.multi import run_multi_model  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    n = len(models)
    console.print(
        f"[dim]Running multi-model comparison: {n} models, "
        f"{n*(n-1)//2} pair(s)…[/dim]"
    )

    with console.status("[dim]Calling models…[/dim]", spinner="dots"):
        report = await run_multi_model(
            prompt,
            models=models,
            semantic=semantic,
            config=config,
            concurrency=concurrency,
            cache=cache,
        )

    # ── Header ───────────────────────────────────────────────────────────────
    from rich.rule import Rule  # noqa: PLC0415
    console.print(Rule(style="dim cyan"))
    from rich.text import Text as RichText  # noqa: PLC0415
    header = RichText()
    header.append("  llm-diff", style="bold cyan")
    header.append("  •  Multi-model comparison", style="dim")
    console.print(header)
    prompt_display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    console.print(RichText(f"  Prompt: {prompt_display!r}", style="dim white"))
    console.print(Rule(style="dim cyan"))

    # ── Pairwise matrix table ─────────────────────────────────────────────────
    score_col = "Semantic" if semantic else "Word similarity"
    table = Table(
        title=f"Pairwise {score_col}",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column("Model A", style="yellow")
    table.add_column("Model B", style="magenta")
    table.add_column(score_col, justify="right")

    for pair in report.ranked_pairs():
        score = pair.primary_score
        if score >= 0.8:
            style = "bold green"
        elif score >= 0.5:
            style = "bold yellow"
        else:
            style = "bold red"
        table.add_row(pair.model_a, pair.model_b, RichText(f"{score:.2%}", style=style))

    console.print(table)

    # ── Per-model responses ───────────────────────────────────────────────────
    console.print()
    console.print(RichText("  Individual responses:", style="dim"))
    for m in models:
        resp_text = report.responses.get(m, "")
        mr = report.model_responses.get(m)
        tokens_info = f"  ({mr.total_tokens} tokens, {mr.latency_ms:.0f}ms)" if mr else ""
        console.print(
            RichText(f"\n  [{m}]{tokens_info}", style="bold cyan"),
        )
        console.print(
            RichText(
                "  " + (resp_text[:300] + "…" if len(resp_text) > 300 else resp_text),
                style="white",
            )
        )

    console.print(Rule(style="dim cyan"))


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------


def _die(console: Console, message: str, exit_code: int = 1) -> NoReturn:
    """Print an error message to stderr and exit."""
    # Use a dedicated stderr console so the error is always visible even
    # when stdout has been redirected (e.g. --json piped to a file).
    err_console = Console(stderr=True, no_color=console.no_color)
    err_text = Text()
    err_text.append("Error: ", style="bold red")
    err_text.append(message)
    err_console.print(err_text)
    sys.exit(exit_code)
