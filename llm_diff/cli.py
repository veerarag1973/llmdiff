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
    # Keep third-party libraries quieter even in verbose mode.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Shared options (reusable decorator group)
# ---------------------------------------------------------------------------

_model_options = [
    click.option(
        "--model-a", "-a", "model_a",
        metavar="MODEL",
        help="Model identifier for side A (e.g. gpt-4o)",
    ),
    click.option(
        "--model-b", "-b", "model_b",
        metavar="MODEL",
        help="Model identifier for side B (e.g. claude-3-5-sonnet)",
    ),
    click.option(
        "--model", "-m", "model",
        metavar="MODEL",
        default=None,
        help="Use the same model for both sides (prompt-diff mode).",
    ),
]

_output_options = [
    click.option(
        "--semantic", "-s", "mode",
        flag_value="semantic",
        help="Semantic diff using embedding similarity.",
    ),
    click.option(
        "--json", "-j", "mode",
        flag_value="json",
        help="Output raw JSON diff to stdout.",
    ),
    click.option(
        "--out", "-o", "out",
        metavar="PATH",
        default=None,
        help="Save HTML report to this path.",
    ),
    click.option(
        "--save",
        is_flag=True,
        default=False,
        help="Auto-save HTML report to ./diffs/ directory.",
    ),
]

_request_options = [
    click.option(
        "--temperature", "-t",
        default=None,
        type=float,
        metavar="FLOAT",
        help="Temperature (default 0.7).",
    ),
    click.option(
        "--max-tokens",
        default=None,
        type=int,
        metavar="INT",
        help="Max tokens per response (default 1024).",
    ),
    click.option(
        "--timeout",
        default=None,
        type=int,
        metavar="SECS",
        help="Request timeout in seconds (default 30).",
    ),
]

_display_options = [
    click.option(
        "--no-color",
        is_flag=True,
        default=False,
        help="Disable terminal colour output.",
    ),
    click.option(
        "--verbose", "-v",
        is_flag=True,
        default=False,
        help="Show full API metadata.",
    ),
]


def _add_options(options: list) -> click.decorators.FC:  # pragma: no cover
    """Decorator factory to attach a list of click options to a command."""
    def decorator(f: click.decorators.FC) -> click.decorators.FC:  # pragma: no cover
        for option in reversed(options):  # pragma: no cover
            f = option(f)  # pragma: no cover
        return f  # pragma: no cover
    return decorator  # pragma: no cover


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
    type=click.Choice(["word", "json"], case_sensitive=False),
    show_default=True,
    help="Diff mode.",
)
@click.option(
    "--semantic", "-s",
    is_flag=True, default=False,
    help="Compute and show embedding-based similarity score.",
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
    batch: str | None,
    out: str | None,
    save: bool,
    temperature: float | None,
    max_tokens: int | None,
    timeout: int | None,
    no_color: bool,
    verbose: bool,
) -> None:
    """Compare two LLM responses — semantically, visually, and at scale.

    \b
    Examples:
      llm-diff "Explain recursion" -a gpt-4o -b claude-3-5-sonnet
      llm-diff "Explain recursion" -a gpt-4o -b claude-3-5-sonnet --semantic
      llm-diff --prompt-a v1.txt --prompt-b v2.txt --model gpt-4o
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
                    out=out,
                    config=cfg,
                    console=console,
                    verbose=verbose,
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
                    out=out,
                    config=cfg,
                    console=console,
                    verbose=verbose,
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
    out: str | None,
    config: LLMDiffConfig,
    console: Console,
    verbose: bool,
) -> None:
    """Orchestrate the batch diff pipeline asynchronously.

    Loads all :class:`~llm_diff.batch.BatchItem` objects from *batch*, runs the
    full diff pipeline for each, renders terminal output, and optionally writes
    a combined HTML report to *out*.
    """
    from llm_diff.batch import BatchResult, load_batch  # noqa: PLC0415

    items = load_batch(batch)

    batch_results: list[BatchResult] = []

    for i, item in enumerate(items, 1):
        console.rule(f"[dim][{i}/{len(items)}] {item.id}[/dim]")

        with console.status("[dim]Calling models…[/dim]", spinner="dots"):
            comparison: ComparisonResult = await compare_models(
                prompt_a=item.prompt_text,
                prompt_b=item.prompt_text,
                model_a=model_a,
                model_b=model_b,
                config=config,
            )

        diff_result: DiffResult = word_diff(
            comparison.response_a.text,
            comparison.response_b.text,
        )

        semantic_score: float | None = None
        if semantic:
            from llm_diff.semantic import compute_semantic_similarity  # noqa: PLC0415

            semantic_score = compute_semantic_similarity(
                comparison.response_a.text,
                comparison.response_b.text,
            )

        render_diff(
            prompt=item.prompt_text[:60],
            result=comparison,
            diff_result=diff_result,
            console=console,
            semantic_score=semantic_score,
        )

        if verbose:
            _render_verbose(comparison, console)

        batch_results.append(
            BatchResult(
                item=item,
                comparison=comparison,
                diff_result=diff_result,
                semantic_score=semantic_score,
            )
        )

    console.rule("[dim]Batch complete[/dim]")
    console.print(f"[dim]Processed {len(items)} prompt(s).[/dim]")

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
    out: str | None,
    config: LLMDiffConfig,
    console: Console,
    verbose: bool,
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
        )

    # ── Compute diff ─────────────────────────────────────────────────────────
    diff_result: DiffResult = word_diff(
        comparison.response_a.text,
        comparison.response_b.text,
    )

    # ── Semantic similarity ──────────────────────────────────────────────────
    semantic_score: float | None = None
    if semantic:
        from llm_diff.semantic import compute_semantic_similarity  # noqa: PLC0415
        semantic_score = compute_semantic_similarity(
            comparison.response_a.text,
            comparison.response_b.text,
        )

    # ── Render ───────────────────────────────────────────────────────────────
    if mode == "json":
        _render_json(
            comparison, diff_result, console,
            prompt=display_prompt,
            semantic_score=semantic_score,
        )
    else:
        render_diff(
            prompt=display_prompt,
            result=comparison,
            diff_result=diff_result,
            console=console,
            semantic_score=semantic_score,
        )

    if verbose:
        _render_verbose(comparison, console)

    # ── Save HTML report ─────────────────────────────────────────────────────
    if out or config.save:
        from llm_diff.report import auto_save_report, build_report, save_report  # noqa: PLC0415

        html = build_report(
            prompt=display_prompt,
            result=comparison,
            diff_result=diff_result,
            semantic_score=semantic_score,
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
