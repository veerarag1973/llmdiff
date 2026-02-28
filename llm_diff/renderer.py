"""Terminal output rendering for llm-diff.

Uses the ``rich`` library to produce a colour-coded, readable diff in any
modern terminal.  All rendering logic is isolated here so the CLI layer
stays thin and so that the renderer can be unit-tested independently.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from llm_diff.diff import DiffChunk, DiffResult, DiffType
from llm_diff.providers import ComparisonResult

if TYPE_CHECKING:  # pragma: no cover
    pass

# Colour palette — chosen to be legible on both dark and light terminals.
_COLOUR_INSERT = "bold green"
_COLOUR_DELETE = "bold red"
_COLOUR_EQUAL = "white"
_COLOUR_HEADER = "bold cyan"
_COLOUR_META = "dim white"
_COLOUR_SCORE_HIGH = "bold green"    # ≥ 80 %
_COLOUR_SCORE_MID = "bold yellow"    # 50–79 %
_COLOUR_SCORE_LOW = "bold red"       # < 50 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score_colour(similarity: float) -> str:
    if similarity >= 0.8:
        return _COLOUR_SCORE_HIGH
    if similarity >= 0.5:
        return _COLOUR_SCORE_MID
    return _COLOUR_SCORE_LOW


def _build_diff_text(chunks: list[DiffChunk]) -> Text:
    """Build a :class:`rich.text.Text` with inline colour markup."""
    out = Text()
    for chunk in chunks:
        if chunk.type == DiffType.INSERT:
            out.append(chunk.text, style=_COLOUR_INSERT)
        elif chunk.type == DiffType.DELETE:
            out.append(chunk.text, style=_COLOUR_DELETE)
        else:
            out.append(chunk.text, style=_COLOUR_EQUAL)
    return out


def _token_badge(label: str, count: int) -> Text:
    t = Text()
    t.append(label, style="dim")
    t.append(str(count), style="bold white")
    return t


def _latency_badge(label: str, ms: float) -> Text:
    t = Text()
    t.append(label, style="dim")
    t.append(f"{ms:.0f}ms", style="bold white")
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_diff(
    *,
    prompt: str,
    result: ComparisonResult,
    diff_result: DiffResult,
    console: Console,
    semantic_score: float | None = None,
) -> None:
    """Write the full diff output to *console*.

    Parameters
    ----------
    prompt:
        The original prompt text shown in the header.
    result:
        The raw API responses from both models.
    diff_result:
        Pre-computed word-level diff of the two responses.
    console:
        A :class:`rich.console.Console` instance.  Pass ``no_color=True``
        when constructing it to honour the ``--no-color`` flag.
    semantic_score:
        Optional semantic cosine similarity (0.0–1.0).  When provided, an
        extra "Semantic:" line is added to the footer.
    """
    ra = result.response_a
    rb = result.response_b

    # ── Header ──────────────────────────────────────────────────────────────
    console.print(Rule(style="dim cyan"))
    header = Text()
    header.append("  llm-diff", style=_COLOUR_HEADER)
    header.append("  •  ", style="dim")
    header.append(ra.model, style="bold yellow")
    header.append("  vs  ", style="dim")
    header.append(rb.model, style="bold magenta")
    console.print(header)

    prompt_display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    console.print(Text(f"  Prompt: {prompt_display!r}", style=_COLOUR_META))
    console.print(Rule(style="dim cyan"))

    # ── Side-by-side labels ──────────────────────────────────────────────────
    label_table = Table.grid(expand=True, padding=(0, 2))
    label_table.add_column(ratio=1)
    label_table.add_column(ratio=1)
    label_a = Text()
    label_a.append("  [A] ", style="dim")
    label_a.append(ra.model, style="bold yellow")
    label_b = Text()
    label_b.append("  [B] ", style="dim")
    label_b.append(rb.model, style="bold magenta")
    label_table.add_row(label_a, label_b)
    console.print(label_table)

    # ── Diff body ───────────────────────────────────────────────────────────
    # We present a unified view — deletions (A only) in red, insertions
    # (B only) in green, and unchanged text in plain white.
    diff_text = _build_diff_text(diff_result.chunks)
    console.print()
    console.print(diff_text, soft_wrap=True)
    console.print()

    # ── Footer metrics ───────────────────────────────────────────────────────
    score_style = _score_colour(diff_result.similarity)
    footer = Text()
    footer.append("  Similarity: ", style="dim")
    footer.append(diff_result.similarity_pct, style=score_style)
    if semantic_score is not None:
        sem_style = _score_colour(semantic_score)
        footer.append("  |  Semantic: ", style="dim")
        footer.append(f"{semantic_score:.2%}", style=sem_style)
    footer.append("  |  Tokens: ", style="dim")
    footer.append(f"{ra.total_tokens}", style="bold white")
    footer.append(" / ", style="dim")
    footer.append(f"{rb.total_tokens}", style="bold white")
    footer.append("  |  Latency: ", style="dim")
    footer.append(f"{ra.latency_ms:.0f}ms", style="bold white")
    footer.append(" / ", style="dim")
    footer.append(f"{rb.latency_ms:.0f}ms", style="bold white")
    console.print(footer)
    console.print(Rule(style="dim cyan"))
