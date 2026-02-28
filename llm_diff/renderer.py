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
    from llm_diff.semantic import ParagraphScore

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
    paragraph_scores: list[ParagraphScore] | None = None,
    bleu_score: float | None = None,
    rouge_l_score: float | None = None,
    judge_result: object = None,
    cost_a: object = None,
    cost_b: object = None,
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
    paragraph_scores:
        Optional list of :class:`~llm_diff.semantic.ParagraphScore` objects.
        When provided, a per-paragraph similarity table is printed after the
        footer.
    bleu_score:
        Optional BLEU score (0.0–1.0).
    rouge_l_score:
        Optional ROUGE-L F1 score (0.0–1.0).
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
    if bleu_score is not None:
        bleu_style = _score_colour(bleu_score)
        footer.append("  |  BLEU: ", style="dim")
        footer.append(f"{bleu_score:.2%}", style=bleu_style)
    if rouge_l_score is not None:
        rouge_style = _score_colour(rouge_l_score)
        footer.append("  |  ROUGE-L: ", style="dim")
        footer.append(f"{rouge_l_score:.2%}", style=rouge_style)
    footer.append("  |  Tokens: ", style="dim")
    footer.append(f"{ra.total_tokens}", style="bold white")
    footer.append(" / ", style="dim")
    footer.append(f"{rb.total_tokens}", style="bold white")
    footer.append("  |  Latency: ", style="dim")
    footer.append(f"{ra.latency_ms:.0f}ms", style="bold white")
    footer.append(" / ", style="dim")
    footer.append(f"{rb.latency_ms:.0f}ms", style="bold white")
    console.print(footer)

    # ── Judge result ─────────────────────────────────────────────────────────
    if judge_result is not None:
        winner = getattr(judge_result, "winner", None)
        reasoning = getattr(judge_result, "reasoning", "")
        score_a = getattr(judge_result, "score_a", None)
        score_b = getattr(judge_result, "score_b", None)
        judge_model = getattr(judge_result, "judge_model", "")
        winner_colour = {
            "A": "bold yellow",
            "B": "bold magenta",
            "tie": "bold cyan",
        }.get(winner or "tie", "bold white")
        judge_text = Text()
        judge_text.append("  Judge", style="bold dim")
        judge_text.append(f" [{judge_model}]", style="dim")
        judge_text.append(": ", style="dim")
        judge_text.append(f"Winner = {winner}", style=winner_colour)
        if score_a is not None and score_b is not None:
            judge_text.append(
                f"  (A: {score_a:.1f}  B: {score_b:.1f})", style="dim"
            )
        console.print(judge_text)
        if reasoning:
            console.print(Text(f"  {reasoning}", style="dim italic"))

    # ── Cost breakdown ───────────────────────────────────────────────────────
    if cost_a is not None and cost_b is not None:
        cost_text = Text()
        cost_text.append("  Cost: ", style="dim")
        cost_text.append(
            f"[A] ${getattr(cost_a, 'total_usd', 0.0):.6f}",
            style="bold yellow",
        )
        cost_text.append("  /  ", style="dim")
        cost_text.append(
            f"[B] ${getattr(cost_b, 'total_usd', 0.0):.6f}",
            style="bold magenta",
        )
        known_a = getattr(cost_a, "known_model", False)
        known_b = getattr(cost_b, "known_model", False)
        if not known_a or not known_b:
            cost_text.append("  (estimated — model not in pricing table)", style="dim red")
        console.print(cost_text)

    console.print(Rule(style="dim cyan"))

    # ── Paragraph similarity table ───────────────────────────────────────────
    if paragraph_scores:
        para_table = Table(
            title="Paragraph Similarity",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
        )
        para_table.add_column("§", style="dim", justify="right", no_wrap=True)
        para_table.add_column("Score", justify="right", no_wrap=True)
        para_table.add_column(f"[yellow]{ra.model}[/yellow] (preview)")
        para_table.add_column(f"[magenta]{rb.model}[/magenta] (preview)")

        for ps in paragraph_scores:
            score_style = _score_colour(ps.score)
            preview_a = (ps.text_a[:60] + "…") if len(ps.text_a) > 60 else ps.text_a
            preview_b = (ps.text_b[:60] + "…") if len(ps.text_b) > 60 else ps.text_b
            para_table.add_row(
                str(ps.index + 1),
                Text(f"{ps.score:.0%}", style=score_style),
                preview_a or Text("(empty)", style="dim"),
                preview_b or Text("(empty)", style="dim"),
            )

        console.print()
        console.print(para_table)

# ---------------------------------------------------------------------------
# JSON-struct diff renderer
# ---------------------------------------------------------------------------


def render_json_struct_diff(
    *,
    prompt: str,
    result: ComparisonResult,
    json_struct_result: object,
    diff_result: DiffResult,
    console: Console,
    judge_result: object = None,
    cost_a: object = None,
    cost_b: object = None,
) -> None:
    """Render a structured JSON key/value diff to *console*."""
    from llm_diff.diff import JsonChangeType  # noqa: PLC0415

    ra = result.response_a
    rb = result.response_b

    # Header
    console.print(Rule(style="dim cyan"))
    header = Text()
    header.append("  llm-diff", style=_COLOUR_HEADER)
    header.append("  •  JSON-struct  ", style="dim")
    header.append(ra.model, style="bold yellow")
    header.append("  vs  ", style="dim")
    header.append(rb.model, style="bold magenta")
    console.print(header)
    prompt_display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    console.print(Text(f"  Prompt: {prompt_display!r}", style=_COLOUR_META))
    console.print(Rule(style="dim cyan"))

    # Check validity
    valid_a = getattr(json_struct_result, "is_valid_json_a", False)
    valid_b = getattr(json_struct_result, "is_valid_json_b", False)

    if not valid_a or not valid_b:
        # Fall back to word diff when one/both are not valid JSON
        console.print(
            Text(
                "  ⚠ One or both responses are not valid JSON — "
                "falling back to word diff.",
                style="bold yellow",
            )
        )
        wdr = getattr(json_struct_result, "word_diff_result", diff_result)
        if wdr is not None:
            diff_text = _build_diff_text(wdr.chunks)
            console.print()
            console.print(diff_text, soft_wrap=True)
        console.print(Rule(style="dim cyan"))
        return

    entries = getattr(json_struct_result, "entries", [])

    _CHANGE_COLOUR: dict[object, str] = {
        JsonChangeType.ADDED: "bold green",
        JsonChangeType.REMOVED: "bold red",
        JsonChangeType.CHANGED: "bold yellow",
        JsonChangeType.TYPE_CHANGE: "bold magenta",
        JsonChangeType.UNCHANGED: "dim white",
    }

    tbl = Table(
        title="JSON key/value diff",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    tbl.add_column("Key", style="dim white", no_wrap=True)
    tbl.add_column("Change", justify="center", no_wrap=True)
    tbl.add_column(f"[yellow]{ra.model}[/yellow] (A)")
    tbl.add_column(f"[magenta]{rb.model}[/magenta] (B)")

    for entry in entries:
        ct = entry.change_type
        col = _CHANGE_COLOUR.get(ct, "white")
        label = ct.value.upper() if hasattr(ct, "value") else str(ct)
        va = "" if entry.value_a is None else str(entry.value_a)
        vb = "" if entry.value_b is None else str(entry.value_b)
        tbl.add_row(
            entry.path,
            Text(label, style=col),
            Text(va[:80], style="bold yellow" if va else "dim"),
            Text(vb[:80], style="bold magenta" if vb else "dim"),
        )

    console.print(tbl)

    # Summary
    summary = Text()
    summary.append("  ", style="")
    added = getattr(json_struct_result, "added", [])
    removed = getattr(json_struct_result, "removed", [])
    changed = getattr(json_struct_result, "changed", [])
    unchanged = getattr(json_struct_result, "unchanged", [])
    summary.append(f"+{len(added)} ", style="bold green")
    summary.append(f"-{len(removed)} ", style="bold red")
    summary.append(f"~{len(changed)} ", style="bold yellow")
    summary.append(f"={len(unchanged)}", style="dim white")
    console.print(summary)

    # Reuse word-diff similarity for the footer metric
    score_style = _score_colour(diff_result.similarity)
    footer = Text()
    footer.append("  Similarity (word): ", style="dim")
    footer.append(diff_result.similarity_pct, style=score_style)
    console.print(footer)

    # Judge / cost
    if judge_result is not None:
        winner = getattr(judge_result, "winner", None)
        judge_model = getattr(judge_result, "judge_model", "")
        winner_colour = {"A": "bold yellow", "B": "bold magenta", "tie": "bold cyan"}.get(
            winner or "tie", "bold white"
        )
        jt = Text()
        jt.append(f"  Judge [{judge_model}]: ", style="dim")
        jt.append(f"Winner = {winner}", style=winner_colour)
        console.print(jt)
    if cost_a is not None and cost_b is not None:
        ct2 = Text()
        ct2.append("  Cost: ", style="dim")
        ct2.append(f"[A] ${getattr(cost_a, 'total_usd', 0.0):.6f}", style="bold yellow")
        ct2.append("  /  ", style="dim")
        ct2.append(f"[B] ${getattr(cost_b, 'total_usd', 0.0):.6f}", style="bold magenta")
        console.print(ct2)

    console.print(Rule(style="dim cyan"))


# ---------------------------------------------------------------------------
# Multi-model report renderer
# ---------------------------------------------------------------------------


def render_multi_model_report(*, report: object, console: Console) -> None:
    """Render a :class:`~llm_diff.multi.MultiModelReport` to *console*."""
    models: list[str] = getattr(report, "models", [])
    responses: dict = getattr(report, "responses", {})
    model_responses: dict = getattr(report, "model_responses", {})

    console.print(Rule(style="dim cyan"))
    header = Text()
    header.append("  llm-diff", style=_COLOUR_HEADER)
    header.append("  •  Multi-model comparison", style="dim")
    console.print(header)

    prompt = getattr(report, "prompt", "")
    prompt_display = prompt if len(prompt) <= 80 else prompt[:77] + "..."
    console.print(Text(f"  Prompt: {prompt_display!r}", style=_COLOUR_META))
    console.print(Rule(style="dim cyan"))

    # Similarity matrix table
    ranked = report.ranked_pairs()  # type: ignore[union-attr]
    tbl = Table(
        title="Pairwise Similarity (ranked)",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
    )
    tbl.add_column("Model A", style="bold yellow")
    tbl.add_column("Model B", style="bold magenta")
    tbl.add_column("Score", justify="right")

    for pair in ranked:
        sc = pair.primary_score
        sc_style = _score_colour(sc)
        tbl.add_row(pair.model_a, pair.model_b, Text(f"{sc:.2%}", style=sc_style))

    console.print(tbl)

    # Per-model responses summary
    console.print()
    for m in models:
        mr = model_responses.get(m)
        resp_text = responses.get(m, "")
        tokens_info = f" ({mr.total_tokens} tokens, {mr.latency_ms:.0f}ms)" if mr else ""
        console.print(Text(f"\n  [{m}]{tokens_info}", style="bold cyan"))
        excerpt = resp_text[:200] + ("…" if len(resp_text) > 200 else "")
        console.print(Text("  " + excerpt, style="white"))

    console.print(Rule(style="dim cyan"))