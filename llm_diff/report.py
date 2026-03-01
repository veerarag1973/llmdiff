"""HTML report generator for llm-diff.

Builds a fully self-contained, offline-capable HTML report from diff results
using the Jinja2 templates in ``llm_diff/templates/``.

Public API
----------
- :func:`build_report`       — render a single-diff HTML report.
- :func:`build_batch_report` — render a combined batch report for multiple diffs.
- :func:`save_report`        — write an HTML string to an arbitrary file path.
- :func:`auto_save_report`   — save to ``./diffs/`` with an auto-generated name.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader

from llm_diff import __version__

if TYPE_CHECKING:
    from llm_diff.diff import DiffResult
    from llm_diff.providers import ComparisonResult
    from llm_diff.semantic import ParagraphScore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 environment
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "report.html.j2"

_jinja_env: Environment | None = None


def _get_jinja_env() -> Environment:
    """Return a cached Jinja2 :class:`~jinja2.Environment`."""
    global _jinja_env  # noqa: PLW0603
    if _jinja_env is None:
        _jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=True,  # always-on: covers .html.j2 extensions too
            keep_trailing_newline=True,
        )
    return _jinja_env


# ---------------------------------------------------------------------------
# Score helpers (duplicated from renderer to keep report.py independent)
# ---------------------------------------------------------------------------


def _score_class(similarity: float) -> str:
    """Return a CSS class name based on the similarity score."""
    if similarity >= 0.8:
        return "score-high"
    if similarity >= 0.5:
        return "score-mid"
    return "score-low"


def _similarity_pct(similarity: float) -> str:
    return f"{similarity:.2%}"


def _safe_model_slug(model: str) -> str:
    """Convert a model identifier to a filesystem-safe slug."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_report(
    *,
    prompt: str,
    result: ComparisonResult,
    diff_result: DiffResult,
    semantic_score: float | None = None,
    paragraph_scores: list[ParagraphScore] | None = None,
    bleu_score: float | None = None,
    rouge_l_score: float | None = None,
    generated_at: str | None = None,
    judge_result: object = None,
    cost_a: object = None,
    cost_b: object = None,
) -> str:
    """Render the HTML report template and return the HTML string.

    Parameters
    ----------
    prompt:
        The prompt text shown in the report header.
    result:
        The paired model responses from :func:`~llm_diff.providers.compare_models`.
    diff_result:
        The word-level diff computed by :func:`~llm_diff.diff.word_diff`.
    semantic_score:
        Optional cosine similarity score (0.0–1.0) from :mod:`~llm_diff.semantic`.
        When ``None``, the semantic section is hidden in the report.
    paragraph_scores:
        Optional list of per-paragraph scores from
        :func:`~llm_diff.semantic.compute_paragraph_similarity`.  When provided,
        a paragraph-level similarity section is rendered in the report.
    bleu_score:
        Optional BLEU score (0.0–1.0).  When ``None``, hidden in the report.
    rouge_l_score:
        Optional ROUGE-L F1 score (0.0–1.0).  When ``None``, hidden.
    generated_at:
        Human-readable timestamp string.  Defaults to the current UTC time.

    Returns
    -------
    str
        A fully self-contained HTML document (no external CDN requests).
    """
    ra = result.response_a
    rb = result.response_b

    if generated_at is None:
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build chunk list for the template
    chunks = [{"type": chunk.type.value, "text": chunk.text} for chunk in diff_result.chunks]

    context = {
        "version": __version__,
        "generated_at": generated_at,
        "prompt": prompt,
        "model_a": ra.model,
        "model_b": rb.model,
        "response_a": ra.text,
        "response_b": rb.text,
        "tokens_a": ra.total_tokens,
        "tokens_b": rb.total_tokens,
        "latency_a_ms": f"{ra.latency_ms:.0f}",
        "latency_b_ms": f"{rb.latency_ms:.0f}",
        "word_similarity_pct": _similarity_pct(diff_result.similarity),
        "word_score_class": _score_class(diff_result.similarity),
        "semantic_score": semantic_score,
        "semantic_similarity_pct": (
            _similarity_pct(semantic_score) if semantic_score is not None else ""
        ),
        "semantic_score_class": (
            _score_class(semantic_score) if semantic_score is not None else ""
        ),
        "chunks": chunks,
        "has_paragraph_scores": paragraph_scores is not None,
        "paragraph_scores": (
            [
                {
                    "text_a": ps.text_a,
                    "text_b": ps.text_b,
                    "score": ps.score,
                    "score_pct": f"{ps.score:.0%}",
                    "score_class": _score_class(ps.score),
                    "index": ps.index,
                }
                for ps in paragraph_scores
            ]
            if paragraph_scores is not None
            else []
        ),
        "bleu_score": bleu_score,
        "bleu_score_pct": (_similarity_pct(bleu_score) if bleu_score is not None else ""),
        "bleu_score_class": (_score_class(bleu_score) if bleu_score is not None else ""),
        "rouge_l_score": rouge_l_score,
        "rouge_l_score_pct": (
            _similarity_pct(rouge_l_score) if rouge_l_score is not None else ""
        ),
        "rouge_l_score_class": (
            _score_class(rouge_l_score) if rouge_l_score is not None else ""
        ),
        # Judge result
        "has_judge": judge_result is not None,
        "judge_winner": getattr(judge_result, "winner", None),
        "judge_reasoning": getattr(judge_result, "reasoning", ""),
        "judge_score_a": getattr(judge_result, "score_a", None),
        "judge_score_b": getattr(judge_result, "score_b", None),
        "judge_model": getattr(judge_result, "judge_model", ""),
        # Cost
        "has_cost": cost_a is not None and cost_b is not None,
        "cost_a": cost_a.to_dict() if cost_a is not None else None,  # type: ignore[union-attr]
        "cost_b": cost_b.to_dict() if cost_b is not None else None,  # type: ignore[union-attr]
    }

    env = _get_jinja_env()
    template = env.get_template(_TEMPLATE_NAME)
    return template.render(**context)


def save_report(html: str, path: Path) -> Path:
    """Write *html* to *path*, creating all parent directories as needed.

    Parameters
    ----------
    html:
        The rendered HTML string returned by :func:`build_report`.
    path:
        Destination file path.  Must end with ``.html`` by convention but
        this is not enforced.

    Returns
    -------
    Path
        The absolute path to the written file.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    logger.info("Report saved to %s (%d bytes)", path, len(html))

    # Emit report exported schema event (best-effort)
    try:
        from llm_diff.schema_events import (  # noqa: PLC0415
            emit as schema_emit,
        )
        from llm_diff.schema_events import (
            make_report_exported_event,
        )

        schema_emit(
            make_report_exported_event(
                output_path=str(path),
                format="html",
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("Schema event emission failed", exc_info=True)

    return path


def auto_save_report(
    html: str,
    model_a: str,
    model_b: str,
    *,
    diffs_dir: Path | None = None,
) -> Path:
    """Auto-save *html* to the ``./diffs/`` directory.

    The filename is built from the current UTC timestamp and the two model
    names so that multiple runs do not overwrite each other::

        ./diffs/20260228_153042_gpt-4o_vs_claude-3-5-sonnet.html

    Parameters
    ----------
    html:
        Rendered HTML from :func:`build_report`.
    model_a, model_b:
        Model identifiers used to build the filename.
    diffs_dir:
        Override the default ``./diffs/`` directory.  Useful in tests.

    Returns
    -------
    Path
        Absolute path to the written report.
    """
    target_dir = Path(diffs_dir) if diffs_dir else Path.cwd() / "diffs"
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    slug_a = _safe_model_slug(model_a)
    slug_b = _safe_model_slug(model_b)
    filename = f"{timestamp}_{slug_a}_vs_{slug_b}.html"
    return save_report(html, target_dir / filename)


_BATCH_TEMPLATE_NAME = "batch_report.html.j2"


def build_batch_report(
    *,
    results: list,
    model_a: str,
    model_b: str,
    generated_at: str | None = None,
) -> str:
    """Render a combined batch HTML report and return the HTML string.

    Parameters
    ----------
    results:
        List of :class:`~llm_diff.batch.BatchResult` objects produced by
        :func:`~llm_diff.cli._run_batch`.
    model_a, model_b:
        Model identifiers shown in the report header.
    generated_at:
        Human-readable timestamp string.  Defaults to current UTC time.

    Returns
    -------
    str
        Fully self-contained HTML batch report (no external dependencies).
    """
    if generated_at is None:
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rendered: list[dict] = []
    for r in results:
        ra = r.comparison.response_a
        rb = r.comparison.response_b
        chunks = [
            {"type": chunk.type.value, "text": chunk.text}
            for chunk in r.diff_result.chunks
        ]
        p_scores = getattr(r, "paragraph_scores", None)
        r_bleu = getattr(r, "bleu_score", None)
        r_rouge = getattr(r, "rouge_l_score", None)
        rendered.append(
            {
                "id": r.item.id,
                "prompt": r.item.prompt_text,
                "input_label": r.item.input_label,
                "response_a": ra.text,
                "response_b": rb.text,
                "tokens_a": ra.total_tokens,
                "tokens_b": rb.total_tokens,
                "latency_a_ms": f"{ra.latency_ms:.0f}",
                "latency_b_ms": f"{rb.latency_ms:.0f}",
                "word_similarity_pct": _similarity_pct(r.diff_result.similarity),
                "word_score_class": _score_class(r.diff_result.similarity),
                "semantic_score": r.semantic_score,
                "semantic_similarity_pct": (
                    _similarity_pct(r.semantic_score)
                    if r.semantic_score is not None
                    else ""
                ),
                "semantic_score_class": (
                    _score_class(r.semantic_score)
                    if r.semantic_score is not None
                    else ""
                ),
                "bleu_score": r_bleu,
                "bleu_score_pct": (
                    _similarity_pct(r_bleu) if r_bleu is not None else ""
                ),
                "bleu_score_class": (
                    _score_class(r_bleu) if r_bleu is not None else ""
                ),
                "rouge_l_score": r_rouge,
                "rouge_l_score_pct": (
                    _similarity_pct(r_rouge) if r_rouge is not None else ""
                ),
                "rouge_l_score_class": (
                    _score_class(r_rouge) if r_rouge is not None else ""
                ),
                "has_paragraph_scores": p_scores is not None,
                "paragraph_scores": (
                    [
                        {
                            "text_a": ps.text_a,
                            "text_b": ps.text_b,
                            "score": ps.score,
                            "score_pct": f"{ps.score:.0%}",
                            "score_class": _score_class(ps.score),
                            "index": ps.index,
                        }
                        for ps in p_scores
                    ]
                    if p_scores is not None
                    else []
                ),
                "chunks": chunks,
            }
        )

    avg_word = (
        sum(r.diff_result.similarity for r in results) / len(results)
        if results
        else 0.0
    )
    semantic_scores = [
        r.semantic_score for r in results if r.semantic_score is not None
    ]
    avg_semantic = (
        sum(semantic_scores) / len(semantic_scores) if semantic_scores else None
    )

    bleu_scores = [
        getattr(r, "bleu_score", None) for r in results
        if getattr(r, "bleu_score", None) is not None
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None

    rouge_scores = [
        getattr(r, "rouge_l_score", None) for r in results
        if getattr(r, "rouge_l_score", None) is not None
    ]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else None

    has_paragraph = any(
        r.get("has_paragraph_scores") for r in rendered
    )

    context = {
        "version": __version__,
        "generated_at": generated_at,
        "model_a": model_a,
        "model_b": model_b,
        "total": len(results),
        "results": rendered,
        "avg_word_similarity_pct": _similarity_pct(avg_word),
        "avg_word_score_class": _score_class(avg_word),
        "has_semantic": avg_semantic is not None,
        "has_paragraph": has_paragraph,
        "avg_semantic_similarity_pct": (
            _similarity_pct(avg_semantic) if avg_semantic is not None else ""
        ),
        "avg_semantic_score_class": (
            _score_class(avg_semantic) if avg_semantic is not None else ""
        ),
        "has_bleu": avg_bleu is not None,
        "avg_bleu_pct": _similarity_pct(avg_bleu) if avg_bleu is not None else "",
        "avg_bleu_score_class": _score_class(avg_bleu) if avg_bleu is not None else "",
        "has_rouge": avg_rouge is not None,
        "avg_rouge_pct": _similarity_pct(avg_rouge) if avg_rouge is not None else "",
        "avg_rouge_score_class": (
            _score_class(avg_rouge) if avg_rouge is not None else ""
        ),
    }

    env = _get_jinja_env()
    template = env.get_template(_BATCH_TEMPLATE_NAME)
    return template.render(**context)
