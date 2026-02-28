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

from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_diff import __version__

if TYPE_CHECKING:
    from llm_diff.diff import DiffResult
    from llm_diff.providers import ComparisonResult

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
            autoescape=select_autoescape(["html"]),
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
    generated_at: str | None = None,
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
        "avg_semantic_similarity_pct": (
            _similarity_pct(avg_semantic) if avg_semantic is not None else ""
        ),
        "avg_semantic_score_class": (
            _score_class(avg_semantic) if avg_semantic is not None else ""
        ),
    }

    env = _get_jinja_env()
    template = env.get_template(_BATCH_TEMPLATE_NAME)
    return template.render(**context)
