"""llm-toolkit-schema integration for llm-diff.

This module provides a thin, zero-configuration integration between llm-diff
and the ``llm-toolkit-schema`` event envelope.  Every major operation in the
diff pipeline — comparison started/completed, model trace spans, cache
lookups, cost recording, and judge evaluations — now emits a structured,
schema-validated :class:`~llm_toolkit_schema.Event`.

Architecture
------------
A module-level :class:`EventEmitter` singleton collects events.  By default
it operates in *sink* mode (events are built and validated but discarded).
Call :func:`configure_emitter` once at startup to attach an exporter, e.g.::

    from llm_diff.schema_events import configure_emitter
    from llm_toolkit_schema.export.jsonl import JSONLExporter

    configure_emitter(exporter=JSONLExporter("events.jsonl"))

After that every comparison automatically appends schema-valid events to
``events.jsonl``.

Usage (library)
---------------
.. code-block:: python

    import asyncio
    from llm_diff import compare
    from llm_diff.schema_events import configure_emitter, get_emitter
    from llm_toolkit_schema.export.jsonl import JSONLExporter

    configure_emitter(exporter=JSONLExporter("events.jsonl"))
    asyncio.run(compare("Explain recursion", model_a="gpt-4o", model_b="claude-3-5-sonnet"))
    events = get_emitter().events  # list of Event objects collected in memory
"""

from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable

from llm_diff import __version__

logger = logging.getLogger(__name__)

# Source string embedded in every emitted event.
_SOURCE = f"llm-diff@{__version__}"

if TYPE_CHECKING:
    from llm_toolkit_schema import Event


# ---------------------------------------------------------------------------
# Lazy import helpers — keep startup cost low
# ---------------------------------------------------------------------------


def _llm_toolkit() -> Any:
    """Return the top-level ``llm_toolkit_schema`` module."""
    import llm_toolkit_schema  # noqa: PLC0415

    return llm_toolkit_schema


def _event_cls() -> type:
    return _llm_toolkit().Event


def _tags_cls() -> type:
    return _llm_toolkit().Tags


def _event_type() -> Any:
    return _llm_toolkit().EventType


def _diff_ns() -> Any:
    from llm_toolkit_schema.namespaces import diff as _diff  # noqa: PLC0415

    return _diff


def _trace_ns() -> Any:
    from llm_toolkit_schema.namespaces import trace as _trace  # noqa: PLC0415

    return _trace


def _cache_ns() -> Any:
    from llm_toolkit_schema.namespaces import cache as _cache  # noqa: PLC0415

    return _cache


def _cost_ns() -> Any:
    from llm_toolkit_schema.namespaces import cost as _cost  # noqa: PLC0415

    return _cost


def _eval_ns() -> Any:
    from llm_toolkit_schema.namespaces import eval_ as _eval  # noqa: PLC0415

    return _eval


def _ulid_or_empty() -> str:
    return str(uuid.uuid4()).replace("-", "")[:26]


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class EventEmitter:
    """Collects and optionally exports llm-toolkit-schema :class:`Event` objects.

    Parameters
    ----------
    exporter:
        Any callable that accepts a single :class:`~llm_toolkit_schema.Event`
        argument.  By default events are only collected in memory (see
        :attr:`events`).  Pass a ``JSONLExporter`` or any compatible object
        with an ``export`` method (or a plain callable) to also ship events
        to an external backend.
    collect:
        When ``True`` (default), events are appended to the in-memory
        :attr:`events` list.  Disable when memory overhead matters in
        long-running processes.
    """

    def __init__(
        self,
        exporter: Callable[[Any], Any] | None = None,
        *,
        collect: bool = True,
    ) -> None:
        self._exporter = exporter
        self._collect = collect
        self._events: list[Any] = []

    @property
    def events(self) -> list[Any]:
        """Read-only list of all :class:`~llm_toolkit_schema.Event` objects collected."""
        return list(self._events)

    def emit(self, event: Any) -> None:  # noqa: ANN401
        """Validate and emit *event*.

        If ``collect=True``, the event is appended to :attr:`events`.
        If an *exporter* is configured, it is called with the event.
        Errors during export are logged as warnings and do not propagate.
        """
        try:
            event.validate()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Schema validation failed for event %s: %s", event.event_type, exc)
            return

        if self._collect:
            self._events.append(event)

        if self._exporter is not None:
            try:
                # Support both callable exporters and object exporters with .export()
                if hasattr(self._exporter, "export"):
                    result = self._exporter.export(event)
                    # Handle async exporters gracefully by ignoring coroutines in sync context
                    if hasattr(result, "__await__"):
                        import asyncio  # noqa: PLC0415

                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                loop.create_task(result)
                            else:
                                loop.run_until_complete(result)
                        except RuntimeError:
                            pass  # no event loop available — silently skip
                else:
                    self._exporter(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Event export failed for %s: %s", event.event_type, exc)

    def clear(self) -> None:
        """Remove all collected events from memory."""
        self._events.clear()


# ---------------------------------------------------------------------------
# Global emitter singleton
# ---------------------------------------------------------------------------

_emitter: EventEmitter = EventEmitter()


def get_emitter() -> EventEmitter:
    """Return the global :class:`EventEmitter` instance."""
    return _emitter


def configure_emitter(
    exporter: Callable[[Any], Any] | None = None,
    *,
    collect: bool = True,
) -> EventEmitter:
    """Replace the global emitter with a new configured instance.

    Call this exactly once at application startup before running any
    comparisons.

    Parameters
    ----------
    exporter:
        Any callable or object with an ``export`` method that accepts a
        :class:`~llm_toolkit_schema.Event`.
    collect:
        Whether to keep events in memory (default ``True``).

    Returns
    -------
    EventEmitter
        The newly installed global emitter.
    """
    global _emitter  # noqa: PLW0603

    _emitter = EventEmitter(exporter=exporter, collect=collect)
    return _emitter


def emit(event: Any) -> None:  # noqa: ANN401
    """Emit *event* through the global emitter."""
    _emitter.emit(event)


# ---------------------------------------------------------------------------
# Event factory helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type_value: str,
    payload: dict[str, Any],
    *,
    trace_id: str | None = None,
    span_id: str | None = None,
    org_id: str | None = None,
    session_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> Any:
    """Build a :class:`~llm_toolkit_schema.Event` from the given arguments."""
    Event = _event_cls()
    Tags = _tags_cls()

    kwargs: dict[str, Any] = {
        "event_type": event_type_value,
        "source": _SOURCE,
        "payload": payload,
    }
    if trace_id is not None:
        kwargs["trace_id"] = trace_id
    if span_id is not None:
        kwargs["span_id"] = span_id
    if org_id is not None:
        kwargs["org_id"] = org_id
    if session_id is not None:
        kwargs["session_id"] = session_id
    if tags:
        kwargs["tags"] = Tags(**tags)

    return Event(**kwargs)


# ---------------------------------------------------------------------------
# llm.diff.* — Comparison lifecycle events
# ---------------------------------------------------------------------------


def make_comparison_started_event(
    *,
    model_a: str,
    model_b: str,
    prompt: str,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.diff.comparison.started`` event.

    Parameters
    ----------
    model_a:
        Identifier of the first model (e.g. ``"gpt-4o"``).
    model_b:
        Identifier of the second model (e.g. ``"claude-3-5-sonnet"``).
    prompt:
        The full prompt text used for the comparison.
    session_id:
        Optional session identifier for correlation.
    org_id:
        Optional organisation identifier.
    """
    ET = _event_type()
    payload: dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        "prompt_length": len(prompt),
    }
    return _make_event(
        ET.DIFF_COMPARISON_STARTED,
        payload,
        session_id=session_id,
        org_id=org_id,
    )


def make_comparison_completed_event(
    *,
    model_a: str,
    model_b: str,
    diff_type: str = "word-level",
    prompt_diff: str | None = None,
    completion_diff: str | None = None,
    similarity_score: float | None = None,
    base_event_id: str | None = None,
    model_a_text: str | None = None,
    model_b_text: str | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.diff.comparison.completed`` event with a DiffComparisonPayload."""
    ET = _event_type()
    ns = _diff_ns()

    diff_result_dict: dict[str, Any] | None = None
    if completion_diff:
        diff_result_dict = {"unified_diff": completion_diff}
    elif prompt_diff:
        diff_result_dict = {"unified_diff": prompt_diff}

    payload_obj = ns.DiffComparisonPayload(
        source_id=base_event_id or model_a,
        target_id=model_b,
        diff_type=diff_type,
        similarity_score=similarity_score,
        source_text=model_a_text,
        target_text=model_b_text,
        diff_result=diff_result_dict,
    )
    payload = dataclasses.asdict(payload_obj)

    return _make_event(
        ET.DIFF_COMPARISON_COMPLETED,
        payload,
        session_id=session_id,
        org_id=org_id,
    )


def make_report_exported_event(
    *,
    output_path: str,
    format: str = "html",
    comparison_event_id: str = "",
    report_id: str | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.diff.report.exported`` event with DiffReportPayload."""
    ET = _event_type()
    ns = _diff_ns()

    payload_obj = ns.DiffReportPayload(
        report_id=report_id or _ulid_or_empty(),
        comparison_event_id=comparison_event_id or _ulid_or_empty(),
        format=format,
        export_path=output_path,
    )
    payload = dataclasses.asdict(payload_obj)
    return _make_event(
        ET.DIFF_REPORT_EXPORTED,
        payload,
        session_id=session_id,
        org_id=org_id,
    )


# ---------------------------------------------------------------------------
# llm.trace.* — Model span events
# ---------------------------------------------------------------------------


def make_trace_span_event(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int | None = None,
    latency_ms: float,
    finish_reason: str | None = None,
    stream: bool = False,
    provider: str | None = None,
    cost_usd: float | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.trace.span.completed`` event with SpanCompletedPayload.

    Parameters
    ----------
    model:
        Model identifier string (e.g. ``"gpt-4o"``).
    prompt_tokens:
        Number of input tokens consumed.
    completion_tokens:
        Number of output tokens generated.
    total_tokens:
        Total token count; inferred from prompt + completion if ``None``.
    latency_ms:
        End-to-end request latency in milliseconds.
    finish_reason:
        Provider finish reason string (``"stop"``, ``"length"``, etc.).
    stream:
        Whether the response was streamed.
    provider:
        Provider name for tagging (``"openai"``, ``"anthropic"``, etc.).
    """
    ET = _event_type()
    ns = _trace_ns()

    total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
    token_usage = ns.TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
    )
    model_info = ns.ModelInfo(
        name=model,
        provider=provider or "unknown",
        version=None,
    )
    payload_obj = ns.SpanCompletedPayload(
        span_name="llm-diff-model-call",
        status="ok" if finish_reason != "error" else "error",
        duration_ms=latency_ms,
        model=model_info,
        token_usage=token_usage,
        cost_usd=cost_usd,
    )
    payload = dataclasses.asdict(payload_obj)
    payload["finish_reason"] = finish_reason
    payload["stream"] = stream

    tags: dict[str, str] | None = None
    if provider:
        tags = {"provider": provider, "model": model}

    return _make_event(
        ET.TRACE_SPAN_COMPLETED,
        payload,
        session_id=session_id,
        org_id=org_id,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# llm.cache.* — Cache hit/miss events
# ---------------------------------------------------------------------------


def make_cache_event(
    *,
    hit: bool,
    cache_key: str | None = None,
    ttl_seconds: int | None = None,
    backend: str = "disk",
    latency_ms: float | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.cache.hit`` or ``llm.cache.miss`` event.

    Parameters
    ----------
    hit:
        ``True`` → ``CACHE_HIT``; ``False`` → ``CACHE_MISS``.
    cache_key:
        Opaque cache key used for lookup (first 16 chars of SHA-256 digest).
    ttl_seconds:
        Time-to-live of the cached entry, if known.
    backend:
        Cache backend name (default ``"disk"``).
    latency_ms:
        Cache lookup latency in milliseconds, if measured.
    """
    ET = _event_type()
    ns = _cache_ns()

    if hit:
        payload_obj = ns.CacheHitPayload(
            cache_key_hash=cache_key or "unknown",
            cache_store=backend,
            ttl_seconds=ttl_seconds,
        )
        event_type = ET.CACHE_HIT
    else:
        payload_obj = ns.CacheMissPayload(
            cache_key_hash=cache_key or "unknown",
            cache_store=backend,
        )
        event_type = ET.CACHE_MISS

    payload = dataclasses.asdict(payload_obj)
    if latency_ms is not None:
        payload["latency_ms"] = latency_ms
    return _make_event(event_type, payload, session_id=session_id, org_id=org_id)


# ---------------------------------------------------------------------------
# llm.cost.* — Cost recorded events
# ---------------------------------------------------------------------------


def make_cost_recorded_event(
    *,
    input_cost: float,
    output_cost: float,
    total_cost: float,
    currency: str = "USD",
    pricing_tier: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    span_event_id: str | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.cost.recorded`` event with CostRecordedPayload."""
    ET = _event_type()
    ns = _cost_ns()

    payload_obj = ns.CostRecordedPayload(
        span_event_id=span_event_id or _ulid_or_empty(),
        model_name=model or "unknown",
        provider=provider or "unknown",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
        cost_usd=total_cost,
        currency=currency,
    )
    payload = dataclasses.asdict(payload_obj)
    payload["input_cost_usd"] = input_cost
    payload["output_cost_usd"] = output_cost
    if pricing_tier is not None:
        payload["pricing_tier"] = pricing_tier

    return _make_event(ET.COST_RECORDED, payload, session_id=session_id, org_id=org_id)


# ---------------------------------------------------------------------------
# llm.eval.* — Judge / evaluation events
# ---------------------------------------------------------------------------


def make_eval_scenario_event(
    *,
    evaluator: str,
    score: float | None = None,
    scale: str = "1-10",
    label: str | None = None,
    rationale: str | None = None,
    criteria: list[str] | None = None,
    status: str = "passed",
    duration_ms: float | None = None,
    baseline_score: float | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.eval.scenario.completed`` event with EvalScenarioPayload.

    Parameters
    ----------
    status:
        Must be ``"passed"``, ``"failed"``, or ``"skipped"``.
    """
    ET = _event_type()
    ns = _eval_ns()

    metrics: dict[str, float] | None = None
    if score is not None and criteria:
        metrics = {c: score for c in criteria}
    elif score is not None:
        metrics = {"score": score}

    scenario_name = f"llm-diff/{evaluator}"
    if label:
        scenario_name = f"{scenario_name}/{label}"

    payload_obj = ns.EvalScenarioPayload(
        scenario_id=_ulid_or_empty(),
        scenario_name=scenario_name,
        status=status,
        score=score,
        metrics=metrics,
        baseline_score=baseline_score,
        duration_ms=duration_ms,
    )
    payload = dataclasses.asdict(payload_obj)
    payload["scale"] = scale
    if rationale:
        payload["rationale"] = rationale
    if label:
        payload["label"] = label
    return _make_event(
        ET.EVAL_SCENARIO_COMPLETED, payload, session_id=session_id, org_id=org_id
    )


def make_eval_regression_event(
    *,
    scenario_name: str,
    current_score: float,
    baseline_score: float,
    threshold: float,
    metrics: dict[str, float] | None = None,
    session_id: str | None = None,
    org_id: str | None = None,
) -> Any:
    """Build a ``llm.eval.regression.failed`` event with EvalRegressionPayload.

    Emitted when the ``--fail-under`` threshold is not met, indicating that
    the primary similarity/semantic score has regressed below the minimum
    acceptable level.

    Parameters
    ----------
    scenario_name:
        Human-readable name for the scenario that triggered the regression,
        e.g. ``"llm-diff/fail-under/batch"`` or ``"llm-diff/fail-under/single"``.
    current_score:
        The actual similarity or semantic score that was measured.
    baseline_score:
        The minimum acceptable score (i.e. the ``--fail-under`` value).
    threshold:
        The ``--fail-under`` threshold value (same as *baseline_score* here).
    metrics:
        Optional mapping of metric names to values for richer diagnostics.
    session_id:
        Optional session identifier for correlation.
    org_id:
        Optional organisation identifier.
    """
    ET = _event_type()
    ns = _eval_ns()

    payload_obj = ns.EvalRegressionPayload(
        scenario_id=_ulid_or_empty(),
        scenario_name=scenario_name,
        current_score=current_score,
        baseline_score=baseline_score,
        regression_delta=baseline_score - current_score,
        threshold=threshold,
        metrics=metrics,
    )
    payload = dataclasses.asdict(payload_obj)
    return _make_event(
        ET.EVAL_REGRESSION_FAILED, payload, session_id=session_id, org_id=org_id
    )
