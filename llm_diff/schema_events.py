"""AgentOBS SDK integration for llm-diff.

This module provides a thin, zero-configuration integration between llm-diff
and the ``agentobs`` event envelope.  Every major operation in the
diff pipeline — comparison started/completed, model trace spans, cache
lookups, cost recording, and judge evaluations — now emits a structured,
schema-validated :class:`~agentobs.Event`.

Architecture
------------
A module-level :class:`EventEmitter` singleton collects events.  By default
events are kept in memory (see :attr:`~EventEmitter.events`).  In
long-running processes call :func:`configure_emitter` with
``collect=False`` (or a finite ``max_events``) to bound memory use.
Attach an exporter, e.g.::

    from llm_diff.schema_events import configure_emitter
    from agentobs.export.jsonl import JSONLExporter

    configure_emitter(exporter=JSONLExporter("events.jsonl"))

After that every comparison automatically appends schema-valid events to
``events.jsonl``.

Usage (library)
---------------
.. code-block:: python

    import asyncio
    from llm_diff import compare
    from llm_diff.schema_events import configure_emitter, get_emitter
    from agentobs.export.jsonl import JSONLExporter

    configure_emitter(exporter=JSONLExporter("events.jsonl"))
    asyncio.run(compare("Explain recursion", model_a="gpt-4o", model_b="claude-3-5-sonnet"))
    events = get_emitter().events  # list of Event objects collected in memory
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

from llm_diff import __version__

logger = logging.getLogger(__name__)

# Source string embedded in every emitted event.
_SOURCE = f"llm-diff@{__version__}"

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lazy import helpers — keep startup cost low
# ---------------------------------------------------------------------------


def _agentobs() -> Any:
    """Return the top-level ``agentobs`` module."""
    import agentobs  # noqa: PLC0415

    return agentobs


def _event_cls() -> type:
    return _agentobs().Event


def _tags_cls() -> type:
    return _agentobs().Tags


def _event_type() -> Any:
    return _agentobs().EventType


def _diff_ns() -> Any:
    from agentobs.namespaces import diff as _diff  # noqa: PLC0415

    return _diff


def _trace_ns() -> Any:
    from agentobs.namespaces import trace as _trace  # noqa: PLC0415

    return _trace


def _cache_ns() -> Any:
    from agentobs.namespaces import cache as _cache  # noqa: PLC0415

    return _cache


def _cost_ns() -> Any:
    from agentobs.namespaces import cost as _cost  # noqa: PLC0415

    return _cost


def _eval_ns() -> Any:
    from agentobs.namespaces import eval_ as _eval  # noqa: PLC0415

    return _eval


def _ulid_or_empty() -> str:
    from agentobs.ulid import generate  # noqa: PLC0415

    return generate()


def _gen_span_id() -> str:
    """Generate a 16-char lowercase hex OTel span ID."""
    return os.urandom(8).hex()


def _gen_trace_id() -> str:
    """Generate a 32-char lowercase hex OTel trace ID."""
    return os.urandom(16).hex()


def _parse_scale(scale: str) -> tuple[float, float]:
    """Parse a scale string like ``'1-10'`` or ``'0-1'`` into (min, max)."""
    try:
        parts = scale.split("-", 1)
        return float(parts[0]), float(parts[1])
    except (IndexError, ValueError):
        return 0.0, 1.0


# Mapping from old diff_type values to AgentOBS valid values.
_DIFF_TYPE_MAP: dict[str, str] = {
    "word-level": "response",
    "completion": "response",
    "prompt": "prompt",
    "both": "response",
    "response": "response",
    "template": "template",
    "token_usage": "token_usage",
    "cost": "cost",
}


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class EventEmitter:
    """Collects and optionally exports AgentOBS :class:`~agentobs.Event` objects.

    Parameters
    ----------
    exporter:
        Any callable that accepts a single :class:`~agentobs.Event`
        argument.  By default events are only collected in memory (see
        :attr:`events`).  Pass a ``JSONLExporter`` or any compatible object
        with an ``export`` method (or a plain callable) to also ship events
        to an external backend.
    collect:
        When ``True`` (default), events are appended to the in-memory
        :attr:`events` list.  Disable when memory overhead matters in
        long-running processes.
    max_events:
        Maximum number of events to keep in memory.  When the buffer is full
        the oldest event is silently dropped (ring-buffer semantics).  ``None``
        (default) means unbounded.
    """

    def __init__(
        self,
        exporter: Callable[[Any], Any] | None = None,
        *,
        collect: bool = True,
        max_events: int | None = None,
    ) -> None:
        self._exporter = exporter
        self._collect = collect
        # Bounded ring-buffer: deque(maxlen=N) automatically discards the
        # oldest entry once the capacity is reached (PERF-02).
        self._events: deque[Any] = deque(maxlen=max_events)

    @property
    def events(self) -> list[Any]:
        """Read-only list of all :class:`~agentobs.Event` objects collected."""
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
    max_events: int | None = None,
) -> EventEmitter:
    """Replace the global emitter with a new configured instance.

    Call this exactly once at application startup before running any
    comparisons.

    Parameters
    ----------
    exporter:
        Any callable or object with an ``export`` method that accepts a
        :class:`~agentobs.Event`.
    collect:
        Whether to keep events in memory (default ``True``).
    max_events:
        Maximum number of events to retain in memory (ring-buffer).
        ``None`` (default) means unbounded.

    Returns
    -------
    EventEmitter
        The newly installed global emitter.
    """
    global _emitter  # noqa: PLW0603

    _emitter = EventEmitter(exporter=exporter, collect=collect, max_events=max_events)
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
    """Build a :class:`~agentobs.Event` from the given arguments."""
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
    """Build a ``x.llm-diff.comparison.started`` event.

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
    payload: dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        "prompt_length": len(prompt),
    }
    return _make_event(
        "x.llm-diff.comparison.started",
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
    """Build a ``llm.diff.computed`` event with a DiffComputedPayload."""
    ET = _event_type()
    ns = _diff_ns()

    sdk_diff_type = _DIFF_TYPE_MAP.get(diff_type, "response")

    payload_obj = ns.DiffComputedPayload(
        ref_event_id=base_event_id or _ulid_or_empty(),
        target_event_id=_ulid_or_empty(),
        diff_type=sdk_diff_type,
        similarity_score=similarity_score if similarity_score is not None else 0.0,
    )
    payload = payload_obj.to_dict()

    if model_a_text is not None:
        payload["source_text"] = model_a_text
    if model_b_text is not None:
        payload["target_text"] = model_b_text
    if completion_diff:
        payload["diff_result"] = {"unified_diff": completion_diff}
    elif prompt_diff:
        payload["diff_result"] = {"unified_diff": prompt_diff}

    return _make_event(
        ET.DIFF_COMPUTED,
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
    """Build a ``x.llm-diff.report.exported`` event."""
    payload: dict[str, Any] = {
        "report_id": report_id or _ulid_or_empty(),
        "comparison_event_id": comparison_event_id or _ulid_or_empty(),
        "format": format,
        "export_path": output_path,
    }
    return _make_event(
        "x.llm-diff.report.exported",
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
    """Build a ``llm.trace.span.completed`` event with SpanPayload.

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
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total,
    )
    model_info = ns.ModelInfo(
        system=provider or "custom",
        name=model,
    )

    end_ns = int(time.time() * 1_000_000_000)
    start_ns = end_ns - int(latency_ms * 1_000_000)

    payload_obj = ns.SpanPayload(
        span_id=_gen_span_id(),
        trace_id=_gen_trace_id(),
        span_name="llm-diff-model-call",
        operation=ns.GenAIOperationName.CHAT,
        span_kind=ns.SpanKind.CLIENT,
        status="ok" if finish_reason != "error" else "error",
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        duration_ms=latency_ms,
        model=model_info,
        token_usage=token_usage,
        finish_reason=finish_reason,
    )
    payload = payload_obj.to_dict()
    payload["stream"] = stream
    if cost_usd is not None:
        payload["cost_usd"] = cost_usd

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
        Cache backend name (default ``"disk"``); used as the namespace label.
    latency_ms:
        Cache lookup latency in milliseconds, if measured.
    """
    ET = _event_type()
    ns = _cache_ns()

    namespace = backend or "disk"

    if hit:
        payload_obj = ns.CacheHitPayload(
            key_hash=cache_key or "unknown",
            namespace=namespace,
            similarity_score=1.0,
            ttl_remaining_seconds=ttl_seconds,
            lookup_duration_ms=latency_ms,
        )
        event_type = ET.CACHE_HIT
    else:
        payload_obj = ns.CacheMissPayload(
            key_hash=cache_key or "unknown",
            namespace=namespace,
            lookup_duration_ms=latency_ms,
        )
        event_type = ET.CACHE_MISS

    payload = payload_obj.to_dict()
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
    """Build a ``llm.cost.token.recorded`` event with CostTokenRecordedPayload."""
    ET = _event_type()
    trace_ns = _trace_ns()
    cost_ns = _cost_ns()

    cost = trace_ns.CostBreakdown(
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
        currency=currency,
    )
    token_usage = trace_ns.TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens or (prompt_tokens + completion_tokens),
    )
    model_info = trace_ns.ModelInfo(
        system=provider or "custom",
        name=model or "unknown",
    )

    payload_obj = cost_ns.CostTokenRecordedPayload(
        cost=cost,
        token_usage=token_usage,
        model=model_info,
        span_id=span_event_id,
    )
    payload = payload_obj.to_dict()
    if pricing_tier is not None:
        payload["pricing_tier"] = pricing_tier

    return _make_event(ET.COST_TOKEN_RECORDED, payload, session_id=session_id, org_id=org_id)


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
    """Build a ``llm.eval.score.recorded`` event with EvalScoreRecordedPayload.

    Parameters
    ----------
    status:
        Must be ``"passed"``, ``"failed"``, or ``"skipped"``.
    """
    ET = _event_type()
    ns = _eval_ns()

    score_min, score_max = _parse_scale(scale)
    metric_name = label if label else "similarity"

    payload_obj = ns.EvalScoreRecordedPayload(
        evaluator=evaluator,
        metric_name=metric_name,
        score=score or 0.0,
        score_min=score_min,
        score_max=score_max,
        passed=status == "passed",
        rationale=rationale,
        threshold=baseline_score,
    )
    payload = payload_obj.to_dict()
    payload["scale"] = scale
    if criteria:
        payload["criteria"] = dict.fromkeys(criteria, score or 0.0)
    if label:
        payload["label"] = label
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return _make_event(
        ET.EVAL_SCORE_RECORDED, payload, session_id=session_id, org_id=org_id
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
    """Build a ``llm.eval.regression.detected`` event with EvalRegressionDetectedPayload.

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

    delta = current_score - baseline_score
    regression_pct = abs(delta / baseline_score * 100) if baseline_score != 0 else 0.0
    severity = "high" if abs(delta) >= 0.2 else "medium" if abs(delta) >= 0.1 else "low"

    payload_obj = ns.EvalRegressionDetectedPayload(
        metric_name=scenario_name,
        baseline_score=baseline_score,
        current_score=current_score,
        delta=delta,
        regression_pct=regression_pct,
        severity=severity,
    )
    payload = payload_obj.to_dict()
    payload["threshold"] = threshold
    if metrics is not None:
        payload["metrics"] = metrics
    return _make_event(
        ET.EVAL_REGRESSION_DETECTED, payload, session_id=session_id, org_id=org_id
    )
