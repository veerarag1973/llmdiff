# Schema Events — Observability with llm-toolkit-schema

`llm-diff` integrates with
[llm-toolkit-schema](https://pypi.org/project/llm-toolkit-schema/) to emit a
structured, validated event for every significant pipeline operation.  Events
are **collected in memory by default** and can also be exported to JSONL,
forwarded to a custom backend, or have in-memory collection disabled via
`configure_emitter(collect=False)`.

---

## Installation

`llm-toolkit-schema` is a declared dependency, so it is installed automatically:

```bash
pip install "llm-diff[semantic]"
```

---

## Quick start

```python
import asyncio
from llm_diff import compare
from llm_diff.schema_events import configure_emitter, get_emitter

# Events are collected in memory by default — read them via get_emitter().events.
# Call configure_emitter(collect=False) to disable in-memory storage.
configure_emitter()

asyncio.run(
    compare(
        "Explain recursion in one sentence.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        show_cost=True,
    )
)

for evt in get_emitter().events:
    print(f"{evt.event_type:40s}  {evt.event_id}")
```

Sample output:

```
llm.diff.comparison.started              01KJKVRV15TKQWTYZ8A3NRJKJK
llm.trace.span.completed                 01KJKVRW2XGBFPQDM4V5NSJL80
llm.trace.span.completed                 01KJKVRX4YPCHQREN6W7OTMN92
llm.cost.recorded                        01KJKVRY6ZQDIRSE8X9PVUNO04
llm.cost.recorded                        01KJKVRZ8AREKSFTAX0QWVOP16
llm.diff.comparison.completed            01KJKVS1CBSGLTGVCZ2RXWPQ28
```

---

## Export to JSONL

```python
from llm_toolkit_schema.export.jsonl import JSONLExporter
from llm_diff.schema_events import configure_emitter

configure_emitter(exporter=JSONLExporter("llm-diff-events.jsonl"))
```

Each line of the output file is a complete JSON object.  Example:

```json
{
  "event_id": "01KJKVRV15TKQWTYZ8A3NRJKJK",
  "event_type": "llm.diff.comparison.started",
  "source": "llm-diff@1.3.0",
  "timestamp": "2026-03-01T09:15:00.123456Z",
  "payload": {
    "model_a": "gpt-4o",
    "model_b": "gpt-4o-mini",
    "prompt_length": 38
  }
}
```

---

## Export to a custom sink

Any callable or object with an `.export(event)` method works:

```python
import json, requests

def post_to_backend(event):
    requests.post(
        "https://events.example.com/ingest",
        json=event.to_dict(),
        timeout=2,
    )

configure_emitter(exporter=post_to_backend)
```

Export errors are caught and logged as warnings — they never interrupt a
comparison.

---

## Event types reference

### `llm.diff.comparison.started`

Emitted at the very beginning of `compare()` or `compare_prompts()`, before
any model calls.

**Payload fields**

| Field | Type | Description |
|-------|------|-------------|
| `model_a` | `str` | First model identifier |
| `model_b` | `str` | Second model identifier |
| `prompt_length` | `int` | Character length of the prompt |

---

### `llm.diff.comparison.completed`

Emitted once the word-level diff is ready.

**Payload** — conforms to `DiffComparisonPayload`

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `str` | Event ID of the `.started` event (or model A name) |
| `target_id` | `str` | Model B identifier |
| `diff_type` | `str` | `"word-level"` or `"prompt"` |
| `similarity_score` | `float \| None` | Word-level similarity (0–1) |
| `source_text` | `str \| None` | Response from model A |
| `target_text` | `str \| None` | Response from model B |
| `diff_result` | `dict \| None` | `{"unified_diff": "..."}` when diff content is available |

---

### `llm.diff.report.exported`

Emitted by `save_report()` after writing the HTML file.

**Payload** — conforms to `DiffReportPayload`

| Field | Type | Description |
|-------|------|-------------|
| `report_id` | `str` | Auto-generated ULID |
| `comparison_event_id` | `str` | ID of the related completed event |
| `format` | `str` | `"html"` |
| `export_path` | `str` | File path written |
| `export_url` | `str \| None` | URL if applicable |

---

### `llm.trace.span.completed`

Emitted after each model API call returns.

**Payload** — conforms to `SpanCompletedPayload`

| Field | Type | Description |
|-------|------|-------------|
| `span_name` | `str` | `"llm-diff-model-call"` |
| `status` | `str` | `"ok"` or `"error"` |
| `duration_ms` | `float` | Round-trip latency |
| `model` | `dict` | `ModelInfo` — `name`, `provider`, `version` |
| `token_usage` | `dict` | `TokenUsage` — `prompt_tokens`, `completion_tokens`, `total_tokens` |
| `cost_usd` | `float \| None` | Estimated cost if available |
| `finish_reason` | `str \| None` | Provider finish reason (`"stop"`, `"length"`, …) |
| `stream` | `bool` | Whether the response was streamed |

---

### `llm.cache.hit`

Emitted when a cached response is returned.

**Payload** — conforms to `CacheHitPayload`

| Field | Type | Description |
|-------|------|-------------|
| `cache_key_hash` | `str` | First 16 hex chars of the SHA-256 request hash |
| `cache_store` | `str` | `"disk"` |
| `similarity_score` | `float \| None` | Not used for exact-match cache |
| `cached_event_id` | `str \| None` | Not set for disk cache |
| `ttl_seconds` | `int \| None` | Cache TTL if configured |
| `latency_ms` | `float \| None` | Cache lookup latency |

---

### `llm.cache.miss`

Emitted when no cached response exists and the API is called.

**Payload** — conforms to `CacheMissPayload`

| Field | Type | Description |
|-------|------|-------------|
| `cache_key_hash` | `str` | First 16 hex chars of the SHA-256 request hash |
| `cache_store` | `str` | `"disk"` |
| `reason` | `str \| None` | Reason for miss (`"not_found"`, `"expired"`, `"disabled"`) |
| `latency_ms` | `float \| None` | Cache lookup latency |

---

### `llm.cost.recorded`

Emitted when `show_cost=True` and a cost estimate is computed.

**Payload** — conforms to `CostRecordedPayload`

| Field | Type | Description |
|-------|------|-------------|
| `span_event_id` | `str` | ULID correlating to the trace span |
| `model_name` | `str` | Model identifier |
| `provider` | `str` | Provider name |
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `total_tokens` | `int` | Sum |
| `cost_usd` | `float` | Total estimated cost |
| `currency` | `str` | `"USD"` |
| `input_cost_usd` | `float` | Cost for input tokens |
| `output_cost_usd` | `float` | Cost for output tokens |

---

### `llm.eval.scenario.completed`

Emitted after an LLM-as-a-Judge run finishes.

**Payload** — conforms to `EvalScenarioPayload`

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | `str` | Auto-generated ULID |
| `scenario_name` | `str` | `"llm-diff/{judge_model}/{winner}"` |
| `status` | `str` | `"passed"` (judge completed) |
| `score` | `float \| None` | Average of score_a and score_b (1–10 scale) |
| `metrics` | `dict \| None` | Per-criterion scores |
| `scale` | `str` | `"1-10"` |
| `rationale` | `str \| None` | Judge's reasoning paragraph |
| `label` | `str \| None` | Winner (`"A"`, `"B"`, `"tie"`) |
| `baseline_score` | `float \| None` | Not set by default |
| `duration_ms` | `float \| None` | Judge call latency |

---

### `llm.eval.regression.failed`

Emitted when the `--fail-under` threshold is not met.  One event is emitted per
failing item in batch mode; one event in single-comparison mode.

**Payload** — conforms to `EvalRegressionPayload`

| Field | Type | Description |
|-------|------|-------------|
| `scenario_id` | `str` | Auto-generated ULID |
| `scenario_name` | `str` | `"llm-diff/fail-under/single"` or `"llm-diff/fail-under/batch"` |
| `current_score` | `float` | Measured similarity or semantic score |
| `baseline_score` | `float` | The `--fail-under` value (minimum required) |
| `regression_delta` | `float` | `baseline_score − current_score` (positive = shortfall) |
| `threshold` | `float` | Same as `baseline_score` |
| `metrics` | `dict \| None` | `{"similarity": word_similarity}` when semantic score is the primary |

**Example**

```python
from llm_diff.schema_events import configure_emitter, get_emitter

configure_emitter()

# Run a CLI comparison that fails --fail-under threshold (programmatically)
# ... or inspect events after any batch run with --fail-under
regression_events = [
    e for e in get_emitter().events
    if e.event_type == "llm.eval.regression.failed"
]
for evt in regression_events:
    p = evt.payload
    print(f"REGRESSION  score={p['current_score']:.4f}  threshold={p['threshold']:.2f}  delta={p['regression_delta']:.4f}")
```

---

## Event envelope fields

Every event, regardless of type, carries these top-level envelope fields:

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | `str` | Auto-generated 26-character ULID |
| `event_type` | `str` | Dot-separated type string (see above) |
| `source` | `str` | `"llm-diff@{version}"` |
| `timestamp` | `str` | ISO-8601 UTC timestamp |
| `payload` | `dict` | Event-specific data (see per-type tables above) |
| `trace_id` | `str \| None` | Optional correlation ID |
| `span_id` | `str \| None` | Optional span ID |
| `session_id` | `str \| None` | Optional session ID |
| `org_id` | `str \| None` | Optional organisation ID |
| `tags` | `Tags \| None` | Optional key-value tags |

---

## Correlating events

The `comparison.started` event ID is passed as `source_id` in the
`comparison.completed` payload, forming a simple parent-child chain:

```python
events = get_emitter().events

started = next(e for e in events if e.event_type == "llm.diff.comparison.started")
completed = next(e for e in events if e.event_type == "llm.diff.comparison.completed")

assert completed.payload["source_id"] == started.event_id
```

---

## Advanced: custom event factory

You can build your own events with the same source tag and emit them through the
global emitter:

```python
from llm_diff.schema_events import emit, make_comparison_started_event

evt = make_comparison_started_event(
    model_a="gpt-4o",
    model_b="claude-3-7-sonnet",
    prompt="My prompt",
    session_id="sess-abc123",
    org_id="acme",
)
emit(evt)
```

---

## Memory management

By default, events accumulate indefinitely.  In long-running processes:

```python
# Disable in-memory collection entirely
configure_emitter(exporter=my_sink, collect=False)

# Or clear periodically
from llm_diff.schema_events import get_emitter
get_emitter().clear()
```

---

## See also

- [Python API — Schema Events section](api.md#schema-events)
- [Tutorial 11 — Schema Events & Observability](tutorials/11-schema-events.md)
- [llm-toolkit-schema on PyPI](https://pypi.org/project/llm-toolkit-schema/)
