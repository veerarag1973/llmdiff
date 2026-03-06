# Schema Events — Observability with AgentOBS

`llm-diff` integrates with
[AgentOBS](https://pypi.org/project/agentobs/) to emit a
structured, validated event for every significant pipeline operation.  Events
are **collected in memory by default** and can also be exported to JSONL,
forwarded to a custom backend, or have in-memory collection disabled via
`configure_emitter(collect=False)`.

---

## Installation

`agentobs` is a declared dependency, so it is installed automatically:

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
x.llm-diff.comparison.started          01KJKVRV15TKQWTYZ8A3NRJKJK
llm.trace.span.completed               01KJKVRW2XGBFPQDM4V5NSJL80
llm.trace.span.completed               01KJKVRX4YPCHQREN6W7OTMN92
llm.cost.token.recorded                01KJKVRY6ZQDIRSE8X9PVUNO04
llm.cost.token.recorded                01KJKVRZ8AREKSFTAX0QWVOP16
llm.diff.computed                      01KJKVS1CBSGLTGVCZ2RXWPQ28
```

---

## Export to JSONL

```python
from agentobs.export.jsonl import JSONLExporter
from llm_diff.schema_events import configure_emitter

configure_emitter(exporter=JSONLExporter("llm-diff-events.jsonl"))
```

Each line of the output file is a complete JSON object.  Example:

```json
{
  "event_id": "01KJKVRV15TKQWTYZ8A3NRJKJK",
  "event_type": "x.llm-diff.comparison.started",
    "source": "llm-diff@1.3.1",
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

### `x.llm-diff.comparison.started`

Emitted at the very beginning of `compare()` or `compare_prompts()`, before
any model calls.

**Payload fields**

| Field | Type | Description |
|-------|------|-------------|
| `model_a` | `str` | First model identifier |
| `model_b` | `str` | Second model identifier |
| `prompt_length` | `int` | Character length of the prompt |

---

### `llm.diff.computed`

Emitted once the word-level diff is ready.

**Payload** — conforms to `DiffComputedPayload` (`agentobs.namespaces.diff`)

| Field | Type | Description |
|-------|------|-------------|
| `ref_event_id` | `str` | ULID of the `.started` event (baseline) |
| `target_event_id` | `str` | Auto-generated ULID for the target |
| `diff_type` | `str` | `"response"`, `"prompt"`, `"template"`, `"token_usage"`, or `"cost"` |
| `similarity_score` | `float` | Word-level similarity (0–1) |
| `source_text` | `str \| None` | Response from model A |
| `target_text` | `str \| None` | Response from model B |
| `diff_result` | `dict \| None` | `{"unified_diff": "..."}` when diff content is available |

---

### `x.llm-diff.report.exported`

Emitted by `save_report()` after writing the HTML file.

**Payload**

| Field | Type | Description |
|-------|------|-------------|
| `report_id` | `str` | Auto-generated ULID |
| `comparison_event_id` | `str` | ID of the related completed event |
| `format` | `str` | `"html"` |
| `export_path` | `str` | File path written |

---

### `llm.trace.span.completed`

Emitted after each model API call returns.

**Payload** — conforms to `SpanPayload` (`agentobs.namespaces.trace`)

| Field | Type | Description |
|-------|------|-------------|
| `span_name` | `str` | `"llm-diff-model-call"` |
| `status` | `str` | `"ok"` or `"error"` |
| `duration_ms` | `float` | Round-trip latency |
| `model` | `dict` | `ModelInfo` — `system`, `name` |
| `token_usage` | `dict` | `TokenUsage` — `input_tokens`, `output_tokens`, `total_tokens` |
| `cost_usd` | `float \| None` | Estimated cost if available |
| `finish_reason` | `str \| None` | Provider finish reason (`"stop"`, `"length"`, …) |
| `stream` | `bool` | Whether the response was streamed |

---

### `llm.cache.hit`

Emitted when a cached response is returned.

**Payload** — conforms to `CacheHitPayload` (`agentobs.namespaces.cache`)

| Field | Type | Description |
|-------|------|-------------|
| `key_hash` | `str` | First 16 hex chars of the SHA-256 request hash |
| `namespace` | `str` | Cache backend name (e.g. `"disk"`) |
| `similarity_score` | `float` | `1.0` for exact-match disk cache |
| `ttl_remaining_seconds` | `int \| None` | Cache TTL if configured |
| `lookup_duration_ms` | `float \| None` | Cache lookup latency |

---

### `llm.cache.miss`

Emitted when no cached response exists and the API is called.

**Payload** — conforms to `CacheMissPayload` (`agentobs.namespaces.cache`)

| Field | Type | Description |
|-------|------|-------------|
| `key_hash` | `str` | First 16 hex chars of the SHA-256 request hash |
| `namespace` | `str` | Cache backend name (e.g. `"disk"`) |
| `lookup_duration_ms` | `float \| None` | Cache lookup latency |

---

### `llm.cost.token.recorded`

Emitted when `show_cost=True` and a cost estimate is computed.

**Payload** — conforms to `CostTokenRecordedPayload` (`agentobs.namespaces.cost`)

| Field | Type | Description |
|-------|------|-------------|
| `cost` | `dict` | `CostBreakdown` — `input_cost_usd`, `output_cost_usd`, `total_cost_usd` |
| `token_usage` | `dict` | `TokenUsage` — `input_tokens`, `output_tokens`, `total_tokens` |
| `model` | `dict` | `ModelInfo` — `system`, `name` |
| `span_id` | `str \| None` | Correlating span identifier |

---

### `llm.eval.score.recorded`

Emitted after an LLM-as-a-Judge run finishes.

**Payload** — conforms to `EvalScoreRecordedPayload` (`agentobs.namespaces.eval_`)

| Field | Type | Description |
|-------|------|-------------|
| `evaluator` | `str` | Judge model identifier |
| `metric_name` | `str` | Metric label (`"similarity"`, or winner label) |
| `score` | `float` | Average of score_a and score_b (1–10 scale) |
| `score_min` | `float \| None` | Minimum of the scoring scale |
| `score_max` | `float \| None` | Maximum of the scoring scale |
| `passed` | `bool \| None` | Whether the score met the threshold |
| `scale` | `str` | `"1-10"` |
| `rationale` | `str \| None` | Judge's reasoning paragraph |
| `label` | `str \| None` | Winner (`"A"`, `"B"`, `"tie"`) |
| `criteria` | `dict \| None` | Per-criterion scores |

---

### `llm.eval.regression.detected`

Emitted when the `--fail-under` threshold is not met.  One event is emitted per
failing item in batch mode; one event in single-comparison mode.

**Payload** — conforms to `EvalRegressionDetectedPayload` (`agentobs.namespaces.eval_`)

| Field | Type | Description |
|-------|------|-------------|
| `metric_name` | `str` | `"llm-diff/fail-under/single"` or `"llm-diff/fail-under/batch"` |
| `current_score` | `float` | Measured similarity or semantic score |
| `baseline_score` | `float` | The `--fail-under` value (minimum required) |
| `delta` | `float` | `current_score − baseline_score` (negative = shortfall) |
| `regression_pct` | `float` | Percentage regression magnitude |
| `threshold` | `float` | Same as `baseline_score` |
| `severity` | `str \| None` | `"low"`, `"medium"`, or `"high"` |
| `metrics` | `dict \| None` | `{"similarity": word_similarity}` when available |

**Example**

```python
from llm_diff.schema_events import configure_emitter, get_emitter

configure_emitter()

# Run a CLI comparison that fails --fail-under threshold (programmatically)
# ... or inspect events after any batch run with --fail-under
regression_events = [
    e for e in get_emitter().events
    if e.event_type == "llm.eval.regression.detected"
]
for evt in regression_events:
    p = evt.payload
    print(f"REGRESSION  score={p['current_score']:.4f}  threshold={p['threshold']:.2f}  delta={p['delta']:.4f}")
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

The `comparison.started` event ID is passed as `ref_event_id` in the
`llm.diff.computed` payload, forming a simple parent-child chain:

```python
events = get_emitter().events

started = next(e for e in events if e.event_type == "x.llm-diff.comparison.started")
completed = next(e for e in events if e.event_type == "llm.diff.computed")

assert completed.payload["ref_event_id"] == started.event_id
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
- [AgentOBS on PyPI](https://pypi.org/project/agentobs/)
