# Tutorial 11 — Schema Events & Observability

**Time:** ~15 minutes  
**Level:** Advanced  
**Prerequisites:** [Tutorial 10](10-python-api.md) completed

← [10 — Python API](10-python-api.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Understand what schema events are and why they matter
- Opt in to in-memory event collection
- Export events to a JSONL file
- Forward events to any custom backend
- Correlate related events using the event envelope
- Build a simple audit log for an evaluation pipeline
- Capture `--fail-under` regression failures as structured events

---

## Background — Why Schema Events?

Every time `llm-diff` runs a comparison it touches several subsystems: it calls
a model API, checks the cache, computes a diff, optionally runs a judge, and
writes an HTML report.  **Without observability you have no record of any of
this** — you can see the terminal output, but you cannot:

- Answer "how many API calls did we make last week?"
- Track P95 model latency over time
- See cache hit rate trends
- Audit which evaluations produced which scores
- Alert when cost exceeds a budget

`llm-diff` solves this by emitting a structured
[llm-toolkit-schema](https://pypi.org/project/llm-toolkit-schema/) event for
every significant operation.  Events are validated, timestamped, carry a unique
ULID ID, and conform to well-defined payload schemas.

By default events are **collected in memory** — every comparison automatically
appends events to the in-process emitter so you can inspect them at any time.
Call `configure_emitter(collect=False)` to disable in-memory storage (for
example, when only streaming to a file exporter).  The CLI disables in-memory
collection automatically (`collect=False`) to prevent unbounded memory growth
in long-running batch jobs.

---

## Step 1 — In-memory collection

The simplest way to see events is to collect them in memory:

```python
import asyncio
from llm_diff import compare
from llm_diff.schema_events import get_emitter

# No setup needed — collection is on by default.
# Call configure_emitter(collect=False) to turn it off.

asyncio.run(
    compare(
        "Explain recursion in one sentence.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
    )
)

events = get_emitter().events
print(f"Collected {len(events)} events")
for evt in events:
    print(f"  {evt.event_type}")
```

Expected output:

```
Collected 4 events
  llm.diff.comparison.started
  llm.trace.span.completed
  llm.trace.span.completed
  llm.diff.comparison.completed
```

Two `llm.trace.span.completed` events are emitted — one per model call.

---

## Step 2 — Inspect event payloads

Each event carries a `payload` dict with operation-specific fields:

```python
for evt in events:
    print(f"\n{evt.event_type}")
    print(f"  event_id: {evt.event_id}")
    print(f"  timestamp: {evt.timestamp}")
    if evt.event_type == "llm.trace.span.completed":
        p = evt.payload
        model = p["model"]["name"]
        dur   = p["duration_ms"]
        toks  = p["token_usage"]["total_tokens"]
        print(f"  model={model}  latency={dur:.0f}ms  tokens={toks}")
```

Sample output:

```
llm.diff.comparison.started
  event_id: 01KJKVRV15TKQWTYZ8A3NRJKJK
  timestamp: 2026-03-01T09:15:00.123456Z

llm.trace.span.completed
  event_id: 01KJKVRW2XGBFPQDM4V5NSJL80
  timestamp: 2026-03-01T09:15:00.891234Z
  model=gpt-4o  latency=768ms  tokens=165

llm.trace.span.completed
  event_id: 01KJKVRX4YPCHQREN6W7OTMN92
  timestamp: 2026-03-01T09:15:01.043210Z
  model=gpt-4o-mini  latency=612ms  tokens=121
```

---

## Step 3 — Add cost and judge events

When you use `show_cost=True` and `judge=`, more event types appear:

```python
configure_emitter()

asyncio.run(
    compare(
        "Explain closures in JavaScript.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        show_cost=True,
        judge="gpt-4o-mini",
    )
)

for evt in get_emitter().events:
    print(evt.event_type)
```

```
llm.diff.comparison.started
llm.trace.span.completed
llm.trace.span.completed
llm.cost.recorded
llm.cost.recorded
llm.eval.scenario.completed
llm.diff.comparison.completed
```

---

## Step 4 — Export to JSONL

For persistent storage, export to a JSONL file:

```python
from llm_toolkit_schema.export.jsonl import JSONLExporter
from llm_diff.schema_events import configure_emitter

configure_emitter(exporter=JSONLExporter("llm-diff-events.jsonl"))

# Run comparisons as normal — events are written automatically
asyncio.run(
    compare("Explain recursion.", model_a="gpt-4o", model_b="gpt-4o-mini")
)
```

Inspect the file:

```bash
cat llm-diff-events.jsonl | python -m json.tool | head -30
```

Each line is a complete JSON event:

```json
{
  "event_id": "01KJKVRV15TKQWTYZ8A3NRJKJK",
  "event_type": "llm.diff.comparison.started",
  "source": "llm-diff@1.3.0",
  "timestamp": "2026-03-01T09:15:00.123456Z",
  "payload": {
    "model_a": "gpt-4o",
    "model_b": "gpt-4o-mini",
    "prompt_length": 24
  }
}
```

---

## Step 5 — Forward to a custom backend

Any callable works as an exporter — HTTP, database, message queue, anything:

```python
import json, logging

def my_sink(event):
    """Write events to the Python logger."""
    logging.getLogger("llm_diff.events").info(
        json.dumps({"type": event.event_type, "id": event.event_id})
    )

configure_emitter(exporter=my_sink)
```

Export errors are caught quietly and never interrupt a comparison.

---

## Step 6 — Correlate events

Events carry IDs you can use to trace a full comparison lifecycle.  The
`comparison.started` event's `event_id` becomes the `source_id` in the
`comparison.completed` payload:

```python
configure_emitter()

asyncio.run(
    compare("Explain recursion.", model_a="gpt-4o", model_b="gpt-4o-mini")
)

evts = get_emitter().events

started   = next(e for e in evts if e.event_type == "llm.diff.comparison.started")
completed = next(e for e in evts if e.event_type == "llm.diff.comparison.completed")
spans     = [e for e in evts if e.event_type == "llm.trace.span.completed"]

print(f"Started event:    {started.event_id}")
print(f"Completed source: {completed.payload['source_id']}")
print(f"Span IDs:         {[s.event_id for s in spans]}")
print(f"Chain verified:   {completed.payload['source_id'] == started.event_id}")
```

---

## Step 7 — Build an audit log for a batch run

Here is a realistic pattern: run a batch evaluation, collect events, then write
a summary audit log alongside the HTML report:

```python
import asyncio
import json
from pathlib import Path
from llm_diff import compare_batch
from llm_diff.schema_events import configure_emitter, get_emitter

def run_batch_with_audit(
    batch_file: str,
    model_a: str,
    model_b: str,
    audit_path: str = "audit.jsonl",
) -> list:
    configure_emitter()

    reports = asyncio.run(
        compare_batch(
            batch_file,
            model_a=model_a,
            model_b=model_b,
            semantic=True,
            show_cost=True,
            concurrency=4,
        )
    )

    # Write all events to JSONL audit log
    events = get_emitter().events
    with open(audit_path, "w") as f:
        for evt in events:
            f.write(json.dumps(evt.to_dict()) + "\n")

    # Print summary
    spans = [e for e in events if e.event_type == "llm.trace.span.completed"]
    costs = [e for e in events if e.event_type == "llm.cost.recorded"]
    total_cost = sum(e.payload["cost_usd"] for e in costs)
    avg_latency = sum(e.payload["duration_ms"] for e in spans) / len(spans) if spans else 0

    print(f"Batch complete: {len(reports)} prompts")
    print(f"Total events:   {len(events)}")
    print(f"API calls made: {len(spans)}")
    print(f"Avg latency:    {avg_latency:.0f}ms")
    print(f"Total cost:     ${total_cost:.4f}")
    print(f"Audit log:      {audit_path}")

    return reports


if __name__ == "__main__":
    run_batch_with_audit(
        "examples/prompts.yml",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
    )
```

---

## Step 8 — Observe `--fail-under` regression events

When a batch or single comparison fails the `--fail-under` threshold, llm-diff
emits an `llm.eval.regression.failed` event in addition to writing to stderr and
exiting with code 1.  You can intercept it before that exit (or replay from a
JSONL file) to build custom alerting:

```python
from llm_diff.schema_events import configure_emitter, get_emitter
from llm_toolkit_schema.export.jsonl import JSONLExporter

configure_emitter(exporter=JSONLExporter("events.jsonl"))

# ... run your batch comparison ...

regression_events = [
    e for e in get_emitter().events
    if e.event_type == "llm.eval.regression.failed"
]

if regression_events:
    print(f"{len(regression_events)} regression(s) detected:")
    for evt in regression_events:
        p = evt.payload
        print(
            f"  {p['scenario_name']}  "
            f"score={p['current_score']:.4f}  "
            f"threshold={p['threshold']:.2f}  "
            f"delta={p['regression_delta']:.4f}"
        )
```

The `EvalRegressionPayload` fields are:

| Field | Description |
|-------|-------------|
| `scenario_name` | `"llm-diff/fail-under/single"` or `"llm-diff/fail-under/batch"` |
| `current_score` | Measured similarity / semantic score |
| `baseline_score` | The `--fail-under` value |
| `regression_delta` | `baseline − current` (how far below threshold) |
| `threshold` | Same as `baseline_score` |
| `metrics` | `{"similarity": …}` when semantic score was the primary metric |

---

## Step 9 — Memory management in long-running processes

In a long-running application or repeated batch job, events accumulate:

```python
# Option 1: disable in-memory collection (events only go to the exporter)
configure_emitter(exporter=my_sink, collect=False)

# Option 2: clear after each batch
from llm_diff.schema_events import get_emitter

for batch in batches:
    results = await run_batch(batch)
    process(results, get_emitter().events)
    get_emitter().clear()   # free memory before next batch
```

---

## Summary

| Task | Code |
|------|------|
| Memory collection (on by default) | `get_emitter().events` — no setup required |
| Limit buffer size (bounded memory) | `configure_emitter(max_events=1000)` |
| Export to JSONL | `configure_emitter(exporter=JSONLExporter("out.jsonl"))` |
| Custom exporter | `configure_emitter(exporter=my_fn)` |
| Read collected events | `get_emitter().events` |
| Clear events | `get_emitter().clear()` |
| Disable memory collection | `configure_emitter(collect=False)` |
| CLI default | collection disabled automatically (`collect=False`) |
| Filter regression events | `[e for e in get_emitter().events if e.event_type == "llm.eval.regression.failed"]` |

---

## Further reading

- [Schema Events Guide](../schema-events.md) — full payload field reference
- [Python API — Schema Events section](../api.md#schema-events) — function signatures
- [llm-toolkit-schema on PyPI](https://pypi.org/project/llm-toolkit-schema/)

---

← [10 — Python API](10-python-api.md)
