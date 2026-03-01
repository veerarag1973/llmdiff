# Python API Reference

`llm-diff` ships a fully-typed async Python library.  Import from the
top-level `llm_diff` package.

---

## Installation

```bash
pip install "llm-diff[semantic]"
```

---

## Quick example

```python
import asyncio
from llm_diff import compare

report = asyncio.run(
    compare("Explain recursion in one sentence.", model_a="gpt-4o", model_b="gpt-4o-mini")
)
print(f"Word similarity:   {report.word_similarity:.2%}")
print(f"Semantic score:    {report.semantic_score}")
print(f"Response A tokens: {report.comparison.response_a.total_tokens}")
```

---

## Functions

### `compare()`

Run a single model-vs-model comparison.

```python
async def compare(
    prompt: str,
    *,
    model_a: str,
    model_b: str,
    semantic: bool = False,
    paragraph: bool = False,
    bleu: bool = False,
    rouge: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 30,
    no_cache: bool = False,
    judge: str | None = None,
    show_cost: bool = False,
    build_html: bool = False,
) -> ComparisonReport: ...
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | — | Prompt sent to both models. |
| `model_a` | `str` | — | Model identifier for side A (e.g. `"gpt-4o"`). |
| `model_b` | `str` | — | Model identifier for side B. |
| `semantic` | `bool` | `False` | Compute cosine similarity via sentence embeddings. |
| `paragraph` | `bool` | `False` | Per-paragraph similarity breakdown (implies `semantic=True`). |
| `bleu` | `bool` | `False` | Compute BLEU score. |
| `rouge` | `bool` | `False` | Compute ROUGE-L F1 score. |
| `temperature` | `float` | `0.7` | Sampling temperature for both models. |
| `max_tokens` | `int` | `1024` | Max completion tokens per response. |
| `timeout` | `int` | `30` | Request timeout in seconds. |
| `no_cache` | `bool` | `False` | Bypass the response cache. |
| `judge` | `str \| None` | `None` | Model name to use as LLM-as-a-Judge scorer. |
| `show_cost` | `bool` | `False` | Estimate USD cost for each call and attach to report. |
| `build_html` | `bool` | `False` | Render a self-contained HTML report string and attach to `report.html_report`. |

**Returns** `ComparisonReport`

---

### `compare_prompts()`

Compare two different prompts on the same model (prompt-diff mode).

```python
async def compare_prompts(
    prompt_a: str,
    prompt_b: str,
    *,
    model: str,
    semantic: bool = False,
    paragraph: bool = False,
    bleu: bool = False,
    rouge: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 30,
    no_cache: bool = False,
    judge: str | None = None,
    show_cost: bool = False,
    build_html: bool = False,
) -> ComparisonReport: ...
```

**Parameters** — same as `compare()` except:

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt_a` | `str` | First prompt variant. |
| `prompt_b` | `str` | Second prompt variant. |
| `model` | `str` | Single model used for both calls. |

---

### `compare_batch()`

Run a full batch from a YAML file concurrently.

```python
async def compare_batch(
    batch_path: str | Path,
    *,
    model_a: str,
    model_b: str,
    semantic: bool = False,
    paragraph: bool = False,
    bleu: bool = False,
    rouge: bool = False,
    judge: str | None = None,
    show_cost: bool = False,
    build_html: bool = False,
    config: LLMDiffConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    concurrency: int = 4,
) -> list[ComparisonReport]: ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_path` | `str \| Path` | — | Path to a `prompts.yml` file. |
| `config` | `LLMDiffConfig \| None` | `None` | Shared run config; overrides individual fields when set. |
| `concurrency` | `int` | `4` | Maximum number of concurrent `compare()` calls. Increase for large batches; reduce when rate-limits are tight. |
| All others | — | — | Same as `compare()`. |

**Returns** `list[ComparisonReport]` — one entry per batch item, in YAML order.

---

### `estimate_cost()`

Estimate the USD cost of an API call without making it.

```python
def estimate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> CostEstimate: ...
```

---

### `run_multi_model()`

Compare 3 or 4 models concurrently and return all pairwise scores.

```python
async def run_multi_model(
    prompt: str,
    models: list[str],
    *,
    semantic: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    timeout: int = 30,
    no_cache: bool = False,
) -> MultiModelReport: ...
```

---

### `json_struct_diff()`

Compare two strings as JSON objects, returning a per-key diff.

```python
def json_struct_diff(
    response_a: str,
    response_b: str,
) -> JsonStructDiffResult: ...
```

If either string is not valid JSON, raises `ValueError`.

---

## Data classes

### `ComparisonReport`

The primary return type of `compare()`, `compare_prompts()`, and items in
`compare_batch()`.

| Field | Type | Description |
|-------|------|-------------|
| `prompt_a` | `str` | Prompt sent to model A. |
| `prompt_b` | `str` | Prompt sent to model B. |
| `comparison` | `ComparisonResult` | Raw paired model responses. |
| `diff_result` | `DiffResult` | Word-level diff chunks + similarity. |
| `semantic_score` | `float \| None` | Cosine similarity of whole texts (0–1). |
| `paragraph_scores` | `list[ParagraphScore] \| None` | Per-paragraph similarity entries. |
| `bleu_score` | `float \| None` | BLEU score (0–1), n-gram precision. |
| `rouge_l_score` | `float \| None` | ROUGE-L F1 score (0–1), LCS-based. |
| `judge_result` | `JudgeResult \| None` | LLM-as-a-Judge output. Present when `judge=` is set. |
| `cost_a` | `CostEstimate \| None` | Estimated USD cost for model A call. Present when `show_cost=True`. |
| `cost_b` | `CostEstimate \| None` | Estimated USD cost for model B call. Present when `show_cost=True`. |
| `html_report` | `str \| None` | Self-contained HTML string. Present when `build_html=True`. |
| `word_similarity` | `float` | **Property** — shortcut for `diff_result.similarity`. |
| `primary_score` | `float` | **Property** — `semantic_score` if set, else `word_similarity`. |

---

### `ComparisonResult`

Returned by the provider layer.

| Field | Type | Description |
|-------|------|-------------|
| `response_a` | `ModelResponse` | Response from model A. |
| `response_b` | `ModelResponse` | Response from model B. |

### `ModelResponse`

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model identifier as returned by the API. |
| `content` | `str` | Response text. |
| `prompt_tokens` | `int` | Input token count. |
| `completion_tokens` | `int` | Output token count. |
| `total_tokens` | `int` | Sum of prompt + completion. |
| `latency_ms` | `float` | Round-trip time in milliseconds. |
| `cached` | `bool` | `True` if the response was served from the local cache. |

---

### `DiffResult`

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | `list[DiffChunk]` | Ordered list of diff hunks. |
| `similarity` | `float` | Word-level similarity ratio (0–1). |

### `DiffChunk`

| Field | Type | Values |
|-------|------|--------|
| `type` | `str` | `"equal"`, `"delete"`, `"insert"` |
| `text` | `str` | The text span for this chunk. |

---

### `ParagraphScore`

| Field | Type | Description |
|-------|------|-------------|
| `index` | `int` | 1-based paragraph index. |
| `text_a` | `str` | Paragraph text from response A (truncated for display). |
| `score` | `float` | Cosine similarity for this paragraph (0–1). |

---

### `JudgeResult`

Returned when `judge=` is set.

| Field | Type | Description |
|-------|------|-------------|
| `winner` | `str` | `"A"`, `"B"`, or `"tie"`. |
| `score_a` | `int` | Judge's score for model A (1–10). |
| `score_b` | `int` | Judge's score for model B (1–10). |
| `reasoning` | `str` | Judge's explanation paragraph. |
| `judge_model` | `str` | Model used as judge. |

---

### `CostEstimate`

Returned when `show_cost=True`.

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Model identifier. |
| `prompt_tokens` | `int` | Input token count used for estimation. |
| `completion_tokens` | `int` | Output token count used for estimation. |
| `prompt_cost_usd` | `float` | Estimated cost for input tokens in USD. |
| `completion_cost_usd` | `float` | Estimated cost for output tokens in USD. |
| `total_cost_usd` | `float` | Total estimated cost in USD. |
| `known_model` | `bool` | `False` if the model is not in the built-in pricing table. |

---

### `MultiModelReport`

| Field | Type | Description |
|-------|------|-------------|
| `models` | `list[str]` | All model identifiers compared. |
| `responses` | `dict[str, str]` | Model → response text mapping. |
| `pair_scores` | `list[PairScore]` | All pairwise similarity results. |

### `PairScore`

| Field | Type | Description |
|-------|------|-------------|
| `model_a` | `str` | First model in this pair. |
| `model_b` | `str` | Second model in this pair. |
| `word_similarity` | `float` | Word-level similarity (0–1). |
| `semantic_score` | `float \| None` | Semantic similarity (0–1), if computed. |

---

### `JsonStructDiffResult`

| Field | Type | Description |
|-------|------|-------------|
| `entries` | `list[JsonStructEntry]` | Per-key diff entries. |
| `summary` | `dict[str, int]` | Count of each change type. |

### `JsonStructEntry`

| Field | Type | Description |
|-------|------|-------------|
| `key_path` | `str` | Dot-notation key path (e.g. `"address.city"`). |
| `status` | `str` | `"ADDED"`, `"REMOVED"`, `"CHANGED"`, `"TYPE_CHANGE"`, or `"UNCHANGED"`. |
| `value_a` | `Any` | Value in response A (`None` if absent). |
| `value_b` | `Any` | Value in response B (`None` if absent). |

---

## Complete examples

### Semantic comparison

```python
import asyncio
from llm_diff import compare

report = asyncio.run(
    compare(
        "Explain recursion in one sentence.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        semantic=True,
    )
)
print(f"Word:     {report.word_similarity:.2%}")
print(f"Semantic: {report.semantic_score:.2%}")
print(f"Primary:  {report.primary_score:.2%}")
```

### Prompt-diff

```python
from llm_diff import compare_prompts

report = asyncio.run(
    compare_prompts(
        "Explain recursion concisely.",
        "Explain recursion with a real-world analogy.",
        model="gpt-4o",
        semantic=True,
    )
)
print(f"Similarity: {report.primary_score:.2%}")
```

### Batch comparison

```python
from llm_diff import compare_batch

reports = asyncio.run(
    compare_batch(
        "examples/prompts.yml",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        semantic=True,
        concurrency=8,
    )
)
for r in reports:
    print(f"{r.comparison.response_a.model}  {r.word_similarity:.2%}")
```

### LLM-as-a-Judge

```python
from llm_diff import compare

report = asyncio.run(
    compare(
        "Explain closures in JavaScript.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        judge="gpt-4o-mini",
    )
)
if report.judge_result:
    jr = report.judge_result
    print(f"Winner:    {jr.winner}")
    print(f"Score A:   {jr.score_a}/10")
    print(f"Score B:   {jr.score_b}/10")
    print(f"Reasoning: {jr.reasoning}")
```

### Cost tracking

```python
from llm_diff import compare

report = asyncio.run(
    compare(
        "Write a haiku about Python.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        show_cost=True,
    )
)
if report.cost_a:
    print(f"Model A cost: ${report.cost_a.total_cost_usd:.6f}")
if report.cost_b:
    print(f"Model B cost: ${report.cost_b.total_cost_usd:.6f}")
```

### Multi-model comparison

```python
from llm_diff import run_multi_model

multi = asyncio.run(
    run_multi_model(
        "What is the capital of France?",
        ["gpt-4o", "gpt-4o-mini", "mistral-large-latest"],
        semantic=True,
    )
)
for pair in multi.pair_scores:
    print(f"{pair.model_a} vs {pair.model_b}: {pair.semantic_score:.2%}")
```

### JSON struct diff

```python
from llm_diff import json_struct_diff

result = json_struct_diff(
    '{"name": "Alice", "age": 30, "city": "NYC"}',
    '{"name": "Alice", "age": 31, "country": "US"}',
)
for entry in result.entries:
    print(f"{entry.status:15s} {entry.key_path}")
```

### Generate an HTML report

```python
from llm_diff import compare
from llm_diff.report import save_report

report = asyncio.run(
    compare(
        "Explain recursion.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        semantic=True,
        build_html=True,
    )
)
save_report(report.html_report, "report.html")
print("Saved to report.html")
```

---

## Schema Events

Every `llm-diff` operation emits a structured
[llm-toolkit-schema](https://pypi.org/project/llm-toolkit-schema/) event that
can be collected in memory, exported to JSONL, or forwarded to any custom
backend.  The integration is **zero-configuration by default** — events are
built and validated but discarded unless you opt in.

See the full guide at [docs/schema-events.md](schema-events.md).

---

### `configure_emitter()`

Replace the global event emitter.  Call once at application startup.

```python
from llm_diff.schema_events import configure_emitter

configure_emitter()          # in-memory only (default)
```

With a JSONL exporter:

```python
from llm_toolkit_schema.export.jsonl import JSONLExporter
from llm_diff.schema_events import configure_emitter

configure_emitter(exporter=JSONLExporter("llm-diff-events.jsonl"))
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exporter` | callable or object with `.export()` | `None` | Receives every emitted event. |
| `collect` | `bool` | `True` | Keep events in `get_emitter().events`. Set to `False` to reduce memory use in long-running processes. |

**Returns** `EventEmitter`

---

### `get_emitter()`

Return the active global `EventEmitter` instance.

```python
from llm_diff.schema_events import get_emitter

events = get_emitter().events   # list[Event]
```

---

### `emit(event)`

Emit an `Event` through the global emitter.  Useful when constructing custom
events to mix with llm-diff's built-in events.

---

### `EventEmitter`

| Method / Property | Description |
|-------------------|-------------|
| `emit(event)` | Validate and emit *event*; call exporter if configured. |
| `events` | Read-only list of all collected `Event` objects. |
| `clear()` | Remove all collected events from memory. |

---

### Event factory functions

All factory functions are importable from `llm_diff` or `llm_diff.schema_events`.

| Function | Event type emitted | When |
|----------|-------------------|------|
| `make_comparison_started_event(model_a, model_b, prompt)` | `llm.diff.comparison.started` | Before calling models |
| `make_comparison_completed_event(model_a, model_b, diff_type, ...)` | `llm.diff.comparison.completed` | After diff is computed |
| `make_report_exported_event(output_path, format, ...)` | `llm.diff.report.exported` | After `save_report()` |
| `make_trace_span_event(model, prompt_tokens, completion_tokens, latency_ms, ...)` | `llm.trace.span.completed` | After each model API call |
| `make_cache_event(hit, cache_key, backend, ...)` | `llm.cache.hit` / `llm.cache.miss` | On every cache lookup |
| `make_cost_recorded_event(input_cost, output_cost, total_cost, ...)` | `llm.cost.recorded` | When `show_cost=True` |
| `make_eval_scenario_event(evaluator, score, status, ...)` | `llm.eval.scenario.completed` | After LLM-as-a-Judge run |
| `make_eval_regression_event(scenario_name, current_score, baseline_score, threshold, ...)` | `llm.eval.regression.failed` | When `--fail-under` threshold is not met |

---

### Schema events example

```python
import asyncio
from llm_diff import compare
from llm_diff.schema_events import configure_emitter, get_emitter

# Opt in: collect events in memory
configure_emitter()

report = asyncio.run(
    compare(
        "Explain closures in JavaScript.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        show_cost=True,
        judge="gpt-4o-mini",
    )
)

for evt in get_emitter().events:
    print(evt.event_type, evt.event_id)
# llm.diff.comparison.started  01KJKVRV...
# llm.trace.span.completed      01KJKVRW...
# llm.trace.span.completed      01KJKVRX...
# llm.cost.recorded             01KJKVRY...
# llm.cost.recorded             01KJKVRZ...
# llm.eval.scenario.completed   01KJKVS0...
# llm.diff.comparison.completed 01KJKVS1...
```
