# Tutorial 08 — Cost Tracking

**Time:** ~10 minutes  
**Level:** Advanced  
**Prerequisites:** [Tutorial 07](07-multi-model-comparison.md) completed

← [07 — Multi-Model Comparison](07-multi-model-comparison.md) | [09 — JSON Struct Diff →](09-json-struct-diff.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Display per-run cost breakdowns with `--show-cost`
- Read the cost table in HTML reports
- Use `estimate_cost()` from the Python API before running a large batch
- Apply strategies to keep your evaluation costs low
- Calculate quality-per-dollar to make model selection decisions

---

## Why Track Costs?

LLM API costs are invisible by default — they accumulate in your billing
dashboard, not in your terminal output.  For evaluation workloads, costs add
up quickly:

- A 50-case batch against GPT-4o with judge mode: ~$0.30
- 100 PRs a month each triggering that batch: ~$30/month
- Scaled to a team of 10 engineers: ~$300/month just for CI evaluation

Cost tracking makes this visible so you can make deliberate trade-offs —
choosing a smaller model for CI and reserving GPT-4o for final validation,
for example.

---

## Step 1 — Display costs with `--show-cost`

Add `--show-cost` to any comparison:

```bash
llm-diff "Explain database indexing." \
  -a gpt-4o \
  -b gpt-4o-mini \
  --semantic \
  --show-cost
```

Output:

```
  Comparing gpt-4o vs gpt-4o-mini

  Word similarity:     38.4%
  Semantic similarity: 86.2%

  ── Cost breakdown ─────────────────────────────────────────────────

  ┌────────────────┬───────────┬────────────┬──────────┬──────────┐
  │ Model          │  In tokens│ Out tokens │ In cost  │ Out cost │
  ├────────────────┼───────────┼────────────┼──────────┼──────────┤
  │ gpt-4o         │       12  │       287  │  $0.0001 │  $0.0029 │
  │ gpt-4o-mini    │       12  │       234  │  $0.0000 │  $0.0001 │
  ├────────────────┼───────────┼────────────┼──────────┼──────────┤
  │ Total          │       24  │       521  │  $0.0001 │  $0.0030 │
  │ Run total      │           │            │          │  $0.0031 │
  └────────────────┴───────────┴────────────┴──────────┴──────────┘
```

The cost breakdown shows input and output tokens and costs separately, since
output tokens cost more than input tokens for all major providers.

---

## Step 2 — Batch cost summary

```bash
llm-diff batch eval.yml --show-cost --out eval-with-costs.html
```

The terminal output adds a cost summary after the case table:

```
  Cases: 10   Passed: 9   Failed: 1
  Mean semantic similarity: 83.7%

  ── Cost summary ───────────────────────────────────────────────────

  Total tokens:   model_a: 4,821   model_b: 3,940
  Total cost:     model_a: $0.048  model_b: $0.004
  Run total:      $0.052

  Cache hits: 0 / 20 calls
```

The `Cache hits` line shows how many API calls were served from cache.  After
the first run, subsequent identical runs cost nothing:

```
  Cache hits: 20 / 20 calls
  Run total:  $0.000
```

---

## Step 3 — Cost table in HTML reports

Every HTML report generated with `--show-cost` includes a **Cost Table**
section showing:

- Per-model token and cost breakdown
- A total row
- Cache hit count and estimated savings
- Cost per case average

This is useful for documenting evaluation spend in team reports or budgeting
discussions.

---

## Step 4 — Estimate cost before running

Use `estimate_cost()` from the Python API to dry-run a cost estimate before
making API calls:

```python
from llm_diff import estimate_cost

# Estimate a single comparison
result = estimate_cost(
    prompt="Explain the event loop in Node.js.",
    model_a="gpt-4o",
    model_b="claude-3-7-sonnet-20250219",
    expected_output_tokens=300,
)
print(f"Estimated cost: ${result.total:.4f}")
# Estimated cost: $0.0094
```

```python
# Estimate a batch
import yaml
from llm_diff import estimate_batch_cost

with open("eval.yml") as f:
    config = yaml.safe_load(f)

result = estimate_batch_cost(config, expected_output_tokens=350)
print(f"Cases: {result.case_count}")
print(f"Estimated cost: ${result.total:.4f}")
print(f"With judge:     ${result.total_with_judge:.4f}")
# Cases: 20
# Estimated cost: $0.1840
# With judge:     $0.2760
```

> **Note:** `estimate_cost()` and `estimate_batch_cost()` use the official
> provider pricing tables bundled with `llm-diff`.  Run `llm-diff pricing` to
> see the current table.

---

## Step 5 — View the current pricing table

```bash
llm-diff pricing
```

```
  ┌──────────────────────────┬─────────────┬──────────────┐
  │ Model                    │  Input $/1M │  Output $/1M │
  ├──────────────────────────┼─────────────┼──────────────┤
  │ gpt-4o                   │      $2.50  │      $10.00  │
  │ gpt-4o-mini              │      $0.15  │       $0.60  │
  │ claude-3-7-sonnet-20250219        │      $3.00  │      $15.00  │
  │ claude-3-5-haiku-20241022           │      $0.25  │       $1.25  │
  │ gemini-2.0-pro           │      $1.25  │       $5.00  │
  │ gemini-2.0-flash         │      $0.075 │       $0.30  │
  │ mistral-large-latest     │      $2.00  │       $6.00  │
  │ mistral-small-latest     │      $0.20  │       $0.60  │
  └──────────────────────────┴─────────────┴──────────────┘
```

Pricing is updated with each `llm-diff` release.  If a model is not listed,
`llm-diff` will display a warning and skip cost calculation for that model.

---

## Step 6 — Quality-per-dollar analysis

To make model selection decisions that balance quality and cost, calculate
quality per dollar:

```python
from llm_diff import compare_batch, estimate_batch_cost
import asyncio, yaml

async def quality_per_dollar():
    candidates = [
        ("gpt-4o",       "gpt-4o-mini"),
        ("claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"),
        ("gemini-2.0-pro",    "gemini-2.0-flash"),
    ]

    with open("eval.yml") as f:
        config = yaml.safe_load(f)

    print(f"{'Pair':45s} {'Semantic':>10} {'Cost':>10} {'Q/$':>10}")
    print("-" * 80)

    for model_a, model_b in candidates:
        config["model_a"] = model_a
        config["model_b"] = model_b

        results = await compare_batch(config, semantic=True)
        mean_score = sum(r.semantic for r in results) / len(results)

        cost_est = estimate_batch_cost(config, expected_output_tokens=300)
        cost = cost_est.total

        quality_per_dollar = mean_score / cost if cost > 0 else float("inf")

        print(
            f"{model_a} vs {model_b:20s}"
            f"{mean_score:>10.1%}"
            f"   ${cost:>7.4f}"
            f"   {quality_per_dollar:>7.0f}"
        )

asyncio.run(quality_per_dollar())
```

Output:

```
  Pair                                         Semantic       Cost        Q/$
  ──────────────────────────────────────────────────────────────────────────
  gpt-4o vs gpt-4o-mini                          83.4%    $0.0520      1604
  claude-3-7-sonnet-20250219 vs claude-3-5-haiku-20241022             81.9%    $0.0380      2155
  gemini-2.0-pro vs gemini-2.0-flash              80.1%    $0.0120      6675
```

In this example, Gemini Flash delivers the best quality per dollar — 80% of
the quality of the GPT-4o pair at 23% of the cost.

---

## Strategies to Keep Costs Low

| Strategy | Impact |
|----------|--------|
| Warm cache between CI runs | Eliminates repeat API calls entirely |
| Use `--concurrency 1` on free tier | Prevents rate-limit retries (retries cost money) |
| Use small model for smoke tests | 10× cheaper than GPT-4o quality gates |
| Reserve judge mode for final review | Judge adds ~50% to API call count |
| Trim test prompts — avoid padding | Input tokens cost money too |
| Use `estimate_cost()` before large batches | Prevent billing surprises |

---

## Summary

You have now:

- ✅ Used `--show-cost` to display per-run token and cost breakdowns
- ✅ Read the cost table in HTML reports
- ✅ Used `estimate_cost()` and `estimate_batch_cost()` before running
- ✅ Inspected the current pricing table with `llm-diff pricing`
- ✅ Calculated quality-per-dollar using the Python API
- ✅ Applied concrete strategies to keep evaluation costs low

---

## What's Next

All the comparisons so far have been on free-form text responses.  The next
tutorial introduces a specialised mode for comparing structured JSON outputs —
useful when your models return function call results, data extraction payloads,
or API responses.

[Tutorial 09 — JSON Struct Diff →](09-json-struct-diff.md)
