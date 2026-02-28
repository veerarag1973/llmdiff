# Tutorial 10 — Python API

**Time:** ~20 minutes  
**Level:** Advanced  
**Prerequisites:** [Tutorial 09](09-json-struct-diff.md) completed

← [09 — JSON Struct Diff](09-json-struct-diff.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Use `compare()` for single model-diff comparisons in Python
- Use `compare_prompts()` for single prompt-diff comparisons in Python
- Use `compare_batch()` to run an entire eval suite from code
- Build and save HTML reports programmatically
- Use async patterns efficiently
- Integrate `llm-diff` into a pytest evaluation harness

---

## When to Use the Python API

The CLI is the right tool for interactive use and CI scripts.  The Python API
is the right tool when you need to:

- Embed evaluation logic inside an existing Python application or test suite
- Compose dynamic evaluations (prompts generated at runtime, cases from a
  database, thresholds loaded from a config service)
- Post-process results (aggregate across runs, write to a database, send alerts)
- Build custom dashboards or reports beyond what the CLI provides
- Run `llm-diff` inside a Jupyter notebook or data pipeline

---

## Step 1 — `compare()`: single model-diff

The `compare()` function is the Python equivalent of the basic CLI command:

```python
import asyncio
from llm_diff import compare

async def main():
    result = await compare(
        prompt="Explain dependency injection.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        semantic=True,
    )

    print(f"Word similarity:     {result.word_similarity:.1%}")
    print(f"Semantic similarity: {result.semantic_similarity:.1%}")
    print(f"Primary score:       {result.primary_score:.1%}")

asyncio.run(main())
```

### The `CompareResult` object

| Attribute | Type | Description |
|-----------|------|-------------|
| `result.response_a` | `str` | Full response from model A |
| `result.response_b` | `str` | Full response from model B |
| `result.word_similarity` | `float` | Word-level similarity (0.0–1.0) |
| `result.semantic_similarity` | `float \| None` | Semantic similarity if requested |
| `result.bleu` | `float \| None` | BLEU score if requested |
| `result.rouge` | `float \| None` | ROUGE-L F1 if requested |
| `result.primary_score` | `float` | Best available score (semantic > BLEU > ROUGE > word) |
| `result.word_diff` | `str` | Formatted word diff string |
| `result.paragraph_scores` | `list[float] \| None` | Per-paragraph semantic scores |
| `result.judge` | `JudgeResult \| None` | Judge verdict if requested |
| `result.cost` | `CostBreakdown \| None` | Cost breakdown if requested |

---

## Step 2 — `compare()` with all options

```python
result = await compare(
    prompt="Explain the actor model of concurrency.",
    model_a="gpt-4o",
    model_b="claude-3-7-sonnet-20250219",
    semantic=True,
    paragraph=True,
    bleu=True,
    rouge=True,
    judge="gpt-4o",
    show_cost=True,
    temperature=0.0,
    max_tokens=500,
)

# Access judge result
if result.judge:
    print(f"Winner:    {result.judge.winner}")
    print(f"Score A:   {result.judge.score_a}/10")
    print(f"Score B:   {result.judge.score_b}/10")
    print(f"Reasoning: {result.judge.reasoning}")

# Access cost
if result.cost:
    print(f"Total cost: ${result.cost.total:.4f}")

# Access paragraph scores
if result.paragraph_scores:
    for i, score in enumerate(result.paragraph_scores, 1):
        print(f"  Paragraph {i}: {score:.1%}")
```

---

## Step 3 — `compare_prompts()`: single prompt-diff

The `compare_prompts()` function is the Python equivalent of the
`--model --prompt-a --prompt-b` CLI mode:

```python
from llm_diff import compare_prompts

result = await compare_prompts(
    model="gpt-4o",
    prompt_a="Summarise this article concisely.",
    prompt_b=(
        "Summarise this article in 3 bullet points. "
        "Lead with the most important finding."
    ),
    semantic=True,
)

print(f"Semantic similarity: {result.semantic_similarity:.1%}")
```

You can also pass file paths (the function detects them automatically):

```python
result = await compare_prompts(
    model="gpt-4o",
    prompt_a="prompts/system_v1.txt",
    prompt_b="prompts/system_v2.txt",
    semantic=True,
    judge="gpt-4o",
)
```

---

## Step 4 — `compare_batch()`: batch evaluation from code

```python
from llm_diff import compare_batch

config = {
    "model_a": "gpt-4o",
    "model_b": "gpt-4o-mini",
    "semantic": True,
    "fail_under": 0.80,
    "cases": [
        {"prompt": "Explain recursion."},
        {"prompt": "What is a hash collision?"},
        {"prompt": "Explain the difference between == and is in Python."},
    ],
}

results = await compare_batch(config, concurrency=5)

for i, result in enumerate(results, 1):
    status = "PASS" if result.primary_score >= 0.80 else "FAIL"
    print(f"  [{status}] Case {i}: {result.primary_score:.1%}")

passed = sum(1 for r in results if r.primary_score >= 0.80)
print(f"\n{passed}/{len(results)} cases passed")
```

### Loading config from a YAML file

```python
import yaml
from llm_diff import compare_batch

with open("eval.yml") as f:
    config = yaml.safe_load(f)

results = await compare_batch(config)
```

---

## Step 5 — Build and save HTML reports programmatically

```python
from llm_diff import compare, render_report, render_batch_report

# Single-comparison report
result = await compare(
    prompt="Explain event sourcing.",
    model_a="gpt-4o",
    model_b="gpt-4o-mini",
    semantic=True,
)

html = render_report(result)
with open("report.html", "w") as f:
    f.write(html)

print("Saved report.html")
```

```python
# Batch report
results = await compare_batch(config)

html = render_batch_report(results, config)
with open("batch-report.html", "w") as f:
    f.write(html)
```

---

## Step 6 — Async patterns: running comparisons concurrently

When you have multiple independent comparisons to run, use `asyncio.gather()`
to run them concurrently rather than sequentially:

```python
import asyncio
from llm_diff import compare

async def run_all():
    prompts = [
        "Explain dependency injection.",
        "What is a race condition?",
        "Explain the CAP theorem.",
        "What is a Bloom filter?",
        "Explain eventual consistency.",
    ]

    # Run all 5 comparisons concurrently
    tasks = [
        compare(
            prompt=p,
            model_a="gpt-4o",
            model_b="gpt-4o-mini",
            semantic=True,
        )
        for p in prompts
    ]

    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"{result.primary_score:.1%}  {prompt[:50]}")

asyncio.run(run_all())
```

> **Note:** Be mindful of API rate limits when running many concurrent
> requests.  Use a semaphore to limit concurrency:

```python
import asyncio
from llm_diff import compare

async def run_limited(prompts, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def compare_one(prompt):
        async with semaphore:
            return await compare(
                prompt=prompt,
                model_a="gpt-4o",
                model_b="gpt-4o-mini",
                semantic=True,
            )

    return await asyncio.gather(*[compare_one(p) for p in prompts])
```

---

## Step 7 — Integrate into pytest

`llm-diff` integrates naturally into pytest for teams that want evaluation as
part of their test suite:

```python
# tests/test_prompt_quality.py

import pytest
import asyncio
from llm_diff import compare_batch
import yaml

with open("eval.yml") as f:
    EVAL_CONFIG = yaml.safe_load(f)

THRESHOLD = 0.80


@pytest.fixture(scope="session")
def batch_results():
    """Run the full eval suite once per test session."""
    return asyncio.get_event_loop().run_until_complete(
        compare_batch(EVAL_CONFIG, semantic=True)
    )


def test_all_cases_pass_threshold(batch_results):
    """Every case must score above the quality threshold."""
    failures = [
        (i + 1, r.primary_score)
        for i, r in enumerate(batch_results)
        if r.primary_score < THRESHOLD
    ]
    assert not failures, (
        f"{len(failures)} case(s) below threshold {THRESHOLD:.0%}:\n"
        + "\n".join(f"  Case {i}: {s:.1%}" for i, s in failures)
    )


def test_mean_score_above_threshold(batch_results):
    """Mean score across all cases must exceed the threshold."""
    mean = sum(r.primary_score for r in batch_results) / len(batch_results)
    assert mean >= THRESHOLD, f"Mean score {mean:.1%} < {THRESHOLD:.0%}"


@pytest.mark.parametrize("case_index", range(len(EVAL_CONFIG["cases"])))
def test_individual_case(batch_results, case_index):
    """Each case is also individually testable (useful for debugging)."""
    result = batch_results[case_index]
    prompt = EVAL_CONFIG["cases"][case_index].get("prompt", "")[:60]
    assert result.primary_score >= THRESHOLD, (
        f"Case {case_index + 1} ({prompt!r}) scored {result.primary_score:.1%}"
    )
```

Run with:

```bash
pytest tests/test_prompt_quality.py -v
```

---

## Step 8 — Full pipeline example

A complete evaluation pipeline that runs a batch, saves a report, and sends
a summary to a Slack webhook:

```python
import asyncio
import json
import urllib.request
import yaml
from llm_diff import compare_batch, render_batch_report

THRESHOLD = 0.80
SLACK_WEBHOOK = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"


async def evaluate_and_report():
    with open("eval.yml") as f:
        config = yaml.safe_load(f)

    print("Running evaluation...")
    results = await compare_batch(config, semantic=True, show_cost=True)

    # Save HTML report
    html = render_batch_report(results, config)
    with open("latest-report.html", "w") as f:
        f.write(html)

    # Compute summary
    passed = sum(1 for r in results if r.primary_score >= THRESHOLD)
    failed = len(results) - passed
    mean = sum(r.primary_score for r in results) / len(results)
    total_cost = sum(r.cost.total for r in results if r.cost)

    status = "✅ PASS" if failed == 0 else "❌ FAIL"

    summary = (
        f"{status} — LLM eval {passed}/{len(results)} cases passed "
        f"| mean: {mean:.1%} | cost: ${total_cost:.4f}"
    )
    print(summary)

    # Post to Slack
    payload = json.dumps({"text": summary}).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(evaluate_and_report())
    raise SystemExit(0 if success else 1)
```

---

## Summary

You have now:

- ✅ Used `compare()` for single model-diff with all available options
- ✅ Used `compare_prompts()` for single prompt-diff from code
- ✅ Used `compare_batch()` to run eval suites and load YAML configs
- ✅ Built and saved single and batch HTML reports programmatically
- ✅ Run comparisons concurrently with `asyncio.gather()` and a semaphore
- ✅ Integrated `llm-diff` into a pytest test suite
- ✅ Built a full pipeline: evaluate → report → notify

---

## You Have Completed the Tutorial Series

Congratulations — you have worked through all ten tutorials.

| Tutorial | What you learned |
|----------|-----------------|
| [00 Introduction](00-introduction.md) | What llm-diff is and why it exists |
| [01 First Comparison](01-first-comparison.md) | CLI basics, responses, HTML reports, cache |
| [02 Semantic Scoring](02-semantic-scoring.md) | Embeddings, `--semantic`, `--bleu`, `--rouge` |
| [03 Prompt Engineering](03-prompt-engineering.md) | Prompt-diff mode, iteration workflow, `--fail-under` |
| [04 Batch Evaluation](04-batch-evaluation.md) | `prompts.yml`, `{input}`, concurrency, batch reports |
| [05 CI/CD Gate](05-ci-cd-regression-gate.md) | GitHub Actions, cache in CI, threshold guidance |
| [06 LLM-as-a-Judge](06-llm-as-a-judge.md) | `--judge`, winner/score/reasoning, biases |
| [07 Multi-Model](07-multi-model-comparison.md) | `--model-c/d`, pairwise ranking, `run_multi_model()` |
| [08 Cost Tracking](08-cost-tracking.md) | `--show-cost`, `estimate_cost()`, quality-per-dollar |
| [09 JSON Struct Diff](09-json-struct-diff.md) | `--mode json-struct`, field labels, `json_struct_diff()` |
| [10 Python API](10-python-api.md) | `compare()`, `compare_batch()`, pytest harness, async |

### Where to go next

- [API Reference](../api.md) — complete function signatures and return types
- [CLI Reference](../cli-reference.md) — every flag documented
- [Configuration Reference](../configuration.md) — full TOML / YAML schema
- [GitHub Discussions](https://github.com/veerarag1973/llmdiff/discussions) — ask questions, share workflows
- [PyPI](https://pypi.org/project/llm-diff/) — check for new versions
