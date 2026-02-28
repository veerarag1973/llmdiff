# Tutorial 04 — Batch Evaluation

**Time:** ~15 minutes  
**Level:** Intermediate  
**Prerequisites:** [Tutorial 03](03-prompt-engineering.md) completed

← [03 — Prompt Engineering](03-prompt-engineering.md) | [05 — CI/CD Regression Gate →](05-ci-cd-regression-gate.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Write a `prompts.yml` file that defines multiple test cases
- Use file-based inputs with the `{input}` placeholder
- Run an entire prompt suite in one command
- Read the batch summary table and identify failing cases
- Open the batch HTML report and navigate its sections
- Tune `--concurrency` to balance speed and API rate limits

---

## The Problem with One-at-a-Time Testing

In the previous tutorials you compared one pair of prompts at a time.  This
works for exploration, but it does not scale.

Real evaluation scenarios need to cover a range of inputs:

- A summarisation prompt should work on short articles, long articles, and
  articles with unusual structure
- A code-review prompt should handle Python, JavaScript, and SQL
- A customer support prompt should handle billing questions, technical
  questions, and hostile tone

Running 12 comparisons manually, one by one, takes time and produces no
consolidated view of where your prompt is performing well versus poorly.
Batch mode solves this.

---

## Step 1 — Understand the `prompts.yml` format

A batch run is driven by a YAML file.  The minimal structure is:

```yaml
model_a: gpt-4o
model_b: gpt-4o-mini

cases:
  - prompt: "What is the capital of France?"
  - prompt: "Explain what a REST API is."
  - prompt: "Write a haiku about software bugs."
```

Save this as `eval.yml` in your project root.  Then run:

```bash
llm-diff batch eval.yml
```

Output:

```
  Running 3 cases ...

  ┌───┬─────────────────────────────────────────┬────────┬──────────┐
  │ # │ Prompt (truncated)                      │  Word  │ Semantic │
  ├───┼─────────────────────────────────────────┼────────┼──────────┤
  │ 1 │ What is the capital of France?          │  94.1% │   98.2%  │
  │ 2 │ Explain what a REST API is.             │  38.7% │   82.4%  │
  │ 3 │ Write a haiku about software bugs.      │  12.1% │   67.8%  │
  └───┴─────────────────────────────────────────┴────────┴──────────┘

  Cases: 3   Passed: 3   Failed: 0
  Mean semantic similarity: 82.8%
```

---

## Step 2 — Add semantic scoring and a threshold

Extend `eval.yml` with global options:

```yaml
model_a: gpt-4o
model_b: gpt-4o-mini
semantic: true
fail_under: 0.75

cases:
  - prompt: "What is the capital of France?"
  - prompt: "Explain what a REST API is."
  - prompt: "Write a haiku about software bugs."
```

Now cases with a semantic score below 75% will be marked as failures:

```
  ┌───┬─────────────────────────────────────────┬────────┬──────────┬────────┐
  │ # │ Prompt (truncated)                      │  Word  │ Semantic │ Status │
  ├───┼─────────────────────────────────────────┼────────┼──────────┼────────┤
  │ 1 │ What is the capital of France?          │  94.1% │   98.2%  │  PASS  │
  │ 2 │ Explain what a REST API is.             │  38.7% │   82.4%  │  PASS  │
  │ 3 │ Write a haiku about software bugs.      │  12.1% │   67.8%  │  FAIL  │
  └───┴─────────────────────────────────────────┴────────┴──────────┴────────┘

  Cases: 3   Passed: 2   Failed: 1
  Mean semantic similarity: 82.8%
```

The command exits with code 1 because at least one case failed, making it
safe to use as a CI gate.

---

## Step 3 — Use the `{input}` placeholder for parameterised cases

For prompts that share the same template but differ only in their input data,
use the `{input}` placeholder.  This keeps your YAML clean and separates
concerns:

```yaml
model_a: gpt-4o
model_b: gpt-4o-mini
semantic: true
fail_under: 0.80

prompt_template: "Summarise the following article in 3 bullet points:\n\n{input}"

cases:
  - input: inputs/article_a.txt
  - input: inputs/article_b.txt
  - input: inputs/article_c.txt
```

Each case loads its file, substitutes `{input}` in the template, and runs the
comparison.  The input column in the summary table shows the filename rather
than the full prompt text.

> **Note:** The `inputs/` directory already exists in this project with
> `article_a.txt` and `article_b.txt`.  You can use those to try this now.

---

## Step 4 — Per-case overrides

Individual cases can override any top-level setting:

```yaml
model_a: gpt-4o
model_b: gpt-4o-mini
semantic: true
fail_under: 0.80

cases:
  - prompt: "What is 2 + 2?"
    fail_under: 0.95          # stricter threshold for factual questions

  - prompt: "Write a poem about the ocean."
    fail_under: 0.50          # creative tasks — allow more divergence
    semantic: false           # semantic scoring not useful for poetry

  - prompt: "Explain DNS in simple terms."
    # uses global defaults
```

This gives you fine-grained control without duplicating the entire file.

---

## Step 5 — Save the batch HTML report

Add `--out` to save an HTML report covering all cases:

```bash
llm-diff batch eval.yml --out batch-report.html
```

The batch HTML report contains:

| Section | Contents |
|---------|----------|
| Summary header | Total cases, pass/fail count, mean scores |
| Case table | Sortable table with scores and status for every case |
| Per-case cards | Each case expands to show the full word diff and metrics |
| Score distribution | Bar chart of semantic scores across all cases |

Open `batch-report.html` in your browser to review the results.

---

## Step 6 — Tune concurrency

By default, batch runs execute 3 comparisons at a time.  Increase this if your
API rate limits allow:

```bash
llm-diff batch eval.yml --concurrency 10
```

Or reduce it if you are hitting rate limit errors:

```bash
llm-diff batch eval.yml --concurrency 1
```

A `--concurrency` of 10 with 50 cases and a 2-second average response time
will finish in roughly 10 seconds instead of 100 seconds.

**General guidance:**

| API tier | Recommended concurrency |
|----------|------------------------|
| Free / trial | 1–2 |
| Pay-as-you-go | 5–10 |
| Enterprise / high-RPM | 20–50 |

---

## Step 7 — Combine with prompt-diff mode in batch

Batch mode also supports prompt-diff (two prompts, same model) at scale:

```yaml
model: gpt-4o
semantic: true
fail_under: 0.78

prompt_a_template: "Answer this question concisely: {input}"
prompt_b_template: "Answer this question. Lead with the key fact. Then explain in one sentence: {input}"

cases:
  - input: "What is machine learning?"
  - input: "What is a hash table?"
  - input: "What is the difference between TCP and UDP?"
```

This is the recommended format for evaluating a prompt rewrite across a diverse
set of inputs before deploying the new version.

---

## Summary

You have now:

- ✅ Written a `prompts.yml` file with global settings and individual cases
- ✅ Used `fail_under` to automatically flag under-performing cases
- ✅ Used the `{input}` placeholder to separate prompt templates from test data
- ✅ Applied per-case overrides for fine-grained control
- ✅ Saved and explored the batch HTML report
- ✅ Tuned `--concurrency` to match your API rate limits
- ✅ Run prompt-diff in batch mode across multiple inputs

---

## What's Next

You can now evaluate an entire prompt suite with one command.  The next step is
to plug that command into your CI pipeline so that every code change is
automatically tested against your quality thresholds.

[Tutorial 05 — CI/CD Regression Gate →](05-ci-cd-regression-gate.md)
