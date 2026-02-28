# Tutorial 05 — CI/CD Regression Gate

**Time:** ~15 minutes  
**Level:** Intermediate  
**Prerequisites:** [Tutorial 04](04-batch-evaluation.md) completed

← [04 — Batch Evaluation](04-batch-evaluation.md) | [06 — LLM-as-a-Judge →](06-llm-as-a-judge.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Understand what a regression gate is and why it matters for LLM applications
- Add `llm-diff` as a step in a GitHub Actions workflow
- Cache API responses across CI runs to keep costs near zero
- Set sensible `--fail-under` thresholds for different use cases
- Interpret CI failures and act on them

---

## What is a Regression Gate?

A regression gate is a check that runs automatically on every code change and
fails the build if quality drops below a defined threshold.

For traditional software this is a test suite.  For LLM applications it is a
prompt evaluation suite.

Without a regression gate, the following can happen silently:

- A developer refactors the system prompt and introduces an edge-case regression
- A new model version (e.g. `o3`) produces subtly different outputs
- A changed environment variable accidentally switches the model tier
- A dependency upgrade changes tokenisation behaviour

A regression gate catches all of these automatically — before they reach
production.

---

## Step 1 — Understand how `--fail-under` works

The `--fail-under` flag accepts a decimal value between 0.0 and 1.0.  The
command exits with code 1 if the primary similarity score falls below the
threshold:

```bash
# Single comparison
llm-diff "Explain DNS." -a gpt-4o -b o3 \
  --semantic --fail-under 0.90

# Batch evaluation
llm-diff batch eval.yml --fail-under 0.80
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | All comparisons passed the threshold |
| 1 | One or more comparisons failed the threshold |
| 2 | Configuration error (invalid flags, missing API key, etc.) |

CI systems (GitHub Actions, GitLab CI, Jenkins) automatically treat a non-zero
exit code as a build failure.

---

## Step 2 — Create a minimal GitHub Actions workflow

Create `.github/workflows/llm-eval.yml`:

```yaml
name: LLM Regression Gate

on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'llm_diff/**'
      - 'eval.yml'

jobs:
  evaluate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install llm-diff
        run: pip install llm-diff

      - name: Restore response cache
        uses: actions/cache@v4
        with:
          path: ~/.cache/llm_diff
          key: llm-diff-cache-${{ hashFiles('eval.yml', 'prompts/**') }}
          restore-keys: llm-diff-cache-

      - name: Run evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: llm-diff batch eval.yml --semantic --fail-under 0.80
```

This workflow:
1. Triggers only when prompt-related files change (not on every commit)
2. Caches API responses so repeated runs with the same prompts are free
3. Fails the PR if any case scores below 80%

---

## Step 3 — Understand the cache mechanism

`llm-diff` caches every API response keyed on:

```
(model, prompt_text, temperature, max_tokens)
```

The cache is stored at `~/.cache/llm_diff/` by default (configurable via
`LLMDIFF_CACHE_DIR`).

In CI, the `actions/cache` step saves and restores this directory.  The cache
key includes a hash of `eval.yml` and all prompt files — so:

- If nothing changes, the cache hits and the run costs nothing
- If a prompt changes, only the affected cases call the API
- If a new case is added, only that case calls the API

**Cost estimate for a 20-case suite:**

| Scenario | API calls | Approximate cost |
|----------|-----------|-----------------|
| First run (cold cache) | 40 (20 × 2 models) | ~$0.04 |
| Subsequent runs (warm cache) | 0 | $0.00 |
| One prompt changed | 2 | ~$0.004 |

---

## Step 4 — Set appropriate thresholds

There is no universal threshold.  The right value depends on what you are
measuring:

| Use case | Recommended threshold | Reasoning |
|----------|-----------------------|-----------|
| Same model, minor version bump | ≥ 0.90 | Outputs should be nearly identical |
| Same model family, different size | ≥ 0.80 | Smaller models are less precise |
| Different model families | ≥ 0.70 | Architecture differences are expected |
| Creative / generative tasks | ≥ 0.55 | High variance is inherent |
| Factual / structured tasks | ≥ 0.85 | Consistency is critical |
| Prompt rewrite (same intent) | ≥ 0.75 | Allow expression to change |

**How to calibrate:**

1. Run your eval suite on a known-good baseline
2. Note the lowest-scoring case
3. Set `fail_under` at roughly that score minus 5–10 percentage points
4. This gives you a safety margin for natural variation while still catching
   genuine regressions

---

## Step 5 — Advanced: matrix strategy for model version tracking

Track multiple model versions in parallel using a matrix strategy:

```yaml
jobs:
  evaluate:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model_b: [o3, gpt-4o, gpt-4o-mini]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install llm-diff
        run: pip install llm-diff

      - name: Run evaluation vs ${{ matrix.model_b }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          llm-diff batch eval.yml \
            --model-b ${{ matrix.model_b }} \
            --semantic \
            --fail-under 0.80 \
            --out report-${{ matrix.model_b }}.html

      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: llm-diff-report-${{ matrix.model_b }}
          path: report-${{ matrix.model_b }}.html
```

This runs three parallel jobs and uploads separate HTML reports for each model
version, giving you a clear comparison dashboard in GitHub Actions.

---

## Step 6 — GitLab CI equivalent

```yaml
llm-eval:
  image: python:3.11
  stage: test

  cache:
    key: llm-diff-$CI_COMMIT_REF_SLUG
    paths:
      - .cache/llm_diff/

  variables:
    LLMDIFF_CACHE_DIR: .cache/llm_diff

  script:
    - pip install llm-diff
    - llm-diff batch eval.yml --semantic --fail-under 0.80

  only:
    changes:
      - prompts/**
      - eval.yml
```

---

## Summary

You have now:

- ✅ Understood what a regression gate is and why LLM apps need one
- ✅ Used `--fail-under` exit codes to integrate with CI systems
- ✅ Written a GitHub Actions workflow that evaluates on every PR
- ✅ Configured the response cache to keep CI costs near zero
- ✅ Selected appropriate thresholds for different use cases
- ✅ Extended the workflow with matrix strategy for multi-version tracking

---

## What's Next

The metrics used so far — semantic similarity, BLEU, ROUGE — measure *how
similar* two responses are.  They do not tell you *which response is better*.
The next tutorial introduces LLM-as-a-Judge mode, which uses a language model
to make that qualitative judgement.

[Tutorial 06 — LLM-as-a-Judge →](06-llm-as-a-judge.md)
