# CI / CD Integration

`llm-diff` is designed to run inside CI pipelines.  The `--fail-under` flag
turns model comparisons into pass/fail gates: exit code 0 means all prompts
met the threshold; exit code 1 means at least one failed.

---

## The `--fail-under` flag

Pass a decimal threshold (0.0–1.0) to enforce a minimum similarity score:

```bash
llm-diff "Summarise the changelog." \
  -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.85
```

Pass (exit code 0):
```
  Semantic similarity: 91.3%   threshold: 85.0%   PASS
```

Fail (exit code 1):
```
  Semantic similarity: 72.4%   threshold: 85.0%   FAIL
Error: similarity 0.724 < threshold 0.850
```

In **batch mode**, exit code 1 is triggered if *any* single prompt fails:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.80
```

```
  explain-recursion         87.2%   PASS
  code-review/func.py       74.1%   FAIL  <- below 80%
  summarise/article1.txt    83.6%   PASS

Error: 1 prompt(s) failed the similarity threshold (0.80)
```

---

## GitHub Actions — single prompt gate

Gate a push or PR on the similarity between `gpt-4o` and `gpt-4o-mini` for a
single critical prompt:

```yaml
# .github/workflows/llm-check.yml
name: LLM regression check

on: [push, pull_request]

jobs:
  llm-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install llm-diff
        run: pip install "llm-diff[semantic]"

      - name: Run similarity check
        run: |
          llm-diff "Summarise the latest changelog." \
            -a gpt-4o -b gpt-4o-mini \
            --semantic --fail-under 0.85
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## GitHub Actions — batch regression suite

Run a full `prompts.yml` regression suite, upload the HTML report as an
artifact, and fail the build if any prompt drops below the threshold:

```yaml
# .github/workflows/llm-regression.yml
name: LLM regression suite

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 6 * * *"   # daily at 06:00 UTC

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install llm-diff
        run: pip install "llm-diff[semantic]"

      - name: Run batch regression
        run: |
          llm-diff --batch prompts.yml \
            -a gpt-4o -b gpt-4o-mini \
            --semantic --fail-under 0.85 \
            --out diff_report.html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload diff report
        if: always()           # upload even when the check fails
        uses: actions/upload-artifact@v4
        with:
          name: llm-diff-report
          path: diff_report.html
```

The upload step runs even on failure (`if: always()`) so you can inspect the
report to see which prompts diverged.

---

## GitHub Actions — model upgrade validation

When upgrading to a new model version, validate that the new model's outputs
are sufficiently similar to the previous version before merging:

```yaml
# .github/workflows/model-upgrade.yml
name: Model upgrade validation

on:
  pull_request:
    paths:
      - "model_config.yml"    # trigger only when model config changes

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install llm-diff
        run: pip install "llm-diff[semantic]"

      - name: Validate model upgrade similarity
        run: |
          llm-diff --batch prompts.yml \
            -a gpt-4o \
            -b gpt-4o-2024-11-20 \
            --semantic --bleu --rouge \
            --fail-under 0.90 \
            --out model_upgrade_report.html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: model-upgrade-report
          path: model_upgrade_report.html
```

---

## Prompt engineering validation

After changing a system prompt, verify that the new outputs remain similar
to the baseline using prompt-diff mode:

```bash
# In CI
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic --fail-under 0.80
```

Or as a GitHub Actions step:

```yaml
      - name: Validate prompt change
        run: |
          llm-diff --model gpt-4o \
            --prompt-a prompts/system_v1.txt \
            --prompt-b prompts/system_v2.txt \
            --semantic --fail-under 0.80
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

---

## Caching in CI

By default, `llm-diff` caches responses to `~/.cache/llm-diff/`.  In CI this
cache is ephemeral — it persists only within a single job run.

To persist the cache across runs (saving API costs on repeated CI runs):

```yaml
      - name: Cache llm-diff responses
        uses: actions/cache@v4
        with:
          path: ~/.cache/llm-diff
          key: llm-diff-${{ hashFiles('prompts.yml') }}
          restore-keys: |
            llm-diff-
```

To disable caching entirely and always get fresh responses:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini --no-cache
```

---

## Cost management in CI

Use `--show-cost` to log the estimated spend per run:

```yaml
      - name: Run regression with cost tracking
        run: |
          llm-diff --batch prompts.yml \
            -a gpt-4o -b gpt-4o-mini \
            --semantic --show-cost \
            --out report.html
```

Cost estimates appear in the HTML report's per-item diff cards.

To keep CI costs low, consider:
- Using cheaper models (`gpt-4o-mini`, `llama-3.1-8b-instant`) for routine
  regression, and reserving expensive models for release gates
- Enabling the response cache so unchanged prompts aren't re-called
- Using `--concurrency` with a modest value (4–8) to avoid rate-limit
  penalties

---

## Choosing a threshold

| Scenario | Recommended `--fail-under` |
|----------|---------------------------|
| Minor model version bump | 0.90 — high bar, subtle regressions caught |
| Major model upgrade | 0.80 — allows for style changes while catching regressions |
| Prompt engineering iteration | 0.70 — expect larger differences by design |
| Cross-model comparison (A/B) | 0.60 — two different models will naturally diverge |

Use `--semantic` for thresholds above 0.80 — word similarity can
undercount similarity when models rephrase the same meaning.  BLEU and
ROUGE-L are good secondary signals but semantic similarity should be the
primary gate.
