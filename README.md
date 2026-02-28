# llm-diff

**Compare LLM outputs — semantically, visually, and at scale.**

![PyPI](https://img.shields.io/pypi/v/llm-diff)
![CI](https://img.shields.io/github/actions/workflow/status/user/llm-diff/ci.yml?branch=main)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/pypi/pyversions/llm-diff)
![License](https://img.shields.io/pypi/l/llm-diff)
![Status](https://img.shields.io/badge/status-production--stable-brightgreen)

---

## Table of Contents

- [The Problem](#the-problem)
- [How llm-diff Solves It](#how-llm-diff-solves-it)
- [What You Get](#what-you-get)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [1. Compare two models on the same prompt](#1-compare-two-models-on-the-same-prompt)
  - [2. Add semantic similarity scoring](#2-add-semantic-similarity-scoring)
  - [3. Per-paragraph scoring](#3-per-paragraph-scoring)
  - [4. BLEU and ROUGE-L metrics](#4-bleu-and-rouge-l-metrics)
  - [5. Prompt-diff mode](#5-prompt-diff-mode)
  - [6. Save an HTML report](#6-save-an-html-report)
  - [7. JSON output](#7-json-output)
  - [8. Batch mode](#8-batch-mode)
  - [9. CI gate with --fail-under](#9-ci-gate-with---fail-under)
  - [10. Result caching](#10-result-caching)
  - [11. Verbose metadata](#11-verbose-metadata)
- [Full CLI Reference](#full-cli-reference)
- [Programmatic API](#programmatic-api)
- [CI/CD Integration](#cicd-integration)
- [HTML Reports](#html-reports)
- [Community & Feedback](#community--feedback)
- [Contributing](#contributing)

---

## The Problem

LLMs do not produce deterministic output. When you are evaluating models,
iterating on prompts, or trying to understand how a model upgrade affects your
application, you are left doing this manually:

- Copy response A into one text editor
- Copy response B into another
- Scroll through them side-by-side trying to spot differences
- Guess whether the meaning actually changed or just the phrasing
- Repeat for every prompt, every model pair, every iteration

This is time-consuming, error-prone, and does not scale. If you have 20 prompts
and just swapped `gpt-4o` for `gpt-4o-mini`, you are looking at an hour of
tedious comparison with no way to record results or enforce a quality bar.

Teams hit this wall repeatedly:

- **Model evaluation** — "Is `claude-3-5-sonnet` actually better than `gpt-4o`
  for our use case?"
- **Prompt engineering** — "Did changing the system prompt improve consistency,
  or just rearrange words?"
- **Regression testing** — "Our model provider pushed an update. Did anything
  break?"
- **A/B testing** — "We are running two prompt variants in production. How
  different are they really?"

There was no tool built for this job.

---

## How llm-diff Solves It

`llm-diff` is a CLI tool and Python library that automates LLM output
comparison from end to end.

You run one command. `llm-diff`:

1. **Calls both models in parallel** via their APIs (OpenAI, Groq, Ollama,
   LM Studio, or any OpenAI-compatible endpoint)
2. **Diffs the responses word-by-word** — every insertion, deletion, and
   unchanged span is tagged and highlighted
3. **Scores them semantically** using sentence embeddings so you know if the
   *meaning* diverged even when the words look different
4. **Renders the result** as a coloured terminal diff or exports a
   self-contained HTML report
5. **Scales to batch workloads** — run a YAML file of prompts across two
   models concurrently and get a combined summary report
6. **Caches responses** so iterating on thresholds or report settings does not
   burn API credits
7. **Gates CI pipelines** via `--fail-under` — fail the build if similarity
   drops below a threshold

---

## What You Get

| Benefit | Detail |
|---|---|
| **Instant visual diff** | Word-level colour highlighting in the terminal — no copy-paste, no manual comparison |
| **Objective similarity score** | A 0-100% number you can track over time, compare across models, and enforce in CI |
| **Semantic awareness** | Detects when two responses mean the same thing despite different words |
| **Shareable reports** | Single self-contained HTML file, no CDN, works offline |
| **Batch evaluation** | Score all your prompts in one run with a summary table |
| **Zero wasted API calls** | Response cache means re-running after tweaking settings costs nothing |
| **CI integration** | `--fail-under 0.85` turns a model upgrade into a pass/fail regression gate |
| **BLEU & ROUGE-L** | Industry-standard NLP metrics — zero extra dependencies, pure Python |
| **Vendor agnostic** | OpenAI, Groq, Mistral, DeepSeek, Ollama, LM Studio — any OpenAI-compatible API |
| **Scriptable** | Full Python library API with no shell dependency |

---

## Installation

### Core (word-diff + HTML reports)

```bash
pip install llm-diff
```

### With semantic scoring

```bash
pip install "llm-diff[semantic]"
```

The `[semantic]` extra installs `sentence-transformers`. The default model
(`all-MiniLM-L6-v2`, ~80 MB) downloads on first use.

### Development install

```bash
git clone https://github.com/user/llm-diff.git
cd llm-diff
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[semantic,dev]"
```

---

## Configuration

`llm-diff` needs API keys. Provide them as environment variables or in a
`.llmdiff` TOML config file.

### Option A — environment variables

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

### Option B — `.llmdiff` config file

Create `.llmdiff` in your project root or `~/.llmdiff` for global defaults:

```toml
[providers.openai]
api_key = "sk-..."          # overrides OPENAI_API_KEY env var

[providers.groq]
api_key = "gsk_..."         # overrides GROQ_API_KEY env var

[defaults]
temperature = 0.7
max_tokens  = 1024
timeout     = 30
save        = false         # auto-save every report to ./diffs/
no_color    = false
```

All keys are optional. The file is looked up in the current directory first,
then `~/.llmdiff`, then environment variables as the final fallback.

### Provider auto-detection

`llm-diff` detects the provider from the model name — no extra flags needed:

| Model prefix | Provider | Key env var |
|---|---|---|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | `OPENAI_API_KEY` |
| `llama*`, `mixtral*`, `gemma*` | Groq | `GROQ_API_KEY` |
| `mistral*`, `codestral*` | Mistral AI | `MISTRAL_API_KEY` |
| `deepseek*` | DeepSeek | `DEEPSEEK_API_KEY` |
| `claude*` | via LiteLLM proxy | `ANTHROPIC_API_KEY` |
| anything else | Custom (`base_url` required) | — |

### Local models (Ollama / LM Studio)

```bash
ollama pull llama3.2
ollama serve
```

```toml
[providers.custom]
api_key  = "ollama"
base_url = "http://localhost:11434/v1"
```

### Anthropic (via LiteLLM proxy)

```bash
pip install litellm
litellm --model anthropic/claude-3-5-sonnet-20241022 --port 4000
```

```toml
[providers.custom]
api_key  = "sk-ant-..."
base_url = "http://localhost:4000/v1"
```

---

## Usage

All examples below assume `OPENAI_API_KEY` is set.

---

### 1. Compare two models on the same prompt

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-3.5-turbo
```

Output:

```
Model A (gpt-4o):
  Recursion is a technique where a function calls itself with a simpler
  version of the problem until a base case is reached.

Model B (gpt-3.5-turbo):
  Recursion is when a function calls itself repeatedly until a base
  condition is met, solving the problem step by step.

Word diff:
  Recursion is [-a technique where a-] {+when a+} function calls itself
  [-with a simpler version of the problem until a base case is reached.-]
  {+repeatedly until a base condition is met, solving the problem step by step.+}

  Word similarity:  61.3%
  Model A tokens:   28    latency: 0.84s
  Model B tokens:   24    latency: 0.61s
```

`[-deleted-]` spans are text removed from A. `{+inserted+}` spans are text
added in B. In the terminal these render as red and green respectively.

---

### 2. Add semantic similarity scoring

```bash
llm-diff "Explain recursion in one sentence." \
  -a gpt-4o -b gpt-3.5-turbo --semantic
```

Output:

```
  Word similarity:     61.3%
  Semantic similarity: 92.7%   <- same meaning, different words
  Primary score:       92.7%
```

A semantic score above ~85% generally means the responses are equivalent in
meaning even if the wording differs. Requires `pip install "llm-diff[semantic]"`.

---

### 3. Per-paragraph scoring

```bash
llm-diff "Write a 3-paragraph summary of transformer architecture." \
  -a gpt-4o -b gpt-4o-mini --paragraph
```

Output:

```
  Word similarity:     54.1%
  Semantic similarity: 88.6%

  Paragraph similarities:
    #1  Transformers rely on self-attention to...    94.2%
    #2  The encoder processes input tokens in...     81.3%
    #3  Training uses masked language modelling...   79.8%
```

Use `--paragraph` when responses are long enough that a single score does not
tell the whole story.

---

### 4. BLEU and ROUGE-L metrics

Compute industry-standard NLP evaluation metrics with no extra dependencies:

```bash
llm-diff "Explain recursion in one sentence." \
  -a gpt-4o -b gpt-3.5-turbo --bleu --rouge
```

Output:

```
  Word similarity:  61.3%
  BLEU:             42.1%
  ROUGE-L:          68.7%
```

- **BLEU** measures n-gram precision (how many phrases from B appear in A)
  with a brevity penalty. Good for checking surface-level phrase overlap.
- **ROUGE-L** measures the Longest Common Subsequence F1, capturing sentence-
  level structural similarity without requiring consecutive matches.

Both are pure Python, require zero extra packages, and run in under 10 ms.
Combine them with `--semantic` for a full picture:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo \
  --semantic --bleu --rouge
```

```
  Word similarity:     61.3%
  Semantic similarity: 92.7%
  BLEU:                42.1%
  ROUGE-L:             68.7%
```

---

### 5. Prompt-diff mode

Compare how two different prompts affect the same model:

```bash
llm-diff --model gpt-4o \
  --prompt-a "Explain recursion concisely." \
  --prompt-b "Explain recursion with a real-world analogy." \
  --semantic
```

Or using files:

```bash
llm-diff --model gpt-4o --prompt-a v1.txt --prompt-b v2.txt --semantic
```

Output:

```
  Comparing prompts on gpt-4o
  Prompt A: Explain recursion concisely.
  Prompt B: Explain recursion with a real-world analogy.

  Word similarity:     38.4%
  Semantic similarity: 74.1%
```

This is the primary workflow for prompt engineering — see whether your rewrites
actually change the output and by how much.

---

### 6. Save an HTML report

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo \
  --semantic --out report.html
```

Output:

```
  Word similarity:     61.3%
  Semantic similarity: 92.7%
  Report saved -> report.html
```

Open `report.html` in any browser. The file is fully self-contained — no
server, no internet, no CDN dependencies.

To auto-save every run to `./diffs/` without specifying `--out` each time:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo --save
```

Or set `save = true` in your `.llmdiff` config.

---

### 7. JSON output

```bash
llm-diff "Hello." -a gpt-4o -b gpt-3.5-turbo --json
```

Output:

```json
{
  "prompt_a": "Hello.",
  "prompt_b": "Hello.",
  "model_a": "gpt-4o",
  "model_b": "gpt-3.5-turbo",
  "response_a": "Hello! How can I assist you today?",
  "response_b": "Hello! How can I help you?",
  "diff": [
    {"type": "equal",  "text": "Hello! How can I "},
    {"type": "delete", "text": "assist you today"},
    {"type": "insert", "text": "help you"},
    {"type": "equal",  "text": "?"}
  ],
  "word_similarity": 0.714,
  "semantic_score": null,
  "tokens_a": {"prompt": 2, "completion": 9, "total": 11},
  "tokens_b": {"prompt": 2, "completion": 7, "total": 9},
  "latency_a_ms": 812,
  "latency_b_ms": 594
}
```

Extract a single field with `jq`:

```bash
llm-diff "Hello." -a gpt-4o -b gpt-3.5-turbo --json | jq '.word_similarity'
# 0.714
```

---

### 8. Batch mode

Evaluate a set of prompts across two models in one command.

Create `prompts.yml`:

```yaml
prompts:
  - id: explain-recursion
    text: "Explain recursion in one sentence."

  - id: code-review
    text: "Review this Python function for readability: {input}"
    inputs:
      - examples/func.py

  - id: summarise
    text: "Summarise the following: {input}"
    inputs:
      - examples/article1.txt
      - examples/article2.txt
```

- `id` — label shown in the terminal and HTML report
- `text` — prompt template; use `{input}` as a file placeholder
- `inputs` — list of files relative to the YAML; one batch item per file

Run the batch:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-3.5-turbo \
  --semantic --out batch_report.html
```

Output:

```
  Fetching responses for 4 prompt(s) (concurrency=4)...

  explain-recursion          word: 61.3%   semantic: 92.7%
  code-review/func.py        word: 74.8%   semantic: 89.2%
  summarise/article1.txt     word: 48.3%   semantic: 83.1%
  summarise/article2.txt     word: 51.0%   semantic: 85.5%

  Average word similarity:      58.9%
  Average semantic similarity:  87.6%
  Report saved -> batch_report.html
```

By default up to 4 API calls run in parallel. Override with `--concurrency`:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-3.5-turbo \
  --concurrency 8 --out batch_report.html
```

A working example is in [`examples/prompts.yml`](examples/prompts.yml).

---

### 9. CI gate with `--fail-under`

Fail the command if similarity drops below a threshold:

```bash
llm-diff "Summarise the changelog." \
  -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.85
```

Passes (exit code 0):

```
  Semantic similarity: 91.3%   threshold: 85.0%   PASS
```

Fails (exit code 1):

```
  Semantic similarity: 72.4%   threshold: 85.0%   FAIL
Error: similarity 0.724 < threshold 0.850
```

In batch mode, exit 1 is triggered if any single prompt drops below the
threshold:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.80
```

Output:

```
  explain-recursion        87.2%   PASS
  code-review/func.py      74.1%   FAIL  <- below 80%
  summarise/article1.txt   83.6%   PASS

Error: 1 prompt(s) failed the similarity threshold (0.80)
```

---

### 10. Result caching

By default `llm-diff` caches responses to `~/.cache/llm-diff/` keyed on
`(model, prompt, temperature, max_tokens)`. Re-running the same diff is
instant and free:

```bash
# First run — calls the API
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo --semantic
# latency: 0.84s / 0.61s

# Second run — served from cache
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo --semantic
# latency: 0.00s / 0.00s  (cached)
```

Caching is most useful when iterating on thresholds, scoring options, or
report formatting without re-paying for the same API calls.

To bypass the cache and always call the API:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo --no-cache
```

---

### 11. Verbose metadata

```bash
llm-diff "Hello." -a gpt-4o -b gpt-3.5-turbo --verbose
```

Output:

```
  [gpt-4o]
    provider:          openai
    prompt_tokens:     2
    completion_tokens: 9
    total_tokens:      11
    latency:           812ms

  [gpt-3.5-turbo]
    provider:          openai
    prompt_tokens:     2
    completion_tokens: 7
    total_tokens:      9
    latency:           594ms
```

---

## Full CLI Reference

```
Usage: llm-diff [OPTIONS] [PROMPT]

  Compare two LLM responses — semantically, visually, and at scale.

Arguments:
  PROMPT               Prompt text sent to both models.

Model selection:
  -a, --model-a MODEL  Model for side A  (e.g. gpt-4o).
  -b, --model-b MODEL  Model for side B  (e.g. gpt-4o-mini).
  -m, --model   MODEL  Same model for both sides (prompt-diff mode).

Prompt sources:
  --prompt-a PATH      Path to a text file used as prompt for model A.
  --prompt-b PATH      Path to a text file used as prompt for model B.
  --batch    PATH      Path to a prompts.yml file for batch comparison.

Scoring:
  -s, --semantic       Compute embedding-based cosine similarity score.
  -p, --paragraph      Per-paragraph similarity (implies --semantic).
  --bleu               Compute BLEU score (n-gram precision, no extra deps).
  --rouge              Compute ROUGE-L F1 score (LCS-based, no extra deps).

Output:
  -j, --json           Output raw JSON to stdout.
  -o, --out PATH       Save HTML report to this path.
  --save               Auto-save HTML report to ./diffs/.

API settings:
  -t, --temperature FLOAT   Temperature for both models  (default 0.7).
  --max-tokens INT          Max tokens per response      (default 1024).
  --timeout SECS            Request timeout in seconds   (default 30).

Quality gate:
  --fail-under FLOAT   Exit 1 if primary score < threshold.

Performance:
  --concurrency INT    Max parallel API calls in batch mode (default 4).
  --no-cache           Skip cache; always call the API.

Display:
  --no-color           Disable terminal colour output.
  -v, --verbose        Show full API metadata per request.

  --version            Show version and exit.
  -h, --help           Show this message and exit.
```

---

## Programmatic API

Use `llm-diff` as a Python library:

```python
import asyncio
from llm_diff import compare

report = asyncio.run(
    compare("Explain recursion.", model_a="gpt-4o", model_b="gpt-3.5-turbo")
)

print(f"Word similarity:   {report.word_similarity:.2%}")
print(f"Response A tokens: {report.comparison.response_a.total_tokens}")
```

### With semantic scoring

```python
report = asyncio.run(
    compare(
        "Explain recursion.",
        model_a="gpt-4o",
        model_b="gpt-3.5-turbo",
        semantic=True,
    )
)
print(f"Semantic similarity: {report.semantic_score:.2%}")
print(f"Primary score:       {report.primary_score:.2%}")
```

### Prompt-diff (same model, two prompts)

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

### Batch

```python
from llm_diff import compare_batch

reports = asyncio.run(
    compare_batch(
        "examples/prompts.yml",
        model_a="gpt-4o",
        model_b="gpt-3.5-turbo",
        semantic=True,
    )
)
for r in reports:
    print(f"{r.comparison.response_a.model}: {r.word_similarity:.2%}")
```

### ComparisonReport fields

| Field | Type | Description |
|---|---|---|
| `prompt_a` | `str` | Prompt sent to model A |
| `prompt_b` | `str` | Prompt sent to model B |
| `comparison` | `ComparisonResult` | Raw paired model responses |
| `diff_result` | `DiffResult` | Word-level diff chunks + similarity |
| `semantic_score` | `float or None` | Whole-text cosine similarity (0-1) |
| `paragraph_scores` | `list or None` | Per-paragraph similarity scores |
| `bleu_score` | `float or None` | BLEU score (0-1), n-gram precision |
| `rouge_l_score` | `float or None` | ROUGE-L F1 score (0-1), LCS-based |
| `html_report` | `str or None` | Self-contained HTML (when `build_html=True`) |
| `word_similarity` | `float` | Property — `diff_result.similarity` |
| `primary_score` | `float` | Property — semantic if set, else word |

---

## CI/CD Integration

```yaml
# .github/workflows/llm-check.yml
name: LLM regression check

on: [push, pull_request]

jobs:
  llm-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install llm-diff
        run: pip install "llm-diff[semantic]"

      - name: Run similarity check
        run: |
          llm-diff --batch prompts.yml \
            -a gpt-4o -b gpt-4o-mini \
            --semantic --fail-under 0.85 \
            --out diff_report.html
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: diff-report
          path: diff_report.html
```

This makes model upgrades and prompt changes reviewable, auditable, and
enforceable — the same discipline applied to code, now applied to LLM output.

---

## HTML Reports

Pass `--out report.html` to generate a fully self-contained HTML report.
No CDN dependencies — works offline in air-gapped environments.

Single-diff reports include:
- Side-by-side responses with coloured word-diff
- Word similarity score and optional semantic score
- Optional per-paragraph similarity table
- Token usage and latency metadata

Batch reports add:
- Summary table with all prompts, scores, and pass/fail highlighting
- Per-item expandable diff cards
- Average similarity across the batch

---

## Community & Feedback

`llm-diff` is open source and community-driven. Your feedback, bug reports,
and feature ideas directly shape the roadmap.

| Action | Link |
|---|---|
| **Report a bug** | [Open a bug report](https://github.com/user/llm-diff/issues/new?labels=bug&template=bug_report.md) |
| **Request a feature** | [Open a feature request](https://github.com/user/llm-diff/issues/new?labels=enhancement&template=feature_request.md) |
| **Browse open issues** | [github.com/user/llm-diff/issues](https://github.com/user/llm-diff/issues) |
| **Join the discussion** | [GitHub Discussions](https://github.com/user/llm-diff/discussions) |
| **View the roadmap** | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |

When filing an issue, please include:
- The `llm-diff` version (`llm-diff --version`)
- The command you ran
- The full error output (if applicable)
- Your OS and Python version

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, testing instructions,
and PR guidelines.

---

## License

MIT — see [LICENSE](LICENSE).
