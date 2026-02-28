# llm-diff

**Compare LLM outputs — semantically, visually, and at scale.**

`llm-diff` is a CLI tool and Python library that sends the same prompt to two
different models (or the same model with two different prompts), diffs the
responses word-by-word, and optionally scores them with sentence embeddings.
Results are rendered as a rich terminal diff or exported as a fully
self-contained HTML report.

![PyPI](https://img.shields.io/pypi/v/llm-diff)
![CI](https://img.shields.io/github/actions/workflow/status/user/llm-diff/ci.yml?branch=main)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![Python](https://img.shields.io/pypi/pyversions/llm-diff)
![License](https://img.shields.io/github/license/user/llm-diff)

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Provider Configuration](#provider-configuration)
- [Batch Mode](#batch-mode)
- [Programmatic API](#programmatic-api)
- [CI/CD Integration](#cicd-integration)
- [HTML Reports](#html-reports)
- [Configuration File](#configuration-file)
- [Contributing](#contributing)

---

## Features

- **Word-level diff** — coloured insert / delete / equal spans rendered in the terminal
- **Semantic similarity** — whole-text cosine similarity via `sentence-transformers`
- **Paragraph scoring** — per-paragraph similarity table (ideal for long documents)
- **Batch mode** — run a YAML file of prompts across two models; get a combined HTML report
- **`--fail-under`** — fail CI if similarity drops below a threshold
- **Programmatic API** — use as a library with no Click dependency
- **Self-contained HTML reports** — single file, no CDN dependencies, offline-ready
- **Multi-provider** — OpenAI, Groq, Ollama, LM Studio, any OpenAI-compatible endpoint

---

## Installation

### Core (word-diff only)

```bash
pip install llm-diff
```

### With semantic scoring

```bash
pip install "llm-diff[semantic]"
```

The `[semantic]` extra adds `sentence-transformers` (~400 MB first run for the
default `all-MiniLM-L6-v2` model).

### Development / editable install

```bash
git clone https://github.com/user/llm-diff.git
cd llm-diff
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[semantic,dev]"
```

---

## Quick Start

**Step 1 — set your API key**

```bash
export OPENAI_API_KEY="sk-..."
```

**Step 2 — run your first diff**

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-3.5-turbo
```

You'll see a coloured word-diff in your terminal alongside a similarity score.
That's it — under 60 seconds from `pip install` to first diff.

---

## CLI Reference

```
Usage: llm-diff [OPTIONS] [PROMPT]

  Compare two LLM responses — semantically, visually, and at scale.

Options:
  PROMPT               Prompt text sent to both models.

  -a, --model-a MODEL  Model for side A  (e.g. gpt-4o).
  -b, --model-b MODEL  Model for side B  (e.g. claude-3-5-sonnet).
  -m, --model   MODEL  Same model for both sides (prompt-diff mode).

  --prompt-a PATH      Path to a text file used as prompt for model A.
  --prompt-b PATH      Path to a text file used as prompt for model B.

  --batch PATH         Path to a prompts.yml file for batch comparison.

  -s, --semantic       Compute embedding-based cosine similarity score.
  -p, --paragraph      Per-paragraph similarity (implies --semantic).
  -j, --json           Output raw JSON to stdout.

  -o, --out PATH       Save HTML report to this path.
  --save               Auto-save HTML report to ./diffs/.

  -t, --temperature FLOAT   Temperature for both models  (default 0.7).
  --max-tokens INT          Max tokens per response      (default 1024).
  --timeout SECS            Request timeout in seconds   (default 30).

  --fail-under FLOAT   Exit 1 if primary score < threshold (CI gate).

  --no-color           Disable terminal colour output.
  -v, --verbose        Show full API metadata per request.

  --version            Show version and exit.
  -h, --help           Show this message and exit.
```

### Examples

```bash
# Two models, same prompt
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo

# Same model, two different prompt variants (prompt-diff)
llm-diff --prompt-a v1.txt --prompt-b v2.txt --model gpt-4o

# With semantic scoring
llm-diff "Summarise this article." -a gpt-4o -b claude-3-5-sonnet --semantic

# Per-paragraph scoring
llm-diff "Write a product description." -a gpt-4o -b gpt-4-turbo --paragraph

# Save HTML report
llm-diff "Explain recursion." -a gpt-4o -b gpt-3.5-turbo --out report.html

# JSON output (pipe to jq)
llm-diff "Hello." -a gpt-4o -b gpt-3.5-turbo --json | jq '.word_similarity'

# Batch mode
llm-diff --batch examples/prompts.yml -a gpt-4o -b gpt-3.5-turbo --out batch.html

# CI gate — fail if similarity drops below 80 %
llm-diff "Critical prompt." -a gpt-4o -b gpt-3.5-turbo --semantic --fail-under 0.8
```

---

## Provider Configuration

`llm-diff` uses the `openai` Python SDK with optional `base_url` overrides,
so it works with any OpenAI-compatible endpoint.

### Auto-detection

| Model prefix | Provider detected | Key env var |
|---|---|---|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | `OPENAI_API_KEY` |
| `llama*`, `mixtral*`, `gemma*` | Groq | `GROQ_API_KEY` |
| `mistral*`, `codestral*` | Mistral AI | `MISTRAL_API_KEY` |
| `deepseek*` | DeepSeek | `DEEPSEEK_API_KEY` |
| `claude*`, `anthropic*` | via LiteLLM proxy | `ANTHROPIC_API_KEY` |
| anything else | Custom (`base_url` required) | — |

### `.llmdiff` config file

Create `.llmdiff` in your project root (or `~/.llmdiff` for global defaults):

```toml
[providers.openai]
api_key = "sk-..."          # or use OPENAI_API_KEY env var

[defaults]
temperature = 0.7
max_tokens  = 1024
timeout     = 30
```

#### Anthropic (via LiteLLM proxy)

```bash
pip install litellm
litellm --model anthropic/claude-3-5-sonnet-20241022 --port 4000
```

```toml
[providers.custom]
api_key  = "sk-ant-..."
base_url = "http://localhost:4000/v1"
```

#### Ollama (local)

```bash
ollama pull llama3.2
ollama serve
```

```toml
[providers.custom]
api_key  = "ollama"
base_url = "http://localhost:11434/v1"
```

```bash
llm-diff "Explain recursion." -a llama3.2 -b mistral
```

#### Groq

```toml
[providers.groq]
api_key = "gsk_..."         # or set GROQ_API_KEY env var
```

```bash
llm-diff "Explain recursion." -a llama-3.3-70b-versatile -b mixtral-8x7b-32768
```

See [docs/providers.md](docs/providers.md) for the full provider guide.

---

## Batch Mode

Run a list of prompts against two models in one command and get a combined
HTML report.

### `prompts.yml` format

```yaml
prompts:
  - id: explain-recursion
    text: "Explain recursion in one sentence."

  - id: summarise
    text: "Summarise the following article: {input}"
    inputs:
      - article1.txt
      - article2.txt

  - id: code-review
    text: "Review this Python function for readability: {input}"
    inputs:
      - func.py
```

- **`id`** — unique identifier (shown in the report header)
- **`text`** — prompt template; use `{input}` as a placeholder
- **`inputs`** — list of file paths (relative to the YAML file); one batch
  item is generated per file

### Running a batch

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-3.5-turbo --out report.html
```

A working example is in [`examples/prompts.yml`](examples/prompts.yml).

---

## Programmatic API

Use `llm-diff` as a library without any CLI dependency:

```python
import asyncio
from llm_diff import compare

report = asyncio.run(
    compare("Explain recursion.", model_a="gpt-4o", model_b="gpt-3.5-turbo")
)

print(f"Word similarity:    {report.word_similarity:.2%}")
print(f"Response A tokens:  {report.comparison.response_a.total_tokens}")
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

### `ComparisonReport` fields

| Field | Type | Description |
|---|---|---|
| `prompt_a` | `str` | Prompt sent to model A |
| `prompt_b` | `str` | Prompt sent to model B |
| `comparison` | `ComparisonResult` | Raw paired model responses |
| `diff_result` | `DiffResult` | Word-level diff chunks + similarity |
| `semantic_score` | `float \| None` | Whole-text cosine similarity |
| `paragraph_scores` | `list \| None` | Per-paragraph similarity scores |
| `html_report` | `str \| None` | Self-contained HTML (when `build_html=True`) |
| `word_similarity` | `float` | Property — `diff_result.similarity` |
| `primary_score` | `float` | Property — semantic if set, else word |

---

## CI/CD Integration

Use `--fail-under` to gate deployments on response consistency:

```yaml
# .github/workflows/llm-check.yml
- name: Check model consistency
  run: |
    llm-diff "Summarise the changelog." \
      -a gpt-4o -b gpt-4o-mini \
      --semantic --fail-under 0.85
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

- Uses **semantic score** when `--semantic` or `--paragraph` is set
- Uses **word similarity** otherwise
- Exit code `1` triggers the CI failure; exit code `0` means all clear

Batch variant — fail if *any* prompt drops below the threshold:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.80
```

---

## HTML Reports

Pass `--out report.html` (single diff) or `--out batch.html` (batch) to
generate a fully self-contained HTML file.  No CDN dependencies — works
offline or in air-gapped environments.

Single-diff reports include:

- Side-by-side responses with coloured word-diff
- Word similarity score + optional semantic score
- Optional per-paragraph similarity table
- Token usage and latency metadata

Batch reports add:

- Summary table (all prompts, scores, pass/fail highlighting)
- Per-item expandable diff cards
- Average similarity across the batch

---

## Configuration File

`llm-diff` looks for a `.llmdiff` TOML file in the current directory, then
`~/.llmdiff` (user home), then falls back to environment variables.

```toml
[providers.openai]
api_key   = "sk-..."          # overrides OPENAI_API_KEY

[providers.groq]
api_key   = "gsk_..."         # overrides GROQ_API_KEY

[providers.custom]
api_key   = "local"
base_url  = "http://localhost:11434/v1"   # Ollama / LM Studio

[defaults]
temperature = 0.7
max_tokens  = 1024
timeout     = 30
save        = false           # auto-save every report to ./diffs/
no_color    = false
```

All keys are optional; omit any section you don't need.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, testing instructions,
and PR guidelines.

---

## License

MIT — see [LICENSE](LICENSE).
