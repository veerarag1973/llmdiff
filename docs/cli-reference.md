# CLI Reference

```
Usage: llm-diff [OPTIONS] [PROMPT]

  Compare two LLM responses — semantically, visually, and at scale.
```

---

## Arguments

| Argument | Description |
|----------|-------------|
| `PROMPT` | Prompt text sent to both models. Mutually exclusive with `--batch`. |

---

## Option groups

### Model selection

| Flag | Type | Description |
|------|------|-------------|
| `-a`, `--model-a` | `TEXT` | Model for side A (e.g. `gpt-4o`). Required unless `--batch` or `--model` is used. |
| `-b`, `--model-b` | `TEXT` | Model for side B (e.g. `gpt-4o-mini`). Required unless `--model` is used. |
| `-m`, `--model` | `TEXT` | Same model for both sides — activates **prompt-diff mode**. |
| `--model-c` | `TEXT` | Add a third model to enable multi-model comparison. |
| `--model-d` | `TEXT` | Add a fourth model to enable multi-model comparison. |

### Prompt sources

| Flag | Type | Description |
|------|------|-------------|
| `--prompt-a` | `PATH` | Text file used as prompt for model A (prompt-diff mode). |
| `--prompt-b` | `PATH` | Text file used as prompt for model B (prompt-diff mode). |
| `--batch` | `PATH` | Path to a `prompts.yml` file for batch comparison. |

### Scoring

| Flag | Description |
|------|-------------|
| `-s`, `--semantic` | Compute embedding-based cosine similarity score (requires `[semantic]` extra). |
| `-p`, `--paragraph` | Per-paragraph similarity breakdown (implies `--semantic`). |
| `--bleu` | Compute BLEU score — n-gram precision with brevity penalty. No extra dependencies. |
| `--rouge` | Compute ROUGE-L F1 score — Longest Common Subsequence based. No extra dependencies. |

### Evaluation depth

| Flag | Type | Description |
|------|------|-------------|
| `--judge` | `TEXT` | Model name to use as LLM-as-a-Judge scorer. Returns a winner (A/B/tie), per-model scores (1–10), and reasoning. |
| `--show-cost` | — | Show estimated USD cost for each API call. Pricing data built-in for 35+ models. |
| `--mode` | `word`\|`json`\|`json-struct` | Diff mode. Default: `word`. Use `json-struct` for key-by-key comparison of JSON responses. |

### Output

| Flag | Type | Description |
|------|------|-------------|
| `-j`, `--json` | — | Output raw JSON to stdout instead of the terminal diff. Pipe to `jq` for field extraction. |
| `-o`, `--out` | `PATH` | Save a self-contained HTML report to this path. |
| `--save` | — | Auto-save every HTML report to `./diffs/` without specifying `--out` each time. |

### API settings

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-t`, `--temperature` | `FLOAT` | `0.7` | Sampling temperature for both models (0.0–2.0). |
| `--max-tokens` | `INT` | `1024` | Maximum tokens per response. |
| `--timeout` | `INT` | `30` | Request timeout in seconds. |

### Quality gate

| Flag | Type | Description |
|------|------|-------------|
| `--fail-under` | `FLOAT` | Exit with code 1 if the primary similarity score is below this threshold (0.0–1.0). |

### Performance

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--concurrency` | `INT` | `4` | Maximum number of parallel API calls in batch mode. |
| `--no-cache` | — | — | Bypass the response cache; always call the API. |

### Display

| Flag | Description |
|------|-------------|
| `--no-color` | Disable terminal colour output. |
| `-v`, `--verbose` | Show full API metadata (provider, token counts, latency) per request. |
| `--version` | Print the installed version and exit. |
| `-h`, `--help` | Show help message and exit. |

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success — all prompts at or above the `--fail-under` threshold (or no threshold set). |
| `1` | Failure — one or more prompts fell below the `--fail-under` threshold, or a fatal error occurred. |

---

## Operation modes

### Single diff (default)

Pass a prompt string and two model flags:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini
```

### Prompt-diff mode

Use `--model` (same model) with `--prompt-a` / `--prompt-b`:

```bash
llm-diff --model gpt-4o \
  --prompt-a "Explain recursion concisely." \
  --prompt-b "Explain recursion with a real-world analogy." \
  --semantic
```

### Batch mode

Pass a YAML file with `--batch`:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini --semantic --out report.html
```

### Multi-model mode

Add `--model-c` (and optionally `--model-d`):

```bash
llm-diff "What is the capital of France?" \
  -a gpt-4o -b gpt-4o-mini \
  --model-c claude-3-5-sonnet-20241022 --model-d mistral-large-latest
```

All pairwise similarity scores are computed concurrently and displayed as a
ranked table.

### JSON struct diff mode

Use `--mode json-struct` when both responses are JSON objects:

```bash
llm-diff 'Return JSON: {"name": "...", "age": ...}' \
  -a gpt-4o -b gpt-4o-mini --mode json-struct
```

Each key path is labelled `ADDED`, `REMOVED`, `CHANGED`, `TYPE_CHANGE`, or
`UNCHANGED`.  Falls back to word diff if one or both responses are not valid
JSON.

---

## Batch YAML format

```yaml
prompts:
  - id: explain-recursion        # label shown in terminal + HTML report
    text: "Explain recursion."   # prompt template

  - id: code-review
    text: "Review this function: {input}"   # use {input} as file placeholder
    inputs:
      - examples/func.py          # one batch item per file

  - id: summarise
    text: "Summarise: {input}"
    inputs:
      - examples/article_a.txt
      - examples/article_b.txt
```

File paths in `inputs` are resolved relative to the YAML file.

---

## JSON output schema

When `--json` is used, the following object is printed to stdout:

```json
{
  "prompt_a":        "...",
  "prompt_b":        "...",
  "model_a":         "gpt-4o",
  "model_b":         "gpt-4o-mini",
  "response_a":      "...",
  "response_b":      "...",
  "diff": [
    {"type": "equal",  "text": "..."},
    {"type": "delete", "text": "..."},
    {"type": "insert", "text": "..."}
  ],
  "word_similarity":  0.614,
  "semantic_score":   0.927,
  "bleu_score":       0.421,
  "rouge_l_score":    0.687,
  "tokens_a":  {"prompt": 2, "completion": 28, "total": 30},
  "tokens_b":  {"prompt": 2, "completion": 24, "total": 26},
  "latency_a_ms": 840,
  "latency_b_ms": 610,
  "judge_result": null,
  "cost_a": null,
  "cost_b": null
}
```

Extract a single value with `jq`:

```bash
llm-diff "Hello." -a gpt-4o -b gpt-4o-mini --json | jq '.word_similarity'
```
