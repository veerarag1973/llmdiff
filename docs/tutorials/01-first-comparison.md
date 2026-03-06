# Tutorial 01 — Your First Comparison

**Time:** ~10 minutes  
**Level:** Beginner  
**Prerequisites:** Python 3.9+, an OpenAI API key, `llm-diff` installed

← [00 — Introduction](00-introduction.md) | [02 — Semantic Scoring →](02-semantic-scoring.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Run a model-vs-model comparison from the command line
- Read and interpret the terminal diff output
- Understand what word similarity means and what it tells you
- Save a self-contained HTML report and open it in a browser
- Use the response cache to re-run instantly without extra API calls

---

## Step 1 — Install llm-diff

If you haven't installed it yet:

```bash
pip install "llm-diff[semantic]"
```

Verify the install:

```bash
llm-diff --version
# llm-diff, version 1.3.1
```

---

## Step 2 — Set your API key

```bash
export OPENAI_API_KEY="sk-..."          # macOS / Linux
$env:OPENAI_API_KEY = "sk-..."          # Windows PowerShell
```

Or create a `.llmdiff` file in your project root so you don't have to set it
every session:

```toml
# .llmdiff
[providers.openai]
api_key = "sk-..."
```

> **Security note:** Add `.llmdiff` to your `.gitignore` — it contains your API key.

---

## Step 3 — Run your first comparison

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini
```

What the flags mean:

| Flag | Meaning |
|------|---------|
| `"Explain recursion in one sentence."` | The prompt sent to both models |
| `-a gpt-4o` | Model A — the first model to compare |
| `-b gpt-4o-mini` | Model B — the second model |

You should see output similar to this:

```
Model A (gpt-4o):
  Recursion is a programming technique where a function calls itself with a
  progressively simpler version of the problem until a base case is reached.

Model B (gpt-4o-mini):
  Recursion is when a function calls itself repeatedly to solve a problem by
  breaking it down into smaller subproblems until a base case is met.

Word diff:
  Recursion is [-a programming technique where a-] {+when a+} function calls
  itself [-with a progressively simpler version of the problem until a base
  case is reached.-] {+repeatedly to solve a problem by breaking it down into
  smaller subproblems until a base case is met.+}

  Word similarity:  54.2%
  Model A tokens:   32    latency: 0.91s
  Model B tokens:   29    latency: 0.57s
```

---

## Step 4 — Read the output

### The responses

The first two sections show the raw text from each model.  Read these to get a
feel for how different (or similar) the responses are before looking at the diff.

### The word diff

```
Recursion is [-a programming technique where a-] {+when a+} function calls itself
```

The diff highlights every change between the two responses:

| Notation | Meaning | Terminal colour |
|----------|---------|-----------------|
| `[-text-]` | Present in Model A, absent from Model B (deleted) | Red |
| `{+text+}` | Absent from Model A, new in Model B (inserted) | Green |
| `plain text` | Identical in both responses | White / default |

In this example, Model A says *"a programming technique where a function calls
itself with a progressively simpler version of the problem"* while Model B says
*"when a function calls itself repeatedly to solve a problem by breaking it down
into smaller subproblems"* — different phrasing, but both correct.

### Word similarity

```
Word similarity:  54.2%
```

This is the percentage of words shared between the two responses, weighted by
position.  A score of 54% means roughly half the words match.

**Important:** word similarity can be misleading.  Two responses can express the
same meaning in completely different words and score as low as 30–40%.  We will
add semantic scoring in [Tutorial 02](02-semantic-scoring.md) to address this.

A rough guide to word similarity:

| Score | What it usually means |
|-------|-----------------------|
| 90–100% | Nearly identical — possibly the same response with minor rewording |
| 70–89% | Very similar — same structure, some paraphrasing |
| 50–69% | Noticeably different — same topic, different phrasing |
| 30–49% | Substantially different — same answer, very different expression |
| 0–29% | Highly divergent — possibly different interpretations of the prompt |

### Token counts and latency

```
Model A tokens:   32    latency: 0.91s
Model B tokens:   29    latency: 0.57s
```

Token counts help you understand cost — longer responses cost more.  Latency
shows how long each API call took.  Smaller models (`gpt-4o-mini`) are typically
faster than larger ones (`gpt-4o`).

---

## Step 5 — Try a longer prompt

Single-sentence prompts produce short responses that are easy to diff but don't
show the full value of the tool.  Try something longer:

```bash
llm-diff "Write a 3-paragraph explanation of how neural networks learn." \
  -a gpt-4o -b gpt-4o-mini
```

Notice how a longer response produces a more complex diff with more red/green
sections.  The word similarity score will likely be lower — longer responses
have more opportunity to diverge in phrasing.

---

## Step 6 — Save an HTML report

Terminal output is useful for a quick check, but it disappears when you close
the window and can't be shared with a teammate.  Use `--out` to save a
self-contained HTML report:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini \
  --out my-first-report.html
```

Output:

```
  Word similarity:  54.2%
  Report saved -> my-first-report.html
```

Open it in your browser:

```bash
# macOS
open my-first-report.html

# Linux
xdg-open my-first-report.html

# Windows
start my-first-report.html
```

The report contains:
- Both full responses side-by-side
- The word diff with colour highlighting
- Similarity score and token/latency metadata
- Everything inline — no internet connection required, works offline

### Auto-save every run

If you want every run saved automatically without specifying `--out` each time,
use `--save`:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --save
# Report saved -> diffs/20260228_143022_gpt-4o_vs_gpt-4o-mini.html
```

Files are saved to `./diffs/` with a timestamp and model names in the filename.
Or set `save = true` in your `.llmdiff` to make this the default:

```toml
[defaults]
save = true
```

---

## Step 7 — See the cache in action

Run the exact same command a second time:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini
```

Notice the latency:

```
  Word similarity:  54.2%
  Model A tokens:   32    latency: 0.00s   (cached)
  Model B tokens:   29    latency: 0.00s   (cached)
```

Both responses are served instantly from the local cache at
`~/.cache/llm-diff/`.  The cache key is `(model, prompt, temperature, max_tokens)`.
As long as none of those change, you will never pay for the same API call twice.

This is especially useful when you are:
- Experimenting with `--fail-under` thresholds
- Tweaking report settings with `--out`
- Adding scoring flags like `--semantic` or `--bleu` to an existing comparison

To bypass the cache and force a fresh API call:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --no-cache
```

---

## Step 8 — Get verbose metadata

Use `-v` (`--verbose`) to see the full API metadata for each request:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --verbose
```

```
  [gpt-4o]
    provider:          openai
    prompt_tokens:     9
    completion_tokens: 32
    total_tokens:      41
    latency:           0.00ms   (cached)

  [gpt-4o-mini]
    provider:          openai
    prompt_tokens:     9
    completion_tokens: 29
    total_tokens:      38
    latency:           0.00ms   (cached)

  Word similarity:  54.2%
```

This is useful when debugging unexpected responses or tracking token usage.

---

## Step 9 — Get JSON output

Use `--json` to output the full result as a JSON object, suitable for piping
into other tools or saving to a file:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --json
```

```json
{
  "prompt_a": "Explain recursion in one sentence.",
  "prompt_b": "Explain recursion in one sentence.",
  "model_a": "gpt-4o",
  "model_b": "gpt-4o-mini",
  "response_a": "Recursion is a programming technique where...",
  "response_b": "Recursion is when a function calls itself...",
  "diff": [
    {"type": "equal",  "text": "Recursion is "},
    {"type": "delete", "text": "a programming technique where a"},
    {"type": "insert", "text": "when a"},
    ...
  ],
  "word_similarity": 0.542,
  "semantic_score": null,
  "tokens_a": {"prompt": 9, "completion": 32, "total": 41},
  "tokens_b": {"prompt": 9, "completion": 29, "total": 38},
  "latency_a_ms": 0,
  "latency_b_ms": 0
}
```

Extract a single value with `jq`:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --json \
  | jq '.word_similarity'
# 0.542
```

---

## Summary

You have now:

- ✅ Installed `llm-diff` and run your first comparison
- ✅ Read and interpreted the word diff notation (`[-deleted-]`, `{+inserted+}`)
- ✅ Understood what word similarity means and its limitations
- ✅ Saved a self-contained HTML report
- ✅ Used the response cache to re-run without API costs
- ✅ Seen verbose metadata and JSON output modes

---

## What's Next

Word similarity tells you how much the *words* changed, but not whether the
*meaning* changed.  In the next tutorial we add semantic scoring — a much more
reliable signal for whether two responses are equivalent.

[Tutorial 02 — Semantic Scoring →](02-semantic-scoring.md)
