# Tutorial 00 — What Is This All About?

**Time:** ~5 minutes  
**Level:** No prerequisites

---

## 1. The Problem

Large Language Models do not produce deterministic output.

Run the same prompt twice on `gpt-4o`, and you will get two different responses —
different words, different structure, sometimes different facts.  Now imagine you
need to answer one of these questions:

- **"Is `claude-3-7-sonnet-20250219` actually better than `gpt-4o` for our use case?"**
  You could run both models on twenty representative prompts.  But then what?
  Read forty responses and form a gut feeling?

- **"Our prompt engineer rewrote the system prompt.  Did it improve the output?"**
  The new prompt produces longer, more structured responses.  But are they *better*,
  or just *different*?

- **"Our model provider pushed a silent update last night.  Did anything break?"**
  You have no idea.  You would need to diff every prompt against a baseline you
  captured before the update.

- **"We are A/B testing two prompt variants in production.  How different are they?"**
  The responses look similar at a glance.  But are they semantically equivalent,
  or is one consistently missing key information?

Every one of these questions requires the same workflow:

1. Send prompt(s) to model A
2. Send the same prompt(s) to model B
3. Compare the responses — word by word, meaning by meaning
4. Record a score you can act on
5. Repeat for every prompt, every model pair, every iteration

Doing this by hand — copying responses into text editors, scrolling side-by-side,
guessing whether the meaning changed — takes hours and produces no repeatable,
shareable record.  It does not scale past a handful of prompts.  And it gives
you no way to enforce a quality bar in CI.

---

## 2. What Exists Today (and Why It Falls Short)

### Manual copy-paste comparison

The most common approach.  Copy response A into one tab, response B into another,
read them both.  Works for one prompt, one time.  Produces no score, no record,
no way to automate.

### General-purpose diff tools (`diff`, `git diff`, VS Code diff)

These tools compare text character-by-character or line-by-line.  They were built
for source code, where two logically identical programs have identical text.  LLM
responses break this assumption: two responses can mean exactly the same thing
while scoring 40% word similarity, because LLMs naturally paraphrase.

A general diff tool will tell you *what words changed*.  It will not tell you
whether the *meaning* changed.

### Full evaluation frameworks (LangSmith, PromptFoo, RAGAS, …)

These are powerful platforms for teams running large-scale LLM evaluation at
production volume.  They require accounts, servers, databases, and significant
setup.  They are the right tool when you are running thousands of evaluations per
day across complex pipelines.

They are overkill when you want to answer **"did this prompt change improve my
output?"** in thirty seconds from the command line.

### The gap

There was no lightweight, developer-focused tool that:
- runs from the terminal with a single command
- calls both models for you concurrently
- diffs the responses word-by-word *and* semantically
- produces a shareable HTML report with no server required
- scales to batch workloads without extra infrastructure
- integrates with CI/CD as a quality gate

That gap is what `llm-diff` was built to fill.

---

## 3. The Solution — llm-diff

`llm-diff` is a CLI tool and Python library that automates LLM output comparison
from end to end.

You run one command:

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --semantic
```

`llm-diff` then:

1. **Calls both models in parallel** — no waiting for one to finish before the other starts
2. **Diffs the responses word-by-word** — every insertion, deletion, and unchanged span is tagged
3. **Scores semantic similarity** — using sentence embeddings so you know if the *meaning* diverged, not just the words
4. **Renders the result** — colour-highlighted terminal diff or self-contained HTML report
5. **Caches the responses** — re-running to tweak thresholds or scoring options costs nothing
6. **Exits with code 1 if quality drops** — so a CI pipeline can enforce a minimum similarity bar

It works with OpenAI, Groq, Mistral, DeepSeek, Ollama, LM Studio, and any
OpenAI-compatible endpoint.  No account beyond an API key.  No server.  No database.

---

## 4. How llm-diff Helps — Feature by Feature

### Word-level diff

The core feature.  Every response is compared token-by-token using a
sequence-matching algorithm.  The output shows exactly which words were added,
removed, or unchanged — in the terminal with colour highlighting, or in an HTML
report as a side-by-side diff.

```
Recursion is [-a technique where a-] {+when a+} function calls itself
[-with a simpler version of the problem until a base case is reached.-]
{+repeatedly until a base condition is met, solving the problem step by step.+}

Word similarity:  61.3%
```

`[-deleted-]` text was in model A but not model B.  `{+inserted+}` text is new
in model B.

### Semantic similarity

Word overlap is a poor proxy for meaning.  Two responses can express the same
idea in completely different words and score 40% word similarity.  The `--semantic`
flag adds a cosine similarity score computed from sentence embeddings:

```
Word similarity:     61.3%
Semantic similarity: 92.7%   ← same meaning, different words
```

A semantic score above ~85% generally means the responses are equivalent in
meaning, even if the phrasing is very different.

### BLEU and ROUGE-L

Industry-standard NLP metrics used in research and production evaluation:

- **BLEU** — measures n-gram precision: how many phrases from B appear in A.
  Good for surface-level phrase overlap.
- **ROUGE-L** — measures the Longest Common Subsequence F1.  Captures
  structural similarity without requiring consecutive phrase matches.

Both are computed in pure Python with zero extra dependencies.

### Per-paragraph scoring

For long responses (summaries, essays, code reviews), a single similarity score
loses signal.  `--paragraph` breaks the comparison down paragraph-by-paragraph
so you can see exactly which section diverged:

```
Paragraph similarities:
  #1  Transformers rely on self-attention...    94.2%
  #2  The encoder processes input tokens...     81.3%
  #3  Training uses masked language...          58.1%  ← diverges here
```

### Batch evaluation

Instead of running one prompt at a time, define all your prompts in a YAML file
and evaluate them all in one command:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini --semantic --out report.html
```

Results are printed as a summary table and rolled into a single HTML report with
per-item diff cards.

### HTML reports

Every run can produce a fully self-contained HTML file — all CSS, JavaScript, and
diff data inline.  No CDN, no internet required.  Open in any browser, share by
email or Slack, view in an air-gapped environment.

### Response caching

API calls are cached keyed on `(model, prompt, temperature, max_tokens)`.
Re-running the same comparison is instant and costs nothing.  This is especially
useful when iterating on thresholds or report settings.

### CI/CD quality gate

The `--fail-under` flag turns `llm-diff` into a regression gate:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini --semantic --fail-under 0.85
# exit code 0 = all prompts passed
# exit code 1 = at least one fell below 85%
```

Add this to a GitHub Actions workflow and every model upgrade, prompt change, or
dependency bump is automatically validated against your quality bar.

### LLM-as-a-Judge

Use a third model to score which response is better — returning a winner (A / B /
tie), a 1–10 score for each model, and a reasoning paragraph:

```bash
llm-diff "Explain closures in JavaScript." \
  -a gpt-4o -b gpt-4o-mini --judge gpt-4o-mini
```

### Cost tracking

See the estimated USD cost for every API call, with built-in pricing data for
35+ models across OpenAI, Anthropic, Groq, Mistral, and DeepSeek:

```bash
llm-diff "Write a haiku about Python." -a gpt-4o -b gpt-4o-mini --show-cost
```

### Multi-model comparison

Compare 3 or 4 models in one shot, with all pairwise similarity scores computed
concurrently and ranked:

```bash
llm-diff "What is the capital of France?" \
  -a gpt-4o -b gpt-4o-mini --model-c mistral-large-latest --model-d claude-3-7-sonnet-20250219
```

### Structured JSON diff

When responses are JSON objects, compare them key-by-key instead of word-by-word:

```bash
llm-diff 'Return JSON: {"name": "...", "age": ...}' \
  -a gpt-4o -b gpt-4o-mini --mode json-struct
```

### Schema events & observability

Every comparison, model call, cache lookup, cost estimate, judge evaluation, and
`--fail-under` regression failure automatically emits a structured
[llm-toolkit-schema](https://pypi.org/project/llm-toolkit-schema/)
event.  Attach any exporter to ship events to JSONL, a database, or a custom
observability backend with one line of configuration:

```python
from llm_toolkit_schema.export.jsonl import JSONLExporter
from llm_diff.schema_events import configure_emitter

configure_emitter(exporter=JSONLExporter("events.jsonl"))
```

---

## 5. What You Will Learn in These Tutorials

| Tutorial | What you will be able to do |
|----------|----------------------------|
| 01 — First Comparison | Run your first diff, read the output, save an HTML report |
| 02 — Semantic Scoring | Add semantic, BLEU, ROUGE-L metrics and interpret them together |
| 03 — Prompt Engineering | Track the impact of prompt rewrites with prompt-diff mode |
| 04 — Batch Evaluation | Evaluate all your prompts in one command, generate a batch report |
| 05 — CI/CD Gate | Block model upgrades and prompt changes that cause regressions |
| 06 — LLM-as-a-Judge | Score responses with a third model, read winner + reasoning |
| 07 — Multi-Model | Compare 3–4 models concurrently with a pairwise ranking table |
| 08 — Cost Tracking | Estimate and track USD spend per evaluation run |
| 09 — JSON Struct Diff | Key-by-key diff for structured JSON responses |
| 10 — Python API | Integrate `llm-diff` into scripts, tests, and evaluation harnesses |
| 11 — Schema Events | Capture and export structured observability events |

---

## 6. Prerequisites

You need:

- **Python 3.9 or later**
- **An API key** for at least one provider (OpenAI is used in most examples)
- **`llm-diff` installed** — see below

```bash
pip install "llm-diff[semantic]"
```

The `[semantic]` extra installs `sentence-transformers`, which is required for
tutorials 02–10.  The default model (`all-MiniLM-L6-v2`, ~80 MB) downloads
automatically on first use.

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."       # macOS / Linux
$env:OPENAI_API_KEY = "sk-..."       # Windows PowerShell
```

---

## 7. How to Read These Tutorials

**Each tutorial is self-contained.**  You can read them in order for the full
learning path, or jump directly to the topic you need.

**All code blocks are copy-pasteable as written.**  Replace model names with any
models you have API access to — the behaviour is identical.

**Examples use `gpt-4o` and `gpt-4o-mini` throughout.**  If you are on Groq,
substitute `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.  If you are 
using local models via Ollama, substitute `llama3.2` and `mistral`.

**Expected output is shown for every command.**  Your actual responses will
differ (LLMs are non-deterministic), but the structure — scores, diff format,
table layout — will be identical.

---

## Ready?

Start with [Tutorial 01 — Your First Comparison →](01-first-comparison.md)
