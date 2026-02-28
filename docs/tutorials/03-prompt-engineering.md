# Tutorial 03 — Prompt Engineering Workflow

**Time:** ~15 minutes  
**Level:** Intermediate  
**Prerequisites:** [Tutorial 02](02-semantic-scoring.md) completed

← [02 — Semantic Scoring](02-semantic-scoring.md) | [04 — Batch Evaluation →](04-batch-evaluation.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Use prompt-diff mode to compare two prompts on the same model
- Track the impact of prompt rewrites with objective scores
- Load prompts from files instead of typing them inline
- Build an iterative prompt engineering workflow using the response cache
- Recognise when a prompt change improves, degrades, or merely rephrases output

---

## The Prompt Engineering Problem

When you are iterating on a prompt, you face a measurement problem.

You rewrite a system prompt.  The new responses look better to you.  But:

- Are they *actually* better, or just different?
- Did you improve one aspect and unknowingly break another?
- How do you communicate the improvement to a teammate objectively?
- If you make five more iterations, will you remember which version was best?

Without a scoring workflow, prompt engineering is subjective and undocumented.
`llm-diff` solves this by giving every prompt change a reproducible, comparable
score.

---

## Prompt-Diff Mode vs Model-Diff Mode

So far in these tutorials you have used **model-diff mode**: the same prompt
sent to two different models.

```bash
# model-diff mode — same prompt, two models
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini
```

**Prompt-diff mode** is the inverse: two different prompts sent to the same
model.  This is the primary mode for prompt engineering.

```bash
# prompt-diff mode — two prompts, same model
llm-diff --model gpt-4o \
  --prompt-a "Explain recursion." \
  --prompt-b "Explain recursion. Use a real-world analogy. Keep it under 2 sentences."
```

Use `--model` (instead of `-a` / `-b`) to activate prompt-diff mode.

---

## Step 1 — Your first prompt comparison

Suppose you have a baseline prompt and want to see if adding instructions
improves specificity:

```bash
llm-diff --model gpt-4o \
  --prompt-a "Summarise the concept of technical debt." \
  --prompt-b "Summarise the concept of technical debt in 2 sentences. Focus on long-term consequences for engineering teams." \
  --semantic
```

Output:

```
  Comparing prompts on gpt-4o

  Prompt A: Summarise the concept of technical debt.
  Prompt B: Summarise the concept of technical debt in 2 sentences. Focus on
            long-term consequences for engineering teams.

  Word diff:
    Technical debt [-refers to the implied cost of rework caused by choosing-]
    {+is the accumulated cost that engineering teams pay when they choose+}
    [-an easy solution now instead of a better approach that would take longer.-]
    {+speed over code quality, leading to slower development, higher bug rates,
    and increased risk of system failure over time.+}

  Word similarity:     31.2%
  Semantic similarity: 78.4%
  Primary score:       78.4%
```

A semantic score of 78.4% means the responses overlap in topic but differ
meaningfully in content — the second prompt produced a more focused, consequence-
oriented answer.  Whether that is an improvement depends on your use case.

---

## Step 2 — Load prompts from files

Inline prompts work for short experiments, but real prompts are often long —
system prompts with personas, constraints, formatting instructions, and examples.
Use `--prompt-a` and `--prompt-b` with file paths:

```bash
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic --out prompt-diff.html
```

Create the files:

**`prompts/system_v1.txt`**
```
You are a helpful assistant. Answer the user's question clearly and concisely.
```

**`prompts/system_v2.txt`**
```
You are a senior software engineer with 15 years of experience. Answer the
user's question clearly and concisely. Lead with the most important point.
Use bullet points for multi-step answers. Avoid unnecessary jargon.
```

Run the comparison:

```bash
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic --paragraph --out prompt-diff.html
```

The `--paragraph` flag is especially useful here — it reveals whether specific
sections of the response changed or stayed the same.

> **Tip:** File paths are resolved relative to your current working directory.

---

## Step 3 — Build an iterative scoring workflow

The real power of prompt-diff mode comes from iterating and tracking scores
across versions.  Here is a practical workflow:

### The loop

```
v1 → run → baseline score
v2 → run → score vs v1
v3 → run → score vs v2
     └─ cache means re-running v1 or v2 later is free
```

### Example: iterating on a code review prompt

**Iteration 1 — baseline:**

```bash
llm-diff --model gpt-4o \
  --prompt-a "Review this Python function for quality." \
  --prompt-b "Review this Python function for quality. Focus on: readability, error handling, and performance. Format your review with one section per focus area." \
  --semantic --out iter1.html
```

```
  Word similarity:     28.6%
  Semantic similarity: 71.3%   ← large change, check if it's better
```

A 71% semantic score means the two prompts produced substantially different
responses.  Open `iter1.html` and read both — is v2 actually better structured?

**Iteration 2 — refine:**

```bash
llm-diff --model gpt-4o \
  --prompt-a "Review this Python function for quality. Focus on: readability, error handling, and performance. Format your review with one section per focus area." \
  --prompt-b "Review this Python function for quality. Structure your review in three sections: 1) Readability, 2) Error Handling, 3) Performance. Start each section with a one-sentence verdict (Good / Needs Work / Critical), then explain." \
  --semantic --out iter2.html
```

```
  Word similarity:     41.7%
  Semantic similarity: 88.2%   ← refinement, same intent, better structure
```

An 88% semantic score with a v2 that adds structural formatting means the core
content stayed consistent — you improved *how* it's expressed without changing
*what* it covers.  This is a healthy prompt engineering pattern.

### Score patterns to look for

| Pattern | Semantic score | What it means |
|---------|---------------|---------------|
| Large change | < 70% | Responses diverge in content — verify both and decide intentionally |
| Refinement | 80–92% | Same core answer, better structure/format — usually an improvement |
| Minor polish | 93–99% | Cosmetic changes — may not be worth the prompt complexity |
| No change | ~100% | The prompt change had no meaningful effect |

---

## Step 4 — Use the cache strategically

Every response is cached on `(model, prompt, temperature, max_tokens)`.  This
means you can add scoring flags to an already-run comparison at zero cost:

```bash
# First run — calls the API
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic

# Add BLEU and ROUGE for free — responses already cached
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic --bleu --rouge

# Save an HTML report for free
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic --paragraph --out v1-vs-v2.html
```

All three commands above call the API zero times after the first run.

You can also cheaply re-test an old version against a new one at any point in
your iteration cycle:

```bash
# Compare v1 directly against v4 — v1 response is still in cache
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v4.txt \
  --semantic
```

---

## Step 5 — Combine prompt-diff with `--fail-under`

Once you have established a baseline score, use `--fail-under` to prevent
accidental regressions during further iteration:

```bash
llm-diff --model gpt-4o \
  --prompt-a prompts/system_approved.txt \
  --prompt-b prompts/system_candidate.txt \
  --semantic --fail-under 0.80
```

If the candidate prompt produces responses that are less than 80% semantically
similar to the approved baseline, the command exits with code 1:

```
  Semantic similarity: 72.1%   threshold: 80.0%   FAIL
Error: similarity 0.721 < threshold 0.800
```

This is useful in code review — you can add a CI step that validates any prompt
file changes against the approved version before merging.

---

## Step 6 — A complete prompt versioning example

Here is a realistic end-to-end workflow for a team managing a customer support
chatbot prompt:

### Project structure

```
prompts/
├── system_v1.0.txt     ← original release
├── system_v1.1.txt     ← first iteration
├── system_v1.2.txt     ← second iteration (candidate)
└── system_approved.txt ← symlink / copy of current approved version
diffs/                  ← auto-saved HTML reports (--save)
```

### Compare each iteration against the previous

```bash
# v1.0 → v1.1
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.0.txt \
  --prompt-b prompts/system_v1.1.txt \
  --semantic --paragraph --save

# v1.1 → v1.2
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.1.txt \
  --prompt-b prompts/system_v1.2.txt \
  --semantic --paragraph --save
```

### Gate the candidate against the approved baseline

```bash
llm-diff --model gpt-4o \
  --prompt-a prompts/system_approved.txt \
  --prompt-b prompts/system_v1.2.txt \
  --semantic --fail-under 0.82
```

### View the history

The `./diffs/` directory now contains timestamped HTML reports for every
comparison, giving you a full audit trail of how your prompt evolved.

---

## Summary

You have now:

- ✅ Understood the difference between model-diff and prompt-diff mode
- ✅ Used `--model` + `--prompt-a` / `--prompt-b` for prompt comparisons
- ✅ Loaded prompts from files for realistic multi-line system prompts
- ✅ Built an iterative scoring loop: edit → compare → score → repeat
- ✅ Used the response cache to add metrics and reports without extra API calls
- ✅ Gated prompt changes with `--fail-under` to prevent regressions
- ✅ Set up a prompt versioning folder structure with an audit trail

---

## What's Next

You now know how to compare two models and two prompts.  The next step is to
stop running one prompt at a time and evaluate your entire prompt suite in a
single command — batch mode.

[Tutorial 04 — Batch Evaluation →](04-batch-evaluation.md)
