# Tutorial 06 — LLM-as-a-Judge

**Time:** ~15 minutes  
**Level:** Advanced  
**Prerequisites:** [Tutorial 05](05-ci-cd-regression-gate.md) completed

← [05 — CI/CD Regression Gate](05-ci-cd-regression-gate.md) | [07 — Multi-Model Comparison →](07-multi-model-comparison.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Understand what LLM-as-a-Judge means and when similarity scores are not enough
- Use `--judge` to get a qualitative verdict on two responses
- Read the winner, score, and reasoning output
- Locate the judge card in HTML reports
- Understand the limitations and biases of judge mode

---

## The Limits of Similarity Scores

Semantic similarity answers one question: *how similar are these two
responses?*

It does not answer: *which response is better?*

Consider two responses to the prompt `"Explain the CAP theorem"`:

- **Response A (gpt-4o):** Correct definition, clear prose, a well-chosen
  example, mentions the trade-offs.
- **Response B (gpt-4o-mini):** Correct definition, shorter prose, no example,
  omits nuance about network partitions.

Semantic similarity might return 84% — they cover the same ground.  But A is
objectively more useful.  A similarity score cannot tell you that.

**LLM-as-a-Judge** solves this by asking a language model — typically a more
capable one — to read both responses and declare a winner with reasoning.

---

## Step 1 — Your first judge run

Add `--judge MODEL` to any comparison:

```bash
llm-diff "Explain the CAP theorem." \
  -a gpt-4o \
  -b gpt-4o-mini \
  --semantic \
  --judge gpt-4o
```

Output:

```
  Comparing gpt-4o vs gpt-4o-mini

  Word similarity:     41.2%
  Semantic similarity: 84.7%

  ── Judge (gpt-4o) ─────────────────────────────────────────────────

  Winner: Model A (gpt-4o)
  Score:  A=8/10   B=6/10

  Reasoning:
    Model A provides a more complete explanation of the CAP theorem.
    It correctly identifies the three properties, uses a concrete database
    example to illustrate partition tolerance trade-offs, and explains
    why CA systems cannot exist in distributed environments. Model B's
    response is accurate but omits the partition tolerance nuance and
    provides no illustrative example, making it less useful for a reader
    new to the concept.

  ───────────────────────────────────────────────────────────────────
```

The judge provides:

| Field | Description |
|-------|-------------|
| **Winner** | Which model won (A, B, or Tie) |
| **Score** | Absolute quality score for each response (0–10) |
| **Reasoning** | Natural language explanation of the verdict |

---

## Step 2 — Use a different model as judge

You can use any model you have API access to as the judge.  Using a more
capable model than your test subjects often produces better reasoning:

```bash
# GPT-4o judges a GPT-4o-mini vs Claude comparison
llm-diff "Write a Python function to parse an ISO 8601 date." \
  -a gpt-4o-mini \
  -b claude-3-5-haiku-20241022 \
  --judge gpt-4o
```

```bash
# Claude judges a GPT-4o vs Gemini comparison (requires both API keys)
llm-diff "Summarise this article." \
  -a gpt-4o \
  -b gemini-2.0-pro \
  --judge claude-3-7-sonnet-20250219
```

> **Tip:** The judge model does not need to be the same provider as the models
> being compared.  You can use GPT-4o to judge a Mistral vs Llama comparison.

---

## Step 3 — Judge mode in batch

Add `judge` to `eval.yml` to run judge evaluation across all cases:

```yaml
model_a: gpt-4o
model_b: gpt-4o-mini
semantic: true
judge: gpt-4o

cases:
  - prompt: "Explain the CAP theorem."
  - prompt: "What is a race condition?"
  - prompt: "Explain garbage collection in Java."
```

```bash
llm-diff batch eval.yml --out judge-batch.html
```

The batch summary table gains a **Winner** column:

```
  ┌───┬─────────────────────────────────┬──────────┬────────┬────────┐
  │ # │ Prompt (truncated)              │ Semantic │ Winner │ Status │
  ├───┼─────────────────────────────────┼──────────┼────────┼────────┤
  │ 1 │ Explain the CAP theorem.        │   84.7%  │   A    │  PASS  │
  │ 2 │ What is a race condition?       │   91.3%  │  Tie   │  PASS  │
  │ 3 │ Explain garbage collection...   │   79.2%  │   B    │  PASS  │
  └───┴─────────────────────────────────┴──────────┴────────┴────────┘

  Model A wins: 1   Model B wins: 1   Ties: 1
```

---

## Step 4 — The judge card in HTML reports

Every HTML report with judge mode enabled contains a **judge card** for each
comparison.  It shows:

- A coloured banner (green = A wins, blue = B wins, grey = tie)
- The judge model name
- Both quality scores
- The full reasoning text

Open `judge-batch.html` to explore the judge cards.

---

## Step 5 — Use judge mode to choose the better prompt

Judge mode is particularly powerful for prompt-diff comparisons, where you want
to know not just *how different* two prompts' outputs are, but *which prompt
produces the better response*:

```bash
llm-diff --model gpt-4o \
  --prompt-a prompts/system_v1.txt \
  --prompt-b prompts/system_v2.txt \
  --semantic \
  --judge gpt-4o \
  --out judge-prompt-diff.html
```

Combine this with batch mode to validate a prompt rewrite across a diverse
input set, getting both a quantitative score (semantic similarity) and a
qualitative verdict (judge winner) for each case.

---

## Step 6 — Understanding judge limitations

LLM-as-a-Judge is powerful but not infallible.  Be aware of these limitations:

### Position bias
Some judge models prefer the response that appears first (Model A) regardless
of quality.  `llm-diff` mitigates this by running the judge twice in swapped
order and averaging the scores — but it does not eliminate bias entirely.

### Self-preference bias
OpenAI models may slightly favour GPT-4o responses when acting as judge.
Anthropic models may slightly favour Claude responses.  For the most objective
results, use a judge model from a different provider than your test subjects.

### Domain expertise
Judge models score based on their own knowledge.  They may not be reliable
judges for highly specialised domains (medical, legal, niche engineering) where
they lack deep domain knowledge.

### Cost
Judge mode makes an additional API call to the judge model for every
comparison.  A 20-case batch with judge mode enabled makes roughly 60 API
calls instead of 40.

### When to use judge mode

| Scenario | Use judge? |
|----------|-----------|
| Automated CI gate | No — use `--fail-under` instead |
| Choosing between two candidate prompts | Yes |
| Model selection for a new project | Yes |
| Debugging a specific failing case | Yes |
| High-volume nightly regression | No — cost is too high |

---

## Summary

You have now:

- ✅ Understood why similarity scores cannot determine which response is better
- ✅ Used `--judge MODEL` to get winner / score / reasoning output
- ✅ Run judge mode in batch and read the winner summary table
- ✅ Explored the judge card in HTML reports
- ✅ Used judge mode for prompt-diff comparisons
- ✅ Understood position bias, self-preference bias, and cost trade-offs

---

## What's Next

You have been comparing two models throughout these tutorials.  The next
tutorial shows you how to compare three or four models at once and get a
pairwise ranking table.

[Tutorial 07 — Multi-Model Comparison →](07-multi-model-comparison.md)
