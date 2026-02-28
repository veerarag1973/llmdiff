# Tutorial 07 — Multi-Model Comparison

**Time:** ~15 minutes  
**Level:** Advanced  
**Prerequisites:** [Tutorial 06](06-llm-as-a-judge.md) completed

← [06 — LLM-as-a-Judge](06-llm-as-a-judge.md) | [08 — Cost Tracking →](08-cost-tracking.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Compare three or four models in a single command using `--model-c` and `--model-d`
- Read the pairwise ranking table in the terminal output
- Navigate the multi-model HTML report
- Use `run_multi_model()` from the Python API for programmatic model selection
- Design a model selection workflow for a new project

---

## Why Compare More Than Two Models?

Two-model comparison is ideal for tracking regressions or comparing a candidate
against a baseline.  But when you are choosing a model for a new project, you
often want to evaluate three or four candidates at once:

- GPT-4o vs Claude 3.5 Sonnet vs Gemini 1.5 Pro vs Mistral Large
- gpt-4o-mini vs claude-3-5-haiku-20241022 vs gemini-2.0-flash (cost-efficiency candidates)

Running A vs B, then A vs C, then A vs D, then B vs C... is tedious and
produces no consolidated ranking.  Multi-model mode handles all pairs in one run.

---

## Step 1 — Three-model comparison

Add `--model-c` to extend beyond two models:

```bash
llm-diff "Explain how HTTPS works." \
  -a gpt-4o \
  -b claude-3-7-sonnet-20250219 \
  --model-c gemini-2.0-pro \
  --semantic
```

Output:

```
  Comparing 3 models on: "Explain how HTTPS works."

  ── Pairwise scores ────────────────────────────────────────────────

  ┌────────────────────────────┬──────────┬────────┐
  │ Pair                       │ Semantic │  Word  │
  ├────────────────────────────┼──────────┼────────┤
  │ gpt-4o vs claude-3-7-sonnet-20250219│   82.1%  │  44.3% │
  │ gpt-4o vs gemini-2.0-pro   │   87.4%  │  51.7% │
  │ claude-3-7-sonnet-20250219 vs gemini│   79.8%  │  38.2% │
  └────────────────────────────┴──────────┴────────┘

  ── Ranking (by mean pairwise semantic score) ──────────────────────

  1. gpt-4o              mean: 84.8%
  2. gemini-2.0-pro      mean: 83.6%
  3. claude-3-7-sonnet-20250219   mean: 81.0%
```

The ranking is determined by each model's mean pairwise semantic score —
the average of all comparisons that model participates in.

---

## Step 2 — Four-model comparison

Add `--model-d` for a full four-way comparison:

```bash
llm-diff "Write a function to flatten a nested list in Python." \
  -a gpt-4o \
  -b claude-3-7-sonnet-20250219 \
  --model-c gemini-2.0-pro \
  --model-d mistral-large-latest \
  --semantic \
  --judge gpt-4o \
  --out model-selection.html
```

With four models, `llm-diff` runs all 6 pairwise combinations:

```
  A vs B   A vs C   A vs D
           B vs C   B vs D
                    C vs D
```

The HTML report contains:

| Section | Contents |
|---------|----------|
| Ranking table | Models ranked by mean pairwise score |
| Judge wins table | Win/loss/tie count for each model |
| Pairwise matrix | Full 4×4 grid of scores |
| Per-pair diffs | Expandable cards with full word diff for each pair |

---

## Step 3 — Multi-model batch evaluation

Combine multi-model with batch mode to rank models across a diverse input set:

```yaml
model_a: gpt-4o
model_b: claude-3-7-sonnet-20250219
model_c: gemini-2.0-pro
semantic: true
judge: gpt-4o

cases:
  - prompt: "Explain blockchain in simple terms."
  - prompt: "Write a regex to validate an email address."
  - prompt: "What are the SOLID principles?"
  - prompt: "Explain eventual consistency."
  - prompt: "What is a deadlock and how do you prevent it?"
```

```bash
llm-diff batch eval-models.yml --out model-selection-batch.html
```

The batch report adds aggregate win counts across all cases, giving you a
statistically stronger basis for model selection.

---

## Step 4 — Python API: `run_multi_model()`

For programmatic model selection — for example, inside a test harness or a
selection script — use `run_multi_model()`:

```python
import asyncio
from llm_diff import run_multi_model

async def select_model():
    results = await run_multi_model(
        prompt="Explain the difference between a thread and a process.",
        models=["gpt-4o", "claude-3-7-sonnet-20250219", "gemini-2.0-pro"],
        semantic=True,
        judge="gpt-4o",
    )

    print("Ranking:")
    for rank, (model, score) in enumerate(results.ranking, start=1):
        print(f"  {rank}. {model:30s}  mean semantic: {score:.1%}")

    print(f"\nRecommended model: {results.ranking[0][0]}")

asyncio.run(select_model())
```

Output:

```
Ranking:
  1. gpt-4o              mean semantic: 84.8%
  2. gemini-2.0-pro      mean semantic: 83.6%
  3. claude-3-7-sonnet-20250219   mean semantic: 81.0%

Recommended model: gpt-4o
```

The `results` object also exposes `results.pairwise` (a dict of all pair
comparisons) and `results.judge_wins` (win counts per model).

---

## Step 5 — A practical model selection workflow

Here is a recommended workflow when choosing a model for a new project:

### Phase 1 — Shortlist (3–4 models, 5 representative prompts)

```bash
llm-diff batch shortlist.yml \
  --model-c gemini-2.0-pro \
  --semantic \
  --out shortlist.html
```

Pick the top 2 models based on semantic score and judge wins.

### Phase 2 — Deep evaluation (2 models, 20+ representative prompts)

```bash
llm-diff batch deep-eval.yml \
  --semantic --bleu --rouge \
  --judge gpt-4o \
  --out deep-eval.html
```

Review per-case breakdowns in the HTML report.  Check which model wins on the
cases most representative of your production load.

### Phase 3 — Cost validation (Tutorial 08)

Once you have a quality winner, check whether a smaller / cheaper model meets
a relaxed threshold (e.g. 85% of the winner's score at 20% of the cost).

---

## Summary

You have now:

- ✅ Used `--model-c` and `--model-d` to compare three and four models at once
- ✅ Read the pairwise ranking table and understood mean pairwise scoring
- ✅ Explored the multi-model HTML report (ranking, judge wins, pairwise matrix)
- ✅ Combined multi-model with batch mode for aggregate ranking
- ✅ Used `run_multi_model()` from the Python API
- ✅ Followed a three-phase model selection workflow

---

## What's Next

You now know how to choose the best model for quality.  The next tutorial
shows you how to factor in cost — understanding what each run costs and how
to optimise for quality per dollar.

[Tutorial 08 — Cost Tracking →](08-cost-tracking.md)
