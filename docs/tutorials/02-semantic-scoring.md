# Tutorial 02 — Semantic Scoring

**Time:** ~10 minutes  
**Level:** Beginner  
**Prerequisites:** [Tutorial 01](01-first-comparison.md) completed, `llm-diff[semantic]` installed

← [01 — Your First Comparison](01-first-comparison.md) | [03 — Prompt Engineering →](03-prompt-engineering.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Explain why word similarity alone is unreliable for LLM comparison
- Add semantic similarity scoring with `--semantic`
- Add per-paragraph similarity breakdown with `--paragraph`
- Add BLEU and ROUGE-L scores with `--bleu` and `--rouge`
- Combine all four metrics and know when to use each one
- Choose the right primary metric for your use case

---

## The Problem with Word Similarity

In Tutorial 01 you saw that word similarity measures how many words the two
responses share.  It is fast and requires no extra dependencies — but it has a
fundamental flaw: **it measures surface form, not meaning**.

Consider these two responses to *"What is a for loop?"*:

**Model A:**
> A for loop is a control flow construct that iterates over a sequence,
> executing a block of code for each element.

**Model B:**
> For loops allow you to repeat a block of statements for every item in a
> collection, making it easy to process lists or ranges of numbers.

These responses mean exactly the same thing.  Yet they share very few words:
*"a"*, *"for"*, *"loop"*, *"block"*, *"of"* — a word similarity of around 28%.

Now consider two responses that share many words but mean different things:

**Model A:**
> The function returns `True` if the value is found.

**Model B:**
> The function returns `False` if the value is found.

Word similarity: ~90%.  Meaning: opposite.

This is why llm-diff provides semantic scoring — a measure of *meaning*
similarity that is robust to paraphrasing and synonyms.

---

## How Semantic Scoring Works

Semantic scoring uses **sentence embeddings** — a technique from NLP where text
is converted into a vector of numbers that captures its meaning.  Two pieces of
text that mean the same thing will have vectors that point in roughly the same
direction, regardless of the words used.

`llm-diff` uses the `sentence-transformers` library with the `all-MiniLM-L6-v2`
model (a small, fast, general-purpose embedding model, ~80 MB).  The score is
the **cosine similarity** between the two embedding vectors, expressed as a
percentage.

You do not need to understand the math.  What matters is:

- A score near **100%** means the responses are semantically identical
- A score above **~85%** generally means the responses convey the same meaning
- A score below **~70%** suggests the responses diverge in meaning
- The model downloads automatically the first time `--semantic` is used

---

## Step 1 — Add `--semantic` to a comparison

```bash
llm-diff "What is a for loop?" -a gpt-4o -b gpt-4o-mini --semantic
```

Output:

```
Model A (gpt-4o):
  A for loop is a control flow construct that iterates over a sequence,
  executing a block of code for each element until the sequence is exhausted.

Model B (gpt-4o-mini):
  For loops allow you to repeat a block of statements for every item in a
  collection, making it easy to process lists or ranges of numbers.

Word diff:
  [-A for loop is a control flow construct that iterates over a sequence,
  executing a block of code for each element until the sequence is exhausted.-]
  {+For loops allow you to repeat a block of statements for every item in a
  collection, making it easy to process lists or ranges of numbers.+}

  Word similarity:     28.1%
  Semantic similarity: 91.4%   ← same meaning, very different words
  Primary score:       91.4%
```

The word diff is almost entirely red and green — very few shared words.  Yet the
semantic score is 91.4%, correctly identifying that both responses convey the
same concept.

### Primary score

When `--semantic` is used, `llm-diff` promotes semantic similarity as the
**primary score** — the main number used for `--fail-under` thresholds, batch
summaries, and HTML report headers.  Word similarity is still shown as a
secondary metric.

---

## Step 2 — See semantic scoring catch a meaningful difference

Now try a prompt where the responses actually diverge in meaning:

```bash
llm-diff "Should I use a list or a tuple in Python for a fixed collection of items?" \
  -a gpt-4o -b gpt-4o-mini --semantic
```

You may see output like:

```
  Word similarity:     62.3%
  Semantic similarity: 74.1%
  Primary score:       74.1%
```

A semantic score of 74% indicates the responses overlap in topic but diverge in
recommendation or emphasis.  Read both responses — one may recommend tuples
strongly while the other hedges or gives equal weight to both.  The semantic
score picks this up where word similarity would not.

---

## Step 3 — Per-paragraph scoring with `--paragraph`

For long responses — summaries, essays, technical explanations — a single score
can mask which *part* of the response diverged.  Use `--paragraph` to break the
comparison down paragraph-by-paragraph:

```bash
llm-diff "Write a 3-paragraph explanation of how neural networks learn." \
  -a gpt-4o -b gpt-4o-mini --paragraph
```

Output:

```
  Word similarity:     51.3%
  Semantic similarity: 86.2%

  Paragraph similarities:
    #1  Neural networks learn by adjusting weights...    93.4%
    #2  During backpropagation, the gradient of...       88.1%
    #3  Over many training iterations, the network...    62.7%  ← diverges here
```

> **Note:** `--paragraph` implies `--semantic` — you don't need both flags.

The overall semantic score (86.2%) looked healthy, but the per-paragraph
breakdown reveals that paragraph 3 diverged significantly.  Without
`--paragraph` you would not know which part to investigate.

### When to use `--paragraph`

- Responses longer than ~3 sentences
- Summaries, reports, or structured explanations
- Anytime you need to pinpoint *where* a response diverges, not just *whether* it does
- Multi-section technical answers where individual sections may behave differently

---

## Step 4 — BLEU score with `--bleu`

BLEU (Bilingual Evaluation Understudy) is an industry-standard metric from
machine translation research.  It measures **n-gram precision**: how many
short phrases (1–4 words) from response B appear in response A, with a brevity
penalty for very short responses.

```bash
llm-diff "Explain recursion in one sentence." \
  -a gpt-4o -b gpt-4o-mini --bleu
```

Output:

```
  Word similarity:  54.2%
  BLEU:             38.6%
```

BLEU ranges from 0% to 100%:

| BLEU score | What it usually means |
|------------|-----------------------|
| 80–100% | Near-identical phrasing |
| 50–79% | Significant phrase overlap |
| 20–49% | Moderate phrase overlap — similar topic, different phrasing |
| 0–19% | Little phrase overlap — responses may still be semantically equivalent |

**When BLEU is useful:**
- When you specifically care about phrase-level reuse (e.g. regulatory text,
  legal summaries, code documentation where exact phrasing matters)
- As a secondary confirmation alongside semantic scoring
- When comparing outputs that should follow a template closely

**When BLEU is misleading:**
- When paraphrasing is acceptable or expected
- For creative writing or open-ended generation
- As a standalone metric for meaning equivalence

---

## Step 5 — ROUGE-L score with `--rouge`

ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation — Longest Common
Subsequence) measures the **longest common subsequence** F1 between the two
responses.  Unlike BLEU, it does not require consecutive phrase matches — it
rewards responses that share the same sequence of key words even with other
words interspersed.

```bash
llm-diff "Explain recursion in one sentence." \
  -a gpt-4o -b gpt-4o-mini --rouge
```

Output:

```
  Word similarity:  54.2%
  ROUGE-L:          61.8%
```

**When ROUGE-L is useful:**
- Summarisation tasks — checks if the key points appear in the same order
- When you want a metric that is more lenient than BLEU about exact phrase matches
- Long-form content where overall structure matters

---

## Step 6 — Combine all metrics

Use all four metrics together for the most complete picture:

```bash
llm-diff "Explain recursion in one sentence." \
  -a gpt-4o -b gpt-4o-mini \
  --semantic --bleu --rouge
```

Output:

```
  Word similarity:     54.2%
  Semantic similarity: 89.3%
  BLEU:                38.6%
  ROUGE-L:             61.8%
  Primary score:       89.3%
```

### Reading all four together

| Metric | Tells you |
|--------|-----------|
| Word similarity | How many words overlap (surface form) |
| Semantic similarity | Whether the *meaning* is the same |
| BLEU | Whether the same *phrases* appear |
| ROUGE-L | Whether the same *sequence of key ideas* appears |

**Healthy pattern** (same meaning, different words):
- Word similarity: 40–60% ↓
- Semantic similarity: 85–95% ↑
- BLEU: 25–45% (moderate)
- ROUGE-L: 50–70% (moderate)

**Concerning pattern** (different meaning):
- Word similarity: 40–60% (misleadingly normal)
- Semantic similarity: 60–70% ↓
- BLEU: 15–30% ↓
- ROUGE-L: 30–50% ↓

**Identical or near-identical responses:**
- All four metrics: 85–100% ↑

---

## Step 7 — Save a report with all metrics

```bash
llm-diff "Write a 3-paragraph explanation of how neural networks learn." \
  -a gpt-4o -b gpt-4o-mini \
  --paragraph --bleu --rouge \
  --out neural-networks-report.html
```

Open `neural-networks-report.html`.  The report includes:
- All four metric scores in the header panel
- The word diff
- A per-paragraph similarity table with a score for each paragraph

---

## Step 8 — Choose your primary metric

For `--fail-under` gates and batch summary averages, `llm-diff` uses a single
**primary score**.  Here is how to choose:

| Use case | Recommended primary metric |
|----------|---------------------------|
| General model evaluation | `--semantic` |
| Prompt engineering iteration | `--semantic` |
| CI regression gate | `--semantic` |
| Template / exact phrasing compliance | `--bleu` |
| Summarisation quality | `--rouge` |
| Quick check, no extra deps | word similarity (default) |

When `--semantic` is used it always becomes the primary score.  If only `--bleu`
or `--rouge` are used (without `--semantic`), word similarity remains primary.

---

## Summary

You have now:

- ✅ Understood why word similarity is unreliable for measuring LLM output equivalence
- ✅ Added semantic scoring with `--semantic` and interpreted cosine similarity
- ✅ Used `--paragraph` to pinpoint which section of a long response diverged
- ✅ Added BLEU with `--bleu` and understood when phrase matching matters
- ✅ Added ROUGE-L with `--rouge` and understood LCS-based scoring
- ✅ Combined all four metrics and learned to read them together
- ✅ Chosen the right primary metric for different use cases

---

## What's Next

Now that you can measure how different two responses are, the natural next step
is to use that measurement to guide **prompt engineering** — iterating on a
prompt and tracking whether your changes actually improve the output.

[Tutorial 03 — Prompt Engineering Workflow →](03-prompt-engineering.md)
