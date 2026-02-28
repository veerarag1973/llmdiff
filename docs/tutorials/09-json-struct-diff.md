# Tutorial 09 — JSON Struct Diff

**Time:** ~10 minutes  
**Level:** Advanced  
**Prerequisites:** [Tutorial 08](08-cost-tracking.md) completed

← [08 — Cost Tracking](08-cost-tracking.md) | [10 — Python API →](10-python-api.md)

---

## What You Will Learn

By the end of this tutorial you will be able to:

- Understand when JSON struct diff is the right tool
- Use `--mode json-struct` to compare structured JSON responses
- Read the per-field change labels: ADDED, REMOVED, CHANGED, TYPE_CHANGE, UNCHANGED
- Understand the fallback behaviour on invalid JSON
- Use `json_struct_diff()` from the Python API

---

## The Problem with Text Diff for JSON

Suppose you ask two models to extract structured data from an article:

**Prompt:**
```
Extract the following fields from this article as JSON:
title, author, publication_date, word_count, topics (array)
```

**Response A (gpt-4o):**
```json
{
  "title": "The Future of AI in Healthcare",
  "author": "Jane Smith",
  "publication_date": "2025-11-14",
  "word_count": 1420,
  "topics": ["AI", "Healthcare", "Machine Learning"]
}
```

**Response B (gpt-4o-mini):**
```json
{
  "title": "The Future of AI in Healthcare",
  "author": "Jane Smith",
  "publication_date": "November 14, 2025",
  "word_count": "1420",
  "topics": ["AI", "health", "ML"]
}
```

A text diff would highlight that these are 61% similar.  But that tells you
nothing useful.  What you actually want to know is:

- `publication_date`: format differs (`2025-11-14` vs `November 14, 2025`)
- `word_count`: TYPE_CHANGE (integer vs string)
- `topics[1]`: CHANGED (`Healthcare` vs `health`)
- `topics[2]`: CHANGED (`Machine Learning` vs `ML`)

JSON struct diff gives you exactly this field-by-field analysis.

---

## Step 1 — Your first JSON struct diff

```bash
llm-diff "Extract the structured data from this article: [article text]" \
  -a gpt-4o \
  -b gpt-4o-mini \
  --mode json-struct
```

Output:

```
  Comparing gpt-4o vs gpt-4o-mini (JSON struct mode)

  ── Field changes ──────────────────────────────────────────────────

  title              UNCHANGED
  author             UNCHANGED
  publication_date   CHANGED     "2025-11-14"  →  "November 14, 2025"
  word_count         TYPE_CHANGE  1420 (int)   →  "1420" (str)
  topics[0]          UNCHANGED
  topics[1]          CHANGED     "Healthcare"  →  "health"
  topics[2]          CHANGED     "Machine Learning"  →  "ML"

  ── Summary ────────────────────────────────────────────────────────

  UNCHANGED:    2 fields  (28.6%)
  CHANGED:      3 fields  (42.9%)
  TYPE_CHANGE:  1 field   (14.3%)
  ADDED:        0 fields
  REMOVED:      0 fields

  Struct similarity: 28.6%
```

The struct similarity score is the percentage of fields that are UNCHANGED.

---

## Step 2 — All change labels explained

| Label | Meaning |
|-------|---------|
| `UNCHANGED` | Field exists in both responses with the same value and type |
| `CHANGED` | Field exists in both responses but the value differs |
| `TYPE_CHANGE` | Field exists in both but the type changed (e.g. int → str, str → array) |
| `ADDED` | Field exists in Response B but not in Response A |
| `REMOVED` | Field exists in Response A but not in Response B |

---

## Step 3 — Schema completeness comparison

JSON struct diff is also useful for checking schema completeness — whether one
model returns all requested fields:

**Prompt:**
```
Return a JSON object with these fields:
id, name, email, phone, address, city, country, postal_code
```

**Response A:** All 8 fields present  
**Response B:** Missing `phone` and `postal_code`

```
  id            UNCHANGED
  name          UNCHANGED
  email         UNCHANGED
  phone         REMOVED
  address       UNCHANGED
  city          UNCHANGED
  country       UNCHANGED
  postal_code   REMOVED

  UNCHANGED:  6 fields  (75.0%)
  REMOVED:    2 fields  (25.0%)

  Struct similarity: 75.0%
```

`--fail-under` works in JSON struct mode too:

```bash
llm-diff "Generate the user profile JSON." \
  -a gpt-4o \
  -b gpt-4o-mini \
  --mode json-struct \
  --fail-under 0.90
```

---

## Step 4 — Combine with semantic scoring

JSON struct mode and semantic scoring are complementary:

```bash
llm-diff "Extract article metadata as JSON." \
  -a gpt-4o \
  -b gpt-4o-mini \
  --mode json-struct \
  --semantic
```

The output shows both:

```
  Struct similarity:   28.6%   ← structural field-by-field match
  Semantic similarity: 74.3%   ← meaning-level match across the full response
```

A high semantic + low struct score typically means the models extracted the
same information but formatted it differently.  A low semantic + low struct
score means they extracted different information.

---

## Step 5 — Fallback on invalid JSON

If one or both models return invalid JSON, `llm-diff` automatically falls back
to text diff mode and adds a warning:

```
  ⚠  Model B response is not valid JSON. Falling back to text diff mode.

  Word similarity:     52.3%
  Semantic similarity: 79.1%
```

This means your batch runs will not crash if an occasional response is
malformed — they will degrade gracefully.

You can make this stricter by setting `json_strict: true` in `eval.yml`, which
causes the case to fail with an error rather than fall back:

```yaml
mode: json-struct
json_strict: true
fail_under: 0.85
```

---

## Step 6 — Python API: `json_struct_diff()`

```python
from llm_diff import json_struct_diff

response_a = {
    "title": "The Future of AI in Healthcare",
    "author": "Jane Smith",
    "publication_date": "2025-11-14",
    "word_count": 1420,
    "topics": ["AI", "Healthcare", "Machine Learning"]
}

response_b = {
    "title": "The Future of AI in Healthcare",
    "author": "Jane Smith",
    "publication_date": "November 14, 2025",
    "word_count": "1420",
    "topics": ["AI", "health", "ML"]
}

result = json_struct_diff(response_a, response_b)

print(f"Struct similarity: {result.similarity:.1%}")
print(f"UNCHANGED: {result.unchanged_count}")
print(f"CHANGED:   {result.changed_count}")
print(f"Added:     {result.added_count}")
print(f"Removed:   {result.removed_count}")
print(f"TypeChange:{result.type_change_count}")

for field in result.fields:
    print(f"  {field.path:30s}  {field.label}")
```

You can also pass JSON strings directly:

```python
result = json_struct_diff(
    '{"status": "ok", "count": 42}',
    '{"status": "ok", "count": "42", "extra": true}',
)
# UNCHANGED: status
# TYPE_CHANGE: count  (42 int → "42" str)
# ADDED: extra
```

---

## Step 7 — JSON struct diff in batch mode

```yaml
model_a: gpt-4o
model_b: gpt-4o-mini
mode: json-struct
fail_under: 0.85
semantic: true

prompt_template: |
  Extract the following fields from the text as JSON:
  title, date, author, summary (one sentence), tags (array)

  Text: {input}

cases:
  - input: inputs/article_a.txt
  - input: inputs/article_b.txt
  - input: inputs/article_c.txt
```

```bash
llm-diff batch eval-json.yml --out json-eval.html
```

The batch HTML report shows both struct similarity and semantic similarity per
case, along with per-case field-by-field breakdowns.

---

## Summary

You have now:

- ✅ Understood why text diff is inadequate for JSON responses
- ✅ Used `--mode json-struct` to get field-by-field change labels
- ✅ Read ADDED, REMOVED, CHANGED, TYPE_CHANGE, UNCHANGED labels
- ✅ Used `--fail-under` as a schema completeness gate
- ✅ Understood graceful fallback on invalid JSON
- ✅ Used `json_struct_diff()` from the Python API
- ✅ Run JSON struct diff in batch mode

---

## What's Next

You have used `llm-diff` entirely through the CLI so far.  The final tutorial
shows you everything you can do from the Python API — composing evaluations
programmatically, integrating into test harnesses, and building async workflows.

[Tutorial 10 — Python API →](10-python-api.md)
