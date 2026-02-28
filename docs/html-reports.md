# HTML Reports

`llm-diff` can generate fully self-contained HTML reports.  The files contain
all CSS, JavaScript, and diff data inline — no CDN, no network requests, no
server required.  Reports work offline and in air-gapped environments.

---

## Generating a report

### Single-diff report

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini \
  --semantic --out report.html
```

### Batch report

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini \
  --semantic --out batch_report.html
```

### Auto-save to `./diffs/`

Use `--save` to automatically save every run to a timestamped file under
`./diffs/`:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini --save
# Saved -> diffs/20260228_040637_gpt-4o_vs_gpt-4o-mini.html
```

Or set `save = true` in your `.llmdiff` config to make this the default:

```toml
[defaults]
save = true
```

### Via the Python API

```python
from llm_diff import compare
from llm_diff.report import save_report

report = asyncio.run(
    compare(
        "Explain recursion.",
        model_a="gpt-4o",
        model_b="gpt-4o-mini",
        semantic=True,
        build_html=True,
    )
)
save_report(report.html_report, "report.html")
```

---

## Single-diff report anatomy

A single-diff HTML report contains:

### Header
- Report title, generation timestamp
- Model A and Model B identifiers

### Similarity scores panel
- **Word similarity** — always present
- **Semantic similarity** — present when `--semantic` was used
- **BLEU** — present when `--bleu` was used
- **ROUGE-L** — present when `--rouge` was used
- **Primary score** — highlighted metric (semantic if available, else word)

### Per-paragraph scores table
Present when `--paragraph` was used.  Shows the similarity score for each
paragraph pair, helping identify which sections diverged.

### Word diff
Side-by-side view of the two responses with colour-coded diff:
- Red highlights — text present in Model A but absent from Model B (deletions)
- Green highlights — text new in Model B (insertions)
- Unmarked text — identical in both responses

### Token and latency metadata
- For each model: prompt tokens, completion tokens, total tokens
- Round-trip latency in milliseconds
- Cached badge if the response was served from the local cache

### Judge card *(v1.2 — present when `--judge` was used)*
- **Winner**: A, B, or Tie
- **Score A** and **Score B** (1–10 scale)
- **Reasoning** paragraph from the judge model
- **Judge model** identifier

### Cost table *(v1.2 — present when `--show-cost` was used)*
| | Model A | Model B |
|--|--|--|
| Prompt tokens | N | N |
| Completion tokens | N | N |
| Total cost (USD) | $X.XXXXXX | $X.XXXXXX |

If the model is not in the built-in pricing table, `unknown model` is shown
instead of a cost figure.

---

## Batch report anatomy

A batch report replaces the single-diff display with an aggregate view.

### Summary header
- Total number of prompts
- Average word similarity across all items
- Average semantic similarity (when `--semantic` was used)
- `--fail-under` threshold and overall pass/fail result (when set)

### Summary table
One row per batch item:

| Prompt ID | Word sim | Semantic sim | Pass/Fail |
|-----------|----------|--------------|-----------|
| explain-recursion | 61.3% | 92.7% | PASS |
| code-review/func.py | 74.8% | 89.2% | PASS |
| summarise/article1.txt | 48.3% | 65.1% | FAIL |

Pass/Fail colouring is applied when `--fail-under` is used.  Rows that fall
below the threshold are highlighted in red.

### Per-item diff cards
Below the summary table, each prompt has an expandable diff card containing
the same sections as a single-diff report (word diff, scores, metadata, judge
card, cost table).

---

## Customising the templates

Report HTML is rendered via Jinja2 templates located in
`llm_diff/templates/`:

| Template | Used for |
|----------|---------|
| `report.html.j2` | Single-diff reports |
| `batch_report.html.j2` | Batch reports |

To customise the templates, copy them to your project and pass the directory
to `build_report()` in the Python API.  The templates use `autoescape=True`
to prevent XSS from model response content.

---

## Opening reports

Reports are standard HTML files.  Open them in any browser:

```bash
# macOS
open report.html

# Linux
xdg-open report.html

# Windows
start report.html
```

Or drag the file into a browser tab.
