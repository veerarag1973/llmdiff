# Getting Started

This guide walks you through installing `llm-diff`, setting up an API key, and
running your first comparison.

---

## Requirements

- Python 3.9 or later
- An API key for at least one LLM provider (OpenAI, Groq, Mistral, etc.)

---

## Installation

### Core install (word diff + HTML reports)

```bash
pip install llm-diff
```

### With semantic scoring

```bash
pip install "llm-diff[semantic]"
```

The `[semantic]` extra installs `sentence-transformers`.  The default model
(`all-MiniLM-L6-v2`, ~80 MB) downloads automatically on first use.

### Development install

```bash
git clone https://github.com/veerarag1973/llmdiff.git
cd llm-diff
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[semantic,dev]"
```

---

## Set an API key

```bash
export OPENAI_API_KEY="sk-..."
```

Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

See [Provider Setup](providers.md) for Groq, Mistral, Anthropic, Ollama, and
other providers.

---

## Your first comparison

```bash
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini
```

Sample output:

```
Model A (gpt-4o):
  Recursion is a technique where a function calls itself with a simpler
  version of the problem until a base case is reached.

Model B (gpt-4o-mini):
  Recursion is when a function calls itself repeatedly until a base
  condition is met, solving the problem step by step.

Word diff:
  Recursion is [-a technique where a-] {+when a+} function calls itself
  [-with a simpler version of the problem until a base case is reached.-]
  {+repeatedly until a base condition is met, solving the problem step by step.+}

  Word similarity:  61.3%
  Model A tokens:   28    latency: 0.84s
  Model B tokens:   24    latency: 0.61s
```

`[-deleted-]` text is present in Model A but absent from Model B.  `{+inserted+}`
text is new in Model B.  In the terminal these render as red and green.

---

## Add semantic scoring

Word overlap can be misleading — two responses can score 50 % word similarity
while meaning the same thing.  The `--semantic` flag computes a cosine
similarity score using sentence embeddings:

```bash
llm-diff "Explain recursion in one sentence." \
  -a gpt-4o -b gpt-4o-mini --semantic
```

```
  Word similarity:     61.3%
  Semantic similarity: 92.7%   <- same meaning, different words
  Primary score:       92.7%
```

A semantic score above ~85 % generally means the responses are equivalent in
meaning even if the wording differs.

---

## Per-paragraph scoring

When responses are multi-paragraph, a single score loses signal.  Use
`--paragraph` to break scores down by paragraph:

```bash
llm-diff "Write a 3-paragraph summary of transformer architecture." \
  -a gpt-4o -b gpt-4o-mini --paragraph
```

```
  Paragraph similarities:
    #1  Transformers rely on self-attention...    94.2%
    #2  The encoder processes input tokens...     81.3%
    #3  Training uses masked language...          79.8%
```

---

## BLEU and ROUGE-L

Compute industry-standard NLP metrics with no extra dependencies:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini --bleu --rouge
```

```
  Word similarity:  61.3%
  BLEU:             42.1%
  ROUGE-L:          68.7%
```

Combine all metrics:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini --semantic --bleu --rouge
```

---

## Prompt-diff mode

Compare how two different prompts affect the same model — useful for prompt
engineering:

```bash
llm-diff --model gpt-4o \
  --prompt-a "Explain recursion concisely." \
  --prompt-b "Explain recursion with a real-world analogy." \
  --semantic
```

You can also pass file paths to `--prompt-a` / `--prompt-b`.

---

## Save an HTML report

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini \
  --semantic --out report.html
```

The file is fully self-contained — no CDN, no internet, works offline.
See [HTML Reports](html-reports.md) for what the report contains.

---

## Batch mode

Evaluate a set of prompts in one command.  Create `prompts.yml`:

```yaml
prompts:
  - id: explain-recursion
    text: "Explain recursion in one sentence."

  - id: code-review
    text: "Review this Python function: {input}"
    inputs:
      - examples/func.py
```

Run the batch:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini \
  --semantic --out batch_report.html
```

See [examples/prompts.yml](../examples/prompts.yml) for a working example.

---

## CI quality gate

Fail the command if similarity drops below a threshold — useful for CI
pipelines:

```bash
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.85
```

Exit code 0 = all prompts passed.  Exit code 1 = at least one failed.

See [CI / CD Integration](ci-cd.md) for full GitHub Actions examples.

---

## Next steps

| Topic | Guide |
|---|---|
| All CLI flags and options | [CLI Reference](cli-reference.md) |
| Programmatic Python usage | [Python API](api.md) |
| Config file and env vars | [Configuration](configuration.md) |
| Provider setup (Groq, Ollama, …) | [Provider Setup](providers.md) |
| HTML report anatomy | [HTML Reports](html-reports.md) |
| CI/CD pipeline integration | [CI / CD Integration](ci-cd.md) |
| Observability / schema events | [Schema Events](schema-events.md) |
