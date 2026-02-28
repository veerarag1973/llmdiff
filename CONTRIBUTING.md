# Contributing to llm-diff

Thank you for your interest in contributing!  This document covers everything
you need to set up a development environment, run tests, and submit a pull
request.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Making a Pull Request](#making-a-pull-request)
- [Architecture Overview](#architecture-overview)

---

## Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/sriramrathinavelu/llm-diff.git
cd llm-diff
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install in editable mode with all development dependencies

```bash
pip install -e ".[semantic,dev]"
```

This installs:
- `llm-diff` itself in editable mode
- `sentence-transformers` (the `[semantic]` extra)
- `pytest`, `pytest-cov`, `pytest-asyncio`, `ruff`, `numpy` (the `[dev]` extra)

### 4. (Optional) Set up a `.llmdiff` config for manual testing

```toml
# .llmdiff  — place in the project root (git-ignored)
[providers.openai]
api_key = "sk-..."
```

---

## Running Tests

### Full test suite

```bash
pytest
```

### With coverage report

```bash
pytest --cov=llm_diff --cov-report=term-missing
```

The project targets **100 % line coverage**.  Every PR must keep coverage at
100 %.

### Only a specific file

```bash
pytest tests/test_api.py -v
```

### Only tests matching a keyword

```bash
pytest -k "fail_under" -v
```

---

## Code Style

All Python code is linted and formatted with [Ruff](https://docs.astral.sh/ruff/).

### Check for lint errors

```bash
ruff check llm_diff/ tests/
```

### Auto-fix fixable issues

```bash
ruff check --fix llm_diff/ tests/
```

Rules enforced (see `pyproject.toml` for the full list):

| Code | Ruleset |
|------|---------|
| `E`, `W` | pycodestyle |
| `F` | pyflakes |
| `I` | isort |
| `B` | flake8-bugbear |
| `C4` | flake8-comprehensions |
| `UP` | pyupgrade |
| `S` | flake8-bandit (security) |
| `ANN` | flake8-annotations (source files only) |

`ANN` and `S` rules are relaxed in `tests/`.

### Type annotations

All public functions and methods in `llm_diff/` must carry full type
annotations.  Test helpers are exempt.

---

## Making a Pull Request

1. **Fork** the repository and create a branch from `main`.
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** — new features should come with tests.

3. **Run the full check suite** before committing:
   ```bash
   ruff check llm_diff/ tests/
   pytest --cov=llm_diff --cov-report=term-missing
   ```
   Both must pass with 0 errors and 100 % coverage.

4. **Commit** with a short, descriptive message:
   ```
   feat: add --retry flag for transient API errors
   fix: handle empty model response in paragraph scoring
   docs: add LM Studio example to providers guide
   ```
   Prefixes: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

5. **Open a pull request** against `main`.  The PR description should explain
   *what* changed and *why*.

---

## Architecture Overview

```
llm_diff/
├── __init__.py       Public re-exports: __version__, compare, compare_batch, …
├── api.py            Programmatic async API (no Click dependency)
├── batch.py          YAML batch loader — BatchItem, BatchResult, load_batch()
├── cli.py            Click entry point; orchestration (calls providers → diff → render)
├── config.py         LLMDiffConfig dataclass; load_config() reads .llmdiff / env
├── diff.py           word_diff() using difflib; DiffChunk, DiffResult
├── providers.py      compare_models() async; ModelResponse, ComparisonResult;
│                     provider auto-detection (OpenAI / Groq / Mistral / …)
├── renderer.py       render_diff() — Rich terminal output
├── report.py         build_report(), build_batch_report(), save_report()
├── semantic.py       compute_semantic_similarity(), compute_paragraph_similarity()
│                     ParagraphScore dataclass
└── templates/
    ├── report.html.j2        Single-diff HTML report template
    └── batch_report.html.j2  Batch HTML report template
```

### Request flow (single diff)

```
CLI main()
  └─ _resolve_inputs()          validate prompt + model flags
  └─ _run_diff()
       ├─ compare_models()       async: two concurrent LLM calls
       ├─ word_diff()            SequenceMatcher-based word diff
       ├─ compute_*_similarity() optional: sentence-transformers
       ├─ render_diff()          Rich terminal output
       └─ build_report()         optional: Jinja2 HTML
```

### Keeping things testable

- `compare_models()` is the outermost async function that touches the
  network.  Tests patch it with `AsyncMock`.
- `word_diff()` is patched in CLI tests where its return value needs to
  control a specific similarity score.
- `compute_semantic_similarity()` / `compute_paragraph_similarity()` are
  patched wherever sentence-transformers would otherwise be invoked.
- All HTML rendering uses Jinja2 templates; the template environment has
  `autoescape=True` to prevent XSS.

### Adding a new provider

1. Add the model-name detection pattern to `_detect_provider()` in
   `providers.py`.
2. Add corresponding API key env-var loading in `_get_api_key()`.
3. Add tests in `tests/test_providers.py`.
4. Add a `.llmdiff` snippet to `docs/providers.md`.
