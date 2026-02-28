# llm-diff — Implementation Plan

> Based on: `llm-diff_Product_Spec.pdf` v1.0 (February 2026)
>
> **llm-diff** is a CLI tool for comparing two LLM responses — semantically, visually, and at scale. Think `git diff` for language model outputs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Repository Structure](#3-repository-structure)
4. [Phase-by-Phase Implementation](#4-phase-by-phase-implementation)
   - [Phase 1 — v0.1 MVP (Week 1–2)](#phase-1--v01-mvp-week-12)
   - [Phase 2 — v0.2 Modes (Week 3–4)](#phase-2--v02-modes-week-34)
   - [Phase 3 — v0.3 Prompt History (Week 5)](#phase-3--v03-prompt-history-week-5)
   - [Phase 4 — v0.4 Dashboard (Week 6)](#phase-4--v04-dashboard-week-6)
   - [Phase 5 — v1.0 Stable Release (Week 7)](#phase-5--v10-stable-release-week-7)
5. [Module Breakdown & Implementation Details](#5-module-breakdown--implementation-details)
6. [CLI Interface Specification](#6-cli-interface-specification)
7. [JSON Output Schema](#7-json-output-schema)
8. [Non-Functional Requirements](#8-non-functional-requirements)
9. [Open Questions to Resolve](#9-open-questions-to-resolve)
10. [Out of Scope for v1.0](#10-out-of-scope-for-v10)
11. [Success Metrics](#11-success-metrics)

---

## 1. Project Overview

**Problem**: Developers working with LLMs constantly need to compare outputs when switching models, tweaking prompts, or adjusting temperature. There is no fast, open-source standard tool for this.

**Solution**: A CLI tool that diffs LLM responses at three levels:
- **Word level** — character/word additions and deletions with color coding
- **Semantic level** — meaning-based similarity scoring using embeddings
- **Structural/JSON** — machine-readable diff for CI pipelines

**Distribution**: Published as both a **Python package (PyPI)** and a **TypeScript/npm package**.

**Target Users**: AI engineers, developers benchmarking model upgrades, teams evaluating vendor switches, researchers comparing model behavior.

---

## 2. Technology Stack

### Python Package (Primary)
| Component | Library/Tool |
|---|---|
| CLI framework | `click` |
| Word diff engine | `difflib` (stdlib) |
| Semantic diff | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| API layer | `openai` SDK (with `base_url` override for other providers) |
| Terminal rendering | `rich` |
| HTML reports | `Jinja2` + vanilla JS (no external deps) |
| Config loading | TOML (`.llmdiff` file) |
| Packaging | `pyproject.toml` |
| Python version | 3.9+ |

### TypeScript Package (Secondary — npm)
| Component | Detail |
|---|---|
| Runtime | Node.js 18+ |
| Package name | `llm-diff` (check npm availability) |
| API | `import { diff } from 'llm-diff'` |
| Dependency | Standalone — no Python required |

### Dev / Test
- `pytest` for Python tests
- Standard npm test tooling for TypeScript
- GitHub Actions for CI

---

## 3. Repository Structure

```
llm-diff/
├── llm_diff/
│   ├── cli.py           # Click entry point
│   ├── diff.py          # Core diff engine (word-level via difflib)
│   ├── semantic.py      # Embedding-based similarity scoring
│   ├── providers.py     # API abstraction layer (multi-provider)
│   ├── report.py        # HTML report generator
│   └── config.py        # .llmdiff TOML config loader
├── templates/
│   └── report.html.j2   # Jinja2 report template (self-contained)
├── tests/
│   ├── test_diff.py
│   └── test_providers.py
├── pyproject.toml
├── README.md
└── (npm/ts package — separate directory or monorepo subfolder)
```

---

## 4. Phase-by-Phase Implementation

### Phase 1 — v0.1 MVP (Week 1–2) ✅ COMPLETE
**Goal**: Working end-to-end on any two OpenAI-compatible endpoints.

#### Tasks

- [x] **Repo setup**
  - Initialize git repo, `pyproject.toml`, dev environment
  - Add `.gitignore`, `README.md` skeleton, `LICENSE`

- [x] **`config.py` — Config loader**
  - Load `.llmdiff` TOML file from project root or home dir
  - Fall back to environment variables and `.env` file
  - Support `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `base_url` overrides

- [x] **`providers.py` — API abstraction layer**
  - Wrap the `openai` SDK with `base_url` override support
  - Support OpenAI, Anthropic, Groq, Mistral, local (Ollama/LM Studio)
  - Implement **concurrent** (not sequential) API calls via `asyncio.gather`
  - Retry with exponential backoff on rate-limit errors (HTTP 429)
  - Timeout handling with clear `TimeoutError` message
  - Fresh client per retry attempt (prevents closed-connection reuse)

- [x] **`diff.py` — Core word-level diff engine**
  - `difflib.SequenceMatcher` for word-level diffing
  - Tokenize responses into words (preserve trailing whitespace for roundtrip)
  - Produce structured `DiffChunk` output: `[{type: equal|insert|delete, text}]`
  - Performance: < 100ms on 2,000-word outputs ✅

- [x] **`cli.py` — Click-based CLI entry point**
  - All 17 CLI flags fully implemented
  - `asyncio.run()` entry point
  - `_die()` annotated `NoReturn`, outputs to dedicated stderr console

- [x] **`renderer.py` — Terminal rendering**
  - Color-coded diff (green inserts, red deletes, white equal)
  - Header: model names, prompt snippet
  - Footer: similarity %, token counts (A/B), latency (A/B)

- [x] **Tests** — 71 tests, all passing
  - `test_diff.py`: 28 tests covering tokeniser, diff engine, serialisation, performance
  - `test_providers.py`: 14 tests covering retry logic, concurrency, API mocking
  - `test_config.py`: 29 tests covering TOML loading, env vars, .env, priority order

- [x] **Code review fixes**
  - Fixed: client created inside retry loop (was being reused after close)
  - Fixed: `compare_models` had duplicate provider lookup calls
  - Fixed: `prompt_a`/`prompt_b` params added — model B was called twice when prompts differed
  - Fixed: `_merge_toml` made recursive (was only one level deep)
  - Fixed: `_die` used invalid `Console.print(stderr=True)` — now uses dedicated stderr Console
  - Fixed: `NoReturn` type annotation added to `_die`
  - Zero `ruff` lint violations

**v0.1 Success Metric**: Works end-to-end with two real OpenAI-compatible models. ✅

---

### Phase 2 — v0.2 Modes (Week 3–4)
**Goal**: Three diff modes working reliably — `word`, `semantic`, `json`.

#### Tasks

- [ ] **`semantic.py` — Embedding-based similarity scoring**
  - Integrate `sentence-transformers` model `all-MiniLM-L6-v2`
  - On first run: download ~80MB model (show download progress)
  - Compute cosine similarity between full response embeddings
  - *(Decision needed)* Sentence-level vs full-response scoring — see [Open Questions](#9-open-questions-to-resolve)
  - Performance target: < 2s with local model

- [ ] **`--mode` flag and shorthand flags**
  - `--mode word` (default)
  - `--mode semantic` / `-s` shorthand
  - `--mode json` / `-j` shorthand

- [ ] **JSON output mode**
  - Output structured JSON to stdout following the [JSON schema](#7-json-output-schema)
  - Include: prompt, model_a, model_b, similarity_score, tokens, latency_ms, diff array

- [ ] **`--save` flag**
  - Auto-save HTML report to `./diffs/` directory on each run

- [ ] **`--out <path>` flag**
  - Save HTML report to a custom file path

- [ ] **Extend terminal output**
  - Semantic mode: show "Similarity: 78% | Key divergences: tone, structure, 3rd paragraph"
  - Surface divergence highlights (paragraph-level)

**v0.2 Success Metric**: All 3 diff modes (`word`, `semantic`, `json`) working reliably.

---

### Phase 3 — v0.3 Prompt History (Week 5)
**Goal**: Batch mode and prompt file diffing — diff 10+ prompts in one command.

#### Tasks

- [ ] **`--prompt-a` and `--prompt-b` file flags**
  - Accept paths to text files as prompts for model A and B respectively
  - Allows prompt diffing against the same model (`--model`)

- [ ] **`prompts.yml` batch support**
  - Define YAML schema:
    ```yaml
    prompts:
      - id: summarize
        text: "Summarize the following in 3 sentences: {input}"
        inputs: [sample1.txt, sample2.txt]
      - id: rewrite
        text: "Rewrite this in a formal tone: {input}"
        inputs: [sample3.txt]
    ```
  - `--batch <prompts.yml>` runs all prompts and outputs a combined report
  - Support `{input}` template variable with input file injection

- [ ] **Batch output aggregation**
  - Per-prompt diff display in terminal
  - Combined HTML report with all batch results when `--out` flag is used

- [ ] **`--model` flag (same model, different prompts)**
  - When only one model is specified, treat `--prompt-a` and `--prompt-b` as the two sides

**v0.3 Success Metric**: Can diff 10 prompts in one command.

---

### Phase 4 — v0.4 Dashboard (Week 6)
**Goal**: Self-contained, shareable HTML report that loads fully offline.

#### Tasks

- [ ] **`report.py` — HTML report generator**
  - Use `Jinja2` to render `templates/report.html.j2`
  - Embed all CSS and JS inline (no external CDN dependencies)
  - Support batch results (multiple prompt comparisons in one report)

- [ ] **`templates/report.html.j2` — Report template**
  - Side-by-side diff view with color-coded additions/deletions
  - Semantic similarity score per paragraph
  - Prompt metadata table: model, temperature, token count, latency
  - Vanilla JS for interactive toggling (expand/collapse diffs)
  - Works fully offline; shareable via email or Slack

- [ ] **End-to-end test for report output**
  - Verify generated HTML loads without external requests
  - Verify all diff data is embedded correctly

**v0.4 Success Metric**: Report loads offline with no external dependencies.

---

### Phase 5 — v1.0 Stable Release (Week 7)
**Goal**: Full docs, test suite, PyPI + npm publish, README with demos.

#### Tasks

- [ ] **Complete test suite**
  - Achieve meaningful coverage on `diff.py`, `providers.py`, `semantic.py`, `report.py`, `config.py`
  - Add integration tests with mocked API responses for all diff modes
  - Add regression tests for terminal output formatting

- [ ] **Documentation**
  - Write comprehensive `README.md` with:
    - Installation instructions (pip + npm)
    - Quick-start demo (< 60 second to first diff)
    - All CLI flags reference
    - Provider configuration guide
    - Example `prompts.yml`
    - Example HTML report screenshot
  - Add `CONTRIBUTING.md`
  - Add `CHANGELOG.md`

- [ ] **TypeScript/npm package**
  - Implement standalone TypeScript package (no Python dependency)
  - Uses same JSON output schema as Python package
  - `import { diff } from 'llm-diff'` programmatic API
  - Target Node.js 18+
  - Resolve npm package name (`llm-diff` vs scoped `@author/llm-diff`)
  - Publish to npm

- [ ] **PyPI publish**
  - Finalize `pyproject.toml` metadata (name, description, classifiers, entry points)
  - Build and publish to PyPI: `pip install llm-diff`

- [ ] **CI/CD**
  - GitHub Actions: lint, test, build on push/PR
  - Automated PyPI and npm publish on tag

- [ ] **Launch prep**
  - GitHub repository cleanup (topics, description, social preview)
  - Discord community setup (target: 50+ members)

**v1.0 Success Metric**: 100 GitHub stars within 2 weeks of launch.

---

## 5. Module Breakdown & Implementation Details

### `cli.py`
- Built with `click`
- Main command: `llm-diff`
- Handles flag parsing, config loading, orchestrates providers → diff → render pipeline

### `diff.py`
- Word tokenization + `difflib.SequenceMatcher` for word-level diff
- Returns a normalized list of `{type, text}` objects consumed by both the terminal renderer and JSON output
- No external dependencies (stdlib only)

### `semantic.py`
- Lazy-loads `sentence-transformers` model on first use
- Cosine similarity computation between A and B response embeddings
- Returns float score (0.0–1.0) + optional sentence-level breakdown

### `providers.py`
- Thin wrapper around `openai.AsyncOpenAI` with `base_url` override
- Provider routing: detect from model name prefix (`gpt-*` → OpenAI, `claude-*` → Anthropic) **or** explicit `--provider` flag (decision pending)
- Concurrent calls using `asyncio.gather(call_a, call_b)`
- Returns: response text, token counts, latency

### `report.py`
- Takes diff output + metadata, renders via Jinja2
- Inlines CSS/JS from `templates/` at build time
- Produces a single `.html` file

### `config.py`
- Priority order: CLI flags > `.llmdiff` TOML > `.env` file > environment variables
- Never logs or exposes API keys
- Supports per-project and global config (`~/.llmdiff`)

---

## 6. CLI Interface Specification

```bash
# Basic comparison — two models, same prompt
llm-diff "Explain recursion" --a gpt-4o --b claude-3-5-sonnet

# Compare two prompt files against the same model
llm-diff --prompt-a v1.txt --prompt-b v2.txt --model gpt-4o

# Semantic diff with score
llm-diff "Summarize this doc" --a gpt-4o --b mistral-large --semantic

# Batch compare from a YAML prompt file
llm-diff --batch prompts.yml --a gpt-4o --b gpt-4-turbo --out report.html
```

### Full Flag Reference

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--prompt` | `-p` | string | — | Prompt text sent to both models |
| `--prompt-a` | — | path | — | Path to text file used as prompt for model A |
| `--prompt-b` | — | path | — | Path to text file used as prompt for model B |
| `--model-a` | `-a` | string | — | Model identifier for side A (e.g. `gpt-4o`) |
| `--model-b` | `-b` | string | — | Model identifier for side B (e.g. `claude-3-5-sonnet`) |
| `--model` | `-m` | string | — | Same model for both sides (prompt diff mode) |
| `--mode` | — | enum | `word` | Diff mode: `word` \| `semantic` \| `json` |
| `--semantic` | `-s` | flag | false | Shorthand for `--mode semantic` |
| `--json` | `-j` | flag | false | Output raw JSON diff to stdout |
| `--batch` | — | path | — | Path to `prompts.yml` for batch comparison |
| `--out` | `-o` | path | — | Save HTML report to this path |
| `--save` | — | flag | false | Auto-save HTML report to `./diffs/` directory |
| `--temperature` | `-t` | float | `0.7` | Temperature passed to both models |
| `--max-tokens` | — | int | `1024` | Max tokens for each model's response |
| `--timeout` | — | int | `30` | Request timeout in seconds |
| `--no-color` | — | flag | false | Disable terminal color output |
| `--verbose` | `-v` | flag | false | Show full API request/response metadata |
| `--version` | — | flag | — | Print version and exit |

---

## 7. JSON Output Schema

Output when `--json` / `-j` flag is set:

```json
{
  "prompt": "Explain recursion in simple terms",
  "model_a": "gpt-4o",
  "model_b": "claude-3-5-sonnet",
  "similarity_score": 0.84,
  "tokens": { "a": 61, "b": 73 },
  "latency_ms": { "a": 920, "b": 1210 },
  "diff": [
    { "type": "equal", "text": "Recursion is when..." },
    { "type": "delete", "text": "a smaller version" },
    { "type": "insert", "text": "a smaller piece...base case" }
  ]
}
```

---

## 8. Non-Functional Requirements

### Performance
| Requirement | Target |
|---|---|
| Cold-start (no API call) | < 200ms |
| Word diff on 2,000-word outputs | < 100ms |
| Semantic diff (embedding, local model) | < 2s |
| API calls for A and B | Always concurrent, never sequential |

### Reliability
- Retry with exponential backoff on API rate-limit errors (HTTP 429)
- Graceful failure with clear, actionable error messages
- Timeout handling with partial result display

### Compatibility
- Python 3.9+
- Node.js 18+ (npm package)
- macOS, Linux, Windows (WSL tested)
- Works in VS Code terminal, iTerm2, and standard shells

### Security
- API keys **never** logged or stored to disk
- Keys read from environment variables or `.env` file only
- No telemetry without explicit opt-in
- HTML reports contain no external requests (fully self-contained)

---

## 9. Open Questions to Resolve

These decisions are needed before or during v0.2 development:

| # | Question | Options | Recommendation |
|---|---|---|---|
| 1 | **Semantic diff model delivery** | (a) ~80MB local download on first run via `sentence-transformers` vs (b) embedding API call | Default to local model; show progress bar on first download. Offer `--use-embeddings-api` flag as opt-in |
| 2 | **Config file format** | TOML (`.llmdiff`) vs YAML vs `.env` only | Proceed with TOML — cleanest DX. Accept `tomli`/`tomllib` (Python 3.11+ stdlib) dependency |
| 3 | **Provider auth UX** | (a) Auto-detect from model name prefix (`claude-*` → Anthropic key) vs (b) explicit `--provider` flag | Implement auto-detection as default with `--provider` as override |
| 4 | **Semantic scoring granularity** | (a) Cosine similarity of full response embeddings vs (b) sentence-level scoring averaged | Start with full-response for v0.2; add sentence-level as opt-in `--detailed` flag in v0.3 |
| 5 | **npm package name** | `llm-diff` vs `@author/llm-diff` | Check npm registry availability first; prefer unscoped `llm-diff` |

---

## 10. Out of Scope for v1.0

These features are **deliberately excluded** to avoid scope creep:

- GUI or web application (CLI-first)
- Fine-tuning or training integrations
- Image, audio, or multimodal output diffing
- Saved diff history or database persistence
- Team accounts or authentication
- Agent or multi-turn conversation diffing *(v2 candidate)*
- Automated prompt optimization suggestions

---

## 11. Success Metrics

Measured at v1.0 launch and 30 days post-launch:

| Metric | Target |
|---|---|
| GitHub stars at launch week | 100+ |
| Time to first successful diff (new user) | < 60 seconds |
| CLI cold-start latency (no API call) | < 200ms |
| PyPI / npm downloads in month 1 | 500+ |
| Issues opened in first 2 weeks | 10+ (signals real usage) |
| Discord / community members at v1.0 | 50+ |

---

## Summary Timeline

| Week | Phase | Deliverable |
|---|---|---|
| 1–2 | v0.1 MVP | Core diff engine, two-model CLI, colored terminal output |
| 3–4 | v0.2 Modes | Semantic diff, word diff, JSON output, `--save` flag |
| 5 | v0.3 Prompt History | `prompt.yml` support, batch mode, prompt file diffing |
| 6 | v0.4 Dashboard | Self-contained HTML report output |
| 7 | v1.0 Stable | Full docs, test suite, PyPI + npm publish, README with demos |

**Total: ~7 weeks to v1.0**
