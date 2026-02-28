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
   - [Phase 1 — v0.1 MVP (Week 1–2)](#phase-1--v01-mvp-week-12) ✅
   - [Phase 2 — v0.2 Modes (Week 3–4)](#phase-2--v02-modes-week-34) ✅
   - [Phase 3 — v0.3 Batch (Week 5)](#phase-3--v03-batch-week-5) ✅
   - [Phase 4 — v0.4 Dashboard (Week 6)](#phase-4--v04-dashboard-week-6) ✅
   - [Phase 5 — v0.5 Security & Code Cleanup (Week 7)](#phase-5--v05-security--code-cleanup-week-7) ✅
   - [Phase 6 — v0.6 Enterprise Features (Week 8)](#phase-6--v06-enterprise-features-week-8) ✅
   - [Phase 7 — v0.7 Documentation & Packaging (Week 9)](#phase-7--v07-documentation--packaging-week-9) ✅
   - [Phase 8 — v0.8 Robustness & Testing (Week 10)](#phase-8--v08-robustness--testing-week-10) ✅
   - [Phase 9 — v1.0 Stable Release (Week 11–12)](#phase-9--v10-stable-release-week-1112)
   - [Phase 10 — v1.1 Tier 1: Evaluation Depth](#phase-10--v11-tier-1-evaluation-depth)
   - [Phase 11 — v1.2 Tier 2: Developer Experience](#phase-11--v12-tier-2-developer-experience)
   - [Phase 12 — v2.0 Tier 3: Polish & Ecosystem](#phase-12--v20-tier-3-polish--ecosystem)
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

### Phase 2 — v0.2 Modes (Week 3–4) ✅ COMPLETE
**Goal**: Semantic scoring, HTML reports, and output saving.

#### Tasks

- [x] **`semantic.py` — Embedding-based similarity scoring**
  - Integrated `sentence-transformers` model `all-MiniLM-L6-v2` with lazy loading + cache
  - Cosine similarity between full response embeddings (clamped 0–1)
  - `reset_model_cache()` for testing

- [x] **`--semantic` / `-s` flag**
  - Boolean flag to enable semantic similarity scoring alongside word diff

- [x] **`--json` / `-j` output mode**
  - Structured JSON to stdout with prompt, models, similarity, tokens, latency, diff array

- [x] **`--save` flag**
  - Auto-save HTML report to `./diffs/` directory with timestamped filename

- [x] **`--out <path>` flag**
  - Save HTML report to a custom file path

- [x] **`report.py` — Jinja2 HTML report generator**
  - `build_report()`, `save_report()`, `auto_save_report()`
  - Dark-theme self-contained HTML template (`report.html.j2`)

- [x] **Tests** — 226 tests, 100% coverage, 0 ruff errors

**v0.2 Success Metric**: Semantic scoring + HTML reports working reliably. ✅

---

### Phase 3 — v0.3 Batch (Week 5) ✅ COMPLETE
**Goal**: Batch mode and prompt file diffing — diff 10+ prompts in one command.

#### Tasks

- [x] **`batch.py` — YAML batch file loader**
  - `BatchItem(frozen=True)`, `BatchResult` dataclasses
  - `_expand_template()` with `{input}` variable injection
  - `load_batch()` with full YAML schema validation

- [x] **`--batch <prompts.yml>` flag**
  - Runs all prompts and outputs per-prompt terminal diffs
  - Combined HTML report via `--out` flag

- [x] **`batch_report.html.j2` — Combined batch HTML template**
  - Summary bar, TOC navigation, per-item result cards with score pills
  - Self-contained dark theme, fully offline

- [x] **`report.py` — `build_batch_report()`**
  - Aggregates multiple `BatchResult` into combined HTML

- [x] **Tests** — 287 tests, 100% coverage, 0 ruff errors

**v0.3 Success Metric**: Can diff 10 prompts in one command. ✅

---

### Phase 4 — v0.4 Dashboard (Week 6) ✅ COMPLETE
**Goal**: Paragraph-level semantic scoring and interactive HTML reports.

#### Tasks

- [x] **`semantic.py` — Paragraph-level scoring**
  - `ParagraphScore(frozen=True)` dataclass
  - `compute_paragraph_similarity()` — splits on `\n\n`, aligns by index, pads shorter side

- [x] **`renderer.py` — Paragraph similarity table**
  - `render_diff()` gained `paragraph_scores` param
  - Rich terminal table with score-coloured rows

- [x] **`report.html.j2` — Interactive HTML features**
  - Paragraph similarity table section
  - Expand/collapse buttons for responses and paragraphs
  - Score pills with colour grading
  - New CSS for `para-section`, `para-table`, `score-pill`
  - New JS functions `toggleResponses()`, `toggleParagraphs()`

- [x] **`--paragraph` / `-p` CLI flag**
  - Computes paragraph-level + overall semantic scores
  - Passes both to renderer and HTML report

- [x] **Tests** — 330 tests, 100% coverage, 0 ruff errors

**v0.4 Success Metric**: Report loads offline with paragraph-level scoring. ✅

---

### Phase 5 — v0.5 Security & Code Cleanup (Week 7) ✅ COMPLETE
**Goal**: Harden the codebase — fix security gaps, remove dead code, clean up stale artifacts.

#### Tasks

- [x] **Fix Jinja2 autoescape gap (security)**
  - Changed `select_autoescape(["html"])` to `autoescape=True` (always-on, covers `.html.j2`)
  - Eliminated XSS vector for user-controlled text in HTML reports

- [x] **Remove dead code in `cli.py`**
  - Removed `_model_options`, `_output_options`, `_request_options`, `_display_options`, `_add_options` (~90 lines)

- [x] **Remove unused `telemetry` field in `config.py`**
  - Removed from `LLMDiffConfig` dataclass and from TOML loader
  - Stale `telemetry` key in `.llmdiff` files is now gracefully ignored

- [x] **Delete stale `templates/` directory at repo root**
  - Removed empty directory; real templates live in `llm_diff/templates/`

- [x] **Guard against empty `response.choices` in `providers.py`**
  - Raises `RuntimeError` with clear message if `choices` is empty or `content` is `None`

- [x] **Suppress API keys in debug/error logging**
  - Added `sentence_transformers` and `transformers` to suppressed loggers
  - httpx kept at WARNING (prevents auth header leakage in debug logs)
  - Documented suppression rationale in comments

- [x] **Tests** — 334 tests, 100% coverage, 0 ruff errors

**v0.5 Success Metric**: Zero known security issues, zero dead code. ✅

---

### Phase 6 — v0.6 Enterprise Features (Week 8)
**Goal**: Add CI/CD-friendly features and a programmatic API for enterprise integration.

#### Tasks

- [x] **`--fail-under <threshold>` flag**
  - New Click option: `--fail-under` / `-f` accepting a float (0.0–1.0)
  - If `similarity_score` (or `semantic_score` when `--semantic` is used) falls below the threshold, exit with code 1
  - Enables use in CI pipelines: `llm-diff "..." --a gpt-4o --b gpt-4-turbo --semantic --fail-under 0.8`
  - Works in both single-diff and batch mode (any item below threshold → exit 1)

- [x] **Programmatic Python API**
  - Create `llm_diff/api.py` exposing public async functions:
    - `async def compare(prompt, model_a, model_b, **kwargs) -> ComparisonReport`
    - `async def compare_batch(batch_path, model_a, model_b, **kwargs) -> list[ComparisonReport]`
  - `ComparisonReport` dataclass: diff_result, semantic_score, paragraph_scores, html_report, json_output
  - No Click dependency — pure library usage: `from llm_diff import compare`
  - Re-export in `__init__.py` for clean imports

- [x] **`--paragraph` support in batch mode**
  - Currently `_run_batch()` in `cli.py` does not pass paragraph scoring through
  - Wire `--paragraph` flag into batch execution to compute paragraph-level scores per batch item
  - Include paragraph data in batch HTML report

- [x] **Anthropic provider documentation / guidance**
  - `providers.py` always uses `AsyncOpenAI` — works with Anthropic via `base_url` override only
  - Document clearly in README which providers are natively supported vs require `base_url` config
  - Add example `.llmdiff` config snippets for OpenAI, Anthropic (via proxy), Groq, Ollama, LM Studio

- [x] **Tests** — 407 tests, 100% coverage, 0 ruff errors

**v0.6 Success Metric**: `--fail-under` works in CI; library usable without CLI.

---

### Phase 7 — v0.7 Documentation & Packaging (Week 9)
**Goal**: Production-quality documentation and PyPI-ready packaging.

#### Tasks

- [x] **`LICENSE` file**
  - Add MIT license (already declared in `pyproject.toml` but file is missing)

- [x] **Comprehensive `README.md`**
  - Installation instructions (`pip install llm-diff`, `pip install llm-diff[semantic]`)
  - Quick-start demo (< 60 seconds to first diff)
  - All CLI flags reference table
  - Provider configuration guide (OpenAI, Anthropic, Groq, Ollama, LM Studio)
  - `.llmdiff` TOML config file example
  - `prompts.yml` batch file example
  - Programmatic API usage examples
  - CI/CD integration guide (`--fail-under`)
  - Badge placeholders (PyPI version, CI status, coverage)

- [x] **`CHANGELOG.md`**
  - Entries for v0.1 through v0.7.0
  - Follow [Keep a Changelog](https://keepachangelog.com/) format

- [x] **`CONTRIBUTING.md`**
  - Dev setup instructions (clone, venv, install dev deps)
  - Running tests (`pytest`, coverage, ruff)
  - PR guidelines, commit message convention
  - Architecture overview (module responsibilities)

- [x] **Example `prompts.yml`**
  - `examples/prompts.yml` — 3 plain prompts + 2 `{input}` template prompts
  - `examples/inputs/article_a.txt`, `article_b.txt`, `func.py` — sample input files

- [x] **PyPI readiness**
  - `pyproject.toml`: classifier updated to `Development Status :: 4 - Beta`
    (targeting 5 - Production/Stable at v1.0); Python 3.13 added to classifiers;
    `build` and `twine` added to `[dev]` extra
  - Build verified via `python -m build` + `twine check dist/*`

- [x] **Tests** — 407 tests, 100% coverage, 0 ruff errors

**v0.7 Success Metric**: `pip install llm-diff` works; README enables first diff in < 60s.

---

### Phase 8 — v0.8 Robustness & Testing (Week 10)
**Goal**: Harden reliability with integration tests, edge-case handling, and CI automation.

#### Tasks

- [x] **Integration tests with mocked API responses**
  - End-to-end tests exercising the full pipeline: CLI → config → providers → diff → render/report
  - Mock `AsyncOpenAI` at the HTTP level (not just function mocks)
  - Cover all diff modes: word-only, semantic, paragraph, JSON output, HTML report
  - Cover batch mode end-to-end

- [x] **GitHub Actions CI pipeline**
  - `.github/workflows/ci.yml`:
    - Matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13
    - Steps: install deps → ruff lint → pytest with coverage → coverage gate (100%)
    - Run on push and PR to main
  - `.github/workflows/publish.yml`:
    - Trigger on version tag (`v*`)
    - Build and publish to PyPI via trusted publisher

- [x] **Edge-case hardening**
  - Empty responses from models (zero-length text) — warning logged, no crash
  - Unicode / emoji in responses — CJK, Arabic, RTL, mathematical symbols
  - Very long responses (10,000+ words) — performance regression guard (< 1s)
  - Malformed YAML in batch files — tested via CLI path
  - Network timeout scenarios — message contains model name + timeout value

- [x] **Regression tests for terminal output**
  - Inline snapshot tests for Rich terminal rendering (header, scores, diff content)
  - JSON schema regression tests
  - No-color mode regression

- [x] **Tests** — 525 tests, 100% coverage, 0 ruff errors

**v0.8 Success Metric**: CI green on all Python versions; no known edge-case crashes.

---

### Phase 9 — v1.0 Stable Release (Week 11–12)
**Goal**: Final polish, nice-to-have features, and public launch.

#### Must-Complete Tasks

- [x] **Version bump to `1.0.0`**
  - Update `llm_diff/__init__.py` and `pyproject.toml`
  - Final `CHANGELOG.md` entry for v1.0

- [ ] **Final QA pass**
  - Manual end-to-end test on macOS, Linux, Windows
  - Verify HTML reports render correctly in Chrome, Firefox, Safari
  - Verify `pip install llm-diff` from TestPyPI works cleanly

- [ ] **PyPI publish**
  - Build and publish: `pip install llm-diff`
  - Verify installation and CLI entry point on clean venv

#### Nice-to-Have Features

- [x] **Batch concurrency**
  - Currently batch items run sequentially — add `asyncio.gather` for parallel API calls across batch items
  - Add `--concurrency <n>` flag to limit parallel requests (default: 4)
  - Significant speedup for large batch files (10+ prompts)

- [ ] **Model download progress bar**
  - On first `--semantic` / `--paragraph` run, sentence-transformers downloads ~80MB model
  - Show a Rich progress bar during download instead of raw pip/torch output
  - Improves first-run UX significantly

- [x] **Result caching**
  - Cache LLM responses by (model, prompt, temperature, max_tokens) hash
  - Store in `~/.cache/llm-diff/` or configurable path
  - Add `--no-cache` flag to bypass
  - Useful for iterating on report formatting without re-calling APIs

- [x] **Cold-start performance benchmark**
  - Add benchmark script measuring CLI cold-start latency (no API call)
  - Target: < 200ms (per spec)
  - Track in CI to prevent regression

- [ ] **TypeScript / npm package**
  - Standalone TypeScript implementation (no Python dependency)
  - Same JSON output schema as Python package
  - `import { diff } from 'llm-diff'` programmatic API
  - Target Node.js 18+
  - Resolve npm name: `llm-diff` vs `@scope/llm-diff`
  - Publish to npm

- [ ] **Dependency lockfile / SBOM**
  - Generate `requirements-lock.txt` or use `pip-compile` for reproducible installs
  - Optional: SBOM generation for enterprise compliance (CycloneDX or SPDX)

#### Launch Prep

- [x] **GitHub repository polish**
  - Topics, description, social preview image
  - Issue templates, PR template
  - Branch protection rules on `main`

- [ ] **Community setup**
  - Discord server or GitHub Discussions enabled
  - Target: 50+ members at launch

**v1.0 Success Metric**: 100 GitHub stars within 2 weeks of launch.

---

### Phase 10 — v1.1 Tier 1: Evaluation Depth
**Goal**: Add the features that move llm-diff from a diff tool into a serious evaluation framework.

#### Tasks

- [ ] **LLM-as-a-Judge scoring (`--judge MODEL`)**
  - Send both responses to a third model (e.g. `gpt-4o`) with a structured evaluation prompt
  - Return: winner (`A` / `B` / `tie`), reasoning string, optional category scores (accuracy, coherence, relevance)
  - New `JudgeResult` dataclass on `ComparisonReport`
  - Expose in terminal footer, HTML report card, and JSON output
  - Cache judge calls independently (same cache key scheme)

- [ ] **Multi-model comparison (`--model-c`, `--model-d`, ...)**
  - Accept up to N models; generate an NxN similarity matrix
  - Terminal: ranked table of pairwise scores
  - HTML report: heatmap grid with colour-coded cells
  - Avoids the O(n²) manual pairwise workflow

- [ ] **Structured JSON diff mode**
  - Detect when both responses are valid JSON and diff at the key/value level
  - Report: added keys, removed keys, changed values, type changes
  - Array handling: order-sensitive and order-insensitive modes
  - Activated automatically when JSON is detected, or via `--mode json-struct`

- [ ] **Cost tracking (`--show-cost`)**
  - Built-in pricing table for known models (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, etc.)
  - Map `prompt_tokens` + `completion_tokens` to estimated USD per model
  - Display in terminal footer and HTML report
  - Update pricing table via `~/.llmdiff` override for custom/fine-tuned models

**v1.1 Success Metric**: Can answer "which model is better *and* cheaper for my use case?" in one command.

---

### Phase 11 — v1.2 Tier 2: Developer Experience
**Goal**: Make llm-diff indispensable for teams running ongoing model evaluations.

#### Tasks

- [ ] **Determinism / consistency analysis (`--runs N`)**
  - Run the same prompt on the same model N times (default 5) and report output variance
  - Metrics: mean word similarity across run pairs, standard deviation, min/max spread
  - Answers: "How stable is this model at temperature T?"
  - Useful for temperature sensitivity studies and provider reliability checks

- [ ] **Historical tracking**
  - Persist `ComparisonReport` results to a local SQLite DB (`~/.local/share/llm-diff/history.db`) or append-only JSONL file
  - CLI: `llm-diff history` — list past runs with prompt snippet, models, scores, date
  - CLI: `llm-diff history --prompt "Explain recursion"` — show trend line for a prompt over time
  - Opt-out via `--no-history` or `history = false` in `.llmdiff`

- [ ] **Side-by-side HTML layout**
  - Add a toggle button in the HTML report to switch between unified diff view and two-column side-by-side view
  - Two-column view mirrors the GitHub PR split diff experience
  - Improves readability for long-form response comparisons

- [ ] **Export to CSV / Markdown**
  - `--out results.csv` — batch results as CSV (id, word_sim, semantic_sim, bleu, rouge, judge_winner)
  - `--out results.md` — Markdown table suitable for pasting into PRs, Notion, etc.
  - Auto-detect format from file extension; `--format csv|md|html` as explicit override

- [ ] **Native Anthropic / Google / AWS SDK support**
  - Native `anthropic` SDK integration — eliminate the LiteLLM proxy requirement
  - Native `google-generativeai` SDK for Gemini models
  - Optional `boto3` support for AWS Bedrock (Claude, Titan, etc.)
  - Each added as an optional extra: `pip install "llm-diff[anthropic]"`, `pip install "llm-diff[google]"`, `pip install "llm-diff[bedrock]"`

**v1.2 Success Metric**: Teams can track model quality trends over weeks without exporting data manually.

---

### Phase 12 — v2.0 Tier 3: Polish & Ecosystem
**Goal**: Round out the feature surface and expand integration points.

#### Tasks

- [ ] **Custom scoring plugins (`--scorer path/to/scorer.py`)**
  - Plugin protocol: `def score(text_a: str, text_b: str) -> float`
  - Loaded dynamically at runtime
  - Useful for domain-specific checks: medical accuracy, code correctness, format compliance
  - Score displayed alongside built-in metrics; included in HTML reports and JSON output

- [ ] **Watch mode (`--watch`)**
  - `llm-diff --watch prompts.yml -a gpt-4o -b gpt-4o-mini` — re-runs on file changes
  - Uses `watchfiles` or stdlib `stat` polling
  - Useful during active prompt engineering sessions for instant feedback

- [ ] **Diff noise filtering**
  - `--ignore-case` — treat upper/lower-case as equal in word diff
  - `--ignore-punctuation` — strip punctuation before diffing
  - `--ignore-whitespace` — collapse whitespace differences
  - Reduce noise from stylistic variation that doesn't reflect content changes

- [ ] **Response streaming display**
  - Stream tokens from both models as they arrive instead of waiting for completion
  - Show a live side-by-side stream in the terminal while the diff is computed after both finish
  - Improves perceived responsiveness for long responses

- [ ] **Per-metric CI thresholds**
  - `--fail-under-bleu 0.4` — exit 1 if BLEU drops below threshold
  - `--fail-under-rouge 0.6` — exit 1 if ROUGE-L drops below threshold
  - `--fail-under-semantic 0.85` — explicit semantic threshold (alias for current `--fail-under` when `--semantic` is set)
  - Each gate is independent; any failing gate triggers exit 1

- [ ] **TypeScript / npm package**
  - Standalone TypeScript implementation (no Python dependency)
  - Same JSON output schema as Python package
  - `import { diff } from 'llm-diff'` programmatic API
  - Target Node.js 18+; publish to npm as `llm-diff`

**v2.0 Success Metric**: llm-diff is referenced as the standard open-source LLM eval CLI in developer tooling roundups.

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

| Week | Phase | Deliverable | Status |
|---|---|---|---|
| 1–2 | v0.1 MVP | Core diff engine, two-model CLI, colored terminal output | ✅ Complete |
| 3–4 | v0.2 Modes | Semantic scoring, HTML reports, `--save`/`--out` flags | ✅ Complete |
| 5 | v0.3 Batch | YAML batch mode, combined batch HTML report | ✅ Complete |
| 6 | v0.4 Dashboard | Paragraph-level scoring, interactive HTML features | ✅ Complete |
| 7 | v0.5 Security | Autoescape fix, dead code removal, response guards | ✅ Complete |
| 8 | v0.6 Enterprise | `--fail-under`, programmatic API, `--paragraph` in batch | ✅ Complete |
| 9 | v0.7 Docs | LICENSE, README, CHANGELOG, CONTRIBUTING, PyPI metadata | ✅ Complete |
| 10 | v0.8 Testing | Integration tests, GitHub Actions CI, edge-case hardening | ✅ Complete |
| 11–12 | v1.0 Stable | Nice-to-haves, final QA, PyPI publish, launch | ✅ Complete |
| TBD | v1.1 Tier 1 | LLM-as-a-Judge, multi-model matrix, structured JSON diff, cost tracking | ⬜ Planned |
| TBD | v1.2 Tier 2 | Consistency analysis, history tracking, side-by-side HTML, CSV/MD export, native SDKs | ⬜ Planned |
| TBD | v2.0 Tier 3 | Custom plugins, watch mode, diff filters, streaming, per-metric CI gates, npm package | ⬜ Planned |

**Total: ~12 weeks to v1.0** · Tier 1–3 phases are community-driven and tracked on the [GitHub Issues board](https://github.com/user/llm-diff/issues).
