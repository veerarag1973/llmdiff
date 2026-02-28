# Changelog

All notable changes to **llm-diff** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [1.0.0] — 2026-03-01

### Added
- **Batch concurrency** (`--concurrency INT`, default 4) — `_run_batch` now
  fires all API calls concurrently via `asyncio.Semaphore` + `asyncio.gather`,
  cutting wall-clock time for large batch files by up to `N×` (where N is the
  concurrency limit).
- **Response caching** (`--no-cache` flag disables) — `ResultCache` persists
  `ModelResponse` objects on disk under `~/.cache/llm-diff/` keyed by
  SHA-256(`model + prompt + temperature + max_tokens`).  Re-running the same
  prompt/model combination returns the cached result without an API call.
  Pass `--no-cache` to bypass both reads and writes.
- `llm_diff.cache.ResultCache` — public class for programmatic cache access.
- Comprehensive `tests/test_cache.py` achieving 100 % branch coverage of
  `llm_diff/cache.py`.

### Changed
- Version bumped to **1.0.0** (first stable release).
- PyPI classifier updated from `Development Status :: 4 - Beta` to
  `Development Status :: 5 - Production/Stable`.

### Fixed
- Nothing

---

## [0.8.0] — 2026-02-28

### Added
- **Integration tests** (`tests/test_integration.py`) — full end-to-end CLI
  pipeline tests; `AsyncOpenAI` mocked at the class level (no real HTTP calls);
  covers word mode, JSON mode, HTML report, verbose mode, batch mode, fail-under,
  and error-handling paths (~55 tests)
- **Edge-case tests** (`tests/test_edge_cases.py`) — empty/whitespace model
  responses, Unicode/emoji/CJK/Arabic inputs, 10 000-word performance regression
  guard, network timeout message assertions, malformed YAML surfaced via CLI (~40 tests)
- **Regression tests** (`tests/test_regression.py`) — snapshot-style checks for
  terminal output structure (header, similarity %, word counts, diff chunks) and
  JSON schema stability (~35 tests)
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — matrix build on Python
  3.9–3.13; runs ruff → pytest (100% coverage gate); triggers on push/PR to `main`
- **GitHub Actions publish** (`.github/workflows/publish.yml`) — builds wheel +
  sdist and publishes to PyPI via OIDC trusted publisher on version tags (`v*`)

### Changed
- `providers._call_model`: logs a `WARNING` when a model returns an empty or
  whitespace-only response instead of silently returning `text=""`
- Version bumped to **0.8.0**

### Fixed
- Nothing

---

## [0.7.0] — 2026-02-28

### Added
- `LICENSE` file (MIT)
- Comprehensive `README.md` with installation, quick-start, CLI reference,
  provider guide, batch format, programmatic API, and CI/CD examples
- `CHANGELOG.md` (this file) in Keep a Changelog format
- `CONTRIBUTING.md` — dev setup, testing workflow, architecture overview
- `examples/prompts.yml` — working batch example with `{input}` templates
- `examples/inputs/` — sample input files used by the example batch

### Changed
- `pyproject.toml`: classifier upgraded from `3 - Alpha` to `4 - Beta`;
  Python 3.13 added to classifier list; `build` and `twine` added to dev deps
- Version bumped to **0.7.0**

### Fixed
- Nothing

---

## [0.6.0] — 2026-02-28

### Added
- **`--fail-under <threshold>` flag** — exit code 1 when similarity is below
  the given 0.0–1.0 threshold; works in both single-diff and batch mode
- **Programmatic Python API** (`llm_diff/api.py`):
  - `async def compare(...)` → `ComparisonReport`
  - `async def compare_prompts(...)` → `ComparisonReport`
  - `async def compare_batch(...)` → `list[ComparisonReport]`
  - `ComparisonReport` dataclass with `word_similarity` and `primary_score`
    convenience properties
  - Re-exported from `llm_diff` for `from llm_diff import compare` usage
- **`--paragraph` support in batch mode** — paragraph-level scores computed
  and stored per batch item; rendered in the batch HTML report
- **`docs/providers.md`** — full provider configuration guide with `.llmdiff`
  snippets for OpenAI, Anthropic/LiteLLM, Groq, Ollama, LM Studio, Mistral
- `BatchResult.paragraph_scores` field

### Changed
- Version bumped to **0.6.0**

---

## [0.5.0] — 2026-02-28

### Security / Cleanup
- **Jinja2 autoescape** enabled on `Environment` — prevents XSS if user
  content is ever rendered in HTML reports
- Removed dead code paths and unused internal helpers
- Removed telemetry / usage-reporting stubs (never shipped, but removed for
  clarity)
- Added response content guards — empty or None model responses now raise
  `ValueError` with a clear message instead of silently producing broken diffs
- Logging hardened — `httpx` / `openai` / `sentence_transformers` loggers
  clamped to `WARNING` in all modes to prevent API-key leakage in verbose logs

### Changed
- Version bumped to **0.5.0**
- Test count: 334 tests, 100 % line coverage

---

## [0.4.0] — 2026-02-28

### Added
- **Paragraph-level semantic scoring** — `--paragraph` / `-p` flag computes
  per-paragraph cosine similarity and renders a collapsible table in both the
  terminal and HTML report
- `compute_paragraph_similarity()` in `llm_diff/semantic.py`
- `ParagraphScore` dataclass (frozen, `index`, `text_a`, `text_b`, `score`)
- Interactive HTML report enhancements:
  - Collapsible side-by-side response panel (`toggleResponses`)
  - Collapsible deletion spans (`toggleDeletions`)
  - Dark-mode–friendly CSS custom properties
  - Paragraph similarity table with colour-coded score pills

### Changed
- Version bumped to **0.4.0**

---

## [0.3.0] — 2026-02-28

### Added
- **Batch mode** (`--batch PATH`) — load a `prompts.yml` file, run each
  prompt against both models, render terminal output, and optionally write
  a combined HTML report via `--out`
- `llm_diff/batch.py` — `BatchItem`, `BatchResult`, `load_batch()`
- `{input}` template expansion — each prompt can reference external text
  files; one batch item is generated per input file
- `build_batch_report()` in `llm_diff/report.py` — Jinja2-based combined
  HTML report for batch runs
- `batch_report.html.j2` template with summary table and per-item diff cards

### Changed
- Version bumped to **0.3.0**

---

## [0.2.0] — 2026-02-28

### Added
- **Semantic similarity** — `--semantic` / `-s` flag computes whole-text
  cosine similarity using `sentence-transformers` (`all-MiniLM-L6-v2`)
- `llm_diff/semantic.py` — `compute_semantic_similarity()`
- **HTML report generation** — `--out PATH` saves a fully self-contained
  single-file HTML diff report
- **`--save` flag** — auto-save report to `./diffs/` with a timestamped,
  model-named filename
- `build_report()` and `save_report()` / `auto_save_report()` in
  `llm_diff/report.py`
- `report.html.j2` — self-contained Jinja2 template (inline CSS + JS)
- **JSON output mode** — `--json` / `-j` outputs a structured JSON object to
  stdout (diff chunks, similarity, token counts, latency)
- **Prompt-diff mode** — `--prompt-a` / `--prompt-b` + `--model` diffs two
  different prompts against the same model
- `llm_diff/config.py` — `LLMDiffConfig` dataclass, `load_config()` reads
  `.llmdiff` TOML + `.env` + environment variables

### Changed
- Version bumped to **0.2.0**

---

## [0.1.0] — 2026-02-28

### Added
- Initial release
- `llm_diff/diff.py` — `word_diff()` using `difflib.SequenceMatcher`;
  `DiffChunk` (op, text) and `DiffResult` (chunks, similarity) dataclasses
- `llm_diff/providers.py` — `compare_models()` async function; concurrent
  requests via `asyncio.gather`; `ModelResponse` and `ComparisonResult`
  dataclasses; provider auto-detection (OpenAI, Groq, Mistral, DeepSeek)
- `llm_diff/renderer.py` — `render_diff()` Rich terminal renderer with
  coloured word-diff, similarity score, and token/latency metadata
- `llm_diff/cli.py` — Click entry point (`llm-diff`); `--model-a` / `-a`,
  `--model-b` / `-b`; `--verbose`, `--no-color`
- `pyproject.toml` — Hatchling build; `[semantic]` and `[dev]` extras

[Unreleased]: https://github.com/user/llm-diff/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/user/llm-diff/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/user/llm-diff/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/user/llm-diff/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/user/llm-diff/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/user/llm-diff/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/user/llm-diff/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/user/llm-diff/releases/tag/v0.1.0
