# llm-diff

**A CLI tool and Python library for comparing LLM outputs — semantically, visually, and at scale.**

[![PyPI](https://img.shields.io/pypi/v/llm-diff?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/llm-diff/)
[![CI](https://github.com/veerarag1973/llmdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/veerarag1973/llmdiff/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/veerarag1973/llmdiff/branch/main/graph/badge.svg)](https://codecov.io/gh/veerarag1973/llmdiff)
[![Python](https://img.shields.io/pypi/pyversions/llm-diff)](https://pypi.org/project/llm-diff/)
[![License](https://img.shields.io/pypi/l/llm-diff)](LICENSE)

---

`llm-diff` calls two LLM models in parallel, diffs their responses word-by-word,
scores them semantically, and renders results in the terminal or as a
self-contained HTML report.  It scales to batch workloads, caches API responses,
gates CI pipelines via `--fail-under`, and emits structured
[llm-toolkit-schema](https://pypi.org/project/llm-toolkit-schema/) events for
observability tooling.

## What is llm-diff?

LLMs do not produce deterministic output.  Evaluating models, iterating on
prompts, or assessing the impact of a model upgrade all require you to compare
responses — and doing that by hand does not scale.

`llm-diff` automates the entire workflow: it calls both models concurrently,
produces a word-level diff, optionally scores semantic similarity via sentence
embeddings, and outputs results to the terminal or as a shareable HTML report.
It supports batch workloads from a YAML file, caches API calls so iterating on
thresholds costs nothing, and emits exit code 1 when similarity falls below a
threshold — making it a first-class citizen in CI/CD pipelines.

Version 1.2 adds LLM-as-a-Judge scoring, per-call USD cost tracking,
multi-model (3–4 model) comparison, and structured JSON diff.

Version 1.3.0 adds `EVAL_REGRESSION_FAILED` schema event emission — `--fail-under`
gate failures now emit a structured `llm.eval.regression.failed` event (via
`make_eval_regression_event()`) in addition to returning exit code 1,
providing a full audit trail for CI regression gates.

Version 1.2.2 integrates [llm-toolkit-schema](https://pypi.org/project/llm-toolkit-schema/)
as a built-in observability layer: every comparison, model call, cache lookup,
cost record, judge evaluation, and `--fail-under` regression failure now emits a
validated schema event that can be collected in memory, exported to JSONL, or
forwarded to any custom backend.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, API keys, first diff |
| [Tutorials](docs/tutorials/README.md) | Step-by-step learning path from first run to Python API (12 tutorials) |
| [CLI Reference](docs/cli-reference.md) | All flags, option groups, exit codes, YAML format |
| [Python API](docs/api.md) | All public functions, dataclasses, and field descriptions |
| [Schema Events](docs/schema-events.md) | Observability integration with llm-toolkit-schema |
| [Configuration](docs/configuration.md) | `.llmdiff` TOML schema, env vars, config priority |
| [Provider Setup](docs/providers.md) | OpenAI, Groq, Mistral, Ollama, LM Studio, Anthropic |
| [HTML Reports](docs/html-reports.md) | Report anatomy, batch reports, judge card, cost table |
| [CI / CD Integration](docs/ci-cd.md) | GitHub Actions examples, threshold recommendations |

## Quick Start

```bash
# Install with semantic scoring support
pip install "llm-diff[semantic]"

# Install with schema-events observability
pip install "llm-diff[semantic]" llm-toolkit-schema

# Set an API key
export OPENAI_API_KEY="sk-..."

# Compare two models on the same prompt
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --semantic

# Save a self-contained HTML report
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini --semantic --out report.html

# Run a batch from a YAML prompt file and gate on similarity
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini --semantic --fail-under 0.85
```

See [Getting Started](docs/getting-started.md) for quick examples, or work through the
[Tutorials](docs/tutorials/README.md) for a guided learning path covering prompt engineering,
batch evaluation, CI/CD gating, LLM-as-a-Judge, cost tracking, and the Python API.

## Getting Help

| | |
|---|---|
| **Bug reports** | [Open an issue](https://github.com/veerarag1973/llmdiff/issues/new?labels=bug&template=bug_report.md) |
| **Feature requests** | [Open a feature request](https://github.com/veerarag1973/llmdiff/issues/new?labels=enhancement&template=feature_request.md) |
| **Questions & discussion** | [GitHub Discussions](https://github.com/veerarag1973/llmdiff/discussions) |
| **Open issues** | [github.com/veerarag1973/llmdiff/issues](https://github.com/veerarag1973/llmdiff/issues) |
| **PyPI project page** | [pypi.org/project/llm-diff](https://pypi.org/project/llm-diff/) |
| **Roadmap** | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |

When filing a bug, please include: `llm-diff --version`, your OS, Python
version, the full command you ran, and the complete error output.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running the test
suite, code style guidelines, and pull request instructions.

## License

`llm-diff` is distributed under the [MIT License](LICENSE).