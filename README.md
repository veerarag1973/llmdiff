# llm-diff

**A CLI tool and Python library for comparing LLM outputs — semantically, visually, and at scale.**

[![PyPI](https://img.shields.io/pypi/v/llm-diff)](https://pypi.org/project/llm-diff/)
[![CI](https://img.shields.io/github/actions/workflow/status/veerarag1973/llm-diff/ci.yml?branch=main)](https://github.com/veerarag1973/llm-diff/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/veerarag1973/llm-diff)
[![Python](https://img.shields.io/pypi/pyversions/llm-diff)](https://pypi.org/project/llm-diff/)
[![License](https://img.shields.io/pypi/l/llm-diff)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--stable-brightgreen)](CHANGELOG.md)

---

`llm-diff` calls two LLM models in parallel, diffs their responses word-by-word,
scores them semantically, and renders results in the terminal or as a
self-contained HTML report.  It scales to batch workloads, caches API responses,
and gates CI pipelines via `--fail-under`.

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

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, API keys, first diff |
| [CLI Reference](docs/cli-reference.md) | All flags, option groups, exit codes, YAML format |
| [Python API](docs/api.md) | All public functions, dataclasses, and field descriptions |
| [Configuration](docs/configuration.md) | `.llmdiff` TOML schema, env vars, config priority |
| [Provider Setup](docs/providers.md) | OpenAI, Groq, Mistral, Ollama, LM Studio, Anthropic |
| [HTML Reports](docs/html-reports.md) | Report anatomy, batch reports, judge card, cost table |
| [CI / CD Integration](docs/ci-cd.md) | GitHub Actions examples, threshold recommendations |

## Quick Start

```bash
# Install with semantic scoring support
pip install "llm-diff[semantic]"

# Set an API key
export OPENAI_API_KEY="sk-..."

# Compare two models on the same prompt
llm-diff "Explain recursion in one sentence." -a gpt-4o -b gpt-4o-mini --semantic

# Save a self-contained HTML report
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini --semantic --out report.html

# Run a batch from a YAML prompt file and gate on similarity
llm-diff --batch prompts.yml -a gpt-4o -b gpt-4o-mini --semantic --fail-under 0.85
```

See [Getting Started](docs/getting-started.md) for more examples including
prompt-diff mode, BLEU/ROUGE metrics, LLM-as-a-Judge, cost tracking, and
multi-model comparison.

## Getting Help

| | |
|---|---|
| **Bug reports** | [Open an issue](https://github.com/veerarag1973/llm-diff/issues/new?labels=bug&template=bug_report.md) |
| **Feature requests** | [Open a feature request](https://github.com/veerarag1973/llm-diff/issues/new?labels=enhancement&template=feature_request.md) |
| **Questions & discussion** | [GitHub Discussions](https://github.com/veerarag1973/llm-diff/discussions) |
| **Open issues** | [github.com/veerarag1973/llm-diff/issues](https://github.com/veerarag1973/llm-diff/issues) |
| **Roadmap** | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |

When filing a bug, please include: `llm-diff --version`, your OS, Python
version, the full command you ran, and the complete error output.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, running the test
suite, code style guidelines, and pull request instructions.

## License

`llm-diff` is distributed under the [MIT License](LICENSE).