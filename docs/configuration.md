# Configuration Reference

`llm-diff` reads settings from multiple sources in the following priority order
(highest to lowest):

| Priority | Source |
|----------|--------|
| 1 (highest) | CLI flags (`--temperature`, `--max-tokens`, …) |
| 2 | Project config — `.llmdiff` in the current working directory |
| 3 | User config — `~/.llmdiff` |
| 4 | `.env` file in the current working directory |
| 5 (lowest) | Environment variables |

---

## The `.llmdiff` config file

Create `.llmdiff` in your project root or `~/.llmdiff` for global defaults.
The format is [TOML](https://toml.io/).

### Full schema with defaults

```toml
# ── Provider credentials ──────────────────────────────────────────────────────

[providers.openai]
api_key  = ""           # overrides OPENAI_API_KEY env var
base_url = ""           # optional: custom endpoint (e.g. Azure OpenAI)

[providers.groq]
api_key  = ""           # overrides GROQ_API_KEY env var

[providers.mistral]
api_key  = ""           # overrides MISTRAL_API_KEY env var

[providers.deepseek]
api_key  = ""           # overrides DEEPSEEK_API_KEY env var

[providers.custom]
api_key  = ""           # any key required by the endpoint
base_url = ""           # required: e.g. "http://localhost:11434/v1"

# ── Request defaults ──────────────────────────────────────────────────────────

[defaults]
temperature = 0.7       # float 0.0–2.0; controls randomness
max_tokens  = 1024      # int; maximum tokens per response
timeout     = 30        # int; request timeout in seconds

# ── Output defaults ───────────────────────────────────────────────────────────

save     = false        # bool; auto-save every HTML report to ./diffs/
no_color = false        # bool; disable terminal colour output
```

All keys are optional. Omitted keys fall back to the next priority level.

---

## Environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI models (`gpt-*`, `o1-*`, `o3-*`) |
| `GROQ_API_KEY` | API key for Groq models (`llama-*`, `mixtral-*`, `gemma-*`) |
| `MISTRAL_API_KEY` | API key for Mistral AI models (`mistral-*`, `codestral-*`) |
| `DEEPSEEK_API_KEY` | API key for DeepSeek models (`deepseek-*`) |
| `ANTHROPIC_API_KEY` | API key for Anthropic models via LiteLLM proxy |
| `LLM_DIFF_NO_COLOR` | Set to `1` to disable colour output |

Export them in your shell:

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk_..."
```

Windows (PowerShell):

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

Or add them to a `.env` file in your project root:

```dotenv
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

---

## Provider auto-detection

`llm-diff` infers which provider to use from the model name.  No `--provider`
flag is needed:

| Model name prefix | Provider | API key variable |
|-------------------|----------|-----------------|
| `gpt-*`, `o1-*`, `o3-*` | OpenAI | `OPENAI_API_KEY` |
| `llama*`, `mixtral*`, `gemma*` | Groq | `GROQ_API_KEY` |
| `mistral*`, `codestral*` | Mistral AI | `MISTRAL_API_KEY` |
| `deepseek*` | DeepSeek | `DEEPSEEK_API_KEY` |
| `claude*` | LiteLLM proxy (Anthropic) | `ANTHROPIC_API_KEY` |
| anything else | Custom (requires `base_url`) | — |

You can always override auto-detection by setting an explicit `base_url` in
your `.llmdiff` file.

---

## Per-provider examples

### OpenAI

```toml
[providers.openai]
api_key = "sk-..."

[defaults]
temperature = 0.7
max_tokens  = 1024
```

### OpenAI-compatible proxy / Azure OpenAI

```toml
[providers.openai]
api_key  = "sk-..."
base_url = "https://my-resource.openai.azure.com/openai/deployments/gpt-4o"
```

### Groq

```toml
[providers.groq]
api_key = "gsk_..."
```

Model names: `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, etc.

### Mistral AI

```toml
[providers.mistral]
api_key = "..."
```

Model names: `mistral-large-latest`, `mistral-small-latest`, `codestral-latest`.

### DeepSeek

```toml
[providers.deepseek]
api_key = "..."
```

Model names: `deepseek-chat`, `deepseek-coder`.

### Anthropic (via LiteLLM proxy)

Start a LiteLLM proxy that translates the OpenAI API format to Anthropic:

```bash
pip install litellm
litellm --model anthropic/claude-3-5-sonnet-20241022 --port 4000
```

Then configure:

```toml
[providers.custom]
api_key  = "sk-ant-..."
base_url = "http://localhost:4000/v1"
```

### Ollama (local)

```bash
ollama pull llama3.2
ollama serve
```

```toml
[providers.custom]
api_key  = "ollama"          # Ollama ignores the key; the SDK requires one
base_url = "http://localhost:11434/v1"
```

```bash
llm-diff "Explain recursion." -a llama3.2 -b mistral
```

### LM Studio (local)

Enable the local server in LM Studio (default port 1234):

```toml
[providers.custom]
api_key  = "lm-studio"
base_url = "http://localhost:1234/v1"
```

---

## Config file locations

| Path | Purpose |
|------|---------|
| `./.llmdiff` | Project-level config (checked in or git-ignored) |
| `~/.llmdiff` | User-level global config |

The project-level file takes precedence over the user-level file.

### Recommended `.gitignore` entry

Config files often contain API keys. Add the project-level file to
`.gitignore`:

```
.llmdiff
.env
```

---

## Response caching

By default, `llm-diff` caches every API response to `~/.cache/llm-diff/`.
The cache key is: `(model, prompt, temperature, max_tokens)`.

Re-running the same comparison is instant and free:

```bash
# First run — calls the API (latency: ~0.8s)
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini

# Second run — served from disk cache (latency: ~0.0s)
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini
```

Bypass the cache when you want a fresh API response:

```bash
llm-diff "Explain recursion." -a gpt-4o -b gpt-4o-mini --no-cache
```

Cache location: `~/.cache/llm-diff/`

The cache uses JSON files keyed by a SHA-256 hash of the request parameters.
It is safe to delete the entire cache directory at any time.

---

## Observability — Schema Events

`llm-diff` emits structured
[AgentOBS](https://pypi.org/project/agentobs/) events for
every significant operation.  By default events are built, validated, and
**discarded** (zero overhead for existing users).  To opt in, call
`configure_emitter()` once before running any comparisons.

### Collect events in memory

```python
from llm_diff.schema_events import configure_emitter, get_emitter

configure_emitter()    # keep events in memory

# ... run comparisons ...

for evt in get_emitter().events:
    print(evt.event_type, evt.timestamp)
```

### Export to JSONL

```python
from agentobs.export.jsonl import JSONLExporter
from llm_diff.schema_events import configure_emitter

configure_emitter(exporter=JSONLExporter("llm-diff-events.jsonl"))
```

Each line of the JSONL file is a self-contained JSON object conforming to the
`agentobs` envelope.

### Export to a custom sink

Any callable (or object with an `.export(event)` method) works as an exporter:

```python
def my_sink(event):
    requests.post("https://events.example.com", json=event.to_dict())

configure_emitter(exporter=my_sink)
```

### Event types emitted

| Event type | Emitted when |
|------------|--------------|
| `llm.diff.comparison.started` | `compare()` / `compare_prompts()` is called |
| `llm.diff.comparison.completed` | Diff result is ready |
| `llm.diff.report.exported` | `save_report()` writes a file |
| `llm.trace.span.completed` | Each model API call returns |
| `llm.cache.hit` | A cached response is served |
| `llm.cache.miss` | No cache entry found; API is called |
| `llm.cost.recorded` | `show_cost=True` and cost is estimated |
| `llm.eval.scenario.completed` | LLM-as-a-Judge scoring finishes |

See [Schema Events](schema-events.md) for the full guide and payload field
reference.
