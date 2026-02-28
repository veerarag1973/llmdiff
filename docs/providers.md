# Provider Configuration Guide

llm-diff uses the `openai` SDK with a `base_url` override to support any
OpenAI-compatible endpoint.  This means it works natively with OpenAI and any
service that exposes an OpenAI-compatible chat completions API (Groq, Together,
Ollama, LM Studio, etc.).

Anthropic's native API has a different format.  To use Anthropic models, route
them through an OpenAI-compatible proxy such as
[LiteLLM](https://docs.litellm.ai/) or the
[Anthropic OpenAI-compatible endpoint](https://docs.anthropic.com/en/api/openai-sdk)
and configure the `base_url` accordingly.

---

## Configuration Priority

Settings are applied in this order (highest wins):

1. **CLI flags** — `--temperature`, `--max-tokens`, `--timeout`, etc.
2. **Project `.llmdiff`** — TOML file in the current working directory
3. **User `~/.llmdiff`** — TOML file in your home directory
4. **`.env` file** — in the current working directory
5. **Environment variables** — `OPENAI_API_KEY`, etc.

---

## Example `.llmdiff` Config Files

### OpenAI

```toml
[providers.openai]
api_key = "sk-..."          # or set OPENAI_API_KEY env var

[defaults]
temperature = 0.7
max_tokens  = 1024
timeout     = 30
```

### Anthropic via LiteLLM proxy

Run LiteLLM locally:

```bash
pip install litellm
litellm --model anthropic/claude-3-5-sonnet-20241022 --port 4000
```

Then configure llm-diff:

```toml
[providers.custom]
api_key  = "sk-ant-..."
base_url = "http://localhost:4000/v1"

[defaults]
temperature = 0.7
max_tokens  = 1024
```

Usage:

```bash
llm-diff "Explain recursion" -a gpt-4o -b claude-3-5-sonnet-20241022 \
  --model-b "claude-3-5-sonnet-20241022"
```

### Groq

```toml
[providers.groq]
api_key = "gsk_..."         # or set GROQ_API_KEY env var

[defaults]
temperature = 0.7
max_tokens  = 1024
```

Groq model names are auto-detected (e.g. `llama-3.3-70b-versatile`,
`mixtral-8x7b-32768`).

### Ollama (local)

Start Ollama with a model:

```bash
ollama pull llama3.2
ollama serve
```

Configure llm-diff:

```toml
[providers.custom]
api_key  = "ollama"         # Ollama ignores the key but the SDK requires one
base_url = "http://localhost:11434/v1"
```

Usage:

```bash
llm-diff "Explain recursion" -a llama3.2 -b mistral
```

### LM Studio (local)

Enable the local server in LM Studio (default port 1234):

```toml
[providers.custom]
api_key  = "lm-studio"
base_url = "http://localhost:1234/v1"
```

### Mistral AI

```toml
[providers.mistral]
api_key = "..."             # or set MISTRAL_API_KEY env var
```

Mistral model names are auto-detected (e.g. `mistral-large-latest`,
`mistral-small-latest`).

---

## Provider Auto-Detection

llm-diff infers the provider from the model name prefix:

| Model prefix         | Provider  | API key env var        |
|----------------------|-----------|------------------------|
| `gpt-*`, `o1-*`, `o3-*` | openai | `OPENAI_API_KEY`   |
| `claude-*`           | anthropic | `ANTHROPIC_API_KEY`    |
| `mistral-*`          | mistral   | `MISTRAL_API_KEY`      |
| `llama-*`, `mixtral-*` | groq    | `GROQ_API_KEY`         |
| anything else        | custom    | (uses `base_url`)      |

You can always override this with an explicit `base_url` in your `.llmdiff`
file or by routing through a proxy.

---

## CI / CD Integration

Use `--fail-under` to gate pipelines on similarity thresholds:

```bash
# Fail if semantic similarity drops below 85% between model versions
llm-diff "Summarize this contract." \
  -a gpt-4o \
  -b gpt-4o-mini \
  --semantic \
  --fail-under 0.85
echo "Exit code: $?"   # 0 = passed, 1 = below threshold
```

Batch mode also respects `--fail-under` — any single item below the
threshold causes exit code 1:

```bash
llm-diff --batch regression_prompts.yml \
  -a gpt-4o -b gpt-4o-mini \
  --semantic --fail-under 0.80
```
