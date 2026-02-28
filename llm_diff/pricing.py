"""Cost-tracking utilities for llm-diff.

Provides a built-in pricing table for popular models and a cost-estimation
function that converts token counts to estimated USD.  Pricing is per-million
tokens (the industry standard).

Users can override or extend the pricing table via their ``~/.llmdiff`` TOML
config file (see :func:`load_pricing_overrides`).

Usage
-----
.. code-block:: python

    from llm_diff.pricing import estimate_cost, PRICING

    cost_a = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    # cost_a -> 0.0075  (USD, rounded to 6 decimal places)

    print(PRICING.keys())
    # dict_keys(['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', ...])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table — USD per 1 million tokens
# Prices sourced from official provider pages (February 2026).
# ---------------------------------------------------------------------------

# Structure: model_key -> {"prompt": $/1M, "completion": $/1M}
PRICING: dict[str, dict[str, float]] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o":                  {"prompt": 2.50,  "completion": 10.00},
    "gpt-4o-2024-11-20":       {"prompt": 2.50,  "completion": 10.00},
    "gpt-4o-2024-08-06":       {"prompt": 2.50,  "completion": 10.00},
    "gpt-4o-mini":             {"prompt": 0.15,  "completion": 0.60},
    "gpt-4o-mini-2024-07-18":  {"prompt": 0.15,  "completion": 0.60},
    "gpt-4-turbo":             {"prompt": 10.00, "completion": 30.00},
    "gpt-4-turbo-preview":     {"prompt": 10.00, "completion": 30.00},
    "gpt-4":                   {"prompt": 30.00, "completion": 60.00},
    "gpt-3.5-turbo":           {"prompt": 0.50,  "completion": 1.50},
    "gpt-3.5-turbo-0125":      {"prompt": 0.50,  "completion": 1.50},
    "o1":                      {"prompt": 15.00, "completion": 60.00},
    "o1-mini":                 {"prompt": 3.00,  "completion": 12.00},
    "o1-preview":              {"prompt": 15.00, "completion": 60.00},
    "o3-mini":                 {"prompt": 1.10,  "completion": 4.40},

    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-3-5-sonnet-20241022": {"prompt": 3.00,  "completion": 15.00},
    "claude-3-5-sonnet-20240620": {"prompt": 3.00,  "completion": 15.00},
    "claude-3-5-sonnet":          {"prompt": 3.00,  "completion": 15.00},
    "claude-3-5-haiku":           {"prompt": 0.80,  "completion": 4.00},
    "claude-3-5-haiku-20241022":  {"prompt": 0.80,  "completion": 4.00},
    "claude-3-opus":              {"prompt": 15.00, "completion": 75.00},
    "claude-3-opus-20240229":     {"prompt": 15.00, "completion": 75.00},
    "claude-3-sonnet":            {"prompt": 3.00,  "completion": 15.00},
    "claude-3-haiku":             {"prompt": 0.25,  "completion": 1.25},

    # ── Groq (hosted inference) ──────────────────────────────────────────────
    "llama-3.3-70b-versatile":    {"prompt": 0.59,  "completion": 0.79},
    "llama-3.1-8b-instant":       {"prompt": 0.05,  "completion": 0.08},
    "llama-3.1-70b-versatile":    {"prompt": 0.59,  "completion": 0.79},
    "mixtral-8x7b-32768":         {"prompt": 0.24,  "completion": 0.24},
    "gemma2-9b-it":               {"prompt": 0.20,  "completion": 0.20},

    # ── Mistral AI ───────────────────────────────────────────────────────────
    "mistral-large-latest":       {"prompt": 3.00,  "completion": 9.00},
    "mistral-small-latest":       {"prompt": 0.20,  "completion": 0.60},
    "codestral-latest":           {"prompt": 0.70,  "completion": 2.10},
    "open-mistral-nemo":          {"prompt": 0.15,  "completion": 0.15},

    # ── DeepSeek ─────────────────────────────────────────────────────────────
    "deepseek-chat":              {"prompt": 0.27,  "completion": 1.10},
    "deepseek-reasoner":          {"prompt": 0.55,  "completion": 2.19},

    # ── Google Gemini ────────────────────────────────────────────────────────
    "gemini-1.5-pro":             {"prompt": 1.25,  "completion": 5.00},
    "gemini-1.5-flash":           {"prompt": 0.075, "completion": 0.30},
    "gemini-2.0-flash-exp":       {"prompt": 0.075, "completion": 0.30},
}

# Alias map so partial model names match the pricing table.
# Keys are lower-cased prefixes; values are canonical PRICING keys.
_PREFIX_ALIASES: list[tuple[str, str]] = [
    ("claude-3-5-sonnet", "claude-3-5-sonnet"),
    ("claude-3-5-haiku",  "claude-3-5-haiku"),
    ("claude-3-opus",     "claude-3-opus"),
    ("claude-3-sonnet",   "claude-3-sonnet"),
    ("claude-3-haiku",    "claude-3-haiku"),
    ("gpt-4o-mini",       "gpt-4o-mini"),
    ("gpt-4o",            "gpt-4o"),
    ("gpt-4-turbo",       "gpt-4-turbo"),
    ("gpt-4",             "gpt-4"),
    ("gpt-3.5-turbo",     "gpt-3.5-turbo"),
    ("o3-mini",           "o3-mini"),
    ("o1-mini",           "o1-mini"),
    ("o1-preview",        "o1-preview"),
    ("o1",                "o1"),
    ("mistral-large",     "mistral-large-latest"),
    ("mistral-small",     "mistral-small-latest"),
    ("codestral",         "codestral-latest"),
    ("deepseek-chat",     "deepseek-chat"),
    ("deepseek-reasoner", "deepseek-reasoner"),
    ("gemini-1.5-pro",    "gemini-1.5-pro"),
    ("gemini-1.5-flash",  "gemini-1.5-flash"),
    ("gemini-2.0-flash",  "gemini-2.0-flash-exp"),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CostEstimate:
    """USD cost estimate for a single model call.

    Attributes
    ----------
    model:
        The model identifier.
    prompt_tokens:
        Number of prompt (input) tokens consumed.
    completion_tokens:
        Number of completion (output) tokens generated.
    total_usd:
        Estimated total cost in USD.
    prompt_usd:
        Prompt-only component of the cost.
    completion_usd:
        Completion-only component of the cost.
    known_model:
        ``True`` if pricing was found in the built-in table.
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_usd: float
    prompt_usd: float
    completion_usd: float
    known_model: bool = True

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_usd": round(self.total_usd, 6),
            "prompt_usd": round(self.prompt_usd, 6),
            "completion_usd": round(self.completion_usd, 6),
            "known_model": self.known_model,
        }

    @property
    def total_usd_str(self) -> str:
        """Human-readable cost string (e.g. ``'$0.000250'``)."""
        if self.total_usd < 0.000001:
            return "$0.00"
        if self.total_usd < 0.01:
            return f"${self.total_usd:.6f}"
        return f"${self.total_usd:.4f}"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def _lookup_pricing(model: str) -> dict[str, float] | None:
    """Return the pricing dict for *model*, or ``None`` if not found.

    Performs an exact lookup first, then tries lower-cased prefix aliases
    so that version-suffixed names (e.g. ``"gpt-4o-2024-11-20"``) still
    match the canonical entry.
    """
    # 1. Exact match (case-sensitive — model names are case-sensitive)
    if model in PRICING:
        return PRICING[model]

    # 2. Lowercase suffix-aware alias search
    mlower = model.lower()
    for prefix, canonical_key in _PREFIX_ALIASES:
        if mlower.startswith(prefix):
            return PRICING.get(canonical_key)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_cost(
    model: str,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    overrides: dict[str, dict[str, float]] | None = None,
) -> CostEstimate:
    """Estimate the USD cost of a single model call.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``"gpt-4o"``).
    prompt_tokens:
        Number of prompt / input tokens.
    completion_tokens:
        Number of completion / output tokens.
    overrides:
        Optional ``{model: {prompt: price, completion: price}}`` mapping that
        takes precedence over the built-in table.  Useful for fine-tuned or
        private models.

    Returns
    -------
    CostEstimate
        A :class:`CostEstimate` where ``total_usd = 0.0`` and
        ``known_model = False`` if the model is not in the pricing table.
    """
    pricing_entry: dict[str, float] | None = None
    known = True

    # Check overrides first
    if overrides:
        pricing_entry = overrides.get(model) or _lookup_in(overrides, model)

    # Fall back to built-in table
    if pricing_entry is None:
        pricing_entry = _lookup_pricing(model)

    if pricing_entry is None:
        known = False
        logger.debug(
            "No pricing entry found for model '%s' — cost will be $0.00", model
        )
        return CostEstimate(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_usd=0.0,
            prompt_usd=0.0,
            completion_usd=0.0,
            known_model=False,
        )

    prompt_usd = (prompt_tokens / 1_000_000) * pricing_entry["prompt"]
    completion_usd = (completion_tokens / 1_000_000) * pricing_entry["completion"]
    total_usd = prompt_usd + completion_usd

    return CostEstimate(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_usd=total_usd,
        prompt_usd=prompt_usd,
        completion_usd=completion_usd,
        known_model=known,
    )


def _lookup_in(table: dict[str, dict[str, float]], model: str) -> dict[str, float] | None:
    """Look up *model* in an arbitrary pricing table (exact match only)."""
    return table.get(model)


def format_cost_table(
    cost_a: CostEstimate,
    cost_b: CostEstimate,
) -> list[tuple[str, str, str]]:
    """Return rows for a cost comparison table.

    Each row is ``(label, value_a, value_b)``.
    """
    rows = [
        ("Model",              cost_a.model,                 cost_b.model),
        ("Prompt tokens",      str(cost_a.prompt_tokens),    str(cost_b.prompt_tokens)),
        ("Completion tokens",  str(cost_a.completion_tokens), str(cost_b.completion_tokens)),
        ("Prompt cost",        cost_a.total_usd_str,         cost_b.total_usd_str),
    ]
    return rows
