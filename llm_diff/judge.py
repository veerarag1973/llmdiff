"""LLM-as-a-Judge scoring for llm-diff.

Sends both model responses to a third "judge" model with a structured
evaluation prompt and parses the result into a :class:`JudgeResult`.

The judge is prompted with a JSON-output constraint so the result can be
reliably parsed, with a plain-text fallback for models that ignore it.

Usage
-----
.. code-block:: python

    import asyncio
    from llm_diff.judge import run_judge

    result = asyncio.run(
        run_judge(
            prompt="Explain recursion in one sentence.",
            response_a="Recursion is a function calling itself.",
            response_b="Recursion occurs when a function calls itself repeatedly.",
            judge_model="gpt-4o",
        )
    )
    print(result.winner)    # "A", "B", or "tie"
    print(result.reasoning)
    print(result.score_a, result.score_b)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from llm_diff.config import LLMDiffConfig, load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are an expert AI evaluator comparing two assistant responses to the same prompt.
Evaluate Response A and Response B, then return ONLY a valid JSON object — no markdown,
no explanation outside the JSON.

Required JSON structure:
{
  "winner": "A" or "B" or "tie",
  "score_a": <integer 1-10>,
  "score_b": <integer 1-10>,
  "reasoning": "<one or two sentences>"
}

Scoring criteria (in priority order):
1. Accuracy and factual correctness
2. Completeness — does it fully address the prompt?
3. Clarity and coherence
4. Conciseness — penalise unnecessary padding or repetition
"""

_JUDGE_USER_TMPL = """\
Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JudgeResult:
    """Result from the LLM-as-a-Judge evaluation.

    Attributes
    ----------
    winner:
        ``"A"``, ``"B"``, or ``"tie"``.
    reasoning:
        One or two sentences explaining the judge's decision.
    score_a:
        Score for response A on a 1–10 scale, or ``None`` if parsing failed.
    score_b:
        Score for response B on a 1–10 scale, or ``None`` if parsing failed.
    judge_model:
        The model identifier used as the judge.
    raw_response:
        The raw text returned by the judge (useful for debugging).
    """

    winner: str                       # "A", "B", or "tie"
    reasoning: str
    score_a: float | None = field(default=None)
    score_b: float | None = field(default=None)
    judge_model: str = field(default="")
    raw_response: str = field(default="")

    def to_dict(self) -> dict:
        return {
            "winner": self.winner,
            "reasoning": self.reasoning,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "judge_model": self.judge_model,
        }

    def to_schema_payload(self) -> dict:
        """Return a dict conforming to the ``llm.eval.*`` namespace payload.

        Compatible with
        :class:`~agentobs.namespaces.eval_.EvalScoreRecordedPayload` field names.
        The ``score`` is normalised to a ``0-1`` range from the ``1-10`` scale
        returned by the judge prompt, so consumers always get a consistent range.
        """
        # Normalise scores: the judge returns 1-10; schema uses 0-1 by convention
        # when `scale` is set accordingly.  We expose raw scores with proper scale.
        avg_score: float = 0.0
        scale = "1-10"
        if self.score_a is not None and self.score_b is not None:
            avg_score = (self.score_a + self.score_b) / 2.0
        elif self.score_a is not None:
            avg_score = self.score_a
        elif self.score_b is not None:
            avg_score = self.score_b

        return {
            "evaluator": self.judge_model or "unknown",
            "score": avg_score,
            "scale": scale,
            "label": self.winner,
            "rationale": self.reasoning,
            "criteria": ["accuracy", "completeness", "clarity", "conciseness"],
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_judge_response(raw: str) -> dict:
    """Extract the JSON payload from the judge's raw response.

    Tries three strategies in order:

    1. Direct JSON parse of the whole response.
    2. Extract content from a `` ```json … ``` `` code fence.
    3. Extract the first ``{…}`` block via regex.

    Raises :exc:`ValueError` if no valid JSON is found.
    """
    text = raw.strip()

    # 1. Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Code fence  (```json ... ``` or just ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text, re.IGNORECASE)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First { ... } block
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot parse judge response as JSON: {raw[:300]!r}")


def _normalise_winner(raw: str) -> str:
    """Normalise the winner field to ``'A'``, ``'B'``, or ``'tie'``."""
    val = (raw or "").strip().upper()
    if val == "A":
        return "A"
    if val == "B":
        return "B"
    return "tie"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_judge(
    *,
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str,
    config: LLMDiffConfig | None = None,
) -> JudgeResult:
    """Call the judge model and return a :class:`JudgeResult`.

    Parameters
    ----------
    prompt:
        The original prompt posed to both models.
    response_a:
        The response produced by model A.
    response_b:
        The response produced by model B.
    judge_model:
        Model identifier for the judge (e.g. ``"gpt-4o"``).
    config:
        Optional :class:`~llm_diff.config.LLMDiffConfig`.  Loaded from the
        environment / ``.llmdiff`` TOML when ``None``.

    Returns
    -------
    JudgeResult
        Parsed evaluation result.  If the judge's response cannot be parsed
        as JSON, a ``"tie"`` result is returned with the raw text as the
        reasoning so callers are never left with an exception.

    Raises
    ------
    ValueError
        If there is no API key for the judge model's provider.
    TimeoutError
        If the judge call exceeds the configured timeout.
    """
    from llm_diff.providers import _validate_provider, call_model_with_messages  # noqa: PLC0415

    cfg = config if config is not None else load_config()
    _, provider_cfg = _validate_provider(cfg, judge_model)

    provider_name, _ = _validate_provider(cfg, judge_model)

    user_text = _JUDGE_USER_TMPL.format(
        prompt=prompt,
        response_a=response_a,
        response_b=response_b,
    )

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user_text},
    ]

    raw = await call_model_with_messages(
        model=judge_model,
        messages=messages,
        config=cfg,
    )

    try:
        parsed = _parse_judge_response(raw)
    except ValueError:
        logger.warning(
            "Judge model '%s' returned unparseable JSON — falling back to tie: %s",
            judge_model,
            raw[:200],
        )
        return JudgeResult(
            winner="tie",
            reasoning=raw[:500] if raw else "Could not parse judge response.",
            judge_model=judge_model,
            raw_response=raw,
        )

    winner = _normalise_winner(str(parsed.get("winner", "tie")))
    reasoning = str(parsed.get("reasoning", "")).strip()

    score_a: float | None = None
    score_b: float | None = None
    try:
        if parsed.get("score_a") is not None:
            score_a = float(parsed["score_a"])
        if parsed.get("score_b") is not None:
            score_b = float(parsed["score_b"])
    except (TypeError, ValueError):
        pass

    result = JudgeResult(
        winner=winner,
        reasoning=reasoning,
        score_a=score_a,
        score_b=score_b,
        judge_model=judge_model,
        raw_response=raw,
    )

    # Emit schema event for the evaluation
    try:
        from llm_diff.schema_events import emit as schema_emit
        from llm_diff.schema_events import make_eval_scenario_event  # noqa: PLC0415

        schema_emit(
            make_eval_scenario_event(
                evaluator=judge_model,
                score=((score_a or 0.0) + (score_b or 0.0)) / 2.0 if (score_a or score_b) else None,
                scale="1-10",
                label=winner,
                rationale=reasoning,
                criteria=["accuracy", "completeness", "clarity", "conciseness"],
                status="passed",
            )
        )
    except Exception:  # noqa: BLE001
        logger.debug("Schema event emission failed", exc_info=True)

    return result
