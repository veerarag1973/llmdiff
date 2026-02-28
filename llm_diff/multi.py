"""Multi-model comparison for llm-diff.

Runs a single prompt against N models and computes all pairwise similarity
scores, returning a ranked table and a full NxN similarity matrix.

Use this when you need to answer "which of my N candidate models is most
consistent with each other?" or "which pair diverges most?"

Usage
-----
.. code-block:: python

    import asyncio
    from llm_diff.multi import run_multi_model

    report = asyncio.run(
        run_multi_model(
            "Explain recursion in one sentence.",
            models=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
        )
    )
    print(report.ranked_pairs())   # list of (model_a, model_b, similarity)
    for pair in report.matrix:
        print(pair.model_a, "vs", pair.model_b, "→", pair.word_similarity)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from itertools import combinations

from llm_diff.config import LLMDiffConfig
from llm_diff.diff import DiffResult, word_diff
from llm_diff.providers import ModelResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PairScore:
    """Similarity scores for a single pair of models.

    Attributes
    ----------
    model_a, model_b:
        Model identifiers for the pair.
    diff_result:
        Word-level diff result.
    semantic_score:
        Cosine similarity (0–1), or ``None`` if not requested.
    word_similarity:
        Convenience alias for ``diff_result.similarity``.
    """

    model_a: str
    model_b: str
    diff_result: DiffResult
    semantic_score: float | None = field(default=None)

    @property
    def word_similarity(self) -> float:
        return self.diff_result.similarity

    @property
    def primary_score(self) -> float:
        return self.semantic_score if self.semantic_score is not None else self.word_similarity

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "word_similarity": round(self.word_similarity, 4),
            "semantic_score": (
                round(self.semantic_score, 4) if self.semantic_score is not None else None
            ),
        }


@dataclass
class MultiModelReport:
    """Full result of a multi-model comparison run.

    Attributes
    ----------
    prompt:
        The prompt sent to all models.
    models:
        Ordered list of model identifiers (as supplied by the caller).
    responses:
        Mapping from model identifier to the text response it produced.
    model_responses:
        Mapping from model identifier to the full :class:`~llm_diff.providers.ModelResponse`.
    matrix:
        All pairwise :class:`PairScore` objects (len = n*(n-1)/2).
    """

    prompt: str
    models: list[str]
    responses: dict[str, str]
    model_responses: dict[str, ModelResponse]
    matrix: list[PairScore]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def ranked_pairs(self) -> list[PairScore]:
        """Return the pair scores sorted by *primary_score* descending."""
        return sorted(self.matrix, key=lambda p: p.primary_score, reverse=True)

    def similarity_matrix(self) -> dict[tuple[str, str], float]:
        """Return a flat dict ``{(model_a, model_b): similarity}``."""
        return {(p.model_a, p.model_b): p.primary_score for p in self.matrix}

    def most_similar_pair(self) -> PairScore | None:
        """Return the pair with the highest similarity, or ``None`` if no pairs."""
        pairs = self.ranked_pairs()
        return pairs[0] if pairs else None

    def most_divergent_pair(self) -> PairScore | None:
        """Return the pair with the lowest similarity, or ``None`` if no pairs."""
        pairs = self.ranked_pairs()
        return pairs[-1] if pairs else None

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "models": self.models,
            "pairs": [p.to_dict() for p in self.ranked_pairs()],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_multi_model(
    prompt: str,
    *,
    models: list[str],
    semantic: bool = False,
    config: LLMDiffConfig | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    concurrency: int = 4,
    cache: object | None = None,
) -> MultiModelReport:
    """Run *prompt* against all *models* and compute pairwise similarities.

    Parameters
    ----------
    prompt:
        The prompt text sent to every model.
    models:
        List of model identifiers (minimum 2, practical maximum ~8).
    semantic:
        When ``True``, compute cosine semantic similarity for each pair.
        Requires ``pip install "llm-diff[semantic]"``.
    config:
        Optional :class:`~llm_diff.config.LLMDiffConfig`.
    temperature, max_tokens, timeout:
        Override the corresponding values in *config*.
    concurrency:
        Maximum number of simultaneous API calls when fetching responses.
    cache:
        Optional :class:`~llm_diff.cache.ResultCache` instance.

    Returns
    -------
    MultiModelReport
        Full result including all pairwise scores and individual responses.

    Raises
    ------
    ValueError
        If fewer than 2 models are provided.
    """
    if len(models) < 2:
        raise ValueError(
            f"run_multi_model requires at least 2 models, got {len(models)}: {models}"
        )

    from llm_diff.config import load_config  # noqa: PLC0415
    cfg = config if config is not None else load_config()
    if temperature is not None:
        cfg.temperature = temperature
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    if timeout is not None:
        cfg.timeout = timeout

    # ── Fetch all model responses concurrently ───────────────────────────────
    sem = asyncio.Semaphore(concurrency)

    async def _fetch_one(model: str) -> tuple[str, ModelResponse]:
        from llm_diff.providers import _call_or_cache, _validate_provider  # noqa: PLC0415
        async with sem:
            provider_name, provider_cfg = _validate_provider(cfg, model)
            resp = await _call_or_cache(
                model=model,
                prompt=prompt,
                provider_cfg=provider_cfg,
                provider_name=provider_name,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout=cfg.timeout,
                cache=cache,
            )
            return model, resp

    results: list[tuple[str, ModelResponse]] = list(
        await asyncio.gather(*[_fetch_one(m) for m in models])
    )

    model_responses: dict[str, ModelResponse] = dict(results)
    responses: dict[str, str] = {m: r.text for m, r in model_responses.items()}

    # ── Compute all pairwise diffs ───────────────────────────────────────────
    matrix: list[PairScore] = []
    for model_a, model_b in combinations(models, 2):
        text_a = responses[model_a]
        text_b = responses[model_b]

        diff_result: DiffResult = word_diff(text_a, text_b)

        sem_score: float | None = None
        if semantic:
            from llm_diff.semantic import compute_semantic_similarity  # noqa: PLC0415
            sem_score = compute_semantic_similarity(text_a, text_b)

        matrix.append(
            PairScore(
                model_a=model_a,
                model_b=model_b,
                diff_result=diff_result,
                semantic_score=sem_score,
            )
        )

    return MultiModelReport(
        prompt=prompt,
        models=list(models),
        responses=responses,
        model_responses=model_responses,
        matrix=matrix,
    )
