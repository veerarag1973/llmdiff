"""Tests for llm_diff.multi."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_diff.diff import DiffResult, DiffType
from llm_diff.multi import MultiModelReport, PairScore


# ---------------------------------------------------------------------------
# PairScore
# ---------------------------------------------------------------------------

def _make_diff_result(similarity: float = 0.75) -> DiffResult:
    return DiffResult(chunks=[], similarity=similarity, word_count_a=10, word_count_b=10)


def test_pair_score_word_similarity() -> None:
    dr = _make_diff_result(0.65)
    ps = PairScore(
        model_a="gpt-4o",
        model_b="claude-3",
        diff_result=dr,
        semantic_score=None,
    )
    assert ps.word_similarity == pytest.approx(0.65)


def test_pair_score_primary_score_prefers_semantic() -> None:
    dr = _make_diff_result(0.50)
    ps = PairScore(
        model_a="gpt-4o",
        model_b="claude-3",
        diff_result=dr,
        semantic_score=0.90,
    )
    assert ps.primary_score == pytest.approx(0.90)


def test_pair_score_primary_score_falls_back_to_word() -> None:
    dr = _make_diff_result(0.70)
    ps = PairScore(
        model_a="gpt-4o",
        model_b="claude-3",
        diff_result=dr,
        semantic_score=None,
    )
    assert ps.primary_score == pytest.approx(0.70)


def test_pair_score_to_dict() -> None:
    dr = _make_diff_result(0.55)
    ps = PairScore(
        model_a="m1",
        model_b="m2",
        diff_result=dr,
        semantic_score=0.60,
    )
    d = ps.to_dict()
    assert d["model_a"] == "m1"
    assert d["model_b"] == "m2"
    assert d["word_similarity"] == pytest.approx(0.55)
    assert d["semantic_score"] == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# MultiModelReport
# ---------------------------------------------------------------------------

def _make_report(pairs: list[PairScore], models: list[str] | None = None) -> MultiModelReport:
    models = models or ["m1", "m2", "m3"]
    return MultiModelReport(
        prompt="What is 2+2?",
        models=models,
        responses={"m1": "four", "m2": "4", "m3": "it's 4"},
        model_responses={},
        matrix=pairs,
    )


def test_ranked_pairs_highest_first() -> None:
    pairs = [
        PairScore("m1", "m2", _make_diff_result(0.4), None),
        PairScore("m1", "m3", _make_diff_result(0.9), None),
        PairScore("m2", "m3", _make_diff_result(0.6), None),
    ]
    report = _make_report(pairs)
    ranked = report.ranked_pairs()
    scores = [p.primary_score for p in ranked]
    assert scores == sorted(scores, reverse=True)


def test_most_similar_pair() -> None:
    pairs = [
        PairScore("m1", "m2", _make_diff_result(0.3), None),
        PairScore("m1", "m3", _make_diff_result(0.95), None),
        PairScore("m2", "m3", _make_diff_result(0.5), None),
    ]
    report = _make_report(pairs)
    best = report.most_similar_pair()
    assert best is not None
    assert best.primary_score == pytest.approx(0.95)


def test_most_divergent_pair() -> None:
    pairs = [
        PairScore("m1", "m2", _make_diff_result(0.1), None),
        PairScore("m1", "m3", _make_diff_result(0.8), None),
        PairScore("m2", "m3", _make_diff_result(0.5), None),
    ]
    report = _make_report(pairs)
    worst = report.most_divergent_pair()
    assert worst is not None
    assert worst.primary_score == pytest.approx(0.1)


def test_similarity_matrix_shape() -> None:
    models = ["m1", "m2", "m3"]
    pairs = [
        PairScore("m1", "m2", _make_diff_result(0.7), None),
        PairScore("m1", "m3", _make_diff_result(0.5), None),
        PairScore("m2", "m3", _make_diff_result(0.9), None),
    ]
    report = _make_report(pairs, models=models)
    matrix = report.similarity_matrix()
    # similarity_matrix returns {(model_a, model_b): score, ...}
    assert isinstance(matrix, dict)
    assert len(matrix) == 3  # 3 pairs for 3 models
    for key, val in matrix.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert 0.0 <= val <= 1.0


def test_to_dict() -> None:
    pairs = [PairScore("m1", "m2", _make_diff_result(0.7), None)]
    report = _make_report(pairs, models=["m1", "m2"])
    d = report.to_dict()
    assert "prompt" in d
    assert "models" in d
    assert "pairs" in d  # key is 'pairs', not 'matrix'


# ---------------------------------------------------------------------------
# run_multi_model — integration with mocked providers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_multi_model_happy_path() -> None:
    """run_multi_model should call each model and return a MultiModelReport."""
    from llm_diff.config import LLMDiffConfig
    from llm_diff.multi import run_multi_model

    cfg = LLMDiffConfig()  # default config — no real keys needed for mocked calls

    class FakeResp:
        text = "Response from model"
        model = "m"
        provider = "openai"
        prompt_tokens = 10
        completion_tokens = 20
        total_tokens = 30
        latency_ms = 100.0

    async def fake_call(*, model: str, prompt: str, **kwargs: object) -> FakeResp:  # noqa: ANN001
        fr = FakeResp()
        fr.model = model  # type: ignore[misc]
        fr.text = f"Response from {model}"  # type: ignore[misc]
        return fr

    from llm_diff.providers import ProviderConfig  # noqa: PLC0415

    dummy_provider = ("openai", ProviderConfig(api_key="test-key"))

    with patch("llm_diff.providers._call_or_cache", new_callable=AsyncMock, side_effect=fake_call), \
         patch("llm_diff.providers._validate_provider", return_value=dummy_provider):
        report = await run_multi_model(
            "Hello",
            models=["m1", "m2"],
            semantic=False,
            config=cfg,
            concurrency=2,
            cache=None,
        )

    assert isinstance(report, MultiModelReport)
    assert len(report.models) == 2
    # 2 models → 1 pair
    assert len(report.matrix) == 1
