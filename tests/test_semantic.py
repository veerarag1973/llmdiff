"""Tests for llm_diff.semantic — embedding-based similarity scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_diff.semantic import (
    _cosine_similarity,
    _get_model,
    compute_semantic_similarity,
    reset_model_cache,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, index: int) -> np.ndarray:
    """Return a unit vector with 1.0 at *index* and 0.0 elsewhere."""
    v = np.zeros(dim, dtype=np.float32)
    v[index] = 1.0
    return v


def _normalised(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors_return_1(self) -> None:
        v = _normalised(np.array([1.0, 2.0, 3.0]))
        assert pytest.approx(_cosine_similarity(v, v), abs=1e-6) == 1.0

    def test_orthogonal_vectors_return_0(self) -> None:
        a = _unit_vec(4, 0)
        b = _unit_vec(4, 1)
        assert pytest.approx(_cosine_similarity(a, b), abs=1e-6) == 0.0

    def test_opposite_vectors_return_minus_1(self) -> None:
        v = _normalised(np.array([1.0, 2.0, 3.0]))
        assert pytest.approx(_cosine_similarity(v, -v), abs=1e-6) == -1.0

    def test_zero_norm_a_returns_0(self) -> None:
        a = np.zeros(4, dtype=np.float32)
        b = _unit_vec(4, 0)
        assert _cosine_similarity(a, b) == 0.0

    def test_zero_norm_b_returns_0(self) -> None:
        a = _unit_vec(4, 0)
        b = np.zeros(4, dtype=np.float32)
        assert _cosine_similarity(a, b) == 0.0

    def test_general_case(self) -> None:
        a = _normalised(np.array([1.0, 0.0, 1.0]))
        b = _normalised(np.array([1.0, 1.0, 0.0]))
        result = _cosine_similarity(a, b)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# _get_model — lazy loading and caching
# ---------------------------------------------------------------------------


class TestGetModel:
    def setup_method(self) -> None:
        reset_model_cache()

    def teardown_method(self) -> None:
        reset_model_cache()

    def test_import_error_when_not_installed(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="sentence-transformers"):
                _get_model()

    def test_returns_model_after_mock_import(self) -> None:
        fake_model = MagicMock()
        fake_st_module = MagicMock()
        fake_st_module.SentenceTransformer.return_value = fake_model

        with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
            model = _get_model()

        assert model is fake_model

    def test_model_is_cached_on_second_call(self) -> None:
        fake_model = MagicMock()
        fake_st_module = MagicMock()
        fake_st_module.SentenceTransformer.return_value = fake_model

        with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
            m1 = _get_model()
            m2 = _get_model()

        # SentenceTransformer constructor called once only
        assert fake_st_module.SentenceTransformer.call_count == 1
        assert m1 is m2

    def test_reset_clears_cache(self) -> None:
        fake_model = MagicMock()
        fake_st_module = MagicMock()
        fake_st_module.SentenceTransformer.return_value = fake_model

        with patch.dict("sys.modules", {"sentence_transformers": fake_st_module}):
            _get_model()
            reset_model_cache()
            _get_model()

        assert fake_st_module.SentenceTransformer.call_count == 2


# ---------------------------------------------------------------------------
# compute_semantic_similarity
# ---------------------------------------------------------------------------


class TestComputeSemanticSimilarity:
    """All tests mock sentence-transformers so no model download is needed."""

    def setup_method(self) -> None:
        reset_model_cache()

    def teardown_method(self) -> None:
        reset_model_cache()

    def _make_mock_model(self, emb_a: np.ndarray, emb_b: np.ndarray) -> MagicMock:
        """Return a mock SentenceTransformer whose encode() yields two embeddings."""
        model = MagicMock()
        model.encode.return_value = np.array([emb_a, emb_b])
        return model

    def _compute(self, text_a: str, text_b: str, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """Helper: patch _get_model, run compute_semantic_similarity."""
        mock_model = self._make_mock_model(emb_a, emb_b)
        with patch("llm_diff.semantic._get_model", return_value=mock_model):
            return compute_semantic_similarity(text_a, text_b)

    def test_identical_embeddings_return_near_1(self) -> None:
        v = _normalised(np.array([1.0, 2.0, 3.0]))
        score = self._compute("same", "same", v, v)
        assert pytest.approx(score, abs=1e-5) == 1.0

    def test_orthogonal_embeddings_return_near_0(self) -> None:
        a = _unit_vec(4, 0)
        b = _unit_vec(4, 2)
        score = self._compute("hello", "world", a, b)
        assert pytest.approx(score, abs=1e-5) == 0.0

    def test_result_clamped_to_0_when_below_0(self) -> None:
        # Force a slight negative value that can come from floating-point drift
        a = _normalised(np.array([1.0, 2.0, 3.0]))
        b = _normalised(-np.array([1.0, 2.0, 3.0]))  # opposite → cos = -1
        score = self._compute("pos", "neg", a, b)
        assert score == 0.0

    def test_result_within_0_and_1(self) -> None:
        a = _normalised(np.array([0.5, 0.5, 0.5]))
        b = _normalised(np.array([0.5, -0.5, 0.5]))
        score = self._compute("a", "b", a, b)
        assert 0.0 <= score <= 1.0

    def test_encode_called_with_both_texts(self) -> None:
        fake_model = self._make_mock_model(
            _unit_vec(4, 0),
            _unit_vec(4, 1),
        )
        with patch("llm_diff.semantic._get_model", return_value=fake_model):
            compute_semantic_similarity("text A", "text B")
        fake_model.encode.assert_called_once_with(
            ["text A", "text B"], convert_to_numpy=True
        )

    def test_empty_strings_handled_without_crash(self) -> None:
        """Zero-norm embeddings return 0.0 after clamping."""
        zero = np.zeros(4, dtype=np.float32)
        score = self._compute("", "", zero, zero)
        assert score == 0.0

    def test_import_error_propagates(self) -> None:
        with patch("llm_diff.semantic._get_model", side_effect=ImportError("missing")):
            with pytest.raises(ImportError, match="missing"):
                compute_semantic_similarity("a", "b")
