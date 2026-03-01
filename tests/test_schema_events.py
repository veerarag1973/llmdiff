"""Tests for llm_diff.schema_events — llm-toolkit-schema integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_diff.schema_events import (
    EventEmitter,
    configure_emitter,
    emit,
    get_emitter,
    make_cache_event,
    make_comparison_completed_event,
    make_comparison_started_event,
    make_cost_recorded_event,
    make_eval_regression_event,
    make_eval_scenario_event,
    make_report_exported_event,
    make_trace_span_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_emitter() -> EventEmitter:
    """Install a fresh in-memory emitter and return it."""
    return configure_emitter(collect=True)


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------


class TestEventEmitter:
    def setup_method(self) -> None:
        _reset_emitter()

    def test_default_emitter_is_active(self) -> None:
        emitter = get_emitter()
        assert emitter is not None

    def test_configure_emitter_replaces_global(self) -> None:
        e1 = get_emitter()
        configure_emitter()
        e2 = get_emitter()
        assert e1 is not e2

    def test_emit_collects_event(self) -> None:
        emitter = _reset_emitter()
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emitter.emit(evt)
        assert len(emitter.events) == 1

    def test_clear_removes_collected_events(self) -> None:
        emitter = _reset_emitter()
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emitter.emit(evt)
        emitter.clear()
        assert len(emitter.events) == 0

    def test_collect_false_does_not_accumulate(self) -> None:
        emitter = configure_emitter(collect=False)
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emitter.emit(evt)
        assert len(emitter.events) == 0

    def test_exporter_callable_is_called(self) -> None:
        # Use a plain function mock (not MagicMock) to avoid auto-created .export attr
        calls = []

        def plain_exporter(event: object) -> None:
            calls.append(event)

        emitter = configure_emitter(exporter=plain_exporter)
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emitter.emit(evt)
        assert len(calls) == 1
        assert calls[0] is evt

    def test_exporter_object_export_method_is_called(self) -> None:
        mock_exporter = MagicMock()
        mock_exporter.export = MagicMock(return_value=None)
        emitter = configure_emitter(exporter=mock_exporter)
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emitter.emit(evt)
        mock_exporter.export.assert_called_once_with(evt)

    def test_exporter_error_does_not_propagate(self) -> None:
        def bad_exporter(event: object) -> None:
            raise RuntimeError("Export failed")

        emitter = configure_emitter(exporter=bad_exporter)
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        # Should not raise
        emitter.emit(evt)

    def test_events_property_returns_copy(self) -> None:
        emitter = _reset_emitter()
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emitter.emit(evt)
        events1 = emitter.events
        events2 = emitter.events
        assert events1 == events2
        assert events1 is not events2


# ---------------------------------------------------------------------------
# Global emit() function
# ---------------------------------------------------------------------------


class TestGlobalEmit:
    def setup_method(self) -> None:
        _reset_emitter()

    def test_global_emit_routes_to_global_emitter(self) -> None:
        emitter = get_emitter()
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello",
        )
        emit(evt)
        assert len(emitter.events) == 1
        assert emitter.events[0] is evt


# ---------------------------------------------------------------------------
# make_comparison_started_event
# ---------------------------------------------------------------------------


class TestMakeComparisonStartedEvent:
    def test_event_type_is_correct(self) -> None:
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
        )
        assert evt.event_type == "llm.diff.comparison.started"

    def test_source_contains_version(self) -> None:
        from llm_diff import __version__

        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
        )
        assert f"llm-diff@{__version__}" == evt.source

    def test_payload_contains_models(self) -> None:
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Hello world",
        )
        assert evt.payload["model_a"] == "gpt-4o"
        assert evt.payload["model_b"] == "claude-3-5-sonnet"
        assert evt.payload["prompt_length"] == len("Hello world")

    def test_event_id_is_auto_generated_ulid(self) -> None:
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
        )
        assert len(evt.event_id) == 26

    def test_optional_session_id(self) -> None:
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
            session_id="sess-123",
        )
        assert evt.session_id == "sess-123"

    def test_optional_org_id(self) -> None:
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
            org_id="acme",
        )
        assert evt.org_id == "acme"

    def test_event_validates_without_error(self) -> None:
        evt = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
        )
        evt.validate()  # should not raise


# ---------------------------------------------------------------------------
# make_comparison_completed_event
# ---------------------------------------------------------------------------


class TestMakeComparisonCompletedEvent:
    def test_event_type_is_correct(self) -> None:
        evt = make_comparison_completed_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            diff_type="completion",
            similarity_score=0.85,
        )
        assert evt.event_type == "llm.diff.comparison.completed"

    def test_payload_similarity_score(self) -> None:
        evt = make_comparison_completed_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            diff_type="completion",
            similarity_score=0.75,
        )
        assert evt.payload["similarity_score"] == pytest.approx(0.75)

    def test_payload_diff_type(self) -> None:
        for diff_type in ("prompt", "completion", "both"):
            evt = make_comparison_completed_event(
                model_a="gpt-4o",
                model_b="claude-3-5-sonnet",
                diff_type=diff_type,
            )
            assert evt.payload["diff_type"] == diff_type

    def test_payload_completion_diff(self) -> None:
        evt = make_comparison_completed_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            diff_type="completion",
            completion_diff="--- a\n+++ b\n-old\n+new\n",
        )
        # diff_result is a nested dict containing the unified diff
        assert evt.payload["diff_result"] is not None
        assert "--- a" in evt.payload["diff_result"]["unified_diff"]

    def test_base_event_id_embedded(self) -> None:
        started = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Test",
        )
        completed = make_comparison_completed_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            diff_type="completion",
            base_event_id=started.event_id,
        )
        # base_event_id becomes source_id in DiffComparisonPayload
        assert completed.payload["source_id"] == started.event_id

    def test_event_validates(self) -> None:
        evt = make_comparison_completed_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            diff_type="completion",
            similarity_score=0.9,
        )
        evt.validate()  # should not raise


# ---------------------------------------------------------------------------
# make_report_exported_event
# ---------------------------------------------------------------------------


class TestMakeReportExportedEvent:
    def test_event_type(self) -> None:
        evt = make_report_exported_event(
            output_path="/tmp/report.html",
            format="html",
        )
        assert evt.event_type == "llm.diff.report.exported"

    def test_payload_fields(self) -> None:
        evt = make_report_exported_event(
            output_path="/tmp/report.html",
            format="html",
        )
        assert evt.payload["export_path"] == "/tmp/report.html"
        assert evt.payload["format"] == "html"

    def test_event_validates(self) -> None:
        make_report_exported_event(output_path="/tmp/r.html").validate()


# ---------------------------------------------------------------------------
# make_trace_span_event
# ---------------------------------------------------------------------------


class TestMakeTraceSpanEvent:
    def test_event_type(self) -> None:
        evt = make_trace_span_event(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=250.0,
        )
        assert evt.event_type == "llm.trace.span.completed"

    def test_payload_fields(self) -> None:
        evt = make_trace_span_event(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=250.0,
            finish_reason="stop",
            stream=False,
        )
        # model is a nested ModelInfo dict
        assert evt.payload["model"]["name"] == "gpt-4o"
        # token_usage is a nested TokenUsage dict
        assert evt.payload["token_usage"]["prompt_tokens"] == 100
        assert evt.payload["token_usage"]["completion_tokens"] == 50
        assert evt.payload["duration_ms"] == pytest.approx(250.0)
        assert evt.payload["finish_reason"] == "stop"
        assert evt.payload["stream"] is False

    def test_total_tokens_auto_computed(self) -> None:
        evt = make_trace_span_event(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=100.0,
        )
        assert evt.payload["token_usage"]["total_tokens"] == 150

    def test_tags_set_when_provider_given(self) -> None:
        evt = make_trace_span_event(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=100.0,
            provider="openai",
        )
        assert evt.tags is not None
        assert evt.tags.get("provider") == "openai"
        assert evt.tags.get("model") == "gpt-4o"

    def test_event_validates(self) -> None:
        make_trace_span_event(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=100.0,
        ).validate()


# ---------------------------------------------------------------------------
# make_cache_event
# ---------------------------------------------------------------------------


class TestMakeCacheEvent:
    def test_cache_hit_event_type(self) -> None:
        evt = make_cache_event(hit=True)
        assert evt.event_type == "llm.cache.hit"

    def test_cache_miss_event_type(self) -> None:
        evt = make_cache_event(hit=False)
        assert evt.event_type == "llm.cache.miss"

    def test_payload_fields(self) -> None:
        evt = make_cache_event(
            hit=True,
            cache_key="abc123",
            ttl_seconds=3600,
            backend="disk",
            latency_ms=1.5,
        )
        # CacheHitPayload uses cache_key_hash and cache_store field names
        assert evt.payload["cache_key_hash"] == "abc123"
        assert evt.payload["ttl_seconds"] == 3600
        assert evt.payload["cache_store"] == "disk"
        assert evt.payload["latency_ms"] == pytest.approx(1.5)

    def test_event_validates(self) -> None:
        make_cache_event(hit=True, cache_key="k1", backend="disk").validate()
        make_cache_event(hit=False, cache_key="k2", backend="disk").validate()


# ---------------------------------------------------------------------------
# make_cost_recorded_event
# ---------------------------------------------------------------------------


class TestMakeCostRecordedEvent:
    def test_event_type(self) -> None:
        evt = make_cost_recorded_event(
            input_cost=0.001,
            output_cost=0.002,
            total_cost=0.003,
        )
        assert evt.event_type == "llm.cost.recorded"

    def test_payload_fields(self) -> None:
        evt = make_cost_recorded_event(
            input_cost=0.0015,
            output_cost=0.0006,
            total_cost=0.0021,
            currency="USD",
        )
        # CostRecordedPayload uses cost_usd for total; input/output are extra fields
        assert evt.payload["input_cost_usd"] == pytest.approx(0.0015)
        assert evt.payload["output_cost_usd"] == pytest.approx(0.0006)
        assert evt.payload["cost_usd"] == pytest.approx(0.0021)
        assert evt.payload["currency"] == "USD"

    def test_model_optional_field(self) -> None:
        evt = make_cost_recorded_event(
            input_cost=0.001,
            output_cost=0.001,
            total_cost=0.002,
            model="gpt-4o",
        )
        assert evt.payload.get("model_name") == "gpt-4o"

    def test_event_validates(self) -> None:
        make_cost_recorded_event(
            input_cost=0.001, output_cost=0.001, total_cost=0.002
        ).validate()


# ---------------------------------------------------------------------------
# make_eval_scenario_event
# ---------------------------------------------------------------------------


class TestMakeEvalScenarioEvent:
    def test_event_type(self) -> None:
        evt = make_eval_scenario_event(
            evaluator="gpt-4o",
            score=7.5,
        )
        assert evt.event_type == "llm.eval.scenario.completed"

    def test_payload_fields(self) -> None:
        evt = make_eval_scenario_event(
            evaluator="gpt-4o",
            score=8.0,
            scale="1-10",
            label="A",
            rationale="Model A was more concise.",
            criteria=["accuracy", "clarity"],
        )
        # EvalScenarioPayload uses scenario_name to embed evaluator
        assert "gpt-4o" in evt.payload["scenario_name"]
        assert evt.payload["score"] == pytest.approx(8.0)
        assert evt.payload["scale"] == "1-10"
        assert evt.payload["label"] == "A"
        assert "concise" in evt.payload["rationale"]
        # criteria are embedded as keys in the metrics dict
        assert "accuracy" in evt.payload["metrics"]

    def test_event_validates(self) -> None:
        make_eval_scenario_event(evaluator="human", score=0.85, scale="0-1").validate()


# ---------------------------------------------------------------------------
# DiffResult.to_schema_payload()
# ---------------------------------------------------------------------------


class TestDiffResultSchemaPayload:
    def test_to_schema_payload_fields(self) -> None:
        from llm_diff.diff import DiffChunk, DiffResult, DiffType

        result = DiffResult(
            chunks=[
                DiffChunk(type=DiffType.DELETE, text="old "),
                DiffChunk(type=DiffType.INSERT, text="new "),
                DiffChunk(type=DiffType.EQUAL, text="tail"),
            ],
            similarity=0.75,
            word_count_a=3,
            word_count_b=3,
        )
        payload = result.to_schema_payload(base_event_id="01ABC")
        assert payload["diff_type"] == "completion"
        assert payload["similarity_score"] == pytest.approx(0.75)
        assert payload["base_event_id"] == "01ABC"

    def test_as_unified_diff_contains_markers(self) -> None:
        from llm_diff.diff import DiffChunk, DiffResult, DiffType

        result = DiffResult(
            chunks=[
                DiffChunk(type=DiffType.DELETE, text="Hello"),
                DiffChunk(type=DiffType.INSERT, text="Hi"),
            ],
            similarity=0.50,
            word_count_a=1,
            word_count_b=1,
        )
        diff_str = result.as_unified_diff()
        assert "--- model_a" in diff_str
        assert "+++ model_b" in diff_str
        assert "-Hello" in diff_str
        assert "+Hi" in diff_str

    def test_as_unified_diff_empty_for_equal_content(self) -> None:
        from llm_diff.diff import DiffChunk, DiffResult, DiffType

        result = DiffResult(
            chunks=[DiffChunk(type=DiffType.EQUAL, text="hello")],
            similarity=1.0,
            word_count_a=1,
            word_count_b=1,
        )
        assert result.as_unified_diff() == ""


# ---------------------------------------------------------------------------
# CostEstimate.to_schema_payload()
# ---------------------------------------------------------------------------


class TestCostEstimateSchemaPayload:
    def test_to_schema_payload_fields(self) -> None:
        from llm_diff.pricing import CostEstimate

        estimate = CostEstimate(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
            total_usd=0.0075,
            prompt_usd=0.0025,
            completion_usd=0.005,
        )
        payload = estimate.to_schema_payload()
        assert payload["input_cost"] == pytest.approx(0.0025)
        assert payload["output_cost"] == pytest.approx(0.005)
        assert payload["total_cost"] == pytest.approx(0.0075)
        assert payload["currency"] == "USD"


# ---------------------------------------------------------------------------
# JudgeResult.to_schema_payload()
# ---------------------------------------------------------------------------


class TestJudgeResultSchemaPayload:
    def test_to_schema_payload_fields(self) -> None:
        from llm_diff.judge import JudgeResult

        result = JudgeResult(
            winner="A",
            reasoning="Model A was more concise.",
            score_a=8.0,
            score_b=6.0,
            judge_model="gpt-4o",
        )
        payload = result.to_schema_payload()
        assert payload["evaluator"] == "gpt-4o"
        assert payload["label"] == "A"
        assert payload["rationale"] == "Model A was more concise."
        assert "accuracy" in payload["criteria"]
        # score should be avg of 8 and 6 = 7.0
        assert payload["score"] == pytest.approx(7.0)
        assert payload["scale"] == "1-10"

    def test_to_schema_payload_no_scores(self) -> None:
        from llm_diff.judge import JudgeResult

        result = JudgeResult(winner="tie", reasoning="Equivalent.", judge_model="gpt-4o")
        payload = result.to_schema_payload()
        assert payload["score"] == pytest.approx(0.0)

    def test_to_schema_payload_one_score(self) -> None:
        from llm_diff.judge import JudgeResult

        result = JudgeResult(
            winner="B",
            reasoning="B wins.",
            score_a=None,
            score_b=9.0,
            judge_model="gpt-4o",
        )
        payload = result.to_schema_payload()
        assert payload["score"] == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Integration: all diff.* event types form a valid sequence
# ---------------------------------------------------------------------------


class TestDiffEventLifecycle:
    def setup_method(self) -> None:
        _reset_emitter()

    def test_full_diff_lifecycle_events(self) -> None:
        """Emit started → completed → exported and verify all are collected."""
        emitter = get_emitter()

        started = make_comparison_started_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            prompt="Explain recursion",
        )
        emitter.emit(started)

        completed = make_comparison_completed_event(
            model_a="gpt-4o",
            model_b="claude-3-5-sonnet",
            diff_type="completion",
            similarity_score=0.85,
            base_event_id=started.event_id,
        )
        emitter.emit(completed)

        exported = make_report_exported_event(
            output_path="/tmp/report.html",
            format="html",
        )
        emitter.emit(exported)

        events = emitter.events
        assert len(events) == 3

        types = [e.event_type for e in events]
        assert "llm.diff.comparison.started" in types
        assert "llm.diff.comparison.completed" in types
        assert "llm.diff.report.exported" in types

        # Verify chain: completed references started
        completed_evt = next(e for e in events if e.event_type == "llm.diff.comparison.completed")
        # base_event_id is stored as source_id in DiffComparisonPayload
        assert completed_evt.payload["source_id"] == started.event_id

    def test_events_have_unique_ids(self) -> None:
        e1 = make_comparison_started_event(
            model_a="gpt-4o", model_b="claude-3-5-sonnet", prompt="A"
        )
        e2 = make_comparison_started_event(
            model_a="gpt-4o", model_b="claude-3-5-sonnet", prompt="B"
        )
        assert e1.event_id != e2.event_id

    def test_all_events_have_valid_timestamps(self) -> None:
        events = [
            make_comparison_started_event(
                model_a="gpt-4o", model_b="claude-3", prompt="T"
            ),
            make_comparison_completed_event(
                model_a="gpt-4o", model_b="claude-3", diff_type="completion"
            ),
            make_report_exported_event(output_path="/tmp/r.html"),
            make_trace_span_event(
                model="gpt-4o", prompt_tokens=10, completion_tokens=5, latency_ms=50.0
            ),
            make_cache_event(hit=True),
            make_cost_recorded_event(input_cost=0.001, output_cost=0.001, total_cost=0.002),
            make_eval_scenario_event(evaluator="gpt-4o", score=7.0),
        ]
        for evt in events:
            assert evt.timestamp  # non-empty ISO-8601 string
            assert "T" in evt.timestamp  # includes time separator


# ---------------------------------------------------------------------------
# Integration with cache.ResultCache
# ---------------------------------------------------------------------------


class TestCacheSchemaIntegration:
    def setup_method(self) -> None:
        _reset_emitter()

    def test_cache_miss_emits_schema_event(self, tmp_path) -> None:
        from llm_diff.cache import ResultCache

        emitter = get_emitter()
        cache = ResultCache(cache_dir=tmp_path, enabled=True)
        cache.get("nonexistent_key_abc123")

        cache_events = [e for e in emitter.events if "cache" in e.event_type]
        assert len(cache_events) == 1
        assert cache_events[0].event_type == "llm.cache.miss"

    def test_cache_disabled_emits_no_events(self, tmp_path) -> None:
        from llm_diff.cache import ResultCache

        emitter = get_emitter()
        cache = ResultCache(cache_dir=tmp_path, enabled=False)
        cache.get("some_key")

        cache_events = [e for e in emitter.events if "cache" in e.event_type]
        assert len(cache_events) == 0


# ---------------------------------------------------------------------------
# make_eval_regression_event
# ---------------------------------------------------------------------------


class TestMakeEvalRegressionEvent:
    def test_event_type(self) -> None:
        evt = make_eval_regression_event(
            scenario_name="llm-diff/fail-under/single",
            current_score=0.6,
            baseline_score=0.8,
            threshold=0.8,
        )
        assert evt.event_type == "llm.eval.regression.failed"

    def test_payload_fields(self) -> None:
        evt = make_eval_regression_event(
            scenario_name="llm-diff/fail-under/single",
            current_score=0.55,
            baseline_score=0.80,
            threshold=0.80,
            metrics={"similarity": 0.55},
        )
        assert evt.payload["scenario_name"] == "llm-diff/fail-under/single"
        assert evt.payload["current_score"] == pytest.approx(0.55)
        assert evt.payload["baseline_score"] == pytest.approx(0.80)
        assert evt.payload["threshold"] == pytest.approx(0.80)
        assert evt.payload["regression_delta"] == pytest.approx(0.25)
        assert evt.payload["metrics"] == {"similarity": pytest.approx(0.55)}

    def test_regression_delta_computed(self) -> None:
        evt = make_eval_regression_event(
            scenario_name="llm-diff/fail-under/batch",
            current_score=0.70,
            baseline_score=0.90,
            threshold=0.90,
        )
        # regression_delta = baseline - current = 0.20
        assert evt.payload["regression_delta"] == pytest.approx(0.20)

    def test_metrics_optional(self) -> None:
        evt = make_eval_regression_event(
            scenario_name="llm-diff/fail-under/single",
            current_score=0.4,
            baseline_score=0.5,
            threshold=0.5,
        )
        assert evt.payload["metrics"] is None

    def test_scenario_id_generated(self) -> None:
        evt = make_eval_regression_event(
            scenario_name="llm-diff/fail-under/single",
            current_score=0.3,
            baseline_score=0.7,
            threshold=0.7,
        )
        assert isinstance(evt.payload["scenario_id"], str)
        assert len(evt.payload["scenario_id"]) > 0

    def test_event_validates(self) -> None:
        make_eval_regression_event(
            scenario_name="llm-diff/fail-under/single",
            current_score=0.60,
            baseline_score=0.75,
            threshold=0.75,
        ).validate()

    def test_optional_session_and_org(self) -> None:
        evt = make_eval_regression_event(
            scenario_name="llm-diff/fail-under/single",
            current_score=0.5,
            baseline_score=0.9,
            threshold=0.9,
            session_id="sess-abc",
            org_id="org-xyz",
        )
        assert evt.session_id == "sess-abc"
        assert evt.org_id == "org-xyz"
