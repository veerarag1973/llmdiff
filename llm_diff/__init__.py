"""llm-diff — CLI tool for comparing LLM outputs."""

from __future__ import annotations

__version__ = "1.3.0"

from llm_diff.api import ComparisonReport, compare, compare_batch, compare_prompts
from llm_diff.diff import JsonStructDiffResult, json_struct_diff
from llm_diff.judge import JudgeResult
from llm_diff.multi import MultiModelReport, PairScore, run_multi_model
from llm_diff.pricing import CostEstimate
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

__all__ = [
    "__version__",
    "ComparisonReport",
    "compare",
    "compare_batch",
    "compare_prompts",
    "CostEstimate",
    "JudgeResult",
    "json_struct_diff",
    "JsonStructDiffResult",
    "MultiModelReport",
    "PairScore",
    "run_multi_model",
    # Schema events
    "EventEmitter",
    "configure_emitter",
    "emit",
    "get_emitter",
    "make_cache_event",
    "make_comparison_completed_event",
    "make_comparison_started_event",
    "make_cost_recorded_event",
    "make_eval_regression_event",
    "make_eval_scenario_event",
    "make_report_exported_event",
    "make_trace_span_event",
]
