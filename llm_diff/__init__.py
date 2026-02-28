"""llm-diff — CLI tool for comparing LLM outputs."""

from __future__ import annotations

__version__ = "1.2.0"

from llm_diff.api import ComparisonReport, compare, compare_batch, compare_prompts
from llm_diff.diff import JsonStructDiffResult, json_struct_diff
from llm_diff.judge import JudgeResult
from llm_diff.multi import MultiModelReport, PairScore, run_multi_model
from llm_diff.pricing import CostEstimate

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
]
