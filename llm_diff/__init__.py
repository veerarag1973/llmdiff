"""llm-diff — CLI tool for comparing LLM outputs."""

from __future__ import annotations

__version__ = "1.0.0"

from llm_diff.api import ComparisonReport, compare, compare_batch, compare_prompts

__all__ = [
    "__version__",
    "ComparisonReport",
    "compare",
    "compare_batch",
    "compare_prompts",
]
