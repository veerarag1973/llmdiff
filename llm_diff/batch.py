"""Batch prompt loading, YAML schema parsing, and ``{input}`` template expansion.

YAML schema expected at the path passed to :func:`load_batch`::

    prompts:
      - id: summarize
        text: "Summarize the following in 3 sentences: {input}"
        inputs: [sample1.txt, sample2.txt]
      - id: rewrite
        text: "Rewrite this in a formal tone: {input}"
        inputs: [sample3.txt]
      - id: plain
        text: "Explain recursion"   # no {input} needed

Each entry with ``inputs`` expands into one :class:`BatchItem` per file.
Entries without ``inputs`` (or with an empty list) produce a single item.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_diff.diff import DiffResult
    from llm_diff.providers import ComparisonResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchItem:
    """A single fully-resolved prompt ready for model comparison.

    Attributes
    ----------
    id:
        Unique identifier.  When expanded from an input file the id is
        ``"{prompt_id}:{input_filename}"``.
    prompt_text:
        Fully resolved prompt text (``{input}`` substituted if applicable).
    input_label:
        Original input filename or ``""`` when no ``{input}`` substitution.
    """

    id: str
    prompt_text: str
    input_label: str = ""


@dataclass
class BatchResult:
    """Result of running one :class:`BatchItem` through the diff pipeline.

    Attributes
    ----------
    item:
        The resolved prompt item that was compared.
    comparison:
        Raw model responses from :func:`~llm_diff.providers.compare_models`.
    diff_result:
        Word-level diff from :func:`~llm_diff.diff.word_diff`.
    semantic_score:
        Cosine similarity (0–1) when ``--semantic`` was requested; else ``None``.
    paragraph_scores:
        Per-paragraph similarity scores when ``--paragraph`` was requested;
        else ``None``.
    """

    item: BatchItem
    comparison: ComparisonResult
    diff_result: DiffResult
    semantic_score: float | None = field(default=None)
    paragraph_scores: list | None = field(default=None)


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------


def _expand_template(text: str, input_content: str) -> str:
    """Replace the ``{input}`` placeholder in *text* with *input_content*.

    Parameters
    ----------
    text:
        Prompt template that may contain ``{input}``.
    input_content:
        Content to substitute (leading/trailing whitespace already stripped
        by :func:`load_batch`).

    Returns
    -------
    str
        Prompt with ``{input}`` replaced.  Returned unchanged if no placeholder.
    """
    return text.replace("{input}", input_content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_batch(path: str | Path) -> list[BatchItem]:
    """Parse a ``prompts.yml`` file and return a flat list of :class:`BatchItem`.

    Parameters
    ----------
    path:
        Path to the YAML batch file.

    Returns
    -------
    list[BatchItem]
        Flat list of fully-resolved batch items ready for comparison.

    Raises
    ------
    FileNotFoundError
        When the batch file or a referenced input file does not exist.
    ValueError
        On schema violations: missing keys, wrong types, empty ``prompts`` list.
    ImportError
        When ``pyyaml`` is not installed.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Batch mode requires PyYAML.  Install it with:  pip install pyyaml"
        ) from exc

    batch_path = Path(path).resolve()
    if not batch_path.is_file():
        raise FileNotFoundError(f"Batch file not found: {path}")

    raw = batch_path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse batch YAML: {exc}") from exc

    if not isinstance(data, dict) or "prompts" not in data:
        raise ValueError(
            "Batch file must have a top-level 'prompts' key.  "
            "See the documentation for the expected schema."
        )

    prompts_raw = data["prompts"]
    if not isinstance(prompts_raw, list):
        raise ValueError("'prompts' must be a list of prompt entries.")

    if not prompts_raw:
        raise ValueError("'prompts' list is empty — nothing to process.")

    items: list[BatchItem] = []
    base_dir = batch_path.parent  # input file paths are resolved relative to the YAML

    for idx, entry in enumerate(prompts_raw):
        if not isinstance(entry, dict):
            raise ValueError(
                f"prompts[{idx}] must be a mapping (dict), "
                f"got {type(entry).__name__}."
            )

        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError(
                f"prompts[{idx}] must have a non-empty string 'id'."
            )

        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"prompts[{idx}] ('{entry_id}') must have a non-empty string 'text'."
            )

        inputs = entry.get("inputs")

        if inputs is None or (isinstance(inputs, list) and len(inputs) == 0):
            # No input expansion — use text verbatim
            items.append(BatchItem(id=entry_id, prompt_text=text))
        else:
            if not isinstance(inputs, list):
                raise ValueError(
                    f"prompts[{idx}] ('{entry_id}'): 'inputs' must be a list, "
                    f"got {type(inputs).__name__}."
                )
            for inp in inputs:
                inp_path = base_dir / str(inp)
                if not inp_path.is_file():
                    raise ValueError(
                        f"Input file not found: '{inp}' "
                        f"(referenced in prompt '{entry_id}')."
                    )
                content = inp_path.read_text(encoding="utf-8").strip()
                expanded = _expand_template(text, content)
                items.append(
                    BatchItem(
                        id=f"{entry_id}:{inp}",
                        prompt_text=expanded,
                        input_label=str(inp),
                    )
                )

    logger.info("Loaded %d batch item(s) from %s", len(items), batch_path)
    return items
