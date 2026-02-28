"""Tests for llm_diff.batch — YAML batch loader and template expansion."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_diff.batch import BatchItem, BatchResult, _expand_template, load_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, content: str) -> Path:
    """Write *content* to *path* and return *path*."""
    path.write_text(content, encoding="utf-8")
    return path


def _make_batch_yaml(tmp_path: Path, content: str) -> Path:
    return _write_yaml(tmp_path / "prompts.yml", content)


# ---------------------------------------------------------------------------
# _expand_template
# ---------------------------------------------------------------------------


class TestExpandTemplate:
    def test_replaces_input_placeholder(self) -> None:
        result = _expand_template("Summarize: {input}", "hello world")
        assert result == "Summarize: hello world"

    def test_no_placeholder_returns_unchanged(self) -> None:
        result = _expand_template("No placeholder here", "ignored")
        assert result == "No placeholder here"

    def test_multiple_placeholders_replaced(self) -> None:
        result = _expand_template("{input} and {input}", "X")
        assert result == "X and X"

    def test_empty_input_content(self) -> None:
        result = _expand_template("Prefix: {input} suffix", "")
        assert result == "Prefix:  suffix"

    def test_empty_text_returns_empty(self) -> None:
        result = _expand_template("", "anything")
        assert result == ""


# ---------------------------------------------------------------------------
# BatchItem dataclass
# ---------------------------------------------------------------------------


class TestBatchItem:
    def test_frozen_cannot_be_mutated(self) -> None:
        item = BatchItem(id="x", prompt_text="hello")
        with pytest.raises((AttributeError, TypeError)):
            item.id = "y"  # type: ignore[misc]

    def test_default_input_label_is_empty(self) -> None:
        item = BatchItem(id="x", prompt_text="hello")
        assert item.input_label == ""

    def test_explicit_input_label(self) -> None:
        item = BatchItem(id="x", prompt_text="hello", input_label="file.txt")
        assert item.input_label == "file.txt"

    def test_equality(self) -> None:
        a = BatchItem(id="x", prompt_text="hello", input_label="")
        b = BatchItem(id="x", prompt_text="hello", input_label="")
        assert a == b

    def test_inequality_on_id(self) -> None:
        a = BatchItem(id="x", prompt_text="hello")
        b = BatchItem(id="y", prompt_text="hello")
        assert a != b

    def test_has_required_fields(self) -> None:
        field_names = {f.name for f in fields(BatchItem)}
        assert {"id", "prompt_text", "input_label"} == field_names


# ---------------------------------------------------------------------------
# BatchResult dataclass
# ---------------------------------------------------------------------------


class TestBatchResult:
    def _make(self, semantic_score: float | None = None) -> BatchResult:
        item = BatchItem(id="t", prompt_text="test")
        comparison = MagicMock()
        diff_result = MagicMock()
        return BatchResult(
            item=item,
            comparison=comparison,
            diff_result=diff_result,
            semantic_score=semantic_score,
        )

    def test_semantic_score_defaults_to_none(self) -> None:
        br = self._make()
        assert br.semantic_score is None

    def test_semantic_score_stored(self) -> None:
        br = self._make(semantic_score=0.85)
        assert br.semantic_score == pytest.approx(0.85)

    def test_item_accessible(self) -> None:
        br = self._make()
        assert br.item.id == "t"

    def test_mutable(self) -> None:
        br = self._make()
        br.semantic_score = 0.5
        assert br.semantic_score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# load_batch — happy paths
# ---------------------------------------------------------------------------


class TestLoadBatchHappy:
    def test_single_prompt_no_inputs(self, tmp_path: Path) -> None:
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: q1\n"
            "    text: 'Explain gravity'\n",
        )
        items = load_batch(yaml_path)
        assert len(items) == 1
        assert items[0].id == "q1"
        assert items[0].prompt_text == "Explain gravity"
        assert items[0].input_label == ""

    def test_two_prompts_no_inputs(self, tmp_path: Path) -> None:
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: a\n"
            "    text: First\n"
            "  - id: b\n"
            "    text: Second\n",
        )
        items = load_batch(yaml_path)
        assert len(items) == 2
        assert items[0].id == "a"
        assert items[1].id == "b"

    def test_single_input_expands_template(self, tmp_path: Path) -> None:
        inp = tmp_path / "doc.txt"
        inp.write_text("My document content", encoding="utf-8")
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: summarize\n"
            "    text: 'Summarize: {input}'\n"
            "    inputs: [doc.txt]\n",
        )
        items = load_batch(yaml_path)
        assert len(items) == 1
        assert items[0].id == "summarize:doc.txt"
        assert "My document content" in items[0].prompt_text
        assert items[0].input_label == "doc.txt"

    def test_multiple_inputs_per_prompt(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("Content A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("Content B", encoding="utf-8")
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: rewrite\n"
            "    text: 'Rewrite: {input}'\n"
            "    inputs: [a.txt, b.txt]\n",
        )
        items = load_batch(yaml_path)
        assert len(items) == 2
        assert items[0].id == "rewrite:a.txt"
        assert "Content A" in items[0].prompt_text
        assert items[1].id == "rewrite:b.txt"
        assert "Content B" in items[1].prompt_text

    def test_empty_inputs_list_no_expansion(self, tmp_path: Path) -> None:
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: plain\n"
            "    text: 'No expansion needed'\n"
            "    inputs: []\n",
        )
        items = load_batch(yaml_path)
        assert len(items) == 1
        assert items[0].id == "plain"
        assert items[0].input_label == ""

    def test_mixed_prompts_with_and_without_inputs(self, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("data", encoding="utf-8")
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: plain\n"
            "    text: 'Static prompt'\n"
            "  - id: dynamic\n"
            "    text: 'Expand: {input}'\n"
            "    inputs: [data.txt]\n",
        )
        items = load_batch(yaml_path)
        assert len(items) == 2
        assert items[0].id == "plain"
        assert items[1].id == "dynamic:data.txt"

    def test_input_file_stripped_of_whitespace(self, tmp_path: Path) -> None:
        inp = tmp_path / "padded.txt"
        inp.write_text("  content here  \n\n", encoding="utf-8")
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: t\n"
            "    text: '{input}'\n"
            "    inputs: [padded.txt]\n",
        )
        items = load_batch(yaml_path)
        assert items[0].prompt_text == "content here"

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: x\n"
            "    text: Hello\n",
        )
        items = load_batch(yaml_path)  # Path object (not string)
        assert len(items) == 1

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: x\n"
            "    text: Hello\n",
        )
        items = load_batch(str(yaml_path))
        assert len(items) == 1

    def test_returns_list_of_batch_items(self, tmp_path: Path) -> None:
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: x\n"
            "    text: Hello\n",
        )
        items = load_batch(yaml_path)
        assert isinstance(items, list)
        assert all(isinstance(item, BatchItem) for item in items)

    def test_large_batch_ten_prompts(self, tmp_path: Path) -> None:
        lines = ["prompts:"]
        for i in range(10):
            lines += [f"  - id: p{i}", f"    text: Prompt {i}"]
        yaml_path = _make_batch_yaml(tmp_path, "\n".join(lines) + "\n")
        items = load_batch(yaml_path)
        assert len(items) == 10

    def test_no_input_placeholder_with_inputs(self, tmp_path: Path) -> None:
        """Text without {input} is used verbatim even when inputs are listed."""
        (tmp_path / "x.txt").write_text("ignored", encoding="utf-8")
        yaml_path = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: novar\n"
            "    text: 'Static text'\n"
            "    inputs: [x.txt]\n",
        )
        items = load_batch(yaml_path)
        assert items[0].prompt_text == "Static text"


# ---------------------------------------------------------------------------
# load_batch — error cases
# ---------------------------------------------------------------------------


class TestLoadBatchErrors:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Batch file not found"):
            load_batch(tmp_path / "nonexistent.yml")

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yml"
        # An unclosed flow sequence is a genuine YAML parse error
        bad.write_text("[unclosed bracket", encoding="utf-8")
        with pytest.raises(ValueError, match="Failed to parse batch YAML"):
            load_batch(bad)

    def test_missing_prompts_key_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(tmp_path, "other_key: []\n")
        with pytest.raises(ValueError, match="'prompts' key"):
            load_batch(f)

    def test_prompts_not_a_list_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(tmp_path, "prompts: 'not a list'\n")
        with pytest.raises(ValueError, match="'prompts' must be a list"):
            load_batch(f)

    def test_empty_prompts_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(tmp_path, "prompts: []\n")
        with pytest.raises(ValueError, match="empty"):
            load_batch(f)

    def test_entry_not_a_dict_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(tmp_path, "prompts:\n  - just_a_string\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_batch(f)

    def test_missing_id_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(
            tmp_path,
            "prompts:\n  - text: Hello\n",
        )
        with pytest.raises(ValueError, match="'id'"):
            load_batch(f)

    def test_empty_id_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(
            tmp_path,
            "prompts:\n  - id: '   '\n    text: Hello\n",
        )
        with pytest.raises(ValueError, match="'id'"):
            load_batch(f)

    def test_missing_text_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(
            tmp_path,
            "prompts:\n  - id: p1\n",
        )
        with pytest.raises(ValueError, match="'text'"):
            load_batch(f)

    def test_empty_text_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(
            tmp_path,
            "prompts:\n  - id: p1\n    text: ''\n",
        )
        with pytest.raises(ValueError, match="'text'"):
            load_batch(f)

    def test_inputs_not_a_list_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(
            tmp_path,
            "prompts:\n  - id: p1\n    text: Hello\n    inputs: 'not a list'\n",
        )
        with pytest.raises(ValueError, match="'inputs' must be a list"):
            load_batch(f)

    def test_input_file_not_found_raises(self, tmp_path: Path) -> None:
        f = _make_batch_yaml(
            tmp_path,
            "prompts:\n"
            "  - id: p1\n"
            "    text: '{input}'\n"
            "    inputs: [nonexistent.txt]\n",
        )
        with pytest.raises(ValueError, match="Input file not found"):
            load_batch(f)

    def test_null_yaml_raises(self, tmp_path: Path) -> None:
        """An empty / null YAML document raises ValueError."""
        f = _make_batch_yaml(tmp_path, "")
        with pytest.raises(ValueError, match="'prompts' key"):
            load_batch(f)

    def test_import_error_when_pyyaml_not_installed(self, tmp_path: Path) -> None:
        """A clear ImportError is raised when pyyaml is not installed."""
        import sys
        from unittest.mock import patch

        f = _make_batch_yaml(tmp_path, "prompts:\n  - id: x\n    text: Hello\n")
        with patch.dict(sys.modules, {"yaml": None}):
            with pytest.raises(ImportError, match="pyyaml"):
                load_batch(f)
