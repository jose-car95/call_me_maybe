"""Tests for the command-line interface."""

import json
from pathlib import Path

import pytest

from src.interfaces import main


class SingleFunctionModel:
    """Language model that selects the first available function."""

    def encode(self, text: str) -> list[int]:
        """Encode prompts and function names predictably."""
        tokens = {
            "fn_add_numbers": [1],
            "fn_greet": [2],
            "fn_reverse_string": [3],
            "fn_get_square_root": [4],
            "fn_substitute_string_with_regex": [5]
        }

        if text in tokens:
            return tokens[text]

        return [10, 20]

    def decode(self, token_ids: list[int]) -> str:
        """Decode is not needed by the CLI test."""
        return ""

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Select token 1."""
        return [0.0, 1.0, 0.5, 0.4, 0.3, 0.2]


def test_cli_returns_zero_and_writes_output(
    tmp_path: Path,
) -> None:
    """A valid CLI execution returns success and creates output."""
    output_path = tmp_path / "results.json"

    exit_code = main(
        [
            "--input",
            "data/input/function_calling_tests.json",
            "--functions",
            "data/input/functions_definition.json",
            "--output",
            str(output_path)
        ],
        SingleFunctionModel()
    )

    assert exit_code == 0
    assert len(json.loads(output_path.read_text(encoding="utf-8"))) == 11


def test_cli_returns_one_for_missing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An expected input error becomes exit code one."""
    missing_path = tmp_path / "missing.json"

    exit_code = main(
        ["--input", str(missing_path)],
        SingleFunctionModel()
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.out
    assert "does not exist" in captured.out
