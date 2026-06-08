"""Tests for the command-line interface."""

import json
from pathlib import Path

import pytest

from src.interfaces import main


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
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert len(json.loads(output_path.read_text(encoding="utf-8"))) == 11


def test_cli_returns_one_for_missing_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An expected input error becomes exit code one."""
    missing_path = tmp_path / "missing.json"

    exit_code = main(["--input", str(missing_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error:" in captured.out
    assert "does not exist" in captured.out
