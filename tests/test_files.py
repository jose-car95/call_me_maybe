"""Tests for JSON file infrastructure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.models import (
    FunctionCallResult,
    FunctionDefinitionError,
    InputFileError,
    InputValidationError,
)
from src.files import (
    load_function_definitions,
    load_prompt_cases,
    write_results,
)


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON fixture to a temporary path."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_function(name: str = "fn_example") -> dict[str, Any]:
    """Return a valid function definition fixture."""
    return {
        "name": name,
        "description": "Example function.",
        "parameters": {"value": {"type": "string"}},
        "returns": {"type": "string"},
    }


def test_loads_current_project_inputs() -> None:
    """The provided project inputs satisfy the declared schemas."""
    prompts = load_prompt_cases(
        Path("data/input/function_calling_tests.json")
    )
    functions = load_function_definitions(
        Path("data/input/function_definitions.json")
    )

    assert len(prompts) == 11
    assert len(functions) == 5


def test_missing_file_raises_input_file_error(tmp_path: Path) -> None:
    """A missing input file produces a project-specific error."""
    with pytest.raises(InputFileError, match="does not exist"):
        load_prompt_cases(tmp_path / "missing.json")


def test_invalid_json_raises_input_file_error(tmp_path: Path) -> None:
    """Malformed JSON is reported before schema validation."""
    path = tmp_path / "invalid.json"
    path.write_text("[{", encoding="utf-8")

    with pytest.raises(InputFileError, match="invalid JSON"):
        load_prompt_cases(path)


def test_non_array_root_is_rejected(tmp_path: Path) -> None:
    """Input documents must contain a JSON array at their root."""
    path = tmp_path / "prompts.json"
    _write_json(path, {"prompt": "Hello"})

    with pytest.raises(InputValidationError, match="JSON array"):
        load_prompt_cases(path)


def test_invalid_prompt_reports_its_index(tmp_path: Path) -> None:
    """Validation errors identify the failing array element."""
    path = tmp_path / "prompts.json"
    _write_json(path, [{"prompt": "Valid"}, {"prompt": ""}])

    with pytest.raises(InputValidationError, match="index 1"):
        load_prompt_cases(path)


def test_empty_function_list_is_rejected(tmp_path: Path) -> None:
    """Processing requires at least one available function."""
    path = tmp_path / "functions.json"
    _write_json(path, [])

    with pytest.raises(FunctionDefinitionError, match="at least one"):
        load_function_definitions(path)


def test_duplicate_function_names_are_rejected(tmp_path: Path) -> None:
    """Function names must identify exactly one definition."""
    path = tmp_path / "functions.json"
    _write_json(path, [_valid_function(), _valid_function()])

    with pytest.raises(FunctionDefinitionError, match="duplicate"):
        load_function_definitions(path)


def test_write_results_creates_valid_json(tmp_path: Path) -> None:
    """Result models are serialized to a JSON array."""
    path = tmp_path / "nested" / "results.json"
    result = FunctionCallResult(
        prompt="Greet Ada",
        fn_name="fn_greet",
        args={"name": "Ada"},
    )

    write_results(path, [result])

    assert json.loads(path.read_text(encoding="utf-8")) == [
        {
            "prompt": "Greet Ada",
            "fn_name": "fn_greet",
            "args": {"name": "Ada"},
        }
    ]
