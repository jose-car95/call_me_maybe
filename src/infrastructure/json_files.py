"""JSON file adapters for project inputs and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.domain import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
    InputFileError,
    InputValidationError,
    OutputFileError,
    PromptCase
)


DEFAULT_INPUT_DIR = Path("data/input")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")
DEFAULT_TESTS_PATH = DEFAULT_INPUT_DIR / "function_calling_tests.json"
DEFAULT_FUNCTIONS_PATH = DEFAULT_INPUT_DIR / "functions_definition.json"

ModelT = TypeVar("ModelT", bound=BaseModel)


def _load_json(path: Path) -> Any:
    """Load a JSON document from the filesystem."""
    if not path.exists():
        raise InputFileError(f"input file does not exist: {path}")
    if not path.is_file():
        raise InputFileError(f"input path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        message = f"invalid JSON in {path}: {exc.msg}"
        raise InputFileError(message) from exc
    except OSError as exc:
        raise InputFileError(f"could not read {path}: {exc}") from exc


def _validate_array(
    raw_data: Any,
    path: Path,
    model: type[ModelT],
    item_name: str
) -> list[ModelT]:
    """Validate a JSON array against one Pydantic model."""
    if not isinstance(raw_data, list):
        raise InputValidationError(
            f"{path} must contain a JSON array"
        )

    validated: list[ModelT] = []
    for index, item in enumerate(raw_data):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            message = f"invalid {item_name} at index {index}: {exc}"
            raise InputValidationError(message) from exc
    return validated


def load_prompt_cases(path: Path) -> list[PromptCase]:
    """Load and validate natural-language prompt test cases."""
    return _validate_array(
        _load_json(path),
        path,
        PromptCase,
        "prompt"
    )


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate available function definitions."""
    functions = _validate_array(
        _load_json(path),
        path,
        FunctionDefinition,
        "function"
    )
    if not functions:
        raise FunctionDefinitionError(
            "at least one function definition is required"
        )

    names = [function.name for function in functions]
    duplicate_names = sorted(
        name for name in set(names) if names.count(name) > 1
    )
    if duplicate_names:
        duplicates = ", ".join(duplicate_names)
        raise FunctionDefinitionError(
            f"duplicate function names are not allowed: {duplicates}"
        )
    return functions


def write_results(path: Path, results: list[FunctionCallResult]) -> None:
    """Write function calling results as a JSON array."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [result.model_dump(mode="json") for result in results]
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        raise OutputFileError(
            f"could not write output file {path}: {exc}"
        ) from exc
