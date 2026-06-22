"""JSON file adapters for project inputs and outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ValidationError

from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
    InputFileError,
    InputValidationError,
    OutputFileError,
    PromptCase
)


def _load_json(path: Path) -> Any:
    """Load a JSON document from the filesystem.

    Args:
        path: File expected to contain JSON data.

    Returns:
        Decoded JSON value.

    Raises:
        InputFileError: If the path is invalid, unreadable, or malformed.
    """
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
    model: type[BaseModel],
    item_name: str
) -> list[BaseModel]:
    """Validate a JSON array against one Pydantic model.

    Args:
        raw_data: Decoded JSON root value.
        path: Source path used in error messages.
        model: Pydantic model applied to each item.
        item_name: Human-readable item name used in errors.

    Returns:
        Validated model instances in input order.

    Raises:
        InputValidationError: If the root or any item is invalid.
    """
    if not isinstance(raw_data, list):
        raise InputValidationError(
            f"{path} must contain a JSON array"
        )

    validated: list[BaseModel] = []
    for index, item in enumerate(raw_data):
        try:
            validated.append(model.model_validate(item))
        except ValidationError as exc:
            message = f"invalid {item_name} at index {index}: {exc}"
            raise InputValidationError(message) from exc
    return validated


def load_prompt_cases(path: Path) -> list[PromptCase]:
    """Load and validate natural-language prompt test cases.

    Args:
        path: JSON file containing prompt entries.

    Returns:
        Validated prompt cases.

    Raises:
        InputFileError: If the file cannot be decoded.
        InputValidationError: If a prompt entry is invalid.
    """
    return cast(
        list[PromptCase],
        _validate_array(
            _load_json(path),
            path,
            PromptCase,
            "prompt"
        )
    )


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate available function definitions.

    Args:
        path: JSON file containing function definitions.

    Returns:
        Validated functions in input order.

    Raises:
        InputFileError: If the file cannot be decoded.
        InputValidationError: If a definition is invalid.
        FunctionDefinitionError: If no usable unique functions exist.
    """
    functions = cast(
        list[FunctionDefinition],
        _validate_array(
            _load_json(path),
            path,
            FunctionDefinition,
            "function"
        )
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
    """Write function-calling results as a JSON array.

    Args:
        path: Destination file path.
        results: Validated results to serialize.

    Raises:
        OutputFileError: If the destination cannot be created or written.
    """
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
