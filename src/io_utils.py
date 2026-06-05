"""Input and output helpers for JSON project files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import FunctionCallResult, FunctionDefinition, PromptCase


DEFAULT_INPUT_DIR = Path("data/input")
DEFAULT_OUTPUT_PATH = Path("data/output/function_calling_results.json")
DEFAULT_TESTS_PATH = DEFAULT_INPUT_DIR / "function_calling_tests.json"
DEFAULT_FUNCTIONS_PATH = DEFAULT_INPUT_DIR / "functions_definition.json"


def load_json_file(path: Path) -> Any:
    """Load a JSON file and return its decoded value."""
    if not path.exists():
        raise ValueError(f"input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"input path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    except OSError as exc:
        raise ValueError(f"could not read {path}: {exc}") from exc


def load_prompt_cases(path: Path) -> list[PromptCase]:
    """Load and validate natural-language prompt test cases."""
    raw_data: Any = load_json_file(path)
    if not isinstance(raw_data, list):
        raise ValueError(f"{path} must contain a JSON array")

    prompts: list[PromptCase] = []
    for index, item in enumerate(raw_data):
        try:
            prompts.append(PromptCase.model_validate(item))
        except ValidationError as exc:
            message = f"invalid prompt at index {index}: {exc}"
            raise ValueError(message) from exc
    return prompts


def load_function_definitions(path: Path) -> list[FunctionDefinition]:
    """Load and validate available function definitions."""
    raw_data: Any = load_json_file(path)
    if not isinstance(raw_data, list):
        raise ValueError(f"{path} must contain a JSON array")

    functions: list[FunctionDefinition] = []
    for index, item in enumerate(raw_data):
        try:
            functions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            message = f"invalid function at index {index}: {exc}"
            raise ValueError(message) from exc
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
        raise ValueError(f"could not write output file {path}: {exc}") from exc
