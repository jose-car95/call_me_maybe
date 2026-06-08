"""Tests for domain model validation."""

import pytest
from pydantic import ValidationError

from src.domain import FunctionDefinition, PromptCase


def test_prompt_rejects_blank_text() -> None:
    """A whitespace-only prompt is not a usable request."""
    with pytest.raises(ValidationError):
        PromptCase(prompt="   ")


def test_prompt_rejects_additional_keys() -> None:
    """Prompt input must match the schema exactly."""
    with pytest.raises(ValidationError):
        PromptCase.model_validate({"prompt": "Hello", "extra": True})


def test_function_rejects_unknown_parameter_type() -> None:
    """Function parameter types are restricted to supported JSON types."""
    raw_function = {
        "name": "fn_example",
        "description": "Example function.",
        "parameters": {"value": {"type": "unknown"}},
        "returns": {"type": "string"},
    }

    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(raw_function)


def test_function_rejects_blank_parameter_name() -> None:
    """Every function parameter needs a meaningful name."""
    raw_function = {
        "name": "fn_example",
        "description": "Example function.",
        "parameters": {" ": {"type": "string"}},
        "returns": {"type": "string"},
    }

    with pytest.raises(ValidationError):
        FunctionDefinition.model_validate(raw_function)
