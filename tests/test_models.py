"""Tests for domain model validation."""

import pytest
from pydantic import ValidationError

from src.models import FunctionDefinition, PromptCase


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


def test_prompt_accepts_subject_plain_string_form() -> None:
    """Prompt cases accept the subject's plain-string format."""
    assert PromptCase.model_validate("Greet Ada").prompt == "Greet Ada"


def test_function_accepts_recursive_parameter_constraints() -> None:
    """Function parameters support enums and nested JSON schemas."""
    function = FunctionDefinition.model_validate(
        {
            "name": "fn_configure",
            "description": "Configure devices.",
            "parameters": {
                "devices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "firmware": {
                                "type": "string",
                                "enum": ["stable", "beta"]
                            }
                        },
                        "required": ["firmware"]
                    }
                }
            },
            "returns": {"type": "boolean"}
        }
    )

    items = function.parameters["devices"].items
    assert items is not None
    assert items.required == ["firmware"]
