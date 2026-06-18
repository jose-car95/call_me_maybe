"""Tests for argument schema helpers."""

from src.application.argument_schema import (
    build_argument_json_template,
    build_argument_schema
)
from src.domain import FunctionDefinition


def test_builds_argument_schema_from_function_definition() -> None:
    """Argument schemas preserve parameter names and types."""
    function = FunctionDefinition.model_validate(
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
    )

    schema = build_argument_schema(function)

    assert schema.parameters == [
        ("a", "number"),
        ("b", "number")
    ]


def test_builds_argument_json_template() -> None:
    """Argument JSON templates include typed placeholders."""
    function = FunctionDefinition.model_validate(
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
    )

    schema = build_argument_schema(function)
    template = build_argument_json_template(schema)

    assert template == '{"a": <number>, "b": <number>}'
