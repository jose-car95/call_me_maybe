"""Tests for argument schema helpers."""

from src.application.argument_schema import (
    ArgumentGenerationStep,
    build_argument_generation_steps,
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


def test_builds_argument_generation_steps() -> None:
    """Argument generation steps separate literals from values."""
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
    steps = build_argument_generation_steps(schema)

    assert [
        (step.literal, step.parameter_name, step.parameter_type)
        for step in steps
    ] == [
        ("{", None, None),
        ('"a": ', None, None),
        ("", "a", "number"),
        (", ", None, None),
        ('"b": ', None, None),
        ("", "b", "number"),
        ("}", None, None)
    ]


def test_argument_generation_step_identifies_literal_and_value() -> None:
    """Generation steps identify literal and value positions."""
    literal_step = ArgumentGenerationStep(
        literal='"a": ',
        parameter_name=None,
        parameter_type=None
    )
    value_step = ArgumentGenerationStep(
        literal="",
        parameter_name="a",
        parameter_type="number"
    )

    assert literal_step.is_literal()
    assert not literal_step.is_value()
    assert value_step.is_value()
    assert not value_step.is_literal()
