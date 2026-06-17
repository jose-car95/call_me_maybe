"""Tests for argument extraction prompts."""

from src.application import (
    build_argument_extraction_prompt,
    build_empty_arguments,
    extract_arguments
)
from src.domain import FunctionDefinition, JsonType, ParameterSpec, ReturnSpec


def create_function() -> FunctionDefinition:
    """Create a function definition for argument extraction tests."""
    return FunctionDefinition(
        name="fn_greet",
        description="Generate a greeting message for a person by name.",
        parameters={
            "name": ParameterSpec(type="string")
        },
        returns=ReturnSpec(type="string")
    )


def create_function_with_parameters(
    parameters: dict[str, JsonType]
) -> FunctionDefinition:
    """Create a function definition with custom parameters."""
    return FunctionDefinition(
        name="fn_test",
        description="Test function.",
        parameters={
            name: ParameterSpec(type=type_name)
            for name, type_name in parameters.items()
        },
        returns=ReturnSpec(type="string")
    )


def test_builds_argument_extraction_prompt() -> None:
    """The prompt includes function metadata and user request."""
    function = create_function()

    prompt = build_argument_extraction_prompt(
        "Greet john",
        function
    )

    assert "Function:\nfn_greet" in prompt
    assert (
        "Description:\n"
        "Generate a greeting message for a person by name."
    ) in prompt
    assert "- name: string" in prompt
    assert "User request:\nGreet john" in prompt
    assert prompt.endswith("Arguments JSON:\n")


def test_builds_empty_string_arguments() -> None:
    """String parameters receive an empty string."""
    function = create_function_with_parameters(
        {
            "name": "string"
        }
    )

    args = build_empty_arguments(function)

    assert args == {
        "name": ""
    }


def test_builds_empty_number_arguments() -> None:
    """Number parameters receive zero."""
    function = create_function_with_parameters(
        {
            "a": "number",
            "b": "integer"
        }
    )

    args = build_empty_arguments(function)

    assert args == {
        "a": 0,
        "b": 0
    }


def test_builds_empty_boolean_array_and_object_arguments() -> None:
    """Complex JSON types receive schema-compatible empty values."""
    function = create_function_with_parameters(
        {
            "enabled": "boolean",
            "items": "array",
            "metadata": "object"
        }
    )

    args = build_empty_arguments(function)

    assert args == {
        "enabled": False,
        "items": [],
        "metadata": {}
    }


def test_extract_arguments_keeps_empty_values_when_text_is_missing() -> None:
    """Missing values keep schema-compatible empty values."""
    function = create_function_with_parameters(
        {
            "name": "string",
            "age": "integer"
        }
    )

    args = extract_arguments(
        "Greet John",
        function
    )

    assert args == {
        "name": "",
        "age": 0
    }


def test_extract_arguments_extracts_numbers_by_parameter_order() -> None:
    """Number parameters are filled from prompt numbers in order."""
    function = create_function_with_parameters(
        {
            "a": "number",
            "b": "number"
        }
    )

    args = extract_arguments(
        "Add 2 and 3",
        function
    )

    assert args == {
        "a": 2.0,
        "b": 3.0
    }


def test_extract_arguments_extracts_quoted_string() -> None:
    """String parameters are filled from quoted text."""
    function = create_function_with_parameters(
        {
            "text": "string"
        }
    )

    args = extract_arguments(
        "Reverse the string 'hello'",
        function
    )

    assert args == {
        "text": "hello"
    }
