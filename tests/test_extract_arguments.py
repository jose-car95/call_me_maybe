"""Tests for argument extraction prompts."""

from src.application import (
    build_argument_extraction_prompt,
    build_argument_schema_template,
    build_constrained_argument_generation_prompt,
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
            "age": "integer"
        }
    )

    args = extract_arguments(
        "Greet John",
        function
    )

    assert args == {
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


def test_extract_arguments_extracts_last_word_for_unquoted_string() -> None:
    """String parameters can be filled from the last prompt word."""
    function = create_function_with_parameters(
        {
            "name": "string"
        }
    )

    args = extract_arguments(
        "Greet john",
        function
    )

    assert args == {
        "name": "john"
    }


def test_extract_arguments_extracts_unquoted_email() -> None:
    """Email parameters are filled from unquoted email addresses."""
    function = create_function_with_parameters(
        {
            "email": "string"
        }
    )

    args = extract_arguments(
        "Extract the email ada.lovelace@example.com from this request",
        function
    )

    assert args == {
        "email": "ada.lovelace@example.com"
    }


def test_extract_arguments_extracts_boolean_true() -> None:
    """Boolean parameters are filled from true-like words."""
    function = create_function_with_parameters(
        {
            "is_admin": "boolean"
        }
    )

    args = extract_arguments(
        "Create a user profile with admin true",
        function
    )

    assert args == {
        "is_admin": True
    }


def test_extract_arguments_extracts_quoted_strings_by_parameter_order(
) -> None:
    """String parameters are filled from quoted text in order."""
    function = create_function_with_parameters(
        {
            "regex": "string",
            "replacement": "string",
            "source_string": "string"
        }
    )

    args = extract_arguments(
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        function
    )

    assert args == {
        "regex": "cat",
        "replacement": "dog",
        "source_string": "The cat sat on the mat with another cat"
    }


def test_extract_arguments_extracts_substitution_word_arguments() -> None:
    """
    Regex substitution arguments are extracted from word replacement prompts.
    """
    function = create_function_with_parameters(
        {
            "source_string": "string",
            "regex": "string",
            "replacement": "string"
        }
    )

    args = extract_arguments(
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'",
        function
    )

    assert args == {
        "source_string": "The cat sat on the mat with another cat",
        "regex": "cat",
        "replacement": "dog"
    }


def test_extract_arguments_extracts_substitution_number_arguments() -> None:
    """Number replacement prompts become a numeric regex."""
    function = create_function_with_parameters(
        {
            "source_string": "string",
            "regex": "string",
            "replacement": "string"
        }
    )

    args = extract_arguments(
        "Replace all numbers in \"Hello 34 I'm 233 years old\" "
        "with NUMBERS",
        function
    )

    assert args == {
        "source_string": "Hello 34 I'm 233 years old",
        "regex": r"\d+",
        "replacement": "NUMBERS"
    }


def test_extract_arguments_extracts_substitution_vowel_arguments() -> None:
    """Vowel replacement prompts become a vowel regex."""
    function = create_function_with_parameters(
        {
            "source_string": "string",
            "regex": "string",
            "replacement": "string"
        }
    )

    args = extract_arguments(
        "Replace all vowels in 'Programming is fun' with asterisks",
        function
    )

    assert args == {
        "source_string": "Programming is fun",
        "regex": r"[aeiouAEIOU]",
        "replacement": "asterisks"
    }


def test_extract_arguments_extracts_substitution_space_arguments() -> None:
    """Space replacement prompts become a whitespace regex."""
    function = create_function_with_parameters(
        {
            "source_string": "string",
            "regex": "string",
            "replacement": "string"
        }
    )

    args = extract_arguments(
        "Replace all spaces in 'hello world from 42' with underscores",
        function
    )

    assert args == {
        "source_string": "hello world from 42",
        "regex": r"\s+",
        "replacement": "underscores"
    }


def test_builds_argument_schema_template() -> None:
    """Argument schema templates include every required parameter."""
    function = create_function_with_parameters(
        {
            "name": "string",
            "age": "integer",
            "active": "boolean"
        }
    )

    template = build_argument_schema_template(function)

    assert template == {
        "name": "",
        "age": 0,
        "active": False
    }


def test_builds_constrained_argument_generation_prompt() -> None:
    """Argument generation prompts include schema templates."""
    function = create_function_with_parameters(
        {
            "a": "number",
            "b": "number"
        }
    )

    prompt = build_constrained_argument_generation_prompt(
        "What is the sum of 2 and 3?",
        function
    )

    assert "Function:\nfn_test" in prompt
    assert "- a: number" in prompt
    assert "- b: number" in prompt
    assert "User request:\nWhat is the sum of 2 and 3?" in prompt
    assert '"a": 0' in prompt
    assert '"b": 0' in prompt
    assert prompt.endswith("Arguments JSON:\n")
