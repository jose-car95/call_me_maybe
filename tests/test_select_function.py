"""Tests for function selection use cases."""

from src.application import build_function_selection_prompt
from src.domain import FunctionDefinition, ReturnSpec


def create_function(
    name: str,
    description: str
) -> FunctionDefinition:
    """Create a function definition for tests."""
    return FunctionDefinition(
        name=name,
        description=description,
        parameters={},
        returns=ReturnSpec(type="string")
    )


def test_builds_function_selection_prompt() -> None:
    """The prompt includes functions and the user request."""
    functions = [
        create_function(
            "fn_add_numbers",
            "Add two numbers."
        ),
        create_function(
            "fn_greet",
            "Generate a greeting."
        )
    ]

    prompt = build_function_selection_prompt(
        "Greet john",
        functions
    )

    assert "- fn_add_numbers: Add two numbers." in prompt
    assert "- fn_greet: Generate a greeting." in prompt
    assert "User request:\nGreet john" in prompt
    assert prompt.endswith("Function:\n")


def test_preserves_function_order() -> None:
    """Functions appear in the same order they were provided."""
    functions = [
        create_function("fn_first", "First function."),
        create_function("fn_second", "Second function.")
    ]

    prompt = build_function_selection_prompt(
        "Example request",
        functions
    )

    first_position = prompt.index("fn_first")
    second_position = prompt.index("fn_second")

    assert first_position < second_position
