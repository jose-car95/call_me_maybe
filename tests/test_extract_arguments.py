"""Tests for argument extraction prompts."""

from src.application import build_argument_extraction_prompt
from src.domain import FunctionDefinition, ParameterSpec, ReturnSpec


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
