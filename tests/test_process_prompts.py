"""Tests for the prompt processing use case."""

import pytest

from src.application import process_prompts
from src.domain import (
    FunctionDefinition,
    FunctionDefinitionError,
    PromptCase
)


class SingleFunctionModel:
    """Language model that selects the only available function."""

    def encode(self, text: str) -> list[int]:
        """Encode prompts and function names predictably."""
        if text == "fn_example":
            return [1]

        return [
            ord(character)
            for character in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        """Decode character code points into text."""
        return "".join(
            chr(token_id)
            for token_id in token_ids
        )

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Select token 1."""
        return [0.0, 1.0]


def test_process_prompts_returns_one_result_per_prompt() -> None:
    """The use case is independent from files and terminal arguments."""
    prompts = [PromptCase(prompt="First"), PromptCase(prompt="Second")]
    function = FunctionDefinition.model_validate(
        {
            "name": "fn_example",
            "description": "Example function.",
            "parameters": {
                "text": {"type": "string"},
                "enabled": {"type": "boolean"}
            },
            "returns": {"type": "string"}
        }
    )

    results = process_prompts(
        SingleFunctionModel(),
        prompts,
        [function]
    )

    assert [result.prompt for result in results] == ["First", "Second"]
    assert all(result.fn_name == "fn_example" for result in results)
    assert [result.args for result in results] == [
        {
            "text": "First",
            "enabled": False
        },
        {
            "text": "Second",
            "enabled": False
        }
    ]


def test_process_prompts_requires_a_function() -> None:
    """The use case fails clearly when no function is available."""
    with pytest.raises(FunctionDefinitionError, match="at least one"):
        process_prompts(
            SingleFunctionModel(),
            [PromptCase(prompt="Hello")],
            []
        )


def test_process_prompts_extracts_arguments_for_selected_function() -> None:
    """Prompt processing extracts arguments for the selected function."""
    prompts = [
        PromptCase(prompt="Add 2 and 3")
    ]
    function = FunctionDefinition.model_validate(
        {
            "name": "fn_example",
            "description": "Add two numbers.",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
    )

    results = process_prompts(
        SingleFunctionModel(),
        prompts,
        [function]
    )

    assert len(results) == 1
    assert results[0].prompt == "Add 2 and 3"
    assert results[0].fn_name == "fn_example"
    assert results[0].args == {
        "a": 2.0,
        "b": 3.0
    }
