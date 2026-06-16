"""Tests for the prompt processing use case."""

import pytest

from src.application import process_prompts
from src.domain import (
    FunctionDefinition,
    FunctionDefinitionError,
    PromptCase,
)


class SingleFunctionModel:
    """Language model that selects the only available function."""

    def encode(self, text: str) -> list[int]:
        """Encode prompts and function names predictably."""
        if text == "fn_example":
            return [1]

        return [10, 20]

    def decode(self, token_ids: list[int]) -> str:
        """Decode is not needed by prompt processing."""
        return ""

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
                "enabled": {"type": "boolean"},
            },
            "returns": {"type": "string"},
        }
    )

    results = process_prompts(
        SingleFunctionModel(),
        prompts,
        [function]
    )

    assert [result.prompt for result in results] == ["First", "Second"]
    assert all(result.fn_name == "fn_example" for result in results)
    assert all(
        result.args == {"text": "", "enabled": False}
        for result in results
    )


def test_process_prompts_requires_a_function() -> None:
    """The use case fails clearly when no function is available."""
    with pytest.raises(FunctionDefinitionError, match="at least one"):
        process_prompts(
            SingleFunctionModel(),
            [PromptCase(prompt="Hello")],
            []
        )
