"""Tests for function selection use cases."""

import pytest

from src.application import (
    build_function_selection_prompt,
    tokenize_function_names,
    find_allowed_next_tokens,
    select_best_allowed_token
)
from src.domain import FunctionDefinition, ReturnSpec, ModelInferenceError


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


class FakeLanguageModel:
    """Predictable tokenizer for function names."""

    def encode(self, text: str) -> list[int]:
        """Return token IDs associated with each function."""
        tokens = {
            "fn_add": [1, 2],
            "fn_greet": [1, 3]
        }

        return tokens[text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode is not needed by this test."""
        return ""

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Logits are not needed by this test."""
        return []


def test_tokenizes_every_function_name() -> None:
    """Each function name is converted into its token sequence."""
    functions = [
        create_function("fn_add", "Add values."),
        create_function("fn_greet", "Greet a person.")
    ]

    tokenized = tokenize_function_names(
        FakeLanguageModel(),
        functions
    )

    assert tokenized == {
        "fn_add": [1, 2],
        "fn_greet": [1, 3]
    }


def test_find_first_allowed_token() -> None:
    """All candidates share the same first token."""
    tokenized = {
        "fn_add": [1, 2],
        "fn_greet": [1, 3],
        "fn_reverse": [1, 4, 5]
    }

    allowed = find_allowed_next_tokens(tokenized, [])

    assert allowed == {1}


def test_finds_tokens_after_shared_prefix() -> None:
    """Every valid continuation after the prefix is returned."""
    tokenized = {
        "fn_add": [1, 2],
        "fn_greet": [1, 3],
        "fn_reverse": [1, 4, 5]
    }

    allowed = find_allowed_next_tokens(tokenized, [1])

    assert allowed == {2, 3, 4}


def test_finds_token_after_longer_prefix() -> None:
    """Only candidates matching the full prefix remain."""
    tokenized = {
        "fn_add": [1, 2],
        "fn_greet": [1, 3],
        "fn_reverse": [1, 4, 5]
    }

    allowed = find_allowed_next_tokens(tokenized, [1, 4])

    assert allowed == {5}


def test_returns_empty_set_for_invalid_prefix() -> None:
    """No continuation exists for an unknown prefix."""
    tokenized = {
        "fn_add": [1, 2],
        "fn_greet": [1, 3]
    }

    allowed = find_allowed_next_tokens(tokenized, [99])

    assert allowed == set()


def test_returns_empty_set_for_completed_name() -> None:
    """A completed function name has no next token."""
    tokenized = {
        "fn_add": [1, 2],
        "fn_greet": [1, 3]
    }

    allowed = find_allowed_next_tokens(tokenized, [1, 2])

    assert allowed == set()


def test_selects_highest_logit_among_allowed_tokens() -> None:
    """Disallowed tokens cannot win even with a higher logit."""
    logits = [0.1, 0.7, 0.9, 0.4, 0.8]
    allowed_tokens = {1, 3, 4}

    selected = select_best_allowed_token(
        logits,
        allowed_tokens
    )

    assert selected == 4


def test_rejects_empty_allowed_tokens() -> None:
    """Selection requires at least one valid continuation."""
    with pytest.raises(
        ModelInferenceError,
        match="no allowed tokens"
    ):
        select_best_allowed_token(
            [0.1, 0.2],
            set()
        )


def test_rejects_token_outside_logits() -> None:
    """Allowed token IDs must exist in the model vocabulary."""
    with pytest.raises(
        ModelInferenceError,
        match="outside"
    ):
        select_best_allowed_token(
            [0.1, 0.2],
            {2}
        )
