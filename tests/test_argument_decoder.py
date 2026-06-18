"""Tests for constrained argument decoding helpers."""

from src.application.argument_decoder import (
    find_next_literal_token,
    tokenize_generation_step_literals,
    tokenize_literal_step
)
from src.application.argument_schema import ArgumentGenerationStep
from src.domain import ModelInferenceError

import pytest


class FakeModel:
    """Language model with predictable tokenization."""

    def encode(self, text: str) -> list[int]:
        """Encode text as character code points."""
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
        """Logits are not needed for literal tokenization."""
        return []


def test_tokenizes_literal_step() -> None:
    """Literal steps are converted into model tokens."""
    step = ArgumentGenerationStep(
        literal='"a": ',
        parameter_name=None,
        parameter_type=None
    )

    tokens = tokenize_literal_step(
        FakeModel(),
        step
    )

    assert tokens == [
        ord('"'),
        ord("a"),
        ord('"'),
        ord(":"),
        ord(" ")
    ]


def test_rejects_value_step_when_tokenizing_literal() -> None:
    """Value steps cannot be tokenized as fixed JSON text."""
    step = ArgumentGenerationStep(
        literal="",
        parameter_name="a",
        parameter_type="number"
    )

    with pytest.raises(ModelInferenceError, match="only literal"):
        tokenize_literal_step(
            FakeModel(),
            step
        )


def test_tokenizes_generation_step_literals() -> None:
    """Only literal generation steps are tokenized."""
    steps = [
        ArgumentGenerationStep(
            literal="{",
            parameter_name=None,
            parameter_type=None
        ),
        ArgumentGenerationStep(
            literal="",
            parameter_name="a",
            parameter_type="number"
        ),
        ArgumentGenerationStep(
            literal="}",
            parameter_name=None,
            parameter_type=None
        )
    ]

    tokenized_steps = tokenize_generation_step_literals(
        FakeModel(),
        steps
    )

    assert tokenized_steps == [
        [ord("{")],
        [],
        [ord("}")]
    ]


def test_finds_next_literal_token() -> None:
    """The next literal token follows the generated prefix."""
    next_token = find_next_literal_token(
        [10, 20, 30],
        [10]
    )

    assert next_token == 20


def test_returns_none_when_literal_is_complete() -> None:
    """Completed literals have no next token."""
    next_token = find_next_literal_token(
        [10, 20, 30],
        [10, 20, 30]
    )

    assert next_token is None


def test_rejects_invalid_literal_prefix() -> None:
    """Generated literal prefixes must match the expected literal."""
    with pytest.raises(ModelInferenceError, match="expected prefix"):
        find_next_literal_token(
            [10, 20, 30],
            [99]
        )
