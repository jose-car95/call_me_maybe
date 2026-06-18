"""Tests for constrained argument decoding helpers."""

from src.application.argument_decoder import (
    build_constrained_argument_json,
    build_constrained_argument_object,
    build_validated_constrained_argument_object,
    find_next_literal_token,
    tokenize_generation_step_literals,
    tokenize_literal_step,
    tokenize_value_step,
    validate_argument_object
)
from src.application.argument_schema import (
    ArgumentGenerationStep,
    ArgumentSchema
)
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


def test_tokenizes_value_step() -> None:
    """Value steps are converted into JSON value tokens."""
    step = ArgumentGenerationStep(
        literal="",
        parameter_name="a",
        parameter_type="number"
    )

    tokens = tokenize_value_step(
        FakeModel(),
        step,
        {
            "a": 2.0
        }
    )

    assert tokens == [
        ord("2"),
        ord("."),
        ord("0")
    ]


def test_rejects_literal_step_when_tokenizing_value() -> None:
    """Literal steps cannot be tokenized as argument values."""
    step = ArgumentGenerationStep(
        literal="{",
        parameter_name=None,
        parameter_type=None
    )

    with pytest.raises(ModelInferenceError, match="only value"):
        tokenize_value_step(
            FakeModel(),
            step,
            {}
        )


def test_builds_constrained_argument_json() -> None:
    """Constrained argument generation follows literal and value steps."""
    steps = [
        ArgumentGenerationStep(
            literal="{",
            parameter_name=None,
            parameter_type=None
        ),
        ArgumentGenerationStep(
            literal='"a": ',
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

    argument_json = build_constrained_argument_json(
        FakeModel(),
        steps,
        {
            "a": 2.0
        }
    )

    assert argument_json == '{"a": 2.0}'


def test_builds_constrained_argument_object() -> None:
    """Constrained argument generation returns a parsed object."""
    steps = [
        ArgumentGenerationStep(
            literal="{",
            parameter_name=None,
            parameter_type=None
        ),
        ArgumentGenerationStep(
            literal='"name": ',
            parameter_name=None,
            parameter_type=None
        ),
        ArgumentGenerationStep(
            literal="",
            parameter_name="name",
            parameter_type="string"
        ),
        ArgumentGenerationStep(
            literal="}",
            parameter_name=None,
            parameter_type=None
        )
    ]

    argument_object = build_constrained_argument_object(
        FakeModel(),
        steps,
        {
            "name": "john"
        }
    )

    assert argument_object == {
        "name": "john"
    }


def test_validates_argument_object_against_schema() -> None:
    """Argument validation accepts matching keys and value types."""
    schema = ArgumentSchema(
        parameters=[
            ("name", "string"),
            ("age", "integer")
        ]
    )

    arguments = validate_argument_object(
        schema,
        {
            "name": "john",
            "age": 42
        }
    )

    assert arguments == {
        "name": "john",
        "age": 42
    }


def test_rejects_argument_object_with_missing_key() -> None:
    """Argument validation requires every schema key."""
    schema = ArgumentSchema(
        parameters=[
            ("name", "string"),
            ("age", "integer")
        ]
    )

    with pytest.raises(ModelInferenceError, match="missing required"):
        validate_argument_object(
            schema,
            {
                "name": "john"
            }
        )


def test_rejects_argument_object_with_extra_key() -> None:
    """Argument validation rejects keys outside the schema."""
    schema = ArgumentSchema(
        parameters=[
            ("name", "string")
        ]
    )

    with pytest.raises(ModelInferenceError, match="unexpected keys"):
        validate_argument_object(
            schema,
            {
                "name": "john",
                "extra": "value"
            }
        )


def test_rejects_argument_object_with_wrong_type() -> None:
    """Argument validation rejects values with the wrong JSON type."""
    schema = ArgumentSchema(
        parameters=[
            ("age", "integer")
        ]
    )

    with pytest.raises(ModelInferenceError, match="expected type integer"):
        validate_argument_object(
            schema,
            {
                "age": "42"
            }
        )


def test_builds_validated_constrained_argument_object() -> None:
    """Validated constrained generation builds an object from a schema."""
    schema = ArgumentSchema(
        parameters=[
            ("a", "number"),
            ("b", "number")
        ]
    )

    argument_object = build_validated_constrained_argument_object(
        FakeModel(),
        schema,
        {
            "a": 2.0,
            "b": 3.0
        }
    )

    assert argument_object == {
        "a": 2.0,
        "b": 3.0
    }
