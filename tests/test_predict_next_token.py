"""Tests for next-token prediction."""

import pytest

from src.application import predict_next_token
from src.domain import ModelInferenceError


class FakeLanguageModel:
    """Predictable language model for application tests."""

    def encode(self, text: str) -> list[int]:
        """Return predictable input IDs."""
        return [10, 20]

    def decode(self, token_ids: list[int]) -> str:
        """Decode the selected test token."""
        return " selected"

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Make token 1 the most probable token."""
        return [0.1, 0.8, 0.2]


class EmptyLogitsModel(FakeLanguageModel):
    """Language model returning no token probabilities."""

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return an invalid empty logits collection."""
        return []


def test_predicts_token_with_highest_logit() -> None:
    """The token ID corresponds to the highest logit position."""
    prediction = predict_next_token(
        FakeLanguageModel(),
        "Hello"
    )

    assert prediction.input_ids == [10, 20]
    assert prediction.logits_count == 3
    assert prediction.token_id == 1
    assert prediction.token_text == " selected"


def test_rejects_empty_logits() -> None:
    """Prediction fails clearly when the model returns no logits."""
    with pytest.raises(ModelInferenceError, match="no logits"):
        predict_next_token(
            EmptyLogitsModel(),
            "Hello"
        )
