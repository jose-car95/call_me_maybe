"""Application ports shared by the function-calling workflow."""

from typing import Protocol


class LanguageModel(Protocol):
    """Model operations required by constrained decoding."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token identifiers.

        Args:
            text: Text to encode.

        Returns:
            Flat model token identifiers.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert token identifiers into text.

        Args:
            token_ids: Model token identifiers.

        Returns:
            Decoded text.
        """
        ...

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return logits for the next token.

        Args:
            input_ids: Existing model token sequence.

        Returns:
            One logit per model vocabulary token.
        """
        ...
