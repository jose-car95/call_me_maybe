"""Contracts required by the application layer."""

from typing import Protocol


class LanguageModel(Protocol):
    """Operations required from a language model."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token identifiers."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert token identifiers back into text."""
        ...

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return the logits for the next possible token."""
        ...
