"""Adapter between the application and the provided LLM SDK."""

from typing import Protocol, cast

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class EncodedTokens(Protocol):
    """Tensor-like result returned by the SDK tokenizer."""

    def tolist(self) -> list[list[int]]:
        """Convert the encoded tensor into nested Python lists."""
        ...


class SDKLanguageModel(Protocol):
    """Public SDK operations required by the adapter."""

    def encode(self, text: str) -> EncodedTokens:
        """Encode text into a token tensor."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token identifiers into text."""
        ...

    def get_logits_from_input_ids(
        self,
        input_ids: list[int]
    ) -> list[float]:
        """Return logits for the next token."""
        ...


class QwenAdapter:
    """Expose the provided Qwen SDK."""

    def __init__(
        self,
        model: SDKLanguageModel | None = None
    ) -> None:
        """Initialize the adapter with a provided or default model."""
        if model is None:
            model = cast(
                SDKLanguageModel,
                Small_LLM_Model()
            )

        self._model = model

    def encode(self, text: str) -> list[int]:
        """Convert text into a flat list of token identifiers."""
        token_batches = self._model.encode(text).tolist()

        if not token_batches:
            return []

        return [
            int(token_id)
            for token_id in token_batches[0]
        ]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token identifiers back into text."""
        return self._model.decode(token_ids)

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return the logits for the next possible token."""
        return self._model.get_logits_from_input_ids(input_ids)
