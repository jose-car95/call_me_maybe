"""Adapter between the application and the provided LLM SDK."""

from typing import Protocol, cast

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.models import CallMeMaybeError, ModelInferenceError
from src.tokenizer import ByteLevelBPETokenizer


class EncodedTokens(Protocol):
    """Tensor-like result returned by the SDK tokenizer."""

    def tolist(self) -> list[list[int]]:
        """Convert the encoded tensor into nested Python lists.

        Returns:
            One token-identifier list per encoded batch item.
        """
        ...


class SDKLanguageModel(Protocol):
    """Public SDK operations required by the adapter."""

    def encode(self, text: str) -> EncodedTokens:
        """Encode text into a token tensor.

        Args:
            text: Text to encode through the SDK tokenizer.

        Returns:
            Tensor-like encoded token batches.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token identifiers into text.

        Args:
            token_ids: Model token identifiers.

        Returns:
            Decoded text.
        """
        ...

    def get_logits_from_input_ids(
        self,
        input_ids: list[int]
    ) -> list[float]:
        """Return logits for the next token.

        Args:
            input_ids: Existing model token sequence.

        Returns:
            One logit per model vocabulary token.
        """
        ...

    def get_path_to_tokenizer_file(self) -> str:
        """Return the public tokenizer JSON path.

        Returns:
            Filesystem path to ``tokenizer.json``.
        """
        ...


class TextTokenizer(Protocol):
    """Text codec used independently from model inference."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token identifiers.

        Args:
            text: Text to encode.

        Returns:
            Flat model token identifiers.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token identifiers into text.

        Args:
            token_ids: Model token identifiers.

        Returns:
            Decoded text.
        """
        ...


class _SDKTokenizerFallback:
    """Compatibility codec for injected test SDK models."""

    def __init__(self, model: SDKLanguageModel) -> None:
        """Store an injected SDK model used as a compatibility codec.

        Args:
            model: Injected model implementing public SDK tokenization methods.
        """
        self._model = model

    def encode(self, text: str) -> list[int]:
        """Encode text through the injected SDK model.

        Args:
            text: Text to encode.

        Returns:
            Flattened token identifiers from the first batch item.
        """
        batches = self._model.encode(text).tolist()
        if not batches:
            return []
        return [int(token_id) for token_id in batches[0]]

    def decode(self, token_ids: list[int]) -> str:
        """Decode token identifiers through the injected SDK model.

        Args:
            token_ids: Model token identifiers.

        Returns:
            Decoded text.
        """
        return self._model.decode(token_ids)


class QwenAdapter:
    """Expose the provided Qwen SDK."""

    def __init__(
        self,
        model: SDKLanguageModel | None = None,
        device: str = "cpu",
        tokenizer: TextTokenizer | None = None
    ) -> None:
        """Initialize the adapter with a provided or default model.

        Args:
            model: Optional injected public SDK implementation.
            device: Device used when constructing the default Qwen model.
            tokenizer: Optional tokenizer override for tests or integrations.

        Raises:
            CallMeMaybeError: If project tokenizer initialization fails.
            ModelInferenceError: If the SDK model cannot be initialized.
        """
        injected_model = model is not None
        try:
            if model is None:
                model = cast(
                    SDKLanguageModel,
                    Small_LLM_Model(device=device)
                )
            if tokenizer is None:
                if injected_model:
                    tokenizer = _SDKTokenizerFallback(model)
                else:
                    tokenizer = ByteLevelBPETokenizer.from_file(
                        model.get_path_to_tokenizer_file()
                    )
        except CallMeMaybeError:
            raise
        except Exception as exc:
            raise ModelInferenceError(
                f"could not initialize language model: {exc}"
            ) from exc

        self._model = model
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        """Convert text into a flat list of token identifiers.

        Args:
            text: Text to encode with the configured tokenizer.

        Returns:
            Flat token identifiers.
        """
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """Convert token identifiers back into text.

        Args:
            token_ids: Token identifiers to decode.

        Returns:
            Decoded text.
        """
        return self._tokenizer.decode(token_ids)

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return logits for the next possible token.

        Args:
            input_ids: Existing model token sequence.

        Returns:
            One logit per vocabulary token.

        Raises:
            ModelInferenceError: If SDK inference fails.
        """
        try:
            return self._model.get_logits_from_input_ids(input_ids)
        except Exception as exc:
            raise ModelInferenceError(
                f"language model inference failed: {exc}"
            ) from exc
