"""Tests for the Qwen SDK adapter."""

from src.engine import LanguageModel
from src.llm import QwenAdapter


class FakeEncodedTokens:
    """Tensor-like object returned by the fake SDK."""

    def __init__(self, token_batches: list[list[int]]) -> None:
        """Store token batches."""
        self._token_batches = token_batches

    def tolist(self) -> list[list[int]]:
        """Returns tokens as nested lists."""
        return self._token_batches


class FakeSDKModel:
    """Test double implementing the SDK operations."""

    def encode(self, text: str) -> FakeEncodedTokens:
        """Return predictable encoded tokens."""
        return FakeEncodedTokens([[10, 20, 30]])

    def decode(self, token_ids: list[int]) -> str:
        """Return predictable decoded text."""
        return "decoded text"

    def get_logits_from_input_ids(
        self,
        input_ids: list[int]
    ) -> list[float]:
        """Return predictable logits."""
        return [0.1, 0.8, 0.2]

    def get_path_to_tokenizer_file(self) -> str:
        """Return an unused path required by the SDK contract."""
        return "tokenizer.json"


def accepts_language_model(model: LanguageModel) -> LanguageModel:
    """Return a model satisfying the application contract."""
    return model


def test_encode_flattens_sdk_token_batch() -> None:
    """The adapter converts the SDK tensor into a flat token list."""
    adapter = QwenAdapter(FakeSDKModel())

    assert adapter.encode("Hello") == [10, 20, 30]


def test_decode_delegates_to_sdk() -> None:
    """The adapter delegates token decoding to the SDK."""
    adapter = QwenAdapter(FakeSDKModel())

    assert adapter.decode([10, 20]) == "decoded text"


def test_get_logits_delegates_to_sdk() -> None:
    """The adapter exposes SDK logits through the application method."""
    adapter = QwenAdapter(FakeSDKModel())

    assert adapter.get_logits([10, 20]) == [0.1, 0.8, 0.2]


def test_adapter_satisfies_language_model_contract() -> None:
    """QwenAdapter is compatible with the application port."""
    adapter = QwenAdapter(FakeSDKModel())

    language_model = accepts_language_model(adapter)

    assert language_model is adapter
