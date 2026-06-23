"""Tests for the public Byte-Level BPE tokenizer."""

import json
from pathlib import Path

import pytest

from src.tokenizer import ByteLevelBPETokenizer

_PARITY_MODEL = "Qwen/Qwen3-0.6B"
_PARITY_SAMPLES = [
    "What is the sum of 2 and 3?",
    "Reverse the string 'hello'",
    "Add 1000000 and 2500",
    "   leading and  multiple   spaces",
    "trailing spaces   ",
    "tabs\there\ttoo",
    "Convert /usr/local/bin/python to uppercase",
    "café déjà vu — naïve",
    "日本語のテキスト 123",
    "mix:  4.5, -42 and 1,000",
    "snake_case and CamelCase  WORDS",
    "user@example.com  spaced",
]


def test_byte_bpe_tokenizer_applies_merges_and_round_trips(
    tmp_path: Path
) -> None:
    """The tokenizer merges known pairs and reverses byte encoding."""
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer_file.write_text(
        json.dumps(
            {
                "model": {
                    "vocab": {"h": 0, "i": 1, "hi": 2, "Ã": 3, "©": 4},
                    "merges": [["h", "i"]]
                }
            }
        ),
        encoding="utf-8"
    )
    tokenizer = ByteLevelBPETokenizer.from_file(tokenizer_file)

    assert tokenizer.encode("hi") == [2]
    assert tokenizer.decode([2]) == "hi"
    assert tokenizer.decode(tokenizer.encode("é")) == "é"


def test_encoding_matches_reference_tokenizer() -> None:
    """The custom encoder reproduces the reference Qwen token ids.

    The test is skipped when the reference tokenizer or its files are not
    locally available, so the suite stays runnable without network access.
    """
    pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    from transformers import AutoTokenizer

    try:
        path = hub.hf_hub_download(_PARITY_MODEL, "tokenizer.json")
        reference = AutoTokenizer.from_pretrained(_PARITY_MODEL)
    except Exception:  # noqa: BLE001 - any download/load issue means skip
        pytest.skip("reference Qwen tokenizer is not available")

    tokenizer = ByteLevelBPETokenizer.from_file(path)
    for text in _PARITY_SAMPLES:
        expected = reference.encode(text, add_special_tokens=False)
        assert tokenizer.encode(text) == expected, text
