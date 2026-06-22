"""Tests for the public Byte-Level BPE tokenizer."""

import json
from pathlib import Path

from src.tokenizer import ByteLevelBPETokenizer


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
