"""Public Byte-Level BPE tokenizer implementation."""

from functools import lru_cache
import json
from pathlib import Path
import unicodedata
from typing import Any

from src.models import InputFileError


def _bytes_to_unicode() -> dict[int, str]:
    """Build the reversible byte mapping used by GPT-style tokenizers.

    Returns:
        Mapping from every byte value to its printable Unicode representation.
    """
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    byte_values = visible[:]
    unicode_values = visible[:]
    extra_index = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + extra_index)
            extra_index += 1
    return dict(zip(byte_values, map(chr, unicode_values), strict=True))


def _is_letter(character: str) -> bool:
    """Check whether a character belongs to a Unicode letter category.

    Args:
        character: Single character to classify.

    Returns:
        ``True`` for Unicode letters.
    """
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    """Check whether a character belongs to a Unicode number category.

    Args:
        character: Single character to classify.

    Returns:
        ``True`` for Unicode numbers.
    """
    return unicodedata.category(character).startswith("N")


def _is_punctuation(character: str) -> bool:
    """Check whether a character is neither text nor whitespace.

    Args:
        character: Single character to classify.

    Returns:
        ``True`` for punctuation and symbol characters.
    """
    return (
        not character.isspace()
        and not _is_letter(character)
        and not _is_number(character)
    )


def _pretokenize(text: str) -> list[str]:
    """Apply the Qwen tokenizer's Unicode-aware split behavior.

    Args:
        text: NFC-normalized text to split before BPE.

    Returns:
        Pretokenized text fragments in their original order.
    """
    pieces: list[str] = []
    index = 0
    contractions = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")
    while index < len(text):
        lowered = text[index:].lower()
        contraction = next(
            (item for item in contractions if lowered.startswith(item)),
            None
        )
        if contraction is not None:
            pieces.append(text[index:index + len(contraction)])
            index += len(contraction)
            continue

        start = index
        if (
            not _is_letter(text[index])
            and not _is_number(text[index])
            and text[index] not in "\r\n"
            and index + 1 < len(text)
            and _is_letter(text[index + 1])
        ):
            index += 1
        if index < len(text) and _is_letter(text[index]):
            while index < len(text) and _is_letter(text[index]):
                index += 1
            pieces.append(text[start:index])
            continue

        index = start
        if _is_number(text[index]):
            pieces.append(text[index])
            index += 1
            continue

        if (
            text[index] == " "
            and index + 1 < len(text)
            and _is_punctuation(text[index + 1])
        ):
            index += 1
        if index < len(text) and _is_punctuation(text[index]):
            while index < len(text) and _is_punctuation(text[index]):
                index += 1
            while index < len(text) and text[index] in "\r\n":
                index += 1
            pieces.append(text[start:index])
            continue

        index = start
        if text[index] in "\r\n":
            while index < len(text) and text[index] in "\r\n":
                index += 1
            pieces.append(text[start:index])
            continue

        if text[index].isspace():
            while index < len(text) and text[index].isspace():
                index += 1
            pieces.append(text[start:index])
            continue

        pieces.append(text[index])
        index += 1
    return pieces


class ByteLevelBPETokenizer:
    """Encode and decode text with a public Byte-Level BPE vocabulary."""

    def __init__(
        self,
        vocabulary: dict[str, int],
        merges: list[list[str]]
    ) -> None:
        """Initialize vocabulary, merge ranks, and byte mappings.

        Args:
            vocabulary: BPE token strings mapped to model token identifiers.
            merges: Ordered token-pair merge rules.
        """
        self._vocabulary = vocabulary
        self._tokens = {
            token_id: token
            for token, token_id in vocabulary.items()
        }
        self._ranks = {
            (left, right): rank
            for rank, (left, right) in enumerate(merges)
        }
        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {
            character: byte
            for byte, character in self._byte_encoder.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "ByteLevelBPETokenizer":
        """Load tokenizer data from a public SDK path.

        Args:
            path: Public ``tokenizer.json`` path returned by the SDK.

        Returns:
            Initialized Byte-Level BPE tokenizer.

        Raises:
            InputFileError: If tokenizer data is unreadable or malformed.
        """
        try:
            with Path(path).open("r", encoding="utf-8") as file:
                payload: Any = json.load(file)
            vocabulary = payload["model"]["vocab"]
            merges = payload["model"]["merges"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise InputFileError(f"invalid tokenizer file: {path}") from exc
        if not isinstance(vocabulary, dict) or not isinstance(merges, list):
            raise InputFileError(f"invalid tokenizer file: {path}")
        return cls(vocabulary, merges)

    @lru_cache(maxsize=8192)
    def _apply_bpe(self, piece: str) -> tuple[str, ...]:
        """Apply ranked BPE merges to one byte-encoded fragment.

        Args:
            piece: Fragment encoded with the reversible byte alphabet.

        Returns:
            Final vocabulary token strings after all available merges.
        """
        word = tuple(piece)
        while len(word) > 1:
            ranked_pairs = [
                (self._ranks[pair], pair)
                for pair in zip(word, word[1:])
                if pair in self._ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs)
            merged: list[str] = []
            index = 0
            while index < len(word):
                pair = word[index:index + 2]
                if index + 1 < len(word) and pair == best_pair:
                    merged.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
        return word

    def encode(self, text: str) -> list[int]:
        """Encode text into model token identifiers.

        Args:
            text: Unicode text to normalize and tokenize.

        Returns:
            Token identifiers compatible with Qwen.

        Raises:
            InputFileError: If the loaded vocabulary cannot encode a fragment.
        """
        token_ids: list[int] = []
        normalized = unicodedata.normalize("NFC", text)
        for piece in _pretokenize(normalized):
            byte_piece = "".join(
                self._byte_encoder[byte]
                for byte in piece.encode("utf-8")
            )
            try:
                token_ids.extend(
                    self._vocabulary[token]
                    for token in self._apply_bpe(byte_piece)
                )
            except KeyError as exc:
                raise InputFileError(
                    "tokenizer vocabulary cannot encode input"
                ) from exc
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode model token identifiers into Unicode text.

        Args:
            token_ids: Token identifiers from the loaded vocabulary.

        Returns:
            Decoded Unicode text.

        Raises:
            InputFileError: If a token identifier is unknown.
        """
        try:
            encoded = "".join(self._tokens[token_id] for token_id in token_ids)
            raw = bytes(self._byte_decoder[character] for character in encoded)
        except KeyError as exc:
            raise InputFileError("unknown tokenizer token") from exc
        return raw.decode("utf-8", errors="replace")
