"""Token-level constrained decoding over finite text choices."""

from collections.abc import Callable

from src.models import ModelInferenceError
from src.ports import LanguageModel


class ConstrainedDecoder:
    """Select allowed strings by masking all invalid token continuations."""

    def __init__(
        self,
        model: LanguageModel,
        trace: Callable[[str], None] | None = None
    ) -> None:
        """Initialize the decoder with model access and a token cache.

        Args:
            model: Language model used to score allowed next tokens.
            trace: Optional callback receiving generation decisions.
        """
        self._model = model
        self._trace = trace
        self._token_cache: dict[str, list[int]] = {}

    def encode(self, text: str) -> list[int]:
        """Encode text once and reuse its token identifiers.

        Args:
            text: Text to encode.

        Returns:
            Token identifiers produced by the configured model tokenizer.
        """
        if text not in self._token_cache:
            self._token_cache[text] = self._model.encode(text)
        return self._token_cache[text]

    def choose(
        self,
        prompt: str,
        choices: dict[str, object],
        label: str
    ) -> object:
        """Select one candidate through token-level constrained decoding.

        Args:
            prompt: Context given to the language model.
            choices: Allowed serialized texts mapped to their result values.
            label: Human-readable name used in traces and errors.

        Returns:
            Value associated with the completed constrained token sequence.

        Raises:
            ModelInferenceError: If no valid constrained continuation exists.
        """
        if not choices:
            raise ModelInferenceError(f"no valid choices for {label}")
        tokenized = {text: self.encode(text) for text in choices}
        if any(not tokens for tokens in tokenized.values()):
            raise ModelInferenceError(
                f"an empty token sequence exists for {label}"
            )

        prompt_tokens = self._model.encode(prompt)
        generated: list[int] = []
        while True:
            # When one candidate's tokens are a prefix of another's, the
            # completed (shorter) candidate wins: a closed-set decoder has
            # no boundary token to ask the model "stop or continue?".
            for text, tokens in tokenized.items():
                if tokens == generated:
                    if self._trace is not None:
                        self._trace(f"{label}={text}")
                    return choices[text]

            index = len(generated)
            allowed = {
                tokens[index]
                for tokens in tokenized.values()
                if tokens[:index] == generated and index < len(tokens)
            }
            logits = self._model.get_logits(prompt_tokens + generated)
            invalid = {token for token in allowed if token >= len(logits)}
            if not allowed or invalid:
                raise ModelInferenceError(
                    f"invalid constrained tokens for {label}"
                )
            selected = max(allowed, key=lambda token: logits[token])
            if self._trace is not None:
                token_text = self._model.decode([selected])
                self._trace(
                    f"{label} allowed={len(allowed)} "
                    f"token={selected} text={token_text!r}"
                )
            generated.append(selected)
