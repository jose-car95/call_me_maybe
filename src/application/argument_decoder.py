"""Constrained decoding helpers for argument generation."""

from src.application.argument_schema import ArgumentGenerationStep
from src.application.ports import LanguageModel
from src.domain import ModelInferenceError


def tokenize_literal_step(
    model: LanguageModel,
    step: ArgumentGenerationStep
) -> list[int]:
    """Tokenize a literal generation step."""
    if not step.is_literal():
        raise ModelInferenceError(
            "only literal steps can be tokenized as fixed text"
        )

    return model.encode(step.literal)


def tokenize_generation_step_literals(
    model: LanguageModel,
    steps: list[ArgumentGenerationStep]
) -> list[list[int]]:
    """Tokenize literal steps and leave value steps empty."""
    tokenized_steps: list[list[int]] = []

    for step in steps:
        if step.is_literal():
            tokenized_steps.append(
                tokenize_literal_step(
                    model,
                    step
                )
            )
            continue

        tokenized_steps.append([])

    return tokenized_steps


def find_next_literal_token(
    literal_tokens: list[int],
    generated_tokens: list[int]
) -> int | None:
    """Return the next required token for a literal."""
    generated_length = len(generated_tokens)

    if generated_length >= len(literal_tokens):
        return None

    expected_prefix = literal_tokens[:generated_length]
    if generated_tokens != expected_prefix:
        raise ModelInferenceError(
            "generated literal tokens do not match the expected prefix"
        )

    return literal_tokens[generated_length]
