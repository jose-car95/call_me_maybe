"""Constrained decoding helpers for argument generation."""

import json
from typing import Any

from src.application.argument_schema import (
    ArgumentGenerationStep,
    ArgumentSchema,
    build_argument_generation_steps
)
from src.application.ports import LanguageModel
from src.domain import JsonType, ModelInferenceError


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


def tokenize_value_step(
    model: LanguageModel,
    step: ArgumentGenerationStep,
    arguments: dict[str, Any]
) -> list[int]:
    """Tokenize a value generation step from provided arguments."""
    if not step.is_value() or step.parameter_name is None:
        raise ModelInferenceError(
            "only value steps can be tokenized as argument values"
        )

    if step.parameter_name not in arguments:
        raise ModelInferenceError(
            f"missing argument value for parameter: {step.parameter_name}"
        )

    return model.encode(
        json.dumps(arguments[step.parameter_name])
    )


def tokenize_generation_steps(
    model: LanguageModel,
    steps: list[ArgumentGenerationStep],
    arguments: dict[str, Any]
) -> list[list[int]]:
    """Tokenize every generation step into constrained token chunks."""
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

        tokenized_steps.append(
            tokenize_value_step(
                model,
                step,
                arguments
            )
        )

    return tokenized_steps


def build_constrained_argument_json(
    model: LanguageModel,
    steps: list[ArgumentGenerationStep],
    arguments: dict[str, Any]
) -> str:
    """Build JSON text by following constrained argument generation steps."""
    token_chunks = tokenize_generation_steps(
        model,
        steps,
        arguments
    )
    output_tokens = [
        token
        for chunk in token_chunks
        for token in chunk
    ]
    argument_json = model.decode(output_tokens)

    try:
        json.loads(argument_json)
    except json.JSONDecodeError as exc:
        raise ModelInferenceError(
            "constrained argument generation produced invalid JSON"
        ) from exc

    return argument_json


def build_constrained_argument_object(
    model: LanguageModel,
    steps: list[ArgumentGenerationStep],
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """Build a parsed argument object from constrained generation steps."""
    argument_json = build_constrained_argument_json(
        model,
        steps,
        arguments
    )
    decoded: Any = json.loads(argument_json)

    if not isinstance(decoded, dict):
        raise ModelInferenceError(
            "constrained argument generation did not produce an object"
        )

    result: dict[str, Any] = {}
    for key, value in decoded.items():
        if not isinstance(key, str):
            raise ModelInferenceError(
                "constrained argument object contains a non-string key"
            )
        result[key] = value

    return result


def validate_argument_object(
    schema: ArgumentSchema,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate an argument object against an argument schema."""
    expected_types = {
        name: type_name
        for name, type_name in schema.parameters
    }
    expected_names = set(expected_types)
    argument_names = set(arguments)

    missing_names = expected_names - argument_names
    if missing_names:
        sorted_missing_names = sorted(missing_names)
        raise ModelInferenceError(
            f"argument object is missing required keys: {sorted_missing_names}"
        )

    extra_names = argument_names - expected_names
    if extra_names:
        raise ModelInferenceError(
            f"argument object contains unexpected keys: {sorted(extra_names)}"
        )

    for name, type_name in expected_types.items():
        if not _value_matches_json_type(arguments[name], type_name):
            raise ModelInferenceError(
                f"argument {name} does not match expected type {type_name}"
            )

    return arguments


def build_validated_constrained_argument_object(
    model: LanguageModel,
    schema: ArgumentSchema,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """Build and validate an argument object using constrained steps."""
    steps = build_argument_generation_steps(schema)
    argument_object = build_constrained_argument_object(
        model,
        steps,
        arguments
    )

    return validate_argument_object(
        schema,
        argument_object
    )


def _value_matches_json_type(value: Any, type_name: JsonType) -> bool:
    """Return whether a Python value matches a JSON schema type."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)

    return False
