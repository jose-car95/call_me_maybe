"""Use case for processing prompts into function call results."""

from __future__ import annotations

from typing import Any

from src.domain import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
    PromptCase
)


def _placeholder_value(type_name: str) -> Any:
    """Return a neutral placeholder value for a JSON schema type."""
    if type_name in {"number", "integer"}:
        return 0
    if type_name == "boolean":
        return False
    if type_name == "array":
        return []
    if type_name == "object":
        return {}
    return ""


def process_prompts(
    prompts: list[PromptCase],
    functions: list[FunctionDefinition],
) -> list[FunctionCallResult]:
    """Build provisional calls using the first available function."""
    if not functions:
        raise FunctionDefinitionError(
            "at least one function definition is required"
        )

    selected_function = functions[0]
    placeholder_args = {
        name: _placeholder_value(spec.type)
        for name, spec in selected_function.parameters.items()
    }

    return [
        FunctionCallResult(
            prompt=prompt.prompt,
            fn_name=selected_function.name,
            args=placeholder_args.copy()
        )
        for prompt in prompts
    ]
