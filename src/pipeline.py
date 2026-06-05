"""Initial project pipeline.

The real LLM-based function selection will be implemented in later milestones.
This module keeps Hito 1 executable while preserving the final output shape.
"""

from __future__ import annotations

from typing import Any

from src.models import FunctionDefinition, FunctionCallResult, PromptCase


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


def build_initial_results(
    prompts: list[PromptCase],
    functions: list[FunctionDefinition],
) -> list[FunctionCallResult]:
    """Build placeholder results from the first function definition."""
    if not functions:
        raise ValueError("at least one function definition is required")

    first_function = functions[0]
    args = {
        name: _placeholder_value(spec.type)
        for name, spec in first_function.parameters.items()
    }

    return [
        FunctionCallResult(
            prompt=prompt.prompt,
            fn_name=first_function.name,
            args=args,
        )
        for prompt in prompts
    ]
