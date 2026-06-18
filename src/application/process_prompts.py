"""Use case for processing prompts into function call results."""

from typing import Any

from src.application.argument_decoder import (
    build_validated_constrained_argument_object
)
from src.application.argument_schema import build_argument_schema
from src.application.extract_arguments import extract_arguments
from src.application.ports import LanguageModel
from src.application.select_function import select_function_name
from src.domain import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
    ModelInferenceError,
    PromptCase
)


def _find_function_by_name(
    functions: list[FunctionDefinition],
    selected_name: str
) -> FunctionDefinition:
    """Return the function definition matching the selected name."""
    for function in functions:
        if function.name == selected_name:
            return function

    raise FunctionDefinitionError(
        f"selected function does not exist: {selected_name}"
    )


def _build_constrained_arguments(
    model: LanguageModel,
    user_prompt: str,
    function: FunctionDefinition
) -> dict[str, Any]:
    """Build arguments with constrained JSON generation and fallback."""
    fallback_arguments = extract_arguments(
        user_prompt,
        function
    )
    schema = build_argument_schema(function)

    try:
        return build_validated_constrained_argument_object(
            model,
            schema,
            fallback_arguments
        )
    except ModelInferenceError:
        return fallback_arguments


def process_prompts(
    model: LanguageModel,
    prompts: list[PromptCase],
    functions: list[FunctionDefinition]
) -> list[FunctionCallResult]:
    """Build function call results for every prompt."""
    if not functions:
        raise FunctionDefinitionError(
            "at least one function definition is required"
        )

    results: list[FunctionCallResult] = []

    for prompt in prompts:
        selected_name = select_function_name(
            model,
            prompt.prompt,
            functions
        )
        selected_function = _find_function_by_name(
            functions,
            selected_name
        )
        args = _build_constrained_arguments(
            model,
            prompt.prompt,
            selected_function
        )

        results.append(
            FunctionCallResult(
                prompt=prompt.prompt,
                fn_name=selected_function.name,
                args=args
            )
        )

    return results
