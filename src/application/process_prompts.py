"""Use case for processing prompts into function call results."""

from src.application.extract_arguments import build_empty_arguments
from src.application.ports import LanguageModel
from src.application.select_function import select_function_name
from src.domain import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
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
        args = build_empty_arguments(selected_function)

        results.append(
            FunctionCallResult(
                prompt=prompt.prompt,
                fn_name=selected_function.name,
                args=args
            )
        )

    return results
