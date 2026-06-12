"""Use cases for selecting a function with a language model."""

from src.domain import FunctionDefinition


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    """Build the instruction used by the LLM to select a function."""
    function_lines = [
        f"- {function.name}: {function.description}"
        for function in functions
    ]

    available_functions = "\n".join(function_lines)

    return (
        "Choose the function that best matches the user request.\n\n"
        "Available functions:\n"
        f"{available_functions}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Function:\n"
    )
