"""Use cases for extracting function arguments."""

from src.domain import FunctionDefinition


def build_argument_extraction_prompt(
    user_prompt: str,
    function: FunctionDefinition
) -> str:
    """Build the instruction used by the LLM to extract arguments."""
    parameter_lines = [
        f"- {name}: {spec.type}"
        for name, spec in function.parameters.items()
    ]
    parameters_text = "\n".join(parameter_lines)

    return (
        "Extract the arguments for the selected function.\n\n"
        "Function:\n"
        f"{function.name}\n\n"
        "Description:\n"
        f"{function.description}\n\n"
        "Parameters:\n"
        f"{parameters_text}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Arguments JSON:\n"
    )
