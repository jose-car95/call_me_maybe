"""Use cases for extracting function arguments."""

from dataclasses import dataclass
import re
from typing import Any

from src.domain import FunctionDefinition


@dataclass(frozen=True)
class ArgumentPatternMatcher:
    """Extract primitive values from user prompts."""

    number_pattern: str = r"-?\d+(?:\.\d+)?"
    quoted_text_pattern: str = r"'([^']*)'|\"([^\"]*)\""

    def extract_numbers(self, text: str) -> list[float]:
        """Extract numeric values from text."""
        return [
            float(match)
            for match in re.findall(
                self.number_pattern,
                text
            )
        ]

    def extract_first_quoted_text(self, text: str) -> str:
        """Extract the first quoted text from text."""
        match = re.search(
            self.quoted_text_pattern,
            text
        )

        if match is None:
            return ""

        return next(
            group
            for group in match.groups()
            if group is not None
        )


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


def build_empty_arguments(
    function: FunctionDefinition
) -> dict[str, Any]:
    """Build schema-compatible empty arguments for a function."""
    return {
        name: _empty_value_for_type(spec.type)
        for name, spec in function.parameters.items()
    }


def _empty_value_for_type(type_name: str) -> Any:
    """Return an empty value compatible with a JSON type."""
    if type_name in {"number", "integer"}:
        return 0
    if type_name == "boolean":
        return False
    if type_name == "array":
        return []
    if type_name == "object":
        return {}

    return ""


def extract_arguments(
    user_prompt: str,
    function: FunctionDefinition
) -> dict[str, Any]:
    """Extract schema-compatible arguments for a selected function."""
    build_argument_extraction_prompt(
        user_prompt,
        function
    )

    return build_empty_arguments(function)
