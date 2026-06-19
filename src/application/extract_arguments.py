"""Use cases for extracting function arguments."""

from dataclasses import dataclass
import json
import re
from typing import Any

from src.domain import FunctionDefinition


@dataclass(frozen=True)
class ArgumentPatternMatcher:
    """Extract primitive values from user prompts."""

    number_pattern: str = r"-?\d+(?:\.\d+)?"
    quoted_text_pattern: str = r"'([^']*)'|\"([^\"]*)\""
    email_pattern: str = r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"

    def extract_numbers(self, text: str) -> list[float]:
        """Extract numeric values from text."""
        return [
            float(match)
            for match in re.findall(
                self.number_pattern,
                text
            )
        ]

    def extract_quoted_texts(self, text: str) -> list[str]:
        """Extract all quoted texts from text."""
        matches = re.findall(
            self.quoted_text_pattern,
            text
        )

        return [
            single_quoted or double_quoted
            for single_quoted, double_quoted in matches
        ]

    def extract_emails(self, text: str) -> list[str]:
        """Extract email addresses from text."""
        return re.findall(
            self.email_pattern,
            text
        )

    def extract_boolean(self, text: str) -> bool | None:
        """Extract a boolean value from text."""
        lowered_text = text.lower()
        true_words = {"true", "yes", "enabled", "active"}
        false_words = {"false", "no", "disabled", "inactive"}
        words = {
            word.strip(".,!?;:")
            for word in lowered_text.split()
        }

        if words & true_words:
            return True
        if words & false_words:
            return False

        return None

    def extract_last_word(self, text: str) -> str:
        """Extract the last word from text."""
        words = text.strip().split()

        if not words:
            return ""

        return words[-1].strip(".,!?;:")


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


def _extract_argument_value(
    parameter_name: str,
    type_name: str,
    numbers: list[float],
    number_index: int,
    quoted_texts: list[str],
    quoted_text_index: int,
    emails: list[str],
    email_index: int,
    matcher: ArgumentPatternMatcher,
    user_prompt: str
) -> tuple[Any, int, int, int]:
    """Extract one argument value and return the updated indexes."""
    if type_name in {"number", "integer"}:
        if number_index >= len(numbers):
            return (
                _empty_value_for_type(type_name),
                number_index,
                quoted_text_index,
                email_index
            )

        value = numbers[number_index]
        if type_name == "integer":
            value = int(value)

        return value, number_index + 1, quoted_text_index, email_index

    if type_name == "string":
        if "email" in parameter_name and email_index < len(emails):
            return (
                emails[email_index],
                number_index,
                quoted_text_index,
                email_index + 1
            )

        if quoted_text_index < len(quoted_texts):
            return (
                quoted_texts[quoted_text_index],
                number_index,
                quoted_text_index + 1,
                email_index
            )

        return (
            matcher.extract_last_word(user_prompt),
            number_index,
            quoted_text_index,
            email_index
        )

    if type_name == "boolean":
        boolean_value = matcher.extract_boolean(user_prompt)

        if boolean_value is None:
            boolean_value = _empty_value_for_type(type_name)

        return (
            boolean_value,
            number_index,
            quoted_text_index,
            email_index
        )

    return (
        _empty_value_for_type(type_name),
        number_index,
        quoted_text_index,
        email_index
    )


def _is_regex_substitution(function: FunctionDefinition) -> bool:
    """Return whether a function has the regex substitution schema."""
    required_parameters = {
        "source_string",
        "regex",
        "replacement"
    }

    return required_parameters.issubset(function.parameters)


def _extract_replacement_after_with(user_prompt: str) -> str:
    """Extract the replacement text after the word 'with'."""
    marker = " with "
    lowered_prompt = user_prompt.lower()
    marker_index = lowered_prompt.rfind(marker)

    if marker_index == -1:
        return ""

    return user_prompt[marker_index + len(marker):].strip(" .,!?:;")


def _extract_regex_pattern(user_prompt: str) -> str:
    """Extract a reusable regex pattern from natural-language targets."""
    lowered_prompt = user_prompt.lower()
    regex_patterns = {
        "numbers": r"\d+",
        "digits": r"\d+",
        "vowels": r"[aeiouAEIOU]",
        "spaces": r"\s+",
        "whitespace": r"\s+"
    }

    for keyword, pattern in regex_patterns.items():
        if keyword in lowered_prompt:
            return pattern

    return ""


def _extract_regex_substitution_arguments(
    user_prompt: str,
    matcher: ArgumentPatternMatcher
) -> dict[str, Any]:
    """Extract arguments for source, regex and replacement schema."""
    quoted_texts = matcher.extract_quoted_texts(user_prompt)
    regex_pattern = _extract_regex_pattern(user_prompt)

    if regex_pattern and quoted_texts:
        return {
            "source_string": quoted_texts[0],
            "regex": regex_pattern,
            "replacement": _extract_replacement_after_with(user_prompt)
        }

    if len(quoted_texts) >= 3:
        return {
            "source_string": quoted_texts[2],
            "regex": quoted_texts[0],
            "replacement": quoted_texts[1]
        }

    return {
        "source_string": quoted_texts[0] if quoted_texts else "",
        "regex": "",
        "replacement": _extract_replacement_after_with(user_prompt)
    }


def extract_arguments(
    user_prompt: str,
    function: FunctionDefinition
) -> dict[str, Any]:
    """Extract schema-compatible arguments for a selected function."""
    build_argument_extraction_prompt(
        user_prompt,
        function
    )

    matcher = ArgumentPatternMatcher()

    if _is_regex_substitution(function):
        return _extract_regex_substitution_arguments(
            user_prompt,
            matcher
        )

    numbers = matcher.extract_numbers(user_prompt)
    quoted_texts = matcher.extract_quoted_texts(user_prompt)
    emails = matcher.extract_emails(user_prompt)
    number_index: int = 0
    quoted_text_index = 0
    email_index = 0
    arguments: dict[str, Any] = {}

    for name, spec in function.parameters.items():
        value, number_index, quoted_text_index, email_index = (
            _extract_argument_value(
                name,
                spec.type,
                numbers,
                number_index,
                quoted_texts,
                quoted_text_index,
                emails,
                email_index,
                matcher,
                user_prompt
            )
        )
        arguments[name] = value

    return arguments


def build_argument_schema_template(
    function: FunctionDefinition
) -> dict[str, Any]:
    """Build the expected argument JSON shape for a function."""
    return build_empty_arguments(function)


def build_constrained_argument_generation_prompt(
    user_prompt: str,
    function: FunctionDefinition
) -> str:
    """Build the prompt used to generate arguments with schema constraints."""
    parameter_lines = [
        f"- {name}: {spec.type}"
        for name, spec in function.parameters.items()
    ]
    parameters_text = "\n".join(parameter_lines)
    schema_template = json.dumps(
        build_argument_schema_template(function)
    )

    return (
        "Extract arguments as JSON.\n\n"
        "Function:\n"
        f"{function.name}\n\n"
        "Description:\n"
        f"{function.description}\n\n"
        "Parameters:\n"
        f"{parameters_text}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "JSON schema template:\n"
        f"{schema_template}\n\n"
        "Arguments JSON:\n"
    )
