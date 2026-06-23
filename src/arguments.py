"""Prompt argument extraction and candidate generation helpers."""

import json
import re
from typing import Any

from src.models import FunctionDefinition, ParameterSpec


def _unique(values: list[Any]) -> list[Any]:
    """Remove JSON-equivalent duplicates while preserving order.

    Args:
        values: Candidate values to deduplicate.

    Returns:
        Values in their original order without JSON-equivalent duplicates.
    """
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _quoted(prompt: str) -> list[str]:
    """Extract single- and double-quoted spans from a prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Quoted spans in their order of appearance.
    """
    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
    return [left or right for left, right in matches]


def _numbers(prompt: str, integer: bool = False) -> list[int | float]:
    """Extract common JSON-compatible numeric forms from a prompt.

    Args:
        prompt: Natural-language request to inspect.
        integer: Whether to keep only values that can be parsed as integers.

    Returns:
        Parsed numbers in their order of appearance.
    """
    values: list[int | float] = []
    pattern = (
        r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
    )
    for match in re.findall(pattern, prompt):
        normalized = match.replace(",", "")
        try:
            values.append(int(normalized) if integer else float(normalized))
        except ValueError:
            continue
    return values


def _last_word(prompt: str) -> str:
    """Return the final punctuation-trimmed word in a prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        The final word, or an empty string for an empty prompt.
    """
    words = prompt.strip().split()
    return words[-1].strip(".,!?;:") if words else ""


def _labelled_value(prompt: str, name: str) -> str | None:
    """Return the value introduced by a ``"<name>: value"`` label.

    Args:
        prompt: Natural-language request to inspect.
        name: Parameter name used as a case-insensitive label.

    Returns:
        Text after the label, or ``None`` when the label is absent.
    """
    match = re.search(
        rf"\b{re.escape(name)}\s*:\s*(.+)$",
        prompt,
        flags=re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def _value_before(prompt: str, name: str) -> str | None:
    """Return the word that directly precedes the parameter name.

    Args:
        prompt: Natural-language request to inspect.
        name: Parameter name mentioned in the request.

    Returns:
        The preceding word when the name is mentioned, otherwise ``None``.
    """
    match = re.search(
        rf"\b([\w.+-]+)\s+{re.escape(name)}\b",
        prompt,
        flags=re.IGNORECASE
    )
    return match.group(1) if match else None


def _path_like(prompt: str) -> str | None:
    """Return a filesystem-path-shaped token when one is present.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        The path-shaped token, or ``None`` when none is found.
    """
    match = re.search(r"(?:[A-Za-z]:\\|/)[^\s]+", prompt)
    return match.group(0).rstrip(".,!?;:") if match else None


def _replacement(prompt: str) -> str:
    """Extract and normalize a replacement introduced by ``with``.

    Args:
        prompt: Natural-language substitution request.

    Returns:
        Replacement text, including normalized named symbols when applicable.
    """
    marker = " with "
    index = prompt.lower().rfind(marker)
    if index == -1:
        return ""
    value = prompt[index + len(marker):].strip(" .,!?:;")
    return _named_symbol(value)


def _named_symbol(value: str) -> str:
    """Convert a named punctuation symbol to its literal character.

    Args:
        value: Candidate symbol name or literal value.

    Returns:
        Normalized symbol character or the unchanged input value.
    """
    aliases = {
        "asterisk": "*",
        "asterisks": "*",
        "underscore": "_",
        "underscores": "_",
        "hyphen": "-",
        "hyphens": "-",
        "dash": "-",
        "dashes": "-",
        "slash": "/",
        "slashes": "/"
    }
    return aliases.get(value.lower(), value)


def _boolean_value(prompt: str) -> bool | None:
    """Extract an explicit true-like or false-like value.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Parsed boolean, or ``None`` when the request is ambiguous.
    """
    words = {word.lower().strip(".,!?;:") for word in prompt.split()}
    true_words = {"true", "yes", "enabled", "active", "on"}
    false_words = {"false", "no", "disabled", "inactive", "off"}
    if words & true_words and not words & false_words:
        return True
    if words & false_words and not words & true_words:
        return False
    return None


def _regex_arguments(prompt: str) -> dict[str, Any]:
    """Resolve arguments for the supported regex-substitution schema.

    Args:
        prompt: Natural-language substitution request.

    Returns:
        Source text, regex pattern, and replacement value.
    """
    quoted = _quoted(prompt)
    aliases = {
        "numbers": r"\d+",
        "digits": r"\d+",
        "vowels": r"[aeiouAEIOU]",
        "spaces": r"\s+",
        "whitespace": r"\s+"
    }
    pattern = next(
        (
            regex
            for word, regex in aliases.items()
            if word in prompt.lower()
        ),
        ""
    )
    if pattern and quoted:
        return {
            "source_string": quoted[0],
            "regex": pattern,
            "replacement": _replacement(prompt)
        }
    if len(quoted) >= 3:
        regex = quoted[0]
        if "word" in prompt.lower():
            regex = rf"\b{re.escape(regex)}\b"
        return {
            "source_string": quoted[2],
            "regex": regex,
            "replacement": quoted[1]
        }
    return {
        "source_string": quoted[0] if quoted else "",
        "regex": "",
        "replacement": _replacement(prompt)
    }


def _empty_value(spec: ParameterSpec) -> Any:
    """Build a schema-compatible empty value.

    Args:
        spec: Parameter schema whose JSON type determines the value.

    Returns:
        An empty value matching the declared JSON type.
    """
    return {
        "string": "",
        "number": 0,
        "integer": 0,
        "boolean": False,
        "array": [],
        "object": {}
    }[spec.type]


def extract_arguments(
    prompt: str,
    function: FunctionDefinition
) -> dict[str, Any]:
    """Extract unambiguous arguments without model inference.

    Args:
        prompt: Natural-language request containing argument values.
        function: Selected function and its parameter schema.

    Returns:
        One initial value for every declared parameter.
    """
    if {"source_string", "regex", "replacement"}.issubset(
        function.parameters
    ):
        return _regex_arguments(prompt)

    numbers = _numbers(prompt)
    quoted = _quoted(prompt)
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", prompt)
    number_index = quoted_index = email_index = 0
    arguments: dict[str, Any] = {}

    for name, spec in function.parameters.items():
        if spec.type in {"number", "integer"}:
            if number_index >= len(numbers):
                arguments[name] = _empty_value(spec)
            else:
                value = numbers[number_index]
                arguments[name] = (
                    int(value)
                    if spec.type == "integer"
                    else value
                )
                number_index += 1
            continue

        if spec.type == "string":
            text_value = _labelled_value(prompt, name)
            if (
                text_value is None
                and "email" in name
                and email_index < len(emails)
            ):
                text_value = emails[email_index]
                email_index += 1
            if text_value is None and quoted_index < len(quoted):
                text_value = quoted[quoted_index]
                quoted_index += 1
            if text_value is None:
                text_value = _value_before(prompt, name)
            if text_value is None:
                text_value = _path_like(prompt)
            if text_value is None:
                text_value = _last_word(prompt)
            arguments[name] = text_value
            continue

        if spec.type == "boolean":
            boolean_value = _boolean_value(prompt)
            arguments[name] = (
                boolean_value
                if boolean_value is not None
                else False
            )
            continue

        arguments[name] = _empty_value(spec)

    return arguments


def _json_candidates(prompt: str, expected_type: type[Any]) -> list[Any]:
    """Extract embedded JSON arrays or objects from a prompt.

    Args:
        prompt: Natural-language request that may contain JSON.
        expected_type: Either ``list`` or ``dict``.

    Returns:
        Parsed values matching the requested container type.
    """
    values: list[Any] = []
    opening, closing = ("[", "]") if expected_type is list else ("{", "}")
    for start, character in enumerate(prompt):
        if character != opening:
            continue
        for end in range(len(prompt), start, -1):
            if prompt[end - 1] != closing:
                continue
            try:
                value = json.loads(prompt[start:end])
            except json.JSONDecodeError:
                continue
            if isinstance(value, expected_type):
                values.append(value)
                break
    return values


def value_candidates(prompt: str, spec: ParameterSpec) -> list[Any]:
    """Build schema-compatible candidates for one argument.

    Args:
        prompt: Natural-language request containing possible values.
        spec: Schema for the argument being resolved.

    Returns:
        Deduplicated candidates that satisfy the declared primitive type.
    """
    if spec.enum is not None:
        return _unique(spec.enum)
    if spec.type == "integer":
        return _unique(_numbers(prompt, integer=True))
    if spec.type == "number":
        return _unique(_numbers(prompt))
    if spec.type == "boolean":
        value = _boolean_value(prompt)
        if value is not None:
            return [value]
        return [True, False]
    if spec.type == "array":
        return _unique(_json_candidates(prompt, list) + [[]])
    if spec.type == "object":
        return _unique(_json_candidates(prompt, dict) + [{}])

    candidates = _quoted(prompt)
    if ":" in prompt:
        candidates.append(prompt.split(":", 1)[1].strip())
    prompt_words = re.findall(r"[^\s,;:!?]+", prompt)
    for size in range(1, min(8, len(prompt_words)) + 1):
        candidates.extend(
            " ".join(prompt_words[start:start + size]).strip(".,")
            for start in range(len(prompt_words) - size + 1)
        )
    lowered = {word.lower().strip(".,") for word in prompt_words}
    candidates.extend(
        normalized
        for word in lowered
        if (normalized := _named_symbol(word)) != word
    )
    return _unique(candidates + [""])


def ambiguous_names(
    prompt: str,
    function: FunctionDefinition,
    arguments: dict[str, Any]
) -> set[str]:
    """Identify arguments that need constrained model selection.

    Args:
        prompt: Original natural-language request.
        function: Selected function definition.
        arguments: Values produced by deterministic extraction.

    Returns:
        Parameter names whose extracted values are not reliable enough.
    """
    ambiguous: set[str] = set()
    string_names = [
        name
        for name, spec in function.parameters.items()
        if spec.type == "string"
    ]
    values = [arguments[name] for name in string_names]
    if len(string_names) > len(_quoted(prompt)):
        duplicates = {value for value in values if values.count(value) > 1}
        ambiguous.update(
            name for name in string_names if arguments[name] in duplicates
        )

    parameter_words = {
        part.rstrip("s").lower()
        for name in function.parameters
        for part in name.split("_")
    }
    for name in string_names:
        value = arguments[name]
        if not value or value.lower().rstrip("s") in (
            parameter_words - {name.rstrip("s").lower()}
        ):
            ambiguous.add(name)

    for name, spec in function.parameters.items():
        if spec.enum is not None and len(spec.enum) > 1:
            ambiguous.add(name)
        if spec.type in {"array", "object"} and not arguments[name]:
            if len(value_candidates(prompt, spec)) > 1:
                ambiguous.add(name)
    return ambiguous
