"""Schema validation helpers for generated function arguments."""

from typing import Any

from src.models import FunctionDefinition, ModelInferenceError, ParameterSpec


def _matches_spec(value: Any, spec: ParameterSpec) -> bool:
    """Check a value against a recursive parameter schema.

    Args:
        value: Python value to validate.
        spec: Recursive JSON-compatible schema.

    Returns:
        ``True`` when the value satisfies type and schema constraints.
    """
    simple_matches = {
        "string": isinstance(value, str),
        "number": (
            isinstance(value, int | float)
            and not isinstance(value, bool)
        ),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict)
    }
    if not simple_matches[spec.type]:
        return False
    if spec.enum is not None and value not in spec.enum:
        return False
    if spec.type == "array" and spec.items is not None:
        return all(_matches_spec(item, spec.items) for item in value)
    if spec.type == "object" and spec.properties is not None:
        required = set(spec.required or spec.properties)
        if not required.issubset(value) or set(value) - set(spec.properties):
            return False
        return all(
            _matches_spec(value[name], child_spec)
            for name, child_spec in spec.properties.items()
            if name in value
        )
    return True


def validate_arguments(
    function: FunctionDefinition,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate argument keys and values against a function definition.

    Args:
        function: Function whose parameter schema must be satisfied.
        arguments: Generated argument object.

    Returns:
        The unchanged argument object after successful validation.

    Raises:
        ModelInferenceError: If keys, types, enums, or nested values are
            invalid.
    """
    expected = set(function.parameters)
    actual = set(arguments)
    if expected != actual:
        raise ModelInferenceError(
            f"argument keys do not match schema: expected {sorted(expected)}"
        )
    for name, spec in function.parameters.items():
        if not _matches_spec(arguments[name], spec):
            raise ModelInferenceError(
                f"argument {name} does not match its schema"
            )
    return arguments
