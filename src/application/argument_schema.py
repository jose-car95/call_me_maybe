"""Schema helpers for constrained argument generation."""

from dataclasses import dataclass

from src.domain import FunctionDefinition, JsonType


@dataclass(frozen=True)
class ArgumentSchema:
    """Expected argument names and types for a function."""

    parameters: list[tuple[str, JsonType]]


def build_argument_schema(function: FunctionDefinition) -> ArgumentSchema:
    """Build an argument schema from a function definition."""
    return ArgumentSchema(
        parameters=[
            (name, spec.type)
            for name, spec in function.parameters.items()
        ]
    )


def build_argument_json_template(schema: ArgumentSchema) -> str:
    """Build a JSON template with typed argument placeholders."""
    parameters = [
        f'"{name}": <{type_name}>'
        for name, type_name in schema.parameters
    ]

    return "{" + ", ".join(parameters) + "}"
