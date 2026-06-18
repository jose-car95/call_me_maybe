"""Schema helpers for constrained argument generation."""

from dataclasses import dataclass

from src.domain import FunctionDefinition, JsonType


@dataclass(frozen=True)
class ArgumentSchema:
    """Expected argument names and types for a function."""

    parameters: list[tuple[str, JsonType]]


@dataclass(frozen=True)
class ArgumentGenerationStep:
    """One step in the constrained argument generation plan."""

    literal: str
    parameter_name: str | None
    parameter_type: JsonType | None

    def is_literal(self) -> bool:
        """Return whether this step emits fixed JSON text."""
        return self.parameter_name is None

    def is_value(self) -> bool:
        """Return whether this step emits a parameter value."""
        return self.parameter_name is not None


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


def build_argument_generation_steps(
    schema: ArgumentSchema
) -> list[ArgumentGenerationStep]:
    """Build generation steps for an argument JSON object."""
    steps: list[ArgumentGenerationStep] = [
        ArgumentGenerationStep(
            literal="{",
            parameter_name=None,
            parameter_type=None
        )
    ]

    for index, (name, type_name) in enumerate(schema.parameters):
        if index > 0:
            steps.append(
                ArgumentGenerationStep(
                    literal=", ",
                    parameter_name=None,
                    parameter_type=None
                )
            )

        steps.append(
            ArgumentGenerationStep(
                literal=f'"{name}": ',
                parameter_name=None,
                parameter_type=None
            )
        )
        steps.append(
            ArgumentGenerationStep(
                literal="",
                parameter_name=name,
                parameter_type=type_name
            )
        )

    steps.append(
        ArgumentGenerationStep(
            literal="}",
            parameter_name=None,
            parameter_type=None
        )
    )

    return steps
