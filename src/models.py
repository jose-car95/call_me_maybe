"""Domain models for function calling inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator
)


class CallMeMaybeError(Exception):
    """Base exception for expected application failures."""


class InputFileError(CallMeMaybeError):
    """Raised when an input file cannot be read or decoded."""


class InputValidationError(CallMeMaybeError):
    """Raised when input data does not match its expected schema."""


class OutputFileError(CallMeMaybeError):
    """Raised when the result file cannot be written."""


class FunctionDefinitionError(CallMeMaybeError):
    """Raised when function definitions cannot be used."""


class ModelInferenceError(CallMeMaybeError):
    """Raised when constrained model inference cannot continue."""


class ParameterSpec(BaseModel):
    """Describe one primitive or recursive function parameter.

    Attributes:
        type: JSON type accepted by the parameter.
        description: Optional semantic guidance for value selection.
        enum: Optional finite set of allowed values.
        items: Element schema for arrays.
        properties: Child schemas for objects.
        required: Required child property names for objects.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "string",
        "number",
        "integer",
        "boolean",
        "object",
        "array"
    ]
    description: str | None = None
    enum: list[Any] | None = None
    items: ParameterSpec | None = None
    properties: dict[str, ParameterSpec] | None = None
    required: list[str] | None = None

    @model_validator(mode="after")
    def validate_type_constraints(self) -> ParameterSpec:
        """Validate constraints associated with compound JSON types.

        Returns:
            The validated parameter specification.

        Raises:
            ValueError: If a constraint is incompatible with its declared type.
        """
        if self.items is not None and self.type != "array":
            raise ValueError("items is only valid for array parameters")
        if self.properties is not None and self.type != "object":
            raise ValueError("properties is only valid for object parameters")
        if self.required is not None:
            property_names = set(self.properties or {})
            unknown_names = set(self.required) - property_names
            if self.type != "object" or unknown_names:
                raise ValueError(
                    "required must reference declared object properties"
                )
        return self


class ReturnSpec(BaseModel):
    """Describe the declared JSON type of a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "string",
        "number",
        "integer",
        "boolean",
        "object",
        "array"
    ]


class FunctionDefinition(BaseModel):
    """Describe one callable function exposed to the language model.

    Attributes:
        name: Unique function identifier.
        description: Semantic description used during selection.
        parameters: Parameter names mapped to recursive schemas.
        returns: Declared return-value schema.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject function metadata containing only whitespace.

        Args:
            value: Function name or description to validate.

        Returns:
            The original non-blank value.

        Raises:
            ValueError: If the value contains only whitespace.
        """
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def reject_blank_parameter_names(
        cls,
        parameters: dict[str, ParameterSpec],
    ) -> dict[str, ParameterSpec]:
        """Reject empty or whitespace-only parameter names.

        Args:
            parameters: Parameter mapping to validate.

        Returns:
            The original parameter mapping.

        Raises:
            ValueError: If any parameter name is blank.
        """
        if any(not name.strip() for name in parameters):
            raise ValueError("parameter names must not be blank")
        return parameters


class PromptCase(BaseModel):
    """Store one validated natural-language request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def accept_plain_prompt(cls, value: Any) -> Any:
        """Accept the plain-string input form documented by the subject.

        Args:
            value: Raw prompt string or object supplied to Pydantic.

        Returns:
            A prompt object for strings, otherwise the unchanged input.
        """
        if isinstance(value, str):
            return {"prompt": value}
        return value

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        """Reject prompts containing only whitespace.

        Args:
            value: Prompt text to validate.

        Returns:
            The original non-blank prompt.

        Raises:
            ValueError: If the prompt contains only whitespace.
        """
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class FunctionCallResult(BaseModel):
    """Store one schema-valid function-calling result.

    Attributes:
        prompt: Original natural-language request.
        fn_name: Selected function name.
        args: Arguments produced for the selected function.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    fn_name: str
    args: dict[str, Any]
