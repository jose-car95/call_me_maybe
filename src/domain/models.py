"""Domain models for function calling inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator
)


JsonType = Literal[
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array"
]


class ParameterSpec(BaseModel):
    """Schema for one function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class ReturnSpec(BaseModel):
    """Schema for a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: JsonType


class FunctionDefinition(BaseModel):
    """Definition of one callable function exposed to the LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec

    @field_validator("name", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject strings containing only whitespace."""
        if not value.strip():
            raise ValueError("value must not be blank")
        return value

    @field_validator("parameters")
    @classmethod
    def reject_blank_parameter_names(
        cls,
        parameters: dict[str, ParameterSpec],
    ) -> dict[str, ParameterSpec]:
        """Reject empty or whitespace-only parameter names."""
        if any(not name.strip() for name in parameters):
            raise ValueError("parameter names must not be blank")
        return parameters


class PromptCase(BaseModel):
    """One natural-language test prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        """Reject prompts containing only whitespace."""
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


class FunctionCallResult(BaseModel):
    """Output object for one generated function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    fn_name: str
    args: dict[str, Any]


class NextTokenPrediction(BaseModel):
    """Result of predicting one token."""

    model_config = ConfigDict(extra="forbid")

    input_ids: list[int]
    logits_count: int
    token_id: int
    token_text: str
