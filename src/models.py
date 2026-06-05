"""Pydantic models for function calling inputs and outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


JsonType = Literal["string", "number", "integer", "boolean", "object", "array"]


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

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls,
        parameters: dict[str, ParameterSpec],
    ) -> dict[str, ParameterSpec]:
        """Ensure parameter names are non-empty."""
        for name in parameters:
            if not name.strip():
                raise ValueError("parameter names must not be empty")
        return parameters


class PromptCase(BaseModel):
    """One natural-language test prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1)


class FunctionCallResult(BaseModel):
    """Output object for one generated function call."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    fn_name: str
    args: dict[str, Any]
