"""Public domain API for function calling."""

from src.domain.exceptions import (
    CallMeMaybeError,
    FunctionDefinitionError,
    InputFileError,
    InputValidationError,
    OutputFileError
)
from src.domain.models import (
    FunctionCallResult,
    FunctionDefinition,
    JsonType,
    ParameterSpec,
    PromptCase,
    ReturnSpec
)


__all__ = [
    "CallMeMaybeError",
    "FunctionCallResult",
    "FunctionDefinition",
    "FunctionDefinitionError",
    "InputFileError",
    "InputValidationError",
    "JsonType",
    "OutputFileError",
    "ParameterSpec",
    "PromptCase",
    "ReturnSpec"
]
