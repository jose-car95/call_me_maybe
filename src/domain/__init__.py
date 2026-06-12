"""Public domain API for function calling."""

from src.domain.exceptions import (
    CallMeMaybeError,
    FunctionDefinitionError,
    InputFileError,
    InputValidationError,
    OutputFileError,
    ModelInferenceError
)
from src.domain.models import (
    FunctionCallResult,
    FunctionDefinition,
    JsonType,
    ParameterSpec,
    PromptCase,
    ReturnSpec,
    NextTokenPrediction
)


__all__ = [
    "CallMeMaybeError",
    "FunctionDefinitionError",
    "InputFileError",
    "InputValidationError",
    "OutputFileError",
    "ModelInferenceError",
    "FunctionCallResult",
    "FunctionDefinition",
    "JsonType",
    "ParameterSpec",
    "PromptCase",
    "ReturnSpec",
    "NextTokenPrediction"
]
