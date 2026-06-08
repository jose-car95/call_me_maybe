"""Project-specific exceptions."""


class CallMeMaybeError(Exception):
    """Base exception for expected application errors."""


class InputFileError(CallMeMaybeError):
    """Raised when an input file cannot be read or decoded."""


class InputValidationError(CallMeMaybeError):
    """Raised when decoded input does not match the expected schema."""


class OutputFileError(CallMeMaybeError):
    """Raised when the result file cannot be written."""


class FunctionDefinitionError(CallMeMaybeError):
    """Raised when available function definitions are not usable."""
