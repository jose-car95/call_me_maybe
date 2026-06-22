"""Shared constant values exposed through side-effect-free helpers."""


def unable_to_retrieve_fn_name() -> str:
    """Return the fallback function name for unresolved prompts.

    Returns:
        Function name used when a prompt cannot be resolved safely.
    """
    return "Unable to retrieve from 'function_definitions.json'"
