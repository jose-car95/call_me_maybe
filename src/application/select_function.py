"""Use cases for selecting a function with a language model."""

from .ports import LanguageModel
from src.domain import FunctionDefinition, ModelInferenceError


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    """Build the instruction used by the LLM to select a function."""
    function_lines = [
        f"- {function.name}: {function.description}"
        for function in functions
    ]

    available_functions = "\n".join(function_lines)

    return (
        "Choose the function that best matches the user request.\n\n"
        "Available functions:\n"
        f"{available_functions}\n\n"
        "User request:\n"
        f"{user_prompt}\n\n"
        "Function:\n"
    )


def tokenize_function_names(
    model: LanguageModel,
    functions: list[FunctionDefinition]
) -> dict[str, list[int]]:
    """Tokenize every available function name."""
    return {
        function.name: model.encode(function.name)
        for function in functions
    }


def find_allowed_next_tokens(
    tokenized_functions: dict[str, list[int]],
    generated_tokens: list[int]
) -> set[int]:
    """Return tokens that can continue the generated prefix."""
    prefix_length = len(generated_tokens)
    allowed_tokens: set[int] = set()

    for candidate_tokens in tokenized_functions.values():
        candidate_prefix = candidate_tokens[:prefix_length]

        if candidate_prefix != generated_tokens:
            continue

        if prefix_length < len(candidate_tokens):
            allowed_tokens.add(candidate_tokens[prefix_length])

    return allowed_tokens


def select_best_allowed_token(
    logits: list[float],
    allowed_tokens: set[int]
) -> int:
    """Return the allowed token with the highest logit."""
    if not allowed_tokens:
        raise ModelInferenceError(
            "no allowed tokens are available"
        )

    invalid_tokens = [
        token_id
        for token_id in allowed_tokens
        if token_id < 0 or token_id >= len(logits)
    ]

    if invalid_tokens:
        raise ModelInferenceError(
            "an allowed token is outside the model vocabulary"
        )

    return max(
        allowed_tokens,
        key=lambda token_id: logits[token_id]
    )
