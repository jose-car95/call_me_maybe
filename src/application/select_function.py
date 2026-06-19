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


def find_completed_function_name(
    tokenized_functions: dict[str, list[int]],
    generated_tokens: list[int]
) -> str | None:
    """Return the function name if the generated tokens match one."""
    for function_name, function_tokens in tokenized_functions.items():
        if function_tokens == generated_tokens:
            return function_name

    return None


def score_function_name_candidate(
    model: LanguageModel,
    prompt_tokens: list[int],
    candidate_tokens: list[int],
    logits_cache: dict[tuple[int, ...], list[float]] | None = None
) -> float:
    """Score a complete function name candidate with normalized logits."""
    if not candidate_tokens:
        raise ModelInferenceError(
            "function name candidate must contain at least one token"
        )

    total_score = 0.0
    generated_tokens: list[int] = []

    for token_id in candidate_tokens:
        input_tokens = prompt_tokens + generated_tokens
        cache_key = tuple(input_tokens)
        if logits_cache is not None and cache_key in logits_cache:
            logits = logits_cache[cache_key]
        else:
            logits = model.get_logits(input_tokens)
            if logits_cache is not None:
                logits_cache[cache_key] = logits

        if token_id < 0 or token_id >= len(logits):
            raise ModelInferenceError(
                "a candidate token is outside the model vocabulary"
            )

        total_score += logits[token_id]
        generated_tokens.append(token_id)

    return total_score / len(candidate_tokens)


def select_highest_scoring_function_name(
    model: LanguageModel,
    prompt_tokens: list[int],
    tokenized_functions: dict[str, list[int]]
) -> str:
    """Select the function name with the best complete-candidate score."""
    if not tokenized_functions:
        raise ModelInferenceError(
            "at least one function candidate is required"
        )

    logits_cache: dict[tuple[int, ...], list[float]] = {}

    return max(
        tokenized_functions,
        key=lambda function_name: score_function_name_candidate(
            model,
            prompt_tokens,
            tokenized_functions[function_name],
            logits_cache
        )
    )


def select_function_name(
    model: LanguageModel,
    user_prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    """Select the best matching function name with constrained decoding."""
    selection_prompt = build_function_selection_prompt(
        user_prompt,
        functions
    )
    prompt_tokens = model.encode(selection_prompt)
    tokenized_functions = tokenize_function_names(model, functions)
    generated_tokens: list[int] = []

    while True:
        completed_name = find_completed_function_name(
            tokenized_functions,
            generated_tokens
        )

        if completed_name is not None:
            return completed_name

        allowed_tokens = find_allowed_next_tokens(
            tokenized_functions,
            generated_tokens
        )
        logits = model.get_logits(prompt_tokens + generated_tokens)
        selected_token = select_best_allowed_token(
            logits,
            allowed_tokens
        )
        generated_tokens.append(selected_token)
