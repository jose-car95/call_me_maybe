"""Public application API for function calling."""

from src.application.ports import LanguageModel
from src.application.process_prompts import process_prompts
from src.application.predict_next_token import predict_next_token
from src.application.select_function import (
    build_function_selection_prompt,
    find_allowed_next_tokens,
    find_completed_function_name,
    select_best_allowed_token,
    select_function_name,
    tokenize_function_names
)


__all__ = [
    "LanguageModel",
    "process_prompts",
    "predict_next_token",
    "build_function_selection_prompt",
    "tokenize_function_names",
    "find_allowed_next_tokens",
    "select_best_allowed_token",
    "find_completed_function_name",
    "select_function_name"
]
