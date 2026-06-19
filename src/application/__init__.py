"""Public application API for function calling."""

from src.application.extract_arguments import (
    build_argument_extraction_prompt,
    build_argument_schema_template,
    build_constrained_argument_generation_prompt,
    build_empty_arguments,
    extract_arguments
)
from src.application.ports import LanguageModel
from src.application.predict_next_token import predict_next_token
from src.application.process_prompts import process_prompts
from src.application.select_function import (
    build_function_selection_prompt,
    find_allowed_next_tokens,
    find_completed_function_name,
    score_function_name_candidate,
    select_best_allowed_token,
    select_function_name,
    select_highest_scoring_function_name,
    tokenize_function_names
)


__all__ = [
    "LanguageModel",
    "build_argument_extraction_prompt",
    "build_argument_schema_template",
    "build_empty_arguments",
    "build_constrained_argument_generation_prompt",
    "extract_arguments",
    "build_function_selection_prompt",
    "find_allowed_next_tokens",
    "find_completed_function_name",
    "predict_next_token",
    "process_prompts",
    "score_function_name_candidate",
    "select_best_allowed_token",
    "select_function_name",
    "select_highest_scoring_function_name",
    "tokenize_function_names"
]
