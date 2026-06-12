"""Public application API for function calling."""

from src.application.ports import LanguageModel
from src.application.process_prompts import process_prompts
from src.application.predict_next_token import predict_next_token


__all__ = [
    "LanguageModel",
    "process_prompts",
    "predict_next_token"
]
