"""Public application API for function calling."""

from src.application.ports import LanguageModel
from src.application.process_prompts import process_prompts


__all__ = [
    "LanguageModel",
    "process_prompts"
]
