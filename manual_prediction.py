"""Run manual constrained function selection with Qwen."""

from pathlib import Path

from src.application import select_function_name
from src.infrastructure import (
    QwenAdapter,
    load_function_definitions
)


def main() -> None:
    """Load Qwen on CPU and select a function for sample prompt."""
    model = QwenAdapter()
    functions = load_function_definitions(
        Path("data/input/functions_definition.json")
    )

    prompts = [
        "What is the sum of 2 and 3?",
        "Greet john",
        "Reverse the string 'hello'",
        "What is the square root of 16?"
    ]

    for prompt in prompts:
        selected_name = select_function_name(
            model,
            prompt,
            functions
        )

        print(f"Prompt: {prompt}")
        print(f"Selected function: {selected_name}")
        print()


if __name__ == "__main__":
    main()
