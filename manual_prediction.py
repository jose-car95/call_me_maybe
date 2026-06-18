"""Run manual function call generation with Qwen."""

from pathlib import Path

from src.application import process_prompts
from src.infrastructure import (
    QwenAdapter,
    load_function_definitions,
    load_prompt_cases
)


def main() -> None:
    """Load Qwen on CPU and process real input prompts."""
    model = QwenAdapter()
    functions = load_function_definitions(
        Path("data/input/functions_definition.json")
    )
    prompts = load_prompt_cases(
        Path("data/input/function_calling_tests.json")
    )

    results = process_prompts(
        model,
        prompts,
        functions
    )

    for result in results:
        print(f"Prompt: {result.prompt}")
        print(f"Selected function: {result.fn_name}")
        print(f"Arguments: {result.args}")
        print()


if __name__ == "__main__":
    main()
