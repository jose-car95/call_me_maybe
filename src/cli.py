"""Command-line interface for call_me_maybe."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.io_utils import (
    DEFAULT_FUNCTIONS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TESTS_PATH,
    load_function_definitions,
    load_prompt_cases,
    write_results,
)
from src.pipeline import build_initial_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TESTS_PATH,
        help="Path to the prompt test JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the result JSON file will be written.",
    )
    parser.add_argument(
        "--functions",
        type=Path,
        default=DEFAULT_FUNCTIONS_PATH,
        help="Path to the function definitions JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the project CLI."""
    args: argparse.Namespace = parse_args()

    try:
        prompts = load_prompt_cases(args.input)
        functions = load_function_definitions(args.functions)
        results = build_initial_results(prompts, functions)
        write_results(args.output, results)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Processed {len(prompts)} prompts.")
    print(f"Wrote output to {args.output}")
    return 0
