"""Command-line interface for call_me_maybe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.engine import FunctionCallingEngine, LanguageModel
from src.files import (
    load_function_definitions,
    load_prompt_cases,
    write_results
)
from src.llm import QwenAdapter
from src.models import CallMeMaybeError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument sequence, or ``None`` to use process arguments.

    Returns:
        Parsed input, output, function, and tracing options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the prompt test JSON file."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path where the result JSON file will be written."
    )
    parser.add_argument(
        "--functions",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the function definitions JSON file."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show constrained token decisions during generation."
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    model: LanguageModel | None = None
) -> int:
    """Run the command-line workflow.

    Args:
        argv: Optional arguments supplied instead of process arguments.
        model: Optional model override used by tests or alternate integrations.

    Returns:
        Zero on success and one for an expected project error.
    """
    args: argparse.Namespace = parse_args(argv)

    try:
        prompts = load_prompt_cases(args.input)
        functions = load_function_definitions(args.functions)
        if model is None:
            model = QwenAdapter()
        trace = print if args.trace else None
        engine = FunctionCallingEngine(model, functions, trace=trace)
        results = engine.process(prompts)
        write_results(args.output, results)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Processed {len(prompts)} prompts.")
    print(f"Wrote output to {args.output}")
    return 0
