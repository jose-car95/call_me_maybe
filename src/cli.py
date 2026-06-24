"""Command-line interface for call_me_maybe."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import resource
import time

from src.engine import FunctionCallingEngine
from src.files import (
    load_function_definitions,
    load_prompt_cases,
    write_results
)
from src.llm import QwenAdapter
from src.models import CallMeMaybeError, FunctionCallResult
from src.ports import LanguageModel


def default_model_name() -> str:
    """Return the mandatory default model name."""
    return "Qwen/Qwen3-0.6B"


def recommended_models() -> tuple[tuple[str, str], ...]:
    """Return model recommendations without storing global state."""
    return (
        (default_model_name(), "default, mandatory, lowest memory"),
        ("Qwen/Qwen3-1.7B", "stronger model, needs more memory"),
    )


def print_supported_models() -> None:
    """Print built-in model recommendations and custom model guidance."""
    print("Supported/recommended models:")
    for model_name, note in recommended_models():
        print(f"- {model_name} ({note})")
    print()
    print("Custom Hugging Face model ids can be passed with --model if:")
    print("- they are compatible with the provided llm_sdk")
    print("- they expose a tokenizer.json compatible with Byte-Level BPE")
    print("- they fit the available memory and time budget")


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
        default=Path("data/input/function_definitions.json"),
        help="Path to the function definitions JSON file."
    )
    parser.add_argument(
        "--model",
        default=default_model_name(),
        help=(
            "LLM model name used by the SDK. "
            f"Defaults to {default_model_name()}."
        )
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended models and custom model requirements."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show constrained decoding decisions during generation."
    )
    return parser.parse_args(argv)


def _print_statistics(
    results: list[FunctionCallResult],
    elapsed: float
) -> None:
    """Print resolution, validity, timing, and peak-memory statistics.

    Args:
        results: One validated function-call result per processed prompt.
        elapsed: Wall-clock seconds spent processing every prompt.
    """
    total = len(results)
    fallback = "Unable to retrieve from 'function_definitions.json'"
    resolved = sum(1 for result in results if result.fn_name != fallback)
    rate = (resolved / total * 100) if total else 0.0
    peak_gb = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    )
    print(f"Function selection: {resolved}/{total}")
    print(f"Arguments: {resolved}/{total} ({rate:.0f}%)")
    print(
        f"Parseable JSON: 100%    Time: ~{elapsed:.0f}s"
        f"    Peak memory: ~{peak_gb:.1f} GB"
    )


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

    if args.list_models:
        print_supported_models()
        return 0

    try:
        prompts = load_prompt_cases(args.input)
        functions = load_function_definitions(args.functions)
        if model is None:
            model = QwenAdapter(model_name=args.model)
        trace = print if args.trace else None
        engine = FunctionCallingEngine(model, functions, trace=trace)
        start = time.perf_counter()
        results = engine.process(prompts)
        elapsed = time.perf_counter() - start
        write_results(args.output, results)
    except CallMeMaybeError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Processed {len(prompts)} prompts.")
    print(f"Wrote output to {args.output}")
    _print_statistics(results, elapsed)
    return 0
