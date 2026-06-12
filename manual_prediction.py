"""Inspect how Qwen tokenizes the available function names."""

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.infrastructure import QwenAdapter


FUNCTION_NAMES = [
    "fn_add_numbers",
    "fn_greet",
    "fn_reverse_string",
    "fn_get_square_root",
    "fn_substitute_string_with_regex"
]


def main() -> None:
    """Print the tokenization of every function name."""
    sdk_model = Small_LLM_Model(device="cpu")
    model = QwenAdapter(sdk_model)

    for function_name in FUNCTION_NAMES:
        token_ids = model.encode(function_name)

        print(f"Function: {function_name}")
        print(f"Token IDs: {token_ids}")
        print("Tokens:")

        for token_id in token_ids:
            token_text = model.decode([token_id])
            print(f"  {token_id}: {token_text!r}")

        print()


if __name__ == "__main__":
    main()
