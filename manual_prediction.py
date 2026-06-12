"""Run one manual next-token prediction using the real Qwen model."""

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.application import predict_next_token
from src.infrastructure import QwenAdapter


def main() -> None:
    """Load Qwen on CPU and predict one token."""
    sdk_model = Small_LLM_Model(device="cpu")
    model = QwenAdapter(sdk_model)

    prediction = predict_next_token(
        model,
        "The capital of France is"
    )

    print("Input IDs:", prediction.input_ids)
    print("Number of logits:", prediction.logits_count)
    print("Selected token ID:", prediction.token_id)
    print("Selected token text:", repr(prediction.token_text))


if __name__ == "__main__":
    main()
