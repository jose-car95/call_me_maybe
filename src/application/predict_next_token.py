"""Use case for predicting one token with a language model."""

import numpy as np

from src.application.ports import LanguageModel
from src.domain import ModelInferenceError, NextTokenPrediction


def predict_next_token(
    model: LanguageModel,
    text: str
) -> NextTokenPrediction:
    """Predict the most likely token following the provided text."""
    input_ids = model.encode(text)

    if not input_ids:
        raise ModelInferenceError(
            "the prompt did not produce any input tokens"
        )

    logits = model.get_logits(input_ids)

    if not logits:
        raise ModelInferenceError(
            "the language model returned no logits"
        )

    token_id = int(np.argmax(logits))
    token_text = model.decode([token_id])

    return NextTokenPrediction(
        input_ids=input_ids,
        logits_count=len(logits),
        token_id=token_id,
        token_text=token_text
    )
