"""Behavior tests for the function-calling engine."""

import pytest

from src.engine import FunctionCallingEngine
from src.models import (
    FunctionDefinition,
    FunctionDefinitionError,
    ModelInferenceError,
    PromptCase
)


class CharacterModel:
    """Character tokenizer with predictable constrained choices."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)

    def get_logits(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * 256
        logits[ord('"')] = 10.0
        return logits


def _function(
    parameters: dict[str, dict[str, object]],
    name: str = "fn_example",
    description: str = "Example function."
) -> FunctionDefinition:
    return FunctionDefinition.model_validate(
        {
            "name": name,
            "description": description,
            "parameters": parameters,
            "returns": {"type": "string"}
        }
    )


def _process(
    prompt: str,
    function: FunctionDefinition
) -> dict[str, object]:
    engine = FunctionCallingEngine(CharacterModel(), [function])
    result = engine.process([PromptCase(prompt=prompt)])[0]
    return result.args


def test_requires_at_least_one_function() -> None:
    with pytest.raises(FunctionDefinitionError):
        FunctionCallingEngine(CharacterModel(), [])


def test_processes_numbers_in_parameter_order() -> None:
    arguments = _process(
        "Add 2 and 3",
        _function({"a": {"type": "number"}, "b": {"type": "number"}})
    )

    assert arguments == {"a": 2.0, "b": 3.0}


def test_extracts_quoted_and_boolean_values() -> None:
    arguments = _process(
        "Create 'Ada Lovelace' with admin true",
        _function(
            {
                "name": {"type": "string"},
                "admin": {"type": "boolean"}
            }
        )
    )

    assert arguments == {"name": "Ada Lovelace", "admin": True}


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (
            "Replace all vowels in 'Programming is fun' with asterisks",
            {
                "source_string": "Programming is fun",
                "regex": r"[aeiouAEIOU]",
                "replacement": "*"
            }
        ),
        (
            "Substitute the word 'cat' with 'dog' in 'cat and cat'",
            {
                "source_string": "cat and cat",
                "regex": "cat",
                "replacement": "dog"
            }
        )
    ]
)
def test_resolves_regex_substitution(
    prompt: str,
    expected: dict[str, object]
) -> None:
    function = _function(
        {
            "source_string": {"type": "string"},
            "regex": {"type": "string"},
            "replacement": {"type": "string"}
        }
    )

    assert _process(prompt, function) == expected


def test_extracts_contextual_strings() -> None:
    function = _function(
        {"path": {"type": "string"}, "encoding": {"type": "string"}}
    )

    arguments = _process(
        "Read /home/user/data.json with utf-8 encoding",
        function
    )

    assert arguments == {
        "path": "/home/user/data.json",
        "encoding": "utf-8"
    }


def test_enum_output_cannot_leave_declared_values() -> None:
    function = _function(
        {"firmware": {"type": "string", "enum": ["stable", "beta"]}}
    )

    arguments = _process("Use a supported firmware", function)

    assert arguments["firmware"] in {"stable", "beta"}


def test_validates_nested_json_from_prompt() -> None:
    function = _function(
        {
            "config": {
                "type": "object",
                "properties": {
                    "firmware": {
                        "type": "string",
                        "enum": ["stable", "beta"]
                    }
                },
                "required": ["firmware"]
            }
        }
    )

    arguments = _process('Configure {"firmware": "stable"}', function)

    assert arguments == {"config": {"firmware": "stable"}}


class BrokenModel(CharacterModel):
    def get_logits(self, input_ids: list[int]) -> list[float]:
        return []


def test_reports_constrained_inference_failure() -> None:
    function = _function(
        {"firmware": {"type": "string", "enum": ["stable", "beta"]}}
    )
    engine = FunctionCallingEngine(BrokenModel(), [function])

    with pytest.raises(ModelInferenceError):
        engine.process([PromptCase(prompt="Use stable firmware")])


def test_trace_reports_selected_function() -> None:
    messages: list[str] = []
    function = _function({"name": {"type": "string"}})
    engine = FunctionCallingEngine(
        CharacterModel(),
        [function],
        trace=messages.append
    )

    engine.process([PromptCase(prompt="Greet Ada")])

    assert any(message.startswith("function=") for message in messages)
