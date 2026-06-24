"""Function-calling engine orchestration."""

from collections.abc import Callable
import json
from typing import Any, cast

from src.arguments import (
    ambiguous_names,
    extract_arguments,
    value_candidates
)
from src.constrained_decoder import ConstrainedDecoder
from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
    ModelInferenceError,
    ParameterSpec,
    PromptCase
)
from src.ports import LanguageModel
from src.schema_validator import validate_arguments


def _function_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    """Build the constrained function-selection prompt.

    Args:
        user_prompt: Natural-language request to classify.
        functions: Available function definitions.

    Returns:
        Prompt containing function names, descriptions, and the request.
    """
    options = "\n".join(
        f"- {function.name}: {function.description}"
        for function in functions
    )
    return (
        "Choose the function that best matches the user request.\n\n"
        f"Available functions:\n{options}\n\n"
        f"User request:\n{user_prompt}\n\n"
        "Function:\n"
    )


def _value_prompt(
    user_prompt: str,
    function: FunctionDefinition,
    name: str,
    spec: ParameterSpec,
    choices: dict[str, Any]
) -> str:
    """Build a focused prompt for one ambiguous argument.

    Args:
        user_prompt: Original natural-language request.
        function: Selected function definition.
        name: Parameter currently being resolved.
        spec: Parameter schema and optional description.
        choices: Serialized allowed values and their Python representations.

    Returns:
        Prompt instructing the model to select one allowed value.
    """
    values = "\n".join(f"- {text}" for text in choices)
    description = spec.description or "No additional description."
    return (
        "Select the exact value for one function argument.\n"
        "Return one allowed value with no explanation.\n"
        "Named symbols must become their character.\n\n"
        f"Function: {function.name}\n"
        f"Function description: {function.description}\n"
        f"Argument: {name}\n"
        f"Argument type: {spec.type}\n"
        f"Argument description: {description}\n"
        f"User request: {user_prompt}\n\n"
        f"Allowed values:\n{values}\n\n"
        "Value:\n"
    )


class FunctionCallingEngine:
    """Translate natural-language prompts into validated function calls."""

    def __init__(
        self,
        model: LanguageModel,
        functions: list[FunctionDefinition],
        trace: Callable[[str], None] | None = None
    ) -> None:
        """Initialize stable function data and constrained decoding.

        Args:
            model: Language model used for constrained semantic decisions.
            functions: Functions available to every processed prompt.
            trace: Optional callback receiving generation decisions.

        Raises:
            FunctionDefinitionError: If no functions are available.
        """
        if not functions:
            raise FunctionDefinitionError(
                "at least one function definition is required"
            )
        self._functions = functions
        self._by_name = {function.name: function for function in functions}
        self._trace = trace
        self._decoder = ConstrainedDecoder(model, trace=trace)
        for function in functions:
            self._decoder.encode(function.name)

    def _select_function(self, user_prompt: str) -> FunctionDefinition:
        """Select the best available function for one request.

        Args:
            user_prompt: Natural-language request to classify.

        Returns:
            Function selected through constrained decoding.
        """
        choices: dict[str, object] = dict(self._by_name)
        return cast(
            FunctionDefinition,
            self._decoder.choose(
                _function_prompt(user_prompt, self._functions),
                choices,
                "function"
            )
        )

    def _choose_argument(
        self,
        user_prompt: str,
        function: FunctionDefinition,
        name: str
    ) -> Any:
        """Select one ambiguous argument from schema-compatible candidates.

        Args:
            user_prompt: Original natural-language request.
            function: Function owning the parameter.
            name: Parameter name to resolve.

        Returns:
            Candidate selected through constrained decoding.
        """
        spec = function.parameters[name]
        candidates = value_candidates(user_prompt, spec)
        choices: dict[str, object] = {
            json.dumps(value, ensure_ascii=False): value
            for value in candidates
        }
        return self._decoder.choose(
            _value_prompt(user_prompt, function, name, spec, choices),
            choices,
            f"argument[{name}]"
        )

    def _resolve_arguments(
        self,
        user_prompt: str,
        function: FunctionDefinition
    ) -> dict[str, Any]:
        """Resolve and validate every argument for one function call.

        Args:
            user_prompt: Original natural-language request.
            function: Selected function definition.

        Returns:
            Complete schema-valid argument object.

        Raises:
            ModelInferenceError: If constrained recovery cannot build valid
                values.
        """
        arguments = extract_arguments(user_prompt, function)
        ambiguous = ambiguous_names(user_prompt, function, arguments)
        for name in ambiguous:
            arguments[name] = self._choose_argument(
                user_prompt,
                function,
                name
            )
        try:
            return validate_arguments(function, arguments)
        except ModelInferenceError:
            retry = {
                name: self._choose_argument(user_prompt, function, name)
                for name in function.parameters
            }
            return validate_arguments(function, retry)

    def process(self, prompts: list[PromptCase]) -> list[FunctionCallResult]:
        """Process every prompt and preserve its original order.

        Args:
            prompts: Validated natural-language requests.

        Returns:
            One validated function-call result per input prompt.
        """
        results: list[FunctionCallResult] = []
        for prompt in prompts:
            if self._trace is not None:
                self._trace(f"prompt={prompt.prompt!r}")
            try:
                function = self._select_function(prompt.prompt)
                arguments = self._resolve_arguments(prompt.prompt, function)
                if self._trace is not None:
                    self._trace(f"selected_function={function.name}")
                    self._trace(f"validated_args={json.dumps(arguments)}")
                results.append(
                    FunctionCallResult(
                        prompt=prompt.prompt,
                        fn_name=function.name,
                        args=arguments
                    )
                )
            except ModelInferenceError as exc:
                if self._trace is not None:
                    self._trace(f"fallback={exc}")
                results.append(
                    FunctionCallResult(
                        prompt=prompt.prompt,
                        fn_name=(
                            "Unable to retrieve from "
                            "'function_definitions.json'"
                        ),
                        args={}
                    )
                )
        return results
