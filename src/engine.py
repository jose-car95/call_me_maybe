"""Function-calling engine with constrained model choices."""

from collections.abc import Callable
import json
import re
from typing import Any, Protocol, cast

from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    FunctionDefinitionError,
    ModelInferenceError,
    ParameterSpec,
    PromptCase
)


class LanguageModel(Protocol):
    """Model operations required by the engine."""

    def encode(self, text: str) -> list[int]:
        """Convert text into token identifiers.

        Args:
            text: Text to encode.

        Returns:
            Flat model token identifiers.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert token identifiers into text.

        Args:
            token_ids: Model token identifiers.

        Returns:
            Decoded text.
        """
        ...

    def get_logits(self, input_ids: list[int]) -> list[float]:
        """Return logits for the next token.

        Args:
            input_ids: Existing model token sequence.

        Returns:
            One logit per model vocabulary token.
        """
        ...


def _unique(values: list[Any]) -> list[Any]:
    """Remove JSON-equivalent duplicates while preserving order.

    Args:
        values: Candidate values to deduplicate.

    Returns:
        Values in their original order without JSON-equivalent duplicates.
    """
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _quoted(prompt: str) -> list[str]:
    """Extract single- and double-quoted spans from a prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Quoted spans in their order of appearance.
    """
    matches = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)
    return [left or right for left, right in matches]


def _numbers(prompt: str, integer: bool = False) -> list[int | float]:
    """Extract common JSON-compatible numeric forms from a prompt.

    Args:
        prompt: Natural-language request to inspect.
        integer: Whether to keep only values that can be parsed as integers.

    Returns:
        Parsed numbers in their order of appearance.
    """
    values: list[int | float] = []
    pattern = (
        r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)"
        r"(?:\.\d+)?(?:[eE][+-]?\d+)?"
    )
    for match in re.findall(pattern, prompt):
        normalized = match.replace(",", "")
        try:
            values.append(int(normalized) if integer else float(normalized))
        except ValueError:
            continue
    return values


def _last_word(prompt: str) -> str:
    """Return the final punctuation-trimmed word in a prompt.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        The final word, or an empty string for an empty prompt.
    """
    words = prompt.strip().split()
    return words[-1].strip(".,!?;:") if words else ""


def _word_before(prompt: str, keyword: str) -> str | None:
    """Find the word immediately preceding a keyword.

    Args:
        prompt: Natural-language request to inspect.
        keyword: Case-insensitive marker whose preceding value is required.

    Returns:
        The preceding word when present, otherwise ``None``.
    """
    match = re.search(
        rf"\b([\w.-]+)\s+{re.escape(keyword)}\b",
        prompt,
        flags=re.IGNORECASE
    )
    return match.group(1) if match else None


def _after_label(prompt: str, label: str) -> str | None:
    """Extract all content following a colon-terminated label.

    Args:
        prompt: Natural-language request to inspect.
        label: Case-insensitive label introducing the value.

    Returns:
        Content after the label, or ``None`` when the label is absent.
    """
    match = re.search(
        rf"\b{re.escape(label)}\s*:\s*(.+)$",
        prompt,
        flags=re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def _contextual_string(name: str, prompt: str) -> str | None:
    """Extract string values with a recognizable parameter context.

    Args:
        name: Parameter name supplied by the function schema.
        prompt: Natural-language request to inspect.

    Returns:
        A contextual value when recognized, otherwise ``None``.
    """
    if name == "path":
        match = re.search(r"(?:[A-Za-z]:\\|/)[^\s]+", prompt)
        return match.group(0).rstrip(".,!?;:") if match else None
    if name == "template":
        return _after_label(prompt, name)
    if name in {"database", "encoding"}:
        return _word_before(prompt, name)
    return None


def _replacement(prompt: str) -> str:
    """Extract and normalize a replacement introduced by ``with``.

    Args:
        prompt: Natural-language substitution request.

    Returns:
        Replacement text, including normalized named symbols when applicable.
    """
    marker = " with "
    index = prompt.lower().rfind(marker)
    if index == -1:
        return ""
    value = prompt[index + len(marker):].strip(" .,!?:;")
    return _named_symbol(value)


def _named_symbol(value: str) -> str:
    """Convert a named punctuation symbol to its literal character.

    Args:
        value: Candidate symbol name or literal value.

    Returns:
        Normalized symbol character or the unchanged input value.
    """
    aliases = {
        "asterisk": "*",
        "asterisks": "*",
        "underscore": "_",
        "underscores": "_",
        "hyphen": "-",
        "hyphens": "-",
        "dash": "-",
        "dashes": "-",
        "slash": "/",
        "slashes": "/"
    }
    return aliases.get(value.lower(), value)


def _boolean_value(prompt: str) -> bool | None:
    """Extract an explicit true-like or false-like value.

    Args:
        prompt: Natural-language request to inspect.

    Returns:
        Parsed boolean, or ``None`` when the request is ambiguous.
    """
    words = {word.lower().strip(".,!?;:") for word in prompt.split()}
    true_words = {"true", "yes", "enabled", "active", "on"}
    false_words = {"false", "no", "disabled", "inactive", "off"}
    if words & true_words and not words & false_words:
        return True
    if words & false_words and not words & true_words:
        return False
    return None


def _regex_arguments(prompt: str) -> dict[str, Any]:
    """Resolve arguments for the supported regex-substitution schema.

    Args:
        prompt: Natural-language substitution request.

    Returns:
        Source text, regex pattern, and replacement value.
    """
    quoted = _quoted(prompt)
    aliases = {
        "numbers": r"\d+",
        "digits": r"\d+",
        "vowels": r"[aeiouAEIOU]",
        "spaces": r"\s+",
        "whitespace": r"\s+"
    }
    pattern = next(
        (
            regex
            for word, regex in aliases.items()
            if word in prompt.lower()
        ),
        ""
    )
    if pattern and quoted:
        return {
            "source_string": quoted[0],
            "regex": pattern,
            "replacement": _replacement(prompt)
        }
    if len(quoted) >= 3:
        return {
            "source_string": quoted[2],
            "regex": quoted[0],
            "replacement": quoted[1]
        }
    return {
        "source_string": quoted[0] if quoted else "",
        "regex": "",
        "replacement": _replacement(prompt)
    }


def _empty_value(spec: ParameterSpec) -> Any:
    """Build a schema-compatible empty value.

    Args:
        spec: Parameter schema whose JSON type determines the value.

    Returns:
        An empty value matching the declared JSON type.
    """
    return {
        "string": "",
        "number": 0,
        "integer": 0,
        "boolean": False,
        "array": [],
        "object": {}
    }[spec.type]


def _extract_arguments(
    prompt: str,
    function: FunctionDefinition
) -> dict[str, Any]:
    """Extract unambiguous arguments without model inference.

    Args:
        prompt: Natural-language request containing argument values.
        function: Selected function and its parameter schema.

    Returns:
        One initial value for every declared parameter.
    """
    if {"source_string", "regex", "replacement"}.issubset(
        function.parameters
    ):
        return _regex_arguments(prompt)

    numbers = _numbers(prompt)
    quoted = _quoted(prompt)
    emails = re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", prompt)
    number_index = quoted_index = email_index = 0
    arguments: dict[str, Any] = {}

    for name, spec in function.parameters.items():
        if spec.type in {"number", "integer"}:
            if number_index >= len(numbers):
                arguments[name] = _empty_value(spec)
            else:
                value = numbers[number_index]
                arguments[name] = (
                    int(value)
                    if spec.type == "integer"
                    else value
                )
                number_index += 1
            continue

        if spec.type == "string":
            contextual = _contextual_string(name, prompt)
            if contextual is not None:
                arguments[name] = contextual
            elif "email" in name and email_index < len(emails):
                arguments[name] = emails[email_index]
                email_index += 1
            elif quoted_index < len(quoted):
                arguments[name] = quoted[quoted_index]
                quoted_index += 1
            else:
                arguments[name] = _last_word(prompt)
            continue

        if spec.type == "boolean":
            boolean_value = _boolean_value(prompt)
            arguments[name] = (
                boolean_value
                if boolean_value is not None
                else False
            )
            continue

        arguments[name] = _empty_value(spec)

    return arguments


def _json_candidates(prompt: str, expected_type: type[Any]) -> list[Any]:
    """Extract embedded JSON arrays or objects from a prompt.

    Args:
        prompt: Natural-language request that may contain JSON.
        expected_type: Either ``list`` or ``dict``.

    Returns:
        Parsed values matching the requested container type.
    """
    values: list[Any] = []
    opening, closing = ("[", "]") if expected_type is list else ("{", "}")
    for start, character in enumerate(prompt):
        if character != opening:
            continue
        for end in range(len(prompt), start, -1):
            if prompt[end - 1] != closing:
                continue
            try:
                value = json.loads(prompt[start:end])
            except json.JSONDecodeError:
                continue
            if isinstance(value, expected_type):
                values.append(value)
                break
    return values


def _value_candidates(prompt: str, spec: ParameterSpec) -> list[Any]:
    """Build schema-compatible candidates for one argument.

    Args:
        prompt: Natural-language request containing possible values.
        spec: Schema for the argument being resolved.

    Returns:
        Deduplicated candidates that satisfy the declared primitive type.
    """
    if spec.enum is not None:
        return _unique(spec.enum)
    if spec.type == "integer":
        return _unique(_numbers(prompt, integer=True))
    if spec.type == "number":
        return _unique(_numbers(prompt))
    if spec.type == "boolean":
        value = _boolean_value(prompt)
        if value is not None:
            return [value]
        return [True, False]
    if spec.type == "array":
        return _unique(_json_candidates(prompt, list) + [[]])
    if spec.type == "object":
        return _unique(_json_candidates(prompt, dict) + [{}])

    candidates = _quoted(prompt)
    if ":" in prompt:
        candidates.append(prompt.split(":", 1)[1].strip())
    prompt_words = re.findall(r"[^\s,;:!?]+", prompt)
    for size in range(1, min(8, len(prompt_words)) + 1):
        candidates.extend(
            " ".join(prompt_words[start:start + size]).strip(".,")
            for start in range(len(prompt_words) - size + 1)
        )
    lowered = {word.lower().strip(".,") for word in prompt_words}
    candidates.extend(
        normalized
        for word in lowered
        if (normalized := _named_symbol(word)) != word
    )
    return _unique(candidates + [""])


def _ambiguous_names(
    prompt: str,
    function: FunctionDefinition,
    arguments: dict[str, Any]
) -> set[str]:
    """Identify arguments that need constrained model selection.

    Args:
        prompt: Original natural-language request.
        function: Selected function definition.
        arguments: Values produced by deterministic extraction.

    Returns:
        Parameter names whose extracted values are not reliable enough.
    """
    ambiguous: set[str] = set()
    string_names = [
        name
        for name, spec in function.parameters.items()
        if spec.type == "string"
    ]
    values = [arguments[name] for name in string_names]
    if len(string_names) > len(_quoted(prompt)):
        duplicates = {value for value in values if values.count(value) > 1}
        ambiguous.update(
            name for name in string_names if arguments[name] in duplicates
        )

    parameter_words = {
        part.rstrip("s").lower()
        for name in function.parameters
        for part in name.split("_")
    }
    for name in string_names:
        value = arguments[name]
        if not value or value.lower().rstrip("s") in (
            parameter_words - {name.rstrip("s").lower()}
        ):
            ambiguous.add(name)

    for name, spec in function.parameters.items():
        if spec.enum is not None and len(spec.enum) > 1:
            ambiguous.add(name)
        if spec.type in {"array", "object"} and not arguments[name]:
            if len(_value_candidates(prompt, spec)) > 1:
                ambiguous.add(name)
    return ambiguous


def _matches_spec(value: Any, spec: ParameterSpec) -> bool:
    """Check a value against a recursive parameter schema.

    Args:
        value: Python value to validate.
        spec: Recursive JSON-compatible schema.

    Returns:
        ``True`` when the value satisfies type and schema constraints.
    """
    simple_matches = {
        "string": isinstance(value, str),
        "number": (
            isinstance(value, int | float)
            and not isinstance(value, bool)
        ),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict)
    }
    if not simple_matches[spec.type]:
        return False
    if spec.enum is not None and value not in spec.enum:
        return False
    if spec.type == "array" and spec.items is not None:
        return all(_matches_spec(item, spec.items) for item in value)
    if spec.type == "object" and spec.properties is not None:
        required = set(spec.required or spec.properties)
        if not required.issubset(value) or set(value) - set(spec.properties):
            return False
        return all(
            _matches_spec(value[name], child_spec)
            for name, child_spec in spec.properties.items()
            if name in value
        )
    return True


def _validate_arguments(
    function: FunctionDefinition,
    arguments: dict[str, Any]
) -> dict[str, Any]:
    """Validate argument keys and values against a function definition.

    Args:
        function: Function whose parameter schema must be satisfied.
        arguments: Generated argument object.

    Returns:
        The unchanged argument object after successful validation.

    Raises:
        ModelInferenceError: If keys, types, enums, or nested values are
            invalid.
    """
    expected = set(function.parameters)
    actual = set(arguments)
    if expected != actual:
        raise ModelInferenceError(
            f"argument keys do not match schema: expected {sorted(expected)}"
        )
    for name, spec in function.parameters.items():
        if not _matches_spec(arguments[name], spec):
            raise ModelInferenceError(
                f"argument {name} does not match its schema"
            )
    return arguments


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
        """Initialize stable function data and token caches.

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
        self._model = model
        self._functions = functions
        self._by_name = {function.name: function for function in functions}
        self._trace = trace
        self._token_cache: dict[str, list[int]] = {}
        for function in functions:
            self._encode(function.name)

    def _encode(self, text: str) -> list[int]:
        """Encode text once and reuse its token identifiers.

        Args:
            text: Text to encode.

        Returns:
            Token identifiers produced by the configured model tokenizer.
        """
        if text not in self._token_cache:
            self._token_cache[text] = self._model.encode(text)
        return self._token_cache[text]

    def _choose(
        self,
        prompt: str,
        choices: dict[str, object],
        label: str
    ) -> object:
        """Select one candidate through token-level constrained decoding.

        Args:
            prompt: Context given to the language model.
            choices: Allowed serialized texts mapped to their result values.
            label: Human-readable name used in traces and errors.

        Returns:
            Value associated with the completed constrained token sequence.

        Raises:
            ModelInferenceError: If no valid constrained continuation exists.
        """
        if not choices:
            raise ModelInferenceError(f"no valid choices for {label}")
        tokenized = {text: self._encode(text) for text in choices}
        if any(not tokens for tokens in tokenized.values()):
            raise ModelInferenceError(
                f"an empty token sequence exists for {label}"
            )

        prompt_tokens = self._model.encode(prompt)
        generated: list[int] = []
        while True:
            for text, tokens in tokenized.items():
                if tokens == generated:
                    if self._trace is not None:
                        self._trace(f"{label}={text}")
                    return choices[text]

            index = len(generated)
            allowed = {
                tokens[index]
                for tokens in tokenized.values()
                if tokens[:index] == generated and index < len(tokens)
            }
            logits = self._model.get_logits(prompt_tokens + generated)
            invalid = {token for token in allowed if token >= len(logits)}
            if not allowed or invalid:
                raise ModelInferenceError(
                    f"invalid constrained tokens for {label}"
                )
            selected = max(allowed, key=lambda token: logits[token])
            if self._trace is not None:
                self._trace(
                    f"{label} allowed={len(allowed)} token={selected}"
                )
            generated.append(selected)

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
            self._choose(
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
        candidates = _value_candidates(user_prompt, spec)
        choices: dict[str, object] = {
            json.dumps(value, ensure_ascii=False): value
            for value in candidates
        }
        return self._choose(
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
        arguments = _extract_arguments(user_prompt, function)
        ambiguous = _ambiguous_names(user_prompt, function, arguments)
        for name in ambiguous:
            arguments[name] = self._choose_argument(
                user_prompt,
                function,
                name
            )
        try:
            return _validate_arguments(function, arguments)
        except ModelInferenceError:
            retry = {
                name: self._choose_argument(user_prompt, function, name)
                for name in function.parameters
            }
            return _validate_arguments(function, retry)

    def process(self, prompts: list[PromptCase]) -> list[FunctionCallResult]:
        """Process every prompt and preserve its original order.

        Args:
            prompts: Validated natural-language requests.

        Returns:
            One validated function-call result per input prompt.

        Raises:
            ModelInferenceError: If model inference or validation cannot
                continue.
        """
        results: list[FunctionCallResult] = []
        for prompt in prompts:
            function = self._select_function(prompt.prompt)
            results.append(
                FunctionCallResult(
                    prompt=prompt.prompt,
                    fn_name=function.name,
                    args=self._resolve_arguments(prompt.prompt, function)
                )
            )
        return results
