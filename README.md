*Este proyecto ha sido creado como parte del currículo de 42 por jose-car.*

# call_me_maybe

## Description

`call_me_maybe` translates natural-language requests into structured function
calls. It selects an available function and produces its arguments with the names
and types declared in the input schema.

The project uses Qwen3-0.6B through the provided SDK. The output does not rely on
the model freely writing correct JSON: function names are generated over a trie of
allowed tokens, ambiguous arguments are chosen among schema-compatible candidates,
and the final structure is serialized and validated deterministically.

## Instructions

Requirements: Python 3.10 or later, `uv`, and roughly 6 GB of memory to run Qwen
on CPU.

```bash
make install
make run
```

By default the program reads `data/input/function_calling_tests.json` and
`data/input/function_definitions.json`, and writes
`data/output/function_calling_results.json`.

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions data/input/function_definitions.json \
  --output /tmp/function_calling_results.json
```

The model can be changed while keeping Qwen as the default:

```bash
uv run python -m src --model Qwen/Qwen3-0.6B
```

Built-in recommended models:

```bash
uv run python -m src --list-models
uv run python -m src --model Qwen/Qwen3-1.7B
```

To add or try new models, use a Hugging Face identifier compatible with the SDK
that exposes a `tokenizer.json`, is compatible with the project's Byte-Level BPE
tokenizer, and fits the memory and time budget of the evaluation. Qwen/Qwen3-0.6B
remains the mandatory default model.

To observe the constrained decisions and the controlled recovery:

```bash
uv run python -m src --trace
```

Development checks:

```bash
make test
make lint
make lint-strict
```

## Usage example

Input (`data/input/function_calling_tests.json`):

```json
["What is the sum of 2 and 3?", "Reverse the string 'hello'"]
```

Output (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "fn_name": "fn_reverse_string",
    "args": { "s": "hello" }
  }
]
```

## Algorithm

### Function selection

1. A prompt is built with the functions and their descriptions.
2. Each valid name is tokenized and placed into a logical trie.
3. Qwen produces logits for the next token.
4. Only tokens that continue some valid name are considered.
5. The allowed token with the highest logit is appended to the prefix.
6. The process stops when a valid name is complete.

The model takes part in the semantic decision, but it cannot invent a function.

### Argument extraction

First, typed candidates are extracted from the prompt and the schema using generic
rules. Values are located by their JSON type (numbers, quoted strings, booleans,
emails, embedded JSON) and by where they appear relative to the parameter itself: a
`"<name>: value"` label, the word adjacent to the parameter name in the sentence, or
a filesystem-path shape. The parameter name is used dynamically as a hint; there is
no hardcoded list of anticipated parameter names. Unambiguous cases are resolved
directly to avoid unnecessary inference.

When several plausible values exist, a constrained decoder builds a trie with their
JSON representations and queries logits token by token. The model can only finish
on one of those values. Finally, keys, types, enums, and nested structures are
validated.

The decoder is closed-set: if one candidate's tokens are a prefix of another's, the
shorter candidate wins (there is no boundary token to ask the model whether to stop
or continue). In practice the candidates do not overlap at the token level, so the
tie-break is predictable and sufficient.

The schema supports `string`, `number`, `integer`, `boolean`, `array`, `object`,
`enum`, `items`, `properties`, and `required`.

### Tokenizer

The main flow does not use the SDK's `encode` or `decode`. The custom Byte-Level
BPE implementation loads `tokenizer.json` through a public method, applies NFC
normalization, Unicode pre-tokenization, a reversible byte mapping, and BPE merges.

## Architecture

```text
CLI -> files -> engine -> LLM/tokenizer -> result
```

- `models.py`: Pydantic models, schemas, and errors.
- `files.py`: JSON reading and writing.
- `ports.py`: shared contracts, such as the language model.
- `constants.py`: shared values exposed through functions.
- `arguments.py`: argument extraction, candidates, and ambiguity.
- `schema_validator.py`: recursive validation against the schema.
- `constrained_decoder.py`: token-by-token constrained selection.
- `engine.py`: orchestration of the function-calling flow.
- `llm.py`: adaptation of the Qwen SDK.
- `tokenizer.py`: Byte-Level BPE implementation.
- `cli.py`: command line.

The structure is deliberately flat because the project has a single use case. Each
file isolates one responsibility without introducing unnecessary folders or deep
imports.

## Design decisions

- Constrained greedy selection is used because scoring each complete function
  multiplied inference time on CPU.
- Extraction is hybrid: generic rules for unambiguous cases and a constrained LLM
  for ambiguity. Generating every argument with logits exceeded five minutes.
- The JSON structure is built from the schema, not from free text.
- `LanguageModel` decouples the engine from the SDK and enables fast tests.
- Pydantic rejects extra keys and recursively validates the inputs.

## Performance and reliability

Local measurement over a set of 11 prompts, Qwen/Qwen3-0.6B on CPU:

```text
Function selection: 11/11    Arguments: 11/11 (100%)
Time: ~42 s    Parseable JSON: 100%    Peak memory: ~5.2 GB
```

Function names are tokenized once and the decoder keeps a cache. File, validation,
initialization, and inference errors are turned into controlled messages.

If a prompt cannot be resolved without breaking the schema, a controlled fallback
is returned with `Unable to retrieve from 'function_definitions.json'` and empty
arguments, preventing a single isolated case from taking down the whole batch.

## Challenges

- Qwen3-0.6B does not generate reliable structures without constraints.
- Exhaustive scoring improved decisions but broke the CPU budget.
- Regular expressions do not generalize for arbitrary strings; ambiguous cases are
  delegated to the model within a set of valid candidates.
- The tokenizer combines Unicode, bytes, and BPE. The custom implementation was
  compared against the reference tokenizer using text, paths, JSON, symbols,
  whitespace, and several languages (see the parity test).

## Testing strategy

The suite covers Pydantic models, invalid files, constrained selection, shared
prefixes, JSON types, enums, recursive schemas, complex numbers, strings, emails,
regex substitution, SDK errors, the CLI, final writing, Byte-Level BPE, and a
parity check of the custom tokenizer against the reference tokenizer.

## Resources

- Official `call_me_maybe` subject.
- Python documentation for `json`, `typing`, and `unicodedata`.
- Pydantic documentation.
- OpenAI article on Byte Pair Encoding.
- Public Qwen and Byte-Level BPE documentation.

AI was used to review the architecture, explore alternatives, propose edge cases,
and assist with tests and documentation. Everything incorporated was reviewed with
tests, static analysis, and real runs with the model.
