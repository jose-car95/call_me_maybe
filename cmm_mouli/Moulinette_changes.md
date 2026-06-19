# Moulinette Changes

## Objective

Adapt `cmm_mouli` to grade your current project output format without changing your project.

Your project output contract is:

```json
{
  "prompt": "...",
  "fn_name": "...",
  "args": {}
}
```

Fallback for unresolved prompts:

```json
{
  "prompt": "...",
  "fn_name": "Unable to retrieve from 'function_definitions.json'",
  "args": {}
}
```

---

## Plan Implemented

1. Fix moulinette imports and module wiring so files renamed by you are correctly referenced.
2. Align exercise/correction schema from old keys (`name`, `parameters`) to new keys (`fn_name`, `args`).
3. Align generated definitions filename with your project (`function_definitions.json`).
4. Update grader logic to read `fn_name/args` (while remaining backward compatible with `name/parameters`).
5. Add explicit fallback constant and grading behavior for unresolved prompts.
6. Run smoke checks (schema generation + syntax checks).

---

## Files Changed

### 1) `moulinette/__main__.py`

- Fixed import typo:
  - `moulinette.extract_functions_infos` -> `moulinette.extract_functions_info`
- Added constant:
  - `UNABLE_TO_RETRIEVE_FN_NAME = "Unable to retrieve from 'function_definitions.json'"`
- Updated generated input filename:
  - `functions_definition.json` -> `function_definitions.json`
- Corrections file generation now uses `generate_function_calling_corrections(...)` output directly.
- Grading now reads student fields in this order:
  - `fn_name` (fallback to `name` for compatibility)
  - `args` (fallback to `parameters` for compatibility)
- Fallback grading behavior:
  - If student uses fallback and correction expects a real function -> invalid (`fallback used on solvable prompt`).
  - If correction also expected fallback -> valid.
- Updated displayed expected call fields to `fn_name/args`.

### 2) `moulinette/generate_tests_and_corrections.py`

- `Correction` model changed:
  - `name` -> `fn_name`
  - `parameters` -> `args`
- Correction builder now emits `fn_name` and `args`.
- Removed unused import `exercises`.

---

## Compatibility Notes

- Grader is now aligned with your output format (`fn_name`, `args`).
- Grader still accepts old format (`name`, `parameters`) to avoid breaking older outputs.
- `prepare_exercises` now creates `data/input/function_definitions.json`, matching your project loader.

---

## Validation Performed

### A) Schema smoke test

Command executed:

```bash
../call_me_maybe/.venv/Scripts/python.exe -c "from moulinette.functions_definition import get_exercises_by_visibility;from moulinette.generate_tests_and_corrections import generate_function_calling_corrections;corr=generate_function_calling_corrections(get_exercises_by_visibility('public'));print(corr[0].model_dump());print(corr[-1].model_dump())"
```

Verified correction items now use:
- `prompt`
- `fn_name`
- `args`
- `expected_output`

### B) Syntax validation

Command executed:

```bash
../call_me_maybe/.venv/Scripts/python.exe -m py_compile moulinette/*.py
```

Result: no syntax errors.

---

## Environment Note

In this execution environment, `fire` is not installed in the reused virtualenv, so `python -m moulinette ...` was not runnable end-to-end here.
Project dependency declaration already includes `fire` in `moulinette_pyproject.toml`; once dependencies are installed in moulinette's own environment, CLI should run.
