# Cambios en cmm_mouli

El objetivo fue adaptar la moulinette para corregir el formato de salida del proyecto sin modificar el proyecto.
El proyecto emite `fn_name`/`args`; la moulinette original esperaba `name`/`parameters`.

---

## `moulinette/__main__.py`

| Cambio | Original | Modificado | Ubicación |
|--------|----------|------------|-----------|
| Import corregido (typo) | `extract_functions_infos` | `extract_functions_info` | línea 10 |
| Constante nueva añadida | — | `UNABLE_TO_RETRIEVE_FN_NAME = "Unable to retrieve from 'function_definitions.json'"` | línea 19 |
| Nombre del archivo generado | `functions_definition.json` | `function_definitions.json` | línea 53 |
| Generación de correcciones | Código más complejo | Usa directamente el output de `generate_function_calling_corrections(filtered_exercises)` | líneas 75–82 |
| Lectura de respuesta del estudiante | Solo leía `name`/`parameters` | `fn_name` (fallback a `name`) y `args` (fallback a `parameters`) | líneas 141–142 |
| Lógica de fallback nueva | — | Fallback del estudiante en prompt resoluble → fallo; fallback esperado en corrección → válido | líneas 144–152 |
| Campos mostrados en errores | `name`/`parameters` | `fn_name`/`args` | línea 178 |

---

## `moulinette/generate_tests_and_corrections.py`

| Cambio | Original | Modificado | Ubicación |
|--------|----------|------------|-----------|
| Modelo `Correction` — campos renombrados | `name` + `parameters` | `fn_name` + `args` | líneas 14–15 |
| Builder de correcciones — campos emitidos | `name`/`parameters` | `fn_name`/`args` | líneas 35–36 |
| Import eliminado | Importaba `exercises` (sin uso) | Eliminado | — |

---

## Notas de compatibilidad

- El grader acepta tanto el formato nuevo (`fn_name`/`args`) como el viejo (`name`/`parameters`) para no romper outputs anteriores.
- `prepare_exercises` genera ahora `data/input/function_definitions.json`, que es el nombre que consume el proyecto.
