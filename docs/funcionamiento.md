# Funcionamiento completo de call_me_maybe (guía de defensa)

Documento personal para preparar la evaluación. Explica, paso a paso, qué ocurre
desde que entra un prompt hasta que se escribe el JSON final, en qué archivo y
función vive cada cosa, qué preguntas son probables y qué partes del código podrían
pedirte tocar en la recodificación.

---

## 0. El pitch de 30 segundos

> El programa **no le pide al modelo que escriba JSON libre**. Usa el LLM solo para
> decisiones semánticas (qué función, y qué valor cuando hay ambigüedad), pero cada
> decisión está **restringida a opciones válidas** mediante decodificación token a
> token sobre los logits. La estructura final la construye Python desde el esquema y
> se valida con Pydantic antes de escribirla. Por eso el JSON es 100% parseable
> incluso con un modelo de 0.6B.

Tres ideas clave que debes saber defender:
1. **Selección de función = LLM restringido** (no heurística). Cumple el requisito
   "la función debe elegirse usando el LLM".
2. **Decodificación restringida = enmascarar logits**: pongo a −∞ (en la práctica:
   ignoro) todo token que no continúe una opción válida, y elijo el de mayor logit
   entre los permitidos.
3. **Tokenizer propio (bonus)**: no uso `encode`/`decode` del SDK en el flujo real;
   reimplemento Byte-Level BPE leyendo `tokenizer.json` y uso
   `get_logits_from_input_ids`.

---

## 1. Cómo se ejecuta

```bash
uv run python -m src [--input <file>] [--output <file>] [--functions <file>]
                     [--model <hf_id>] [--list-models] [--trace]
```

- `--input`: prompts. Por defecto `data/input/function_calling_tests.json`.
- `--functions`: definiciones. Por defecto `data/input/function_definitions.json`.
- `--output`: salida. Por defecto `data/output/function_calling_results.json`.
- `--model`: modelo del SDK. Por defecto `Qwen/Qwen3-0.6B` (obligatorio).
- `--list-models`: lista modelos recomendados y sale sin cargar nada.
- `--trace`: imprime las decisiones de decodificación restringida (útil en defensa
  para *visualizar* el proceso → cubre el bonus de "visualización").

---

## 2. Mapa de archivos (una responsabilidad por archivo)

```text
src/__main__.py          entrypoint mínimo (python -m src)
src/__init__.py          declara el paquete; no expone atajos con efectos colaterales
src/cli.py               argumentos, orquestación de alto nivel, captura de errores
src/files.py             leer/validar JSON de entrada y escribir el JSON de salida
src/models.py            modelos Pydantic (datos) + jerarquía de excepciones
src/ports.py             Protocol LanguageModel (desacopla motor del SDK)
src/llm.py               QwenAdapter: envuelve el SDK y el tokenizer propio
src/tokenizer.py         ByteLevelBPETokenizer (encode/decode propios)
src/constrained_decoder.py  ConstrainedDecoder.choose(): selección token a token
src/arguments.py         extracción de argumentos y generación de candidatos
src/schema_validator.py  validación recursiva de args contra el esquema
src/engine.py            FunctionCallingEngine: orquesta todo el flujo por prompt
src/constants.py         valores compartidos expuestos como funciones (sin globales)
```

Estructura plana a propósito: un único caso de uso, sin capas innecesarias.

---

## 3. Flujo completo trazado con un ejemplo

Prompt de ejemplo: **`"What is the sum of 2 and 3?"`**
Funciones disponibles: `fn_add_numbers(a: number, b: number)`, `fn_greet`,
`fn_reverse_string`, `fn_get_square_root`, `fn_substitute_string_with_regex`.

### Paso 1 — Arranque y CLI
`src/__main__.py` llama a `main()` en `src/cli.py`.
`main()` (cli.py:97) hace: `parse_args()` → carga prompts y funciones → crea el
modelo → crea el motor → procesa → escribe. Todo dentro de un `try/except
CallMeMaybeError` para no crashear nunca.

### Paso 2 — Cargar entradas (`src/files.py`)
- `load_prompt_cases()` lee el JSON, exige que la raíz sea un array y valida cada
  item con el modelo `PromptCase`. **Acepta tanto `"texto"` como `{"prompt": "texto"}`**
  (lo resuelve `PromptCase.accept_plain_prompt` en models.py:171).
- `load_function_definitions()` valida cada función con `FunctionDefinition` y
  **rechaza listas vacías y nombres duplicados**.
- Errores aquí → `InputFileError` / `InputValidationError` / `FunctionDefinitionError`.

### Paso 3 — Crear el modelo (`src/llm.py`)
`QwenAdapter` construye `Small_LLM_Model` del SDK (Qwen3-0.6B). En su `__init__`:
- Si no se inyecta tokenizer, crea el **tokenizer propio** a partir de la ruta
  pública `model.get_path_to_tokenizer_file()` → `ByteLevelBPETokenizer.from_file()`.
- Expone tres métodos simples: `encode()`, `decode()`, `get_logits()`.
- Cualquier excepción del SDK se convierte en `ModelInferenceError`.

> En tests se inyecta un modelo falso (`CharacterModel`) que implementa el mismo
> `Protocol` de `ports.py`, así las pruebas son rápidas y deterministas.

### Paso 4 — Seleccionar la función (`src/engine.py` → `src/constrained_decoder.py`)
`FunctionCallingEngine._select_function()` (engine.py:117):
1. Construye el prompt de selección con `_function_prompt()` (engine.py:26): lista
   `- nombre: descripción` de cada función y pide "Function:".
2. Llama a `ConstrainedDecoder.choose(prompt, {nombre: FunctionDefinition}, "function")`.

Dentro de `choose()` (constrained_decoder.py:40), por cada opción se tokeniza su
texto (cacheado). Luego, **token a token**:
- Calcula el conjunto `allowed` = primer token de cada opción cuyo prefijo coincide
  con lo generado hasta ahora.
- Pide `logits = get_logits(prompt_tokens + generated)`.
- Elige `selected = max(allowed, key=logit)` → solo entre los permitidos.
- Repite hasta que `generated` sea exactamente los tokens de una opción → devuelve
  esa `FunctionDefinition`.

Resultado: `fn_add_numbers`. El modelo decidió, pero **no pudo inventar** un nombre.

### Paso 5 — Resolver los argumentos (`src/engine.py` → `src/arguments.py`)
`_resolve_arguments()` (engine.py:164):
1. `extract_arguments(prompt, function)` (arguments.py): reglas **genéricas por
   tipo**. Para `fn_add_numbers`, ve dos params `number` y `_numbers()` extrae
   `[2.0, 3.0]` del prompt en orden → `{"a": 2.0, "b": 3.0}`. Sin LLM (caso claro).
2. `ambiguous_names(prompt, function, args)`: decide qué params son dudosos. Aquí
   ninguno (dos números, dos huecos) → no se llama al modelo.
3. `validate_arguments(function, args)` (schema_validator.py): comprueba claves
   exactas, tipos, enums y estructuras anidadas. Si falla, hay un **reintento**:
   resolver TODOS los params vía LLM restringido y volver a validar.

> Para un caso ambiguo (p.ej. enum, o varios strings posibles), `_choose_argument()`
> genera candidatos con `value_candidates()` y usa **el mismo `ConstrainedDecoder`**
> para que el modelo elija solo entre representaciones JSON válidas.

### Paso 6 — Construir el resultado (`src/models.py`)
Se crea un `FunctionCallResult(prompt, fn_name, args)`. Pydantic con
`extra="forbid"` garantiza exactamente esas 3 claves, sin texto libre.

### Paso 7 — Escribir el JSON (`src/files.py`)
`write_results()` crea `data/output/`, serializa con `model_dump(mode="json")` y
`json.dump(indent=2, ensure_ascii=False)`. Salida:

```json
[
  { "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": { "a": 2.0, "b": 3.0 } }
]
```

---

## 4. Decodificación restringida en detalle (el corazón — te preguntarán esto)

Archivo: `src/constrained_decoder.py`, método `choose()`.

Cómo lo explico:
- "En cada paso miro los logits que da el modelo para el siguiente token, pero
  **descarto todos los que no continúan una opción válida**. De los que quedan, cojo
  el de mayor logit. Repito hasta completar exactamente una opción."
- Equivale a poner los logits inválidos a **−∞**; yo directamente solo evalúo el
  subconjunto permitido (`allowed`), que es la misma idea de forma más eficiente.

Detalles que conviene tener claros:
- **`allowed`** (constrained_decoder.py:77): `tokens[index]` de cada opción cuyo
  prefijo `tokens[:index]` coincide con `generated` y que aún tenga token en esa
  posición.
- **Caché** de tokenización por texto (`_token_cache`) → no re-tokenizo nombres.
- **Errores**: si no hay tokens permitidos o un token excede el vocabulario de los
  logits → `ModelInferenceError` (lo recoge el motor y hace fallback).
- **Desempate de prefijos** (limitación conocida y documentada): si los tokens de
  una opción son prefijo de otra, gana la corta. Un conjunto cerrado no tiene token
  de "frontera/EOS" para preguntar al modelo si parar o seguir. En la práctica los
  candidatos no se solapan a nivel de token, así que es predecible. *Si me piden
  arreglarlo de verdad, habría que añadir un token de fin de opción.*

---

## 5. Tokenizer propio en detalle (el bonus — también muy preguntado)

Archivo: `src/tokenizer.py`, clase `ByteLevelBPETokenizer`.

Por qué existe: el bonus pide **no usar `encode`/`decode` del SDK** y trabajar con
`get_logits_from_input_ids` + el vocabulario público. Reimplemento el tokenizador de
Qwen (familia GPT-2 / Byte-Level BPE).

`encode()` (4 fases):
1. **NFC**: normalizo Unicode (`unicodedata.normalize`).
2. **Pre-tokenización** (`_pretokenize`): replico la regex de Qwen con categorías
   Unicode (`unicodedata.category`) porque `re` de Python no soporta `\p{L}`. Separo
   contracciones (`'s`, `'re`…), letras, números (dígito a dígito), puntuación y
   espacios. **Detalle fino**: un espacio simple se **pega** al token siguiente
   (` palabra`), y en rachas de varios espacios, solo el **último** se pega; el resto
   forma su propio token (semántica GPT-2 ` ?\p{L}+`). Esto lo arreglé y lo verifico
   con un test de paridad contra el tokenizer real.
3. **Bytes→Unicode reversible** (`_bytes_to_unicode`): cada byte se mapea a un
   carácter imprimible (el famoso `Ġ` para el espacio).
4. **Merges BPE** (`_apply_bpe`): aplico los merges por ranking (el de menor rango
   primero) hasta no poder más, y traduzco a ids con el vocabulario.

`decode()`: ids → tokens → bytes (mapa inverso) → UTF-8.

**Cómo conecta con la decodificación restringida** (frase para defensa): "tokenizo
cada opción con mi propio BPE, comparo los ids contra los logits que devuelve el
modelo por id, y así enmascaro a nivel de token usando el mismo espacio de ids".

Verificación: `tests/test_tokenizer.py::test_encoding_matches_reference_tokenizer`
compara mi `encode` contra el tokenizer de referencia en texto, rutas, JSON,
símbolos, espacios y varios idiomas (hace skip si el modelo no está disponible).

---

## 6. Extracción de argumentos en detalle

Archivo: `src/arguments.py`.

Públicas:
- `extract_arguments()`: una pasada por los parámetros; asigna valores **genéricos
  por tipo** (no por nombre de parámetro).
- `value_candidates()`: dado un tipo, genera candidatos válidos (números, comillas,
  enum, JSON embebido, n-gramas para strings…).
- `ambiguous_names()`: marca qué params necesitan decisión del modelo (string vacío,
  colisión, enum con >1 valor, array/object vacío con varios candidatos).

Internas relevantes: `_numbers` (enteros, decimales, signo, notación científica,
miles con coma), `_quoted` (comillas simples/dobles), `_boolean_value`,
`_regex_arguments` (sustituciones regex genéricas: detecta patrón + comillas),
`_json_candidates` (arrays/objetos JSON dentro del prompt), `_named_symbol`
(`asterisk`→`*`, etc.), `_empty_value`.

Para strings, el orden de resolución (en `extract_arguments`) es genérico y
**dirigido por el nombre del parámetro de forma dinámica**, no hardcodeado:
1. `_labelled_value(prompt, name)` → valor tras una etiqueta `"<name>: ..."`.
2. email (si el nombre contiene "email") / siguiente string entre comillas.
3. `_value_before(prompt, name)` → palabra adyacente al nombre (p.ej. "production
   database" → `database` = "production").
4. `_path_like(prompt)` → token con forma de ruta (`/...` o `C:\...`).
5. `_last_word(prompt)` como último recurso; si nada es concluyente, el parámetro
   queda ambiguo y lo decide el LLM restringido.

> Decisión consciente y su porqué: **no hay un mapa hardcodeado de nombres
> anticipados** (antes existía `_contextual_string` con `if name == "path"/"template"`,
> que anticipaba funciones concretas — frágil y mala práctica según el subject: "no
> hardcodees", "los inputs pueden cambiar"). La versión actual usa el nombre del
> parámetro como pista genérica ("busca el valor que el usuario asoció a ESE
> parámetro"), lo cual es defendible y, medido contra el set privado de la moulinette,
> recupera 11/11 en argumentos.

Frase para defensa sobre el límite de las regex: "las regex no generalizan para
strings arbitrarios; por eso lo inequívoco se resuelve con reglas simples y lo
ambiguo se delega al modelo dentro de candidatos válidos".

---

## 7. Manejo de errores y fallback (requisito: nunca crashear)

- Toda excepción esperada hereda de `CallMeMaybeError` (models.py) y se captura en
  `cli.main()` → imprime "Error: ..." y devuelve código 1.
- Errores de inferencia en un prompt concreto no tumban el lote: `engine.process()`
  los captura y devuelve un **fallback controlado**:
  ```json
  { "prompt": "...", "fn_name": "Unable to retrieve from 'function_definitions.json'", "args": {} }
  ```
  El texto sale de `constants.unable_to_retrieve_fn_name()` (función, no variable
  global). Defensa: "prefiero un resultado controlado para ese caso a romper toda la
  ejecución; el JSON sigue siendo válido y parseable".

---

## 8. Decisiones de diseño (y por qué)

- **Greedy restringido** en vez de puntuar funciones completas: en CPU, puntuar cada
  función entera multiplicaba el tiempo; con el límite de 5 min no compensaba.
- **Extracción híbrida** (reglas + LLM solo para ambigüedad): generar TODOS los args
  con logits superaba los 5 min.
- **JSON construido desde el esquema**, no desde texto libre del modelo → 100%
  parseable.
- **`Protocol LanguageModel`** desacopla motor y SDK → tests rápidos.
- **Pydantic con `extra="forbid"`** → rechaza claves extra y valida recursivamente.
- **Sin variables globales**: incluso las constantes se exponen como funciones.

---

## 9. Preguntas probables del evaluador (con respuesta)

**¿Esto es realmente decodificación restringida o haces que el modelo escriba JSON?**
Restringida. Construyo el JSON yo; el modelo solo elige entre tokens permitidos.
Demuéstralo con `--trace`.

**¿La función se elige con el LLM o con heurística?**
Con el LLM (`_select_function` → `ConstrainedDecoder`). La heurística solo participa
en *extraer candidatos* de argumentos, nunca en elegir la función.

**¿Por qué reimplementaste el tokenizer si el SDK ya tiene `encode`?**
Es el bonus: no usar `encode`/`decode` del SDK; usar `get_logits_from_input_ids` y el
vocabulario público. Además demuestro cómo encajan tokenización e ids con los logits.

**¿Cómo garantizas que tu tokenizer coincide con Qwen?**
Test de paridad contra el tokenizer de referencia. Tuve un bug con espacios
múltiples y lo arreglé (semántica del espacio que se pega a la palabra).

**¿Qué pasa si el JSON de entrada es inválido o no existe?**
`files.py` lanza `InputFileError`/`InputValidationError`, `cli.main()` los captura,
imprime un mensaje claro y devuelve 1. Nunca crashea. (Pruébalo borrando el input.)

**¿Y si un prompt no encaja con ninguna función / no se puede resolver?**
Fallback controlado con `fn_name = "Unable to retrieve..."` y `args = {}`.

**¿Por qué Pydantic en todo?**
El subject lo exige para validación; lo uso en todos los modelos de datos de entrada
y salida, con `extra="forbid"`.

**¿Soporta argumentos anidados?**
Sí: `ParameterSpec` es recursivo (`items`, `properties`, `required`) y
`schema_validator._matches_spec` valida recursivamente; los valores anidados se
extraen como JSON embebido del prompt o `{}`/`[]` y los valida el esquema.

---

## 10. Recodificación: qué te pueden pedir tocar y dónde

El subject avisa de una posible modificación pequeña en la defensa. Las más
probables, con ubicación exacta y qué vigilar:

1. **Añadir/cambiar una clave del output** (p.ej. añadir `confidence` o renombrar
   `fn_name`).
   - Tocar: `src/models.py` (`FunctionCallResult`), donde se crea en
     `src/engine.py` (`process`), y el ejemplo en tests/`test_files.py`.
   - Vigila: `extra="forbid"` rechaza claves no declaradas; añádela al modelo.

2. **Añadir una restricción de tipo** (p.ej. `minimum`/`maximum` en number, o
   `min_length` en string).
   - Tocar: `src/models.py` (`ParameterSpec`, nuevo campo) y
     `src/schema_validator.py` (`_matches_spec`, nueva comprobación).
   - Vigila: actualizar `value_candidates` si afecta a la generación.

3. **Cambiar el texto del prompt de selección de función**.
   - Tocar: `src/engine.py` (`_function_prompt`). Solo cambia el string.

4. **Cambiar la estrategia de selección de token** (p.ej. elegir el de MENOR logit,
   o añadir desempate, o un token de frontera).
   - Tocar: `src/constrained_decoder.py` (`choose`, línea del `max(...)`).
   - Vigila: el bucle debe seguir terminando (cada paso añade un token que continúa
     una opción válida).

5. **Soportar un nuevo tipo de argumento o un nuevo extractor** (p.ej. fechas).
   - Tocar: `src/arguments.py` (`value_candidates` y/o `extract_arguments`).
   - Vigila: que el tipo exista en `ParameterSpec` (models.py) y se valide en
     `schema_validator`.

6. **Cambiar rutas por defecto o añadir un flag**.
   - Tocar: `src/cli.py` (`parse_args` y `main`).

7. **Cambiar el comportamiento del fallback** (p.ej. omitir el prompt en vez de
   escribir "Unable to...").
   - Tocar: `src/engine.py` (`process`) y `src/constants.py`.

8. **Imprimir estadísticas** (p.ej. cuántos JSON válidos / nº de fallbacks).
   - Tocar: `src/cli.py` (`main`, tras `engine.process`).

Checklist al recodificar en vivo:
- Mantén `flake8`/`mypy` limpios (hay type hints en todo; respeta `--strict`).
- Si tocas modelos o salida, **actualiza el test** correspondiente y corre
  `make test`.
- No introduzcas variables globales (es una restricción autoimpuesta del proyecto).
- Si tocas el tokenizer, corre el test de paridad.
- Verifica con una ejecución real: `uv run python -m src --trace`.

---

## 11. Resumen de una frase por archivo (para responder "¿dónde está X?")

- "¿Dónde eliges la función?" → `engine.py::_select_function` + `constrained_decoder.py::choose`.
- "¿Dónde extraes argumentos?" → `arguments.py` (genéricos) y, si hay ambigüedad, `engine.py::_choose_argument`.
- "¿Dónde validas el esquema?" → `schema_validator.py::validate_arguments`.
- "¿Dónde tokenizas?" → `tokenizer.py::ByteLevelBPETokenizer.encode`.
- "¿Dónde llamas al modelo?" → `llm.py::QwenAdapter.get_logits` → SDK `get_logits_from_input_ids`.
- "¿Dónde lees/escribes JSON?" → `files.py`.
- "¿Dónde está el manejo de errores?" → excepciones en `models.py`, captura en `cli.py::main` y `engine.py::process`.
