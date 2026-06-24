# call_me_maybe — Guía completa de defensa (documento personal)

Todo lo que necesitas para entender el proyecto **línea a línea** y explicarlo sin
miedo: la pipeline completa, qué hace cada archivo y cada función, un ejemplo
trazado de principio a fin, los puntos fuertes (incluidos los que cuentan como
bonus), las preguntas probables y qué te pueden pedir recodificar.

> Este archivo NO se entrega (vive en `docs/`). Aquí hablo claro de todo.

---

## Índice

1. El pitch y las 3 ideas clave
2. Cómo se ejecuta
3. La pipeline de un vistazo
4. Mapa de archivos (una responsabilidad por archivo)
5. Ejemplo trazado COMPLETO (prompt → output), paso a paso
6. Recorrido por el código, archivo por archivo y función por función
7. Decodificación restringida a fondo (con traza de tokens)
8. El tokenizer propio a fondo (las 4 fases)
9. Extracción de argumentos a fondo (precedencia y ambigüedad)
10. Validación de esquema
11. Manejo de errores y fallback
12. Puntos fuertes / extras (los bonus que tenemos y dónde están)
13. Decisiones de diseño y su porqué
14. Preguntas probables del evaluador (con respuesta)
15. Recodificación: qué te pueden pedir y dónde tocar
16. Cómo medir (comandos) y números reales

---

## 1. El pitch y las 3 ideas clave

> El programa **no le pide al modelo que escriba JSON libre**. Usa el LLM solo para
> decisiones semánticas (qué función, y qué valor cuando hay ambigüedad), pero cada
> decisión está **restringida a opciones válidas** mediante decodificación token a
> token sobre los logits. La estructura final la construye Python desde el esquema y
> se valida con Pydantic antes de escribirla. Por eso el JSON es 100% parseable
> incluso con un modelo de 0.6B.

Las 3 ideas que debes saber defender:

1. **La función se elige con el LLM, no con heurística.** Cumple el requisito del
   subject "la función a llamar debe elegirse usando el LLM". → `engine.py` +
   `constrained_decoder.py`.
2. **Decodificación restringida = enmascarar logits.** En cada paso ignoro (pongo a
   −∞) todo token que no continúe una opción válida y elijo el de mayor logit entre
   los permitidos. → `constrained_decoder.py::ConstrainedDecoder.choose`.
3. **Tokenizer propio.** No uso `encode`/`decode` del SDK en el flujo real:
   reimplemento Byte-Level BPE leyendo `tokenizer.json` y uso
   `get_logits_from_input_ids`. → `tokenizer.py`.

---

## 2. Cómo se ejecuta

```bash
uv run python -m src [--input <file>] [--output <file>] [--functions <file>]
                     [--model <hf_id>] [--list-models] [--trace]
```

- `--input`: prompts. Por defecto `data/input/function_calling_tests.json`.
- `--functions`: definiciones. Por defecto `data/input/function_definitions.json`.
- `--output`: salida. Por defecto `data/output/function_calling_results.json`.
- `--model`: modelo del SDK. Por defecto `Qwen/Qwen3-0.6B` (obligatorio). Verificado
  también con `Qwen/Qwen3-1.7B`.
- `--list-models`: lista modelos recomendados y sale sin cargar nada.
- `--trace`: imprime cada decisión de la decodificación restringida (sirve para
  *visualizar* el proceso de generación en la defensa).

El SDK selecciona dispositivo, pero el adaptador fuerza `cpu` por defecto para que
sea reproducible en cualquier máquina de evaluación.

---

## 3. La pipeline de un vistazo

```text
                          ┌─────────────────────────────────────────┐
 prompt(s) ──► cli.main ──► files.load_* (Pydantic valida entrada)   │
                          │                                          │
                          ▼                                          │
                FunctionCallingEngine.process(prompts)               │
                          │                                          │
        ┌── por cada prompt ──────────────────────────────┐         │
        │ 1) _select_function ──► ConstrainedDecoder.choose│         │
        │      (LLM elige el nombre, token a token)        │         │
        │ 2) _resolve_arguments                            │         │
        │      a) extract_arguments  (reglas genéricas)    │         │
        │      b) ambiguous_names    (¿qué falta decidir?) │         │
        │      c) _choose_argument ──► ConstrainedDecoder  │         │
        │      d) validate_arguments (esquema Pydantic)    │         │
        │ 3) FunctionCallResult (prompt, fn_name, args)    │         │
        └──────────────────────────────────────────────────┘         │
                          │                                          │
                          ▼                                          │
                  files.write_results ──► data/output/...json ◄──────┘

   El motor habla con el modelo a través de un Protocol (ports.LanguageModel):
     encode()  ──► tokenizer.py (BPE propio)
     get_logits() ──► llm.py ──► SDK.get_logits_from_input_ids
     decode()  ──► tokenizer.py (BPE propio)
```

Punto clave de integración: **tokenizo con MI tokenizer** y le paso los `input_ids`
al modelo del SDK para pedir `get_logits`. Mismo espacio de ids → puedo enmascarar a
nivel de token.

---

## 4. Mapa de archivos

```text
src/__main__.py            entrypoint mínimo (python -m src)
src/__init__.py            declara el paquete; sin atajos con efectos colaterales
src/cli.py                 argumentos, orquestación de alto nivel, captura de errores
src/files.py               leer/validar JSON de entrada y escribir el JSON de salida
src/models.py              modelos Pydantic (datos) + jerarquía de excepciones
src/ports.py               Protocol LanguageModel (desacopla motor del SDK)
src/llm.py                 QwenAdapter: envuelve el SDK + el tokenizer propio
src/tokenizer.py           ByteLevelBPETokenizer (encode/decode propios)
src/constrained_decoder.py ConstrainedDecoder.choose(): selección token a token
src/arguments.py           extracción de argumentos y generación de candidatos
src/schema_validator.py    validación recursiva de args contra el esquema
src/engine.py              FunctionCallingEngine: orquesta el flujo por prompt
```

Estructura plana a propósito: un único caso de uso, sin capas innecesarias. **Sin
variables globales** (restricción autoimpuesta).

---

## 5. Ejemplo trazado COMPLETO

Prompt: **`"What is the sum of 2 and 3?"`**
Funciones disponibles (las públicas):
`fn_add_numbers(a:number, b:number)`, `fn_greet(name:string)`,
`fn_reverse_string(s:string)`, `fn_get_square_root(a:number)`,
`fn_substitute_string_with_regex(source_string, regex, replacement)`.

### (1) Arranque
`python -m src` ejecuta `src/__main__.py`, que llama a `cli.main()` y convierte su
retorno en código de salida (`raise SystemExit(main())`).

### (2) CLI: `cli.main()`
1. `parse_args()` construye el `Namespace` con defaults.
2. `--list-models` no está → sigue.
3. `load_prompt_cases(args.input)` → lista de `PromptCase`.
4. `load_function_definitions(args.functions)` → lista de `FunctionDefinition`.
5. Crea `QwenAdapter(model_name="Qwen/Qwen3-0.6B")` (no se inyectó modelo).
6. Crea `FunctionCallingEngine(model, functions, trace=None)`.
7. `engine.process(prompts)` → lista de `FunctionCallResult`.
8. `write_results(args.output, results)`.
9. Todo dentro de `try/except CallMeMaybeError` → nunca crashea.

### (3) Carga de entrada (`files.py`)
- `_load_json`: comprueba que el archivo existe y es un archivo; abre con context
  manager; `json.load`. Errores → `InputFileError`.
- `load_prompt_cases` → `_validate_array(..., PromptCase, "prompt")`. Como el item es
  el string `"What is the sum of 2 and 3?"`, `PromptCase.accept_plain_prompt`
  (validador `mode="before"`) lo convierte en `{"prompt": "..."}`. (Acepta tanto
  array de strings como de objetos → cubre el formato del subject.)
- `load_function_definitions` valida cada función y **rechaza listas vacías y nombres
  duplicados**.

### (4) Construcción del motor (`engine.py::__init__`)
- Si `functions` está vacío → `FunctionDefinitionError`.
- `self._by_name = {f.name: f}`.
- `self._decoder = ConstrainedDecoder(model)`.
- Pre-tokeniza cada nombre de función (`self._decoder.encode(f.name)`) para **cachear**
  y que la selección sea rápida.

### (5) Selección de función (`engine._select_function`)
- `_function_prompt(user_prompt, functions)` arma el texto:
  ```text
  Choose the function that best matches the user request.

  Available functions:
  - fn_add_numbers: Add two numbers
  - fn_greet: ...
  - ...

  User request:
  What is the sum of 2 and 3?

  Function:
  ```
- Llama a `decoder.choose(prompt, {nombre: FunctionDefinition}, "function")`.
- Internamente (ver §7): tokeniza cada nombre, y **token a token** solo permite
  continuaciones de nombres válidos, eligiendo el de mayor logit. El modelo "lee" la
  petición y converge a `fn_add_numbers`. **No puede inventar** un nombre.
- Devuelve la `FunctionDefinition` de `fn_add_numbers`.

### (6) Resolución de argumentos (`engine._resolve_arguments`)
1. `extract_arguments(prompt, function)`:
   - No es el esquema `{source_string,regex,replacement}`.
   - `_numbers("What is the sum of 2 and 3?")` → `[2.0, 3.0]`.
   - Param `a` (number) → `2.0`; param `b` (number) → `3.0`.
   - Resultado: `{"a": 2.0, "b": 3.0}` (sin tocar el LLM: caso inequívoco).
2. `ambiguous_names(...)`: no hay strings, ni enums, ni arrays/objetos → conjunto
   vacío. No se llama al modelo para argumentos.
3. `validate_arguments(function, {"a":2.0,"b":3.0})`: claves exactas ✓, tipos
   `number` ✓ → devuelve los args.

### (7) Resultado y escritura
- `FunctionCallResult(prompt=..., fn_name="fn_add_numbers", args={"a":2.0,"b":3.0})`.
  Pydantic con `extra="forbid"` garantiza exactamente esas 3 claves.
- `write_results`: crea `data/output/`, serializa con `model_dump(mode="json")` y
  `json.dump(indent=2, ensure_ascii=False)`:
  ```json
  [
    { "prompt": "What is the sum of 2 and 3?",
      "fn_name": "fn_add_numbers",
      "args": { "a": 2.0, "b": 3.0 } }
  ]
  ```

### Variante con argumento ambiguo (para que lo veas)
Prompt `"Execute SQL query 'SELECT * FROM users' on the production database"` con
`fn_execute_sql_query(query:string, database:string)`:
- `extract_arguments`: `query` → primera cadena entre comillas `"SELECT * FROM users"`;
  `database` → `_value_before(prompt, "database")` = `"production"` (palabra que
  precede al nombre del parámetro). Determinista, sin LLM.
- Si NO se hubiera podido extraer (p.ej. valor entre varias opciones), el parámetro
  se marcaría ambiguo en `ambiguous_names` y se resolvería con
  `_choose_argument` → `ConstrainedDecoder` sobre candidatos JSON válidos.

---

## 6. Recorrido por el código

### `src/__main__.py`
Una sola responsabilidad: `from src.cli import main` y
`raise SystemExit(main())`. Mantiene el entrypoint mínimo y el código de salida del
proceso correcto (0 ok, 1 error esperado).

### `src/__init__.py`
Declara `src` como paquete. **No expone atajos** ni importa el SDK al cargar el
paquete (evita efectos colaterales y arranques lentos/impredecibles).

### `src/cli.py`
- `default_model_name()` → `"Qwen/Qwen3-0.6B"` (modelo obligatorio; función, no
  constante global).
- `recommended_models()` → tupla de `(modelo, nota)` sin estado global.
- `print_supported_models()` → imprime la lista y los requisitos para modelos custom
  (lo que ves con `--list-models`).
- `parse_args(argv)` → define `--input/--output/--functions/--model/--list-models/
  --trace` con `argparse`. `argv=None` usa los del proceso (permite tests).
- `main(argv, model)` → orquesta (ver §5). Acepta un `model` inyectado (tests) para
  no cargar el SDK real. `trace = print if args.trace else None`. Captura
  `CallMeMaybeError` → imprime "Error: ..." y retorna 1.

### `src/files.py`
- `_load_json(path)` → existencia, es-archivo, `json.load` con context manager;
  `json.JSONDecodeError`/`OSError` → `InputFileError`.
- `_validate_array(raw, path, model, item_name)` → exige raíz lista; valida cada item
  con `model.model_validate`; `ValidationError` → `InputValidationError` indicando el
  índice.
- `load_prompt_cases` / `load_function_definitions` → usan lo anterior. La segunda
  además rechaza lista vacía (`FunctionDefinitionError`) y **nombres duplicados**.
- `write_results(path, results)` → `mkdir(parents=True, exist_ok=True)`,
  `model_dump(mode="json")`, `json.dump(indent=2, ensure_ascii=False)` + `\n` final.
  `OSError` → `OutputFileError`.

### `src/models.py` (Pydantic = validación; cumple "todas las clases usan pydantic")
- Jerarquía de errores: `CallMeMaybeError` (base) → `InputFileError`,
  `InputValidationError`, `OutputFileError`, `FunctionDefinitionError`,
  `ModelInferenceError`. Atrapar la base = atrapar todo lo esperado.
- `ParameterSpec` (recursivo): `type` (Literal de los 6 tipos), `description`,
  `enum`, `items` (para arrays), `properties` + `required` (para objetos). Validador
  `validate_type_constraints`: `items` solo en array, `properties` solo en object,
  `required` debe referenciar propiedades declaradas. `extra="forbid"`.
- `ReturnSpec`: solo `type`.
- `FunctionDefinition`: `name`, `description`, `parameters: dict[str, ParameterSpec]`,
  `returns`. Validadores: nombre/descrip no en blanco; nombres de parámetro no en
  blanco.
- `PromptCase`: `prompt` (min_length 1). `accept_plain_prompt` (before) convierte un
  string suelto en `{"prompt": str}`. `reject_blank_prompt` rechaza solo-espacios.
- `FunctionCallResult`: `prompt`, `fn_name`, `args: dict[str, Any]`, `extra="forbid"`.

### `src/ports.py`
`LanguageModel` (Protocol) con `encode`, `decode`, `get_logits`. Es el **contrato**
que usa el motor. Gracias a esto, en tests inyectamos un `CharacterModel` falso,
rápido y determinista, sin cargar Qwen.

### `src/llm.py`
- Protocols auxiliares: `SDKLanguageModel` (lo que ofrece el SDK: `encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_tokenizer_file`), `EncodedTokens`
  (tensor con `.tolist()`), `TextTokenizer`.
- `_SDKTokenizerFallback`: si en tests se inyecta un modelo SDK pero no un tokenizer,
  envuelve `model.encode().tolist()` para dar `list[int]`.
- `QwenAdapter` (implementa `LanguageModel`):
  - `__init__`: si no hay modelo inyectado, crea `Small_LLM_Model(model_name, device)`.
    Si no hay tokenizer, crea **el propio** con
    `ByteLevelBPETokenizer.from_file(model.get_path_to_tokenizer_file())`. Cualquier
    excepción del SDK → `ModelInferenceError`.
  - `encode/decode` → delegan en el tokenizer propio (¡no en el SDK!).
  - `get_logits(input_ids)` → `model.get_logits_from_input_ids(input_ids)`, con
    `try/except` → `ModelInferenceError`.

### `src/tokenizer.py`  → ver §8.
### `src/constrained_decoder.py`  → ver §7.
### `src/arguments.py`  → ver §9.
### `src/schema_validator.py`  → ver §10.

### `src/engine.py`
- `_function_prompt` y `_value_prompt`: construyen los textos para selección de
  función y de valor ambiguo.
- `FunctionCallingEngine.__init__`: guarda funciones, `_by_name`, crea el decoder y
  pre-cachea nombres.
- `_select_function`: `choose` sobre `{name: FunctionDefinition}` → devuelve la
  función. `cast` solo para mypy.
- `_choose_argument`: `value_candidates(prompt, spec)` → `choices = {json.dumps(v): v}`
  → `choose` sobre el `_value_prompt`. Devuelve el valor elegido.
- `_resolve_arguments`: extrae → marca ambiguos → resuelve ambiguos con el LLM →
  `validate_arguments`. Si la validación lanza `ModelInferenceError`, **reintenta**
  resolviendo TODOS los parámetros con el LLM y vuelve a validar.
- `process`: por cada prompt, intenta `select`+`resolve`+`FunctionCallResult`; si hay
  `ModelInferenceError`, añade el **fallback controlado** (ver §11). Conserva el orden
  de entrada.

---

## 7. Decodificación restringida a fondo

Archivo: `src/constrained_decoder.py`, método `choose(prompt, choices, label)`.

```text
choices : dict[str, valor]      # texto permitido -> valor a devolver
```

Algoritmo:

1. `tokenized = {texto: encode(texto)}` para cada opción (con caché por texto).
2. Si alguna se tokeniza a lista vacía → `ModelInferenceError`.
3. `prompt_tokens = model.encode(prompt)`; `generated = []`.
4. Bucle:
   a. Si `generated` == los tokens de alguna opción → devuelvo su valor (fin).
   b. `index = len(generated)`.
   c. `allowed = { tokens[index] }` para cada opción cuyo prefijo `tokens[:index]`
      coincide con `generated` y que aún tenga token en esa posición.
   d. `logits = get_logits(prompt_tokens + generated)`.
   e. Si no hay `allowed`, o algún token permitido excede el tamaño de `logits`
      (vocabulario incompatible) → `ModelInferenceError`.
   f. `selected = max(allowed, key=lambda t: logits[t])` → **solo entre permitidos**.
   g. `generated.append(selected)`.

Cómo lo explico en una frase: *"miro los logits del modelo, pero descarto todos los
tokens que no continúan una opción válida; de los que quedan, cojo el de mayor logit;
repito hasta completar exactamente una opción."* Equivale a poner los logits
inválidos a −∞; yo directamente solo evalúo el subconjunto permitido (más eficiente).

Traza conceptual (función con opciones cuyos tokens son `A=[10,11]`, `B=[10,27]`):
```text
generated=[]      allowed={10}        -> elijo 10            generated=[10]
generated=[10]    allowed={11,27}     -> logits[11] vs [27]  -> elijo el mayor
si elige 11 -> generated=[10,11]=A -> devuelvo A
```

Con `--trace` esto se imprime: `function allowed=2 token=27 text='...'` y al final
`function=fn_add_numbers`.

Caché y errores:
- `_token_cache` evita re-tokenizar las mismas opciones.
- Errores → `ModelInferenceError`, que el motor recoge para reintentar o hacer
  fallback.

**Limitación conocida (¡tenla preparada!):** si los tokens de una opción son prefijo
de los de otra, gana la corta (el chequeo de igualdad salta antes de poder extender).
Un conjunto cerrado no tiene token de "frontera/EOS" para preguntar al modelo
"¿parar o seguir?". En la práctica los candidatos no se solapan a nivel de token, así
que el desempate es predecible. *Si me piden arreglarlo de verdad: añadir un token de
fin de opción y dejar que el modelo elija entre "terminar" y "continuar".*

---

## 8. El tokenizer propio a fondo

Archivo: `src/tokenizer.py`, clase `ByteLevelBPETokenizer`. Reimplemento el
tokenizador de Qwen (familia GPT-2 / Byte-Level BPE) sin usar `encode`/`decode` del
SDK: cargo `tokenizer.json` (vocab + merges) por un método público y trabajo con
`get_logits_from_input_ids`.

`from_file(path)`: abre `tokenizer.json`, lee `model.vocab` (dict token→id) y
`model.merges` (lista de pares). Errores → `InputFileError`.

`encode(text)` — 4 fases:
1. **NFC** (`unicodedata.normalize("NFC", text)`): unifica formas Unicode.
2. **Pre-tokenización** (`_pretokenize`): replico la regex de Qwen con categorías
   Unicode (porque `re` de Python no soporta `\p{L}`). Separo:
   - contracciones (`'s`, `'re`, `'ve`, `'ll`, `'t`, `'m`, `'d`),
   - letras (`\p{L}+`, con un posible carácter no-letra inicial),
   - números **dígito a dígito**,
   - puntuación (`[^\s\p{L}\p{N}]+`, con posible espacio inicial),
   - saltos de línea y espacios.
   **Detalle fino (y bug que arreglé):** un espacio simple se **pega** al token
   siguiente (` palabra`); y en una racha de varios espacios, solo el **último** se
   pega y el resto forma su propio token (semántica GPT-2 ` ?\p{L}+`). Antes agrupaba
   todos los espacios juntos → divergía del tokenizer real con espacios múltiples.
3. **Bytes→Unicode reversible** (`_bytes_to_unicode`): cada byte (0–255) se mapea a un
   carácter imprimible único (de ahí el `Ġ` que representa el espacio). Esto permite
   que el BPE trabaje sobre caracteres y luego volver a bytes exactos.
4. **Merges BPE** (`_apply_bpe`, con `lru_cache`): parto el fragmento en caracteres y
   aplico los merges por **ranking** (el de menor rango primero) hasta que no haya más
   pares fusionables; luego traduzco cada token resultante a su id con `vocab`. Token
   desconocido → `InputFileError`.

`decode(token_ids)`: id→token (`_tokens`), concateno, mapeo inverso carácter→byte
(`_byte_decoder`), y `bytes.decode("utf-8", errors="replace")`.

**Cómo conecta con la decodificación restringida (frase de defensa):** *"tokenizo
cada opción con mi propio BPE, y como el modelo me da un logit por id de token,
comparo en el mismo espacio de ids para enmascarar a nivel de token."*

Verificación: `tests/test_tokenizer.py::test_encoding_matches_reference_tokenizer`
compara mi `encode` con el tokenizer de referencia sobre texto, rutas, JSON,
símbolos, espacios y varios idiomas (hace `skip` si el modelo no está disponible).

---

## 9. Extracción de argumentos a fondo

Archivo: `src/arguments.py`.

### `extract_arguments(prompt, function)`
- Si los parámetros incluyen `{source_string, regex, replacement}` → `_regex_arguments`
  (manejo dedicado de sustituciones, ver abajo).
- Si no, recorre los parámetros una vez:
  - **number/integer**: toma el siguiente de `_numbers(prompt)` (en orden). `integer`
    → `int(...)`. Si no quedan números → `_empty_value`.
  - **string** (precedencia genérica, **dirigida por el nombre, no hardcodeada**):
    1. `_labelled_value(prompt, name)` → valor tras `"<name>: ..."`.
    2. email (si `"email" in name`) / siguiente cadena entre comillas.
    3. `_value_before(prompt, name)` → palabra que precede al nombre del parámetro
       (p.ej. "production database" → "production").
    4. `_path_like(prompt)` → token con forma de ruta (`/...` o `C:\...`).
    5. `_last_word(prompt)` como último recurso.
  - **boolean**: `_boolean_value(prompt)` (true/yes/on… vs false/no/off…) o `False`.
  - resto (array/object) → `_empty_value`.

### Helpers
- `_numbers`: regex que cubre enteros, decimales, signo, notación científica y miles
  con coma (`1,000`).
- `_quoted`: cadenas entre comillas simples o dobles, en orden.
- `_labelled_value` / `_value_before`: usan el **nombre del parámetro** como pista
  dinámica (etiqueta `nombre:` o palabra adyacente). Es genérico: no hay `if
  name=="path"`.
- `_path_like`: forma de ruta por su **aspecto** (valor), no por el nombre.
- `_regex_arguments`: detecta alias de patrón (`numbers`→`\d+`, `vowels`→`[aeiou...]`,
  etc.) + comillas; o el caso "sustituye 'X' por 'Y' en 'Z'"; arma
  `{source_string, regex, replacement}`. `_named_symbol` convierte "asterisk"→"*".
- `_json_candidates`: busca arrays/objetos JSON embebidos en el prompt.
- `_empty_value`: valor vacío según tipo (`""`, `0`, `False`, `[]`, `{}`).

### `value_candidates(prompt, spec)`
Genera el conjunto de candidatos válidos para un argumento ambiguo:
- `enum` → los valores del enum.
- integer/number → números del prompt.
- boolean → el booleano detectado o `[True, False]`.
- array/object → JSON embebido + `[]`/`{}`.
- string → comillas + lo que sigue a `:` + **n-gramas** de 1..8 palabras + símbolos
  nombrados + `""`.

### `ambiguous_names(prompt, function, arguments)`
Marca qué parámetros necesitan decisión del LLM:
- strings: si hay más strings que cadenas entre comillas y se repiten valores
  (colisión); o si el valor está vacío o coincide con el nombre de un parámetro.
- `enum` con más de un valor → ambiguo.
- array/object vacío con más de un candidato → ambiguo.

Idea de defensa: *"lo inequívoco se resuelve con reglas simples y rápido; lo ambiguo
se delega al modelo, pero siempre dentro de candidatos válidos."* El subject solo
obliga a usar el LLM para elegir la **función** (eso sí lo cumplimos al 100%); para
los valores, este enfoque híbrido es rápido y fiable.

---

## 10. Validación de esquema

Archivo: `src/schema_validator.py`.

- `_matches_spec(value, spec)` (recursivo):
  - tipos simples: `string/number/integer/boolean/array/object` (ojo: `bool` no
    cuenta como `number`/`integer`; se excluye explícitamente).
  - `enum`: el valor debe estar en la lista.
  - array con `items`: todos los elementos cumplen el sub-esquema.
  - object con `properties`: claves requeridas presentes, sin claves extra, y cada
    propiedad cumple su sub-esquema.
- `validate_arguments(function, arguments)`:
  - las **claves** deben coincidir exactamente con `function.parameters` (ni de más ni
    de menos).
  - cada valor cumple `_matches_spec`.
  - si algo falla → `ModelInferenceError` (para reintento o fallback).

Esto, junto a Pydantic en la salida, garantiza el "JSON válido al 100% y conforme al
esquema" del subject.

---

## 11. Manejo de errores y fallback

- Todo error esperado hereda de `CallMeMaybeError` y se captura en `cli.main()` →
  imprime "Error: ..." y retorna 1. **Nunca crashea.**
- Errores de inferencia en un prompt concreto no tumban el lote: `engine.process()`
  los captura y añade un **fallback controlado**:
  ```json
  { "prompt": "...", "fn_name": "Unable to retrieve from 'function_definitions.json'", "args": {} }
  ```
  El texto es un literal en `engine.py::process` (único punto donde se usa).
- Defensa: *"prefiero un resultado controlado para ese caso a romper toda la
  ejecución; la salida sigue siendo JSON válido."*

---

## 12. Puntos fuertes / extras (los que cuentan como bonus) y dónde están

- **Recodificar el tokenizer** (no usar `encode`/`decode` del SDK; usar
  `get_logits_from_input_ids` + vocabulario público): `tokenizer.py` + `llm.py`.
- **Implementación pública de `encode` y `decode`**: `tokenizer.py`
  (`ByteLevelBPETokenizer.encode/decode`).
- **Demostración de cómo encode/decode encajan con la decodificación restringida**:
  esta guía + `--trace` + `constrained_decoder.py` (mismo espacio de ids).
- **Visualización del proceso de generación**: flag `--trace` (`cli.py` →
  `engine.py`/`constrained_decoder.py`).
- **Suite de tests amplia**: `tests/` (38 tests), incluida la **paridad** del
  tokenizer con el de referencia.
- **Soporte de múltiples modelos**: `--model` (`cli.py`); verificado con
  `Qwen/Qwen3-0.6B` y `Qwen/Qwen3-1.7B`, ambos 100%.
- **Recuperación de errores avanzada**: reintento en `engine._resolve_arguments` +
  fallback controlado en `engine.process`.
- **Optimizaciones**: caché de tokens (`ConstrainedDecoder._token_cache`), pre-cacheo
  de nombres de función, `lru_cache` en `_apply_bpe`.
- **Argumentos anidados**: `ParameterSpec` recursivo (`items`/`properties`/`required`)
  + `schema_validator._matches_spec` recursivo + extracción de JSON embebido.

---

## 13. Decisiones de diseño y su porqué

- **Greedy restringido** en vez de puntuar funciones completas: en CPU, puntuar cada
  función entera multiplicaba el tiempo; con el límite de 5 min no compensaba.
- **Extracción híbrida** (reglas + LLM solo para ambigüedad): generar TODOS los args
  con logits superaba los 5 min y, además, daba peores resultados que las reglas en
  casos claros.
- **JSON construido desde el esquema**, no desde texto libre del modelo → 100%
  parseable.
- **`Protocol LanguageModel`** desacopla motor y SDK → tests rápidos y deterministas.
- **Pydantic con `extra="forbid"`** → rechaza claves extra y valida recursivamente.
- **Sin variables globales**: incluso las constantes se exponen como funciones.
- **Extracción dirigida por el nombre, no hardcodeada**: defendible y robusta ante
  inputs cambiados (no anticipo funciones concretas).

---

## 14. Preguntas probables del evaluador (con respuesta)

**¿Esto es decodificación restringida real o haces que el modelo escriba JSON?**
Restringida. Construyo el JSON yo; el modelo solo elige entre tokens permitidos.
Pruébalo con `--trace`.

**¿La función se elige con el LLM o con heurística?**
Con el LLM (`_select_function` → `ConstrainedDecoder`). La heurística solo extrae
candidatos de argumentos, nunca elige la función.

**¿Por qué reimplementaste el tokenizer si el SDK ya tiene `encode`?**
Para no depender de `encode`/`decode` del SDK: uso `get_logits_from_input_ids` y el
vocabulario público, y demuestro cómo encajan tokenización, ids y logits.

**¿Cómo garantizas que tu tokenizer coincide con Qwen?**
Test de paridad contra el tokenizer de referencia. Tuve un bug con espacios
múltiples y lo arreglé (el espacio que se pega a la palabra).

**¿Qué pasa si el JSON de entrada es inválido o no existe?**
`files.py` lanza `InputFileError`/`InputValidationError`, `cli.main()` los captura,
imprime un mensaje claro y retorna 1. Nunca crashea.

**¿Y si un prompt no se puede resolver?**
Fallback controlado con `fn_name = "Unable to retrieve..."` y `args = {}`.

**¿Por qué Pydantic?**
El subject lo exige; lo uso en todos los modelos de datos, con `extra="forbid"`.

**¿Soporta argumentos anidados?**
Sí: `ParameterSpec` es recursivo y `_matches_spec` valida recursivamente; los valores
anidados se extraen como JSON embebido o se validan contra el esquema.

**¿Por qué greedy y no muestreo/temperatura?**
Queremos determinismo y fiabilidad; el mayor logit entre los permitidos basta para
estructura/semántica. (Si quieren, se cambia en una línea.)

---

## 15. Recodificación: qué te pueden pedir y dónde tocar

El subject avisa de una posible modificación pequeña en la defensa. Las más
probables, con ubicación y qué vigilar:

1. **Añadir/cambiar una clave del output** (p.ej. `confidence`, o renombrar `fn_name`).
   - Tocar: `models.py` (`FunctionCallResult`), su creación en `engine.process`, y el
     test en `tests/test_files.py`.
   - Vigila: `extra="forbid"` rechaza claves no declaradas; añádela al modelo.

2. **Añadir una restricción de tipo** (p.ej. `minimum`/`maximum`, `min_length`).
   - Tocar: `models.py` (`ParameterSpec`, nuevo campo) y `schema_validator.py`
     (`_matches_spec`, nueva comprobación). Si afecta a generación: `value_candidates`.

3. **Cambiar el texto del prompt de selección de función**.
   - Tocar: `engine.py` (`_function_prompt`). Solo el string.

4. **Cambiar la estrategia de selección de token** (menor logit, desempate, EOS).
   - Tocar: `constrained_decoder.py` (`choose`, la línea del `max(...)`).
   - Vigila: el bucle debe seguir terminando (cada paso añade un token que continúa
     una opción válida).

5. **Soportar un nuevo extractor/tipo de argumento** (p.ej. fechas).
   - Tocar: `arguments.py` (`value_candidates` y/o `extract_arguments`).
   - Vigila: el tipo debe existir en `ParameterSpec` y validarse en
     `schema_validator`.

6. **Cambiar rutas por defecto o añadir un flag**.
   - Tocar: `cli.py` (`parse_args` y `main`).

7. **Cambiar el fallback** (p.ej. omitir el prompt en vez de "Unable to...").
   - Tocar: `engine.py` (`process`).

8. **Imprimir estadísticas** (cuántos JSON válidos / nº de fallbacks).
   - Tocar: `cli.py` (`main`, tras `engine.process`).

Checklist al recodificar en vivo:
- Mantén `flake8`/`mypy` limpios (hay type hints en todo).
- Si tocas modelos o salida, actualiza el test y corre `make test`.
- No introduzcas variables globales.
- Si tocas el tokenizer, corre el test de paridad.
- Verifica con una ejecución real: `uv run python -m src --trace`.

---

## 16. Cómo medir (comandos) y números reales

```bash
# por defecto (Qwen3-0.6B)
uv run python -m src
# otro modelo
uv run python -m src --model Qwen/Qwen3-1.7B
# ver el proceso token a token
uv run python -m src --trace
# calidad
make lint        # flake8 + pydocstyle + mypy
make lint-strict # flake8 + mypy --strict
make test        # pytest (38 tests)
```

Números reales medidos (set de 11 prompts, Qwen3-0.6B en CPU):
```text
Selección de función: 11/11    Argumentos: 11/11 (100%)
JSON parseable: 100%    Tiempo: ~42 s    Memoria pico: ~5.2 GB
```
Qwen3-1.7B (`--model`): también 11/11 en función y argumentos.

---

## 17. Resumen de "¿dónde está X?" (respuesta rápida)

- "¿Dónde eliges la función?" → `engine.py::_select_function` + `constrained_decoder.py::choose`.
- "¿Dónde extraes argumentos?" → `arguments.py` (genéricos) y, si hay ambigüedad, `engine.py::_choose_argument`.
- "¿Dónde validas el esquema?" → `schema_validator.py::validate_arguments`.
- "¿Dónde tokenizas?" → `tokenizer.py::ByteLevelBPETokenizer.encode`.
- "¿Dónde llamas al modelo?" → `llm.py::QwenAdapter.get_logits` → SDK `get_logits_from_input_ids`.
- "¿Dónde lees/escribes JSON?" → `files.py`.
- "¿Dónde está el manejo de errores?" → excepciones en `models.py`, captura en `cli.py::main` y `engine.py::process`.
- "¿Dónde está el fallback?" → `engine.py::process`.
