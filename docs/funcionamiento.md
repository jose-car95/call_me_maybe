# Funcionamiento completo de call_me_maybe

Este documento explica que ocurre desde que se ejecuta el programa hasta que se escribe el JSON final. Tambien indica en que archivo vive cada responsabilidad para que la defensa sea directa.

## Ejecucion

Comando obligatorio:

```bash
uv run python -m src [--input <input_file>] [--output <output_file>]
```

Opciones soportadas:

- `--input`: archivo de prompts. Por defecto `data/input/function_calling_tests.json`.
- `--functions`: archivo de funciones. Por defecto `data/input/function_definitions.json`.
- `--output`: archivo de salida. Por defecto `data/output/function_calling_results.json`.
- `--model`: modelo del SDK. Por defecto `Qwen/Qwen3-0.6B`.
- `--list-models`: muestra modelos recomendados y criterios para modelos custom.
- `--trace`: muestra decisiones de decodificacion restringida.

Modelos recomendados:

```bash
uv run python -m src --list-models
uv run python -m src --model Qwen/Qwen3-0.6B
uv run python -m src --model Qwen/Qwen3-1.7B
```

`Qwen/Qwen3-0.6B` sigue siendo el modelo obligatorio por defecto. Para probar otro modelo, se puede pasar un identificador de Hugging Face con `--model` si cumple estas condiciones:

- Es compatible con `Small_LLM_Model` del SDK.
- Expone un `tokenizer.json` publico.
- El tokenizer es compatible con el Byte-Level BPE implementado en `src/tokenizer.py`.
- Cabe en memoria y respeta el limite de tiempo de evaluacion.

## Arquitectura Final

```text
src/__main__.py
-> src/cli.py
-> src/files.py
-> src/models.py
-> src/engine.py
   -> src/arguments.py
   -> src/schema_validator.py
   -> src/constrained_decoder.py
   -> src/constants.py
   -> src/ports.py
-> src/llm.py
-> src/tokenizer.py
-> data/output/function_calling_results.json
```

La estructura es plana a proposito. El proyecto tiene un caso de uso principal y no necesita carpetas por capas. Los nombres de archivo separan responsabilidades sin hacer mas dificil seguir el flujo.

## 1. Punto De Entrada

Archivo: `src/__main__.py`

Cuando se ejecuta `python -m src`, Python carga `src/__main__.py`. Este archivo llama a `main()` en `src/cli.py` y transforma su retorno en codigo de salida del proceso.

Responsabilidad:

- Mantener el entrypoint minimo.
- Delegar toda la logica en la CLI.

## 2. Paquete Principal

Archivo: `src/__init__.py`

El archivo existe para declarar `src` como paquete Python y para documentar una decision importante: no expone atajos publicos. Esto evita imports con efectos colaterales y mantiene predecible el arranque del programa. Por ejemplo, importar el paquete no debe cargar el SDK ni preparar el modelo.

## 3. CLI

Archivo: `src/cli.py`

Funciones principales:

- `default_model_name()`: devuelve el modelo obligatorio por defecto.
- `recommended_models()`: devuelve modelos recomendados sin usar variables globales.
- `print_supported_models()`: imprime modelos y requisitos para modelos custom.
- `parse_args()`: parsea argumentos de linea de comandos.
- `main()`: coordina la ejecucion completa.

Flujo de `main()`:

1. Parsear argumentos.
2. Si se usa `--list-models`, imprimir la lista y salir sin cargar archivos ni modelo.
3. Leer prompts.
4. Leer definiciones de funciones.
5. Crear `QwenAdapter` si no se inyecto un modelo de tests.
6. Crear `FunctionCallingEngine`.
7. Procesar prompts.
8. Escribir resultados.
9. Capturar errores esperados con mensajes claros.

## 4. Lectura Y Escritura JSON

Archivo: `src/files.py`

Funciones principales:

- `_load_json()`: abre archivos con context manager y parsea JSON.
- `_validate_array()`: exige que la raiz sea un array y valida cada item con Pydantic.
- `load_prompt_cases()`: carga prompts.
- `load_function_definitions()`: carga funciones y rechaza listas vacias o nombres duplicados.
- `write_results()`: crea el directorio de salida y escribe JSON valido.

Errores manejados:

- Archivo ausente.
- Ruta que no es archivo.
- JSON invalido.
- Schema invalido.
- Fallo de escritura.

## 5. Modelos Pydantic Y Errores

Archivo: `src/models.py`

Modelos principales:

- `PromptCase`: representa un prompt validado. Acepta strings simples como indica el subject.
- `ParameterSpec`: representa el schema de un parametro, incluyendo `enum`, `array`, `object`, `items`, `properties` y `required`.
- `ReturnSpec`: representa el tipo de retorno declarado.
- `FunctionDefinition`: representa una funcion disponible.
- `FunctionCallResult`: representa una salida con `prompt`, `fn_name` y `args`.

Errores principales:

- `InputFileError`
- `InputValidationError`
- `OutputFileError`
- `FunctionDefinitionError`
- `ModelInferenceError`

## 6. Puertos De Aplicacion

Archivo: `src/ports.py`

Define `LanguageModel`, el protocolo que necesita el decodificador:

- `encode(text)`
- `decode(token_ids)`
- `get_logits(input_ids)`

Esta separacion evita que el motor dependa directamente del SDK. En tests se pueden usar modelos falsos rapidos y deterministas.

## 7. Adaptador Del LLM

Archivo: `src/llm.py`

Clase: `QwenAdapter`

Responsabilidad:

- Construir `Small_LLM_Model` del SDK con `Qwen/Qwen3-0.6B` por defecto.
- Permitir cambiar el modelo con `--model`.
- Crear el tokenizador propio con el archivo publico del SDK.
- Exponer `encode()`, `decode()` y `get_logits()` con una interfaz simple.
- Convertir errores del SDK en `ModelInferenceError`.

## 8. Tokenizador Propio

Archivo: `src/tokenizer.py`

Clase: `ByteLevelBPETokenizer`

El adaptador carga el `tokenizer.json` publico mediante el SDK y construye un tokenizador propio.

Proceso de `encode()`:

1. Normalizar texto con NFC.
2. Pretokenizar texto Unicode.
3. Convertir bytes a un alfabeto Unicode reversible.
4. Aplicar merges BPE por ranking.
5. Convertir tokens a ids de vocabulario.

Proceso de `decode()`:

1. Convertir ids a tokens.
2. Reconstruir bytes.
3. Decodificar UTF-8.

Esto permite explicar como se conectan tokenizacion, ids y decodificacion restringida.

## 9. Decodificador Restringido

Archivo: `src/constrained_decoder.py`

Clase: `ConstrainedDecoder`

Responsabilidad:

- Recibir un prompt y un conjunto finito de elecciones permitidas.
- Tokenizar cada eleccion.
- Consultar logits del modelo en cada paso.
- Permitir solo tokens que continuan alguna eleccion valida.
- Elegir entre esos tokens permitidos el de mayor logit.
- Repetir hasta completar exactamente una eleccion.

Metodo principal:

```python
choose(prompt, choices, label)
```

Ejemplo conceptual:

```text
choices = {
  "fn_add_numbers": FunctionDefinition(...),
  "fn_reverse_string": FunctionDefinition(...)
}
```

El modelo decide semanticamente usando logits, pero no puede inventar una funcion porque solo se aceptan continuaciones de las opciones validas.

## 10. Extraccion De Argumentos Y Candidatos

Archivo: `src/arguments.py`

Funciones publicas:

- `extract_arguments()`: extrae argumentos claros sin llamar al modelo.
- `value_candidates()`: genera candidatos compatibles con el tipo esperado.
- `ambiguous_names()`: detecta parametros que necesitan decision del modelo.

Funciones internas relevantes:

- `_quoted()`: strings entre comillas.
- `_numbers()`: numeros enteros, decimales, signos y notacion cientifica.
- `_contextual_string()`: rutas, templates, bases de datos y encoding.
- `_regex_arguments()`: sustituciones con regex.
- `_json_candidates()`: arrays u objetos JSON embebidos en el prompt.

Ejemplo:

```text
Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'
```

Produce:

```json
{
  "source_string": "The cat sat on the mat with another cat",
  "regex": "\\bcat\\b",
  "replacement": "dog"
}
```

## 11. Validacion De Schema

Archivo: `src/schema_validator.py`

Funcion publica:

- `validate_arguments()`

Comprueba:

- Que todas las claves requeridas esten presentes.
- Que no haya claves extra.
- Que los tipos coincidan.
- Que `enum` se respete.
- Que arrays y objetos anidados cumplan su schema.

Si algo no encaja, lanza `ModelInferenceError` para que el motor pueda recuperar o devolver fallback.

## 12. Motor Orquestador

Archivo: `src/engine.py`

Clase: `FunctionCallingEngine`

Responsabilidad:

1. Recibir modelo y funciones disponibles.
2. Crear `ConstrainedDecoder`.
3. Seleccionar la funcion con `_select_function()`.
4. Extraer argumentos iniciales con `extract_arguments()`.
5. Detectar ambiguos con `ambiguous_names()`.
6. Resolver ambiguos con `_choose_argument()` y `ConstrainedDecoder`.
7. Validar con `validate_arguments()`.
8. Crear `FunctionCallResult`.
9. Si un prompt falla por inferencia, devolver fallback controlado.

Fallback controlado:

```json
{
  "prompt": "...",
  "fn_name": "Unable to retrieve from 'function_definitions.json'",
  "args": {}
}
```

El valor del fallback sale de `src/constants.py` mediante una funcion, no mediante una variable global.

## 13. Constantes Compartidas

Archivo: `src/constants.py`

No almacena variables globales. Expone funciones pequenas y sin estado.

Funcion actual:

- `unable_to_retrieve_fn_name()`

Esto evita tener valores mutables o constantes globales directas y hace explicito de donde sale el fallback.

## 14. Output Final

Archivo de salida por defecto:

```text
data/output/function_calling_results.json
```

Formato:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

No se escriben claves extra ni texto libre. `json.dump()` garantiza JSON parseable si los modelos Pydantic ya validaron los datos.

## 15. Como Explicarlo En Defensa

Resumen corto:

```text
El programa no pide al modelo que escriba JSON libre. Usa el modelo para tomar decisiones semanticas, pero cada decision se limita a opciones validas. La estructura final la construye Python y se valida contra Pydantic antes de escribirla.
```

Sobre decodificacion restringida:

```text
En cada paso miro los logits del modelo, pero solo permito tokens que continuan una opcion valida. Para funciones, las opciones son los nombres disponibles. Para argumentos ambiguos, las opciones son candidatos JSON compatibles con el schema.
```

Sobre arquitectura:

```text
engine.py solo orquesta. constrained_decoder.py contiene la seleccion token a token. arguments.py extrae y genera candidatos. schema_validator.py valida el resultado. llm.py adapta el SDK y tokenizer.py implementa el Byte-Level BPE propio.
```
