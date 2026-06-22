*Este proyecto ha sido creado como parte del currículo de 42 por jose-car.*

# call_me_maybe

## Descripción

`call_me_maybe` traduce peticiones en lenguaje natural a llamadas de función
estructuradas. Selecciona una función disponible y produce sus argumentos con los
nombres y tipos definidos en el esquema de entrada.

El proyecto utiliza Qwen3-0.6B mediante el SDK proporcionado. La salida no depende
de que el modelo escriba libremente un JSON correcto: los nombres de función se
generan sobre un trie de tokens permitidos, los argumentos ambiguos se eligen entre
candidatos compatibles y la estructura final se serializa y valida de forma
determinista.

## Instrucciones

Requisitos: Python 3.10 o posterior, `uv` y aproximadamente 6 GB de memoria para
ejecutar Qwen en CPU.

```bash
make install
make run
```

Por defecto se leen `data/input/function_calling_tests.json` y
`data/input/function_definitions.json`, y se escribe
`data/output/function_calling_results.json`.

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions data/input/function_definitions.json \
  --output /tmp/function_calling_results.json
```

El modelo puede cambiarse manteniendo Qwen como valor por defecto:

```bash
uv run python -m src --model Qwen/Qwen3-0.6B
```

Modelos recomendados incorporados:

```bash
uv run python -m src --list-models
uv run python -m src --model Qwen/Qwen3-1.7B
```

Para agregar o probar modelos nuevos, usa un identificador de Hugging Face
compatible con el SDK, que exponga `tokenizer.json`, sea compatible con el
tokenizador Byte-Level BPE del proyecto y entre en el presupuesto de memoria y
tiempo de la evaluacion. Qwen/Qwen3-0.6B sigue siendo el modelo obligatorio por
defecto.

Para observar las decisiones restringidas y la recuperación controlada:

```bash
uv run python -m src --trace
```

Comprobaciones de desarrollo:

```bash
make test
make lint
make lint-strict
```

## Algoritmo

### Selección De Función

1. Se construye un prompt con las funciones y sus descripciones.
2. Cada nombre válido se tokeniza y se introduce en un trie lógico.
3. Qwen produce logits para el siguiente token.
4. Solo se comparan tokens que continúan algún nombre válido.
5. El token permitido con mayor logit se añade al prefijo.
6. El proceso termina al completar un nombre válido.

El modelo participa en la decisión semántica, pero no puede inventar una función.

### Extracción De Argumentos

Primero se extraen candidatos tipados a partir del prompt y del esquema. Los casos
inequívocos se resuelven directamente para evitar inferencias innecesarias.

Cuando existen varios valores plausibles, un decodificador restringido crea un trie
con sus representaciones JSON y consulta logits token a token. El modelo solo puede
terminar en uno de esos valores. Finalmente se validan claves, tipos, enums y
estructuras anidadas.

El esquema soporta `string`, `number`, `integer`, `boolean`, `array`, `object`,
`enum`, `items`, `properties` y `required`.

### Tokenizador

El flujo principal no utiliza `encode` ni `decode` del SDK. La implementación propia
de Byte-Level BPE carga `tokenizer.json` mediante un método público, aplica
normalización NFC, pretokenización Unicode, mapeo reversible de bytes y merges BPE.

## Arquitectura

```text
CLI -> archivos -> motor -> LLM/tokenizador -> resultado
```

- `models.py`: modelos Pydantic, esquemas y errores.
- `files.py`: lectura y escritura JSON.
- `ports.py`: contratos compartidos, como el modelo de lenguaje.
- `constants.py`: valores compartidos expuestos mediante funciones.
- `arguments.py`: extracción de argumentos, candidatos y ambigüedad.
- `schema_validator.py`: validación recursiva contra el esquema.
- `constrained_decoder.py`: selección restringida token a token.
- `engine.py`: orquestación del flujo de function calling.
- `llm.py`: adaptación del SDK de Qwen.
- `tokenizer.py`: implementación Byte-Level BPE.
- `cli.py`: línea de comandos.
- `docs/funcionamiento.md`: recorrido completo para defensa.

La estructura es deliberadamente plana porque el proyecto tiene un único caso de
uso. Cada archivo separa una responsabilidad concreta sin introducir carpetas
innecesarias ni imports profundos.

## Decisiones De Diseño

- Se usa selección greedy restringida porque puntuar cada función completa
  multiplicaba el tiempo de inferencia en CPU.
- La extracción es híbrida: reglas genéricas para casos inequívocos y LLM restringido
  para ambigüedad. Generar todos los argumentos con logits superó cinco minutos.
- La estructura JSON se construye desde el esquema, no desde texto libre.
- `LanguageModel` desacopla el motor del SDK y permite pruebas rápidas.
- Pydantic rechaza claves adicionales y valida recursivamente las entradas.

## Rendimiento Y Fiabilidad

Medición local sobre 11 prompts, Qwen3-0.6B en CPU:

```text
Tiempo: 41.87 s
Memoria máxima aproximada: 5.2 GB
JSON parseable: 100%
Resultado público: 11/11
Resultado privado: 11/11
```

Los nombres de función se tokenizan una vez y el decodificador mantiene una caché.
Los errores de archivos, validación, inicialización e inferencia se convierten en
mensajes controlados.

Si un prompt no puede resolverse sin romper el esquema, se devuelve un fallback
controlado con `Unable to retrieve from 'function_definitions.json'` y argumentos
vacíos, evitando que un caso aislado tumbe todo el lote.

## Retos Encontrados

- Qwen3-0.6B no genera estructuras fiables sin restricciones.
- La puntuación exhaustiva mejoraba decisiones, pero incumplía el presupuesto de CPU.
- Las regex no generalizan para strings arbitrarios; los casos ambiguos se delegan al
  modelo dentro de candidatos válidos.
- El tokenizador combina Unicode, bytes y BPE. La implementación propia se comparó
  con el SDK usando texto, rutas, JSON, símbolos y varios idiomas.

## Estrategia De Pruebas

La suite cubre modelos Pydantic, archivos inválidos, selección restringida, prefijos
compartidos, tipos JSON, enums, esquemas recursivos, números complejos, strings,
emails, rutas, plantillas, regex, errores del SDK, CLI, escritura final y Byte-Level
BPE.

## Recursos

- Subject oficial de `call_me_maybe`.
- Documentación de Python para `json`, `typing` y `unicodedata`.
- Documentación de Pydantic.
- Artículo de OpenAI sobre Byte Pair Encoding.
- Documentación pública de Qwen y Byte-Level BPE.

Se utilizó IA para revisar arquitectura, explorar alternativas, proponer casos límite
y asistir en pruebas y documentación. Todo lo incorporado se revisó con pruebas,
análisis estático y ejecuciones reales con el modelo.
