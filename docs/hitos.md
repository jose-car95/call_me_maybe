# Hitos del proyecto call_me_maybe

## Objetivo general

Construir una herramienta de function calling que traduzca prompts en lenguaje natural
en llamadas de funcion estructuradas, generando siempre un JSON valido y compatible
con las definiciones de funciones disponibles.

## Hito 1: Estructura base del proyecto

Crear la estructura minima para poder ejecutar el proyecto con:

```bash
uv run python -m src
```

Tareas:

- Crear `pyproject.toml` en la raiz.
- Crear `src/__main__.py`.
- Crear los primeros modelos de datos con `pydantic`.
- Preparar lectura de archivos JSON de entrada.
- Preparar escritura del archivo de salida.
- Completar el `Makefile` obligatorio.

Objetivo del hito:

Tener un programa ejecutable aunque todavia no use el LLM.

## Hito 2: Validacion de entradas

Validar correctamente los archivos:

- `data/input/function_calling_tests.json`
- `data/input/functions_definition.json`

Tareas:

- Comprobar que los archivos existen.
- Detectar JSON invalido.
- Validar que cada prompt tiene el formato esperado.
- Validar que cada funcion tiene nombre, descripcion, parametros y tipo de retorno.
- Mostrar errores claros sin que el programa falle de forma inesperada.

Objetivo del hito:

Garantizar que el programa entiende bien los datos antes de llamar al modelo.

## Hito 3: Primer contacto con el LLM

Usar el SDK proporcionado para entender el ciclo basico:

```text
prompt -> tokenizacion -> input ids -> modelo -> logits -> siguiente token
```

Tareas:

- Instanciar `Small_LLM_Model`.
- Codificar texto con `encode`.
- Obtener logits con `get_logits_from_input_ids`.
- Decodificar tokens con `decode`.
- Crear pruebas pequenas para observar el comportamiento del modelo.

Objetivo del hito:

Entender de forma practica que el modelo predice tokens, no ejecuta funciones.

## Hito 4: Selector de funcion con LLM

Hacer que el LLM ayude a escoger la funcion correcta entre las disponibles.

Tareas:

- Construir un prompt interno con las funciones disponibles.
- Pedir al modelo que seleccione una funcion.
- Restringir la salida a nombres de funciones validos.
- Validar que la funcion elegida existe en `functions_definition.json`.

Objetivo del hito:

Elegir `fn_name` usando el LLM, sin hardcodear soluciones basadas en los ejemplos.

## Hito 5: Extraccion de argumentos

Generar los argumentos necesarios para la funcion elegida.

Tareas:

- Leer el schema de parametros de la funcion seleccionada.
- Generar un valor para cada argumento requerido.
- Convertir los valores al tipo correcto: `number`, `string`, `boolean`, etc.
- Validar los argumentos generados con `pydantic`.

Objetivo del hito:

Producir un objeto `args` completo y compatible con la funcion seleccionada.

## Hito 6: Decodificacion restringida

Implementar el mecanismo principal del proyecto: guiar la generacion token a token para
que el resultado mantenga una estructura valida.

Tareas:

- Cargar el vocabulario/tokenizer disponible mediante el SDK.
- Mapear tokens a sus representaciones de texto.
- Definir que tokens son validos en cada estado de generacion.
- Forzar estructura JSON valida.
- Forzar que el JSON respete el schema esperado.

Objetivo del hito:

No depender de que el modelo "tenga suerte" generando JSON correcto, sino imponer la
estructura desde el decodificador.

## Hito 7: Output final

Generar el archivo:

```text
data/output/function_calling_results.json
```

Cada entrada debe tener exactamente:

```json
{
  "prompt": "...",
  "fn_name": "...",
  "args": {}
}
```

Tareas:

- Crear el directorio `data/output/` si no existe.
- Escribir un array JSON valido.
- Evitar claves adicionales.
- Asegurar que todos los argumentos requeridos estan presentes.

Objetivo del hito:

Producir una salida valida para todos los prompts de entrada.

## Hito 8: Tests y casos limite

Crear pruebas locales para validar la robustez del proyecto.

Casos a probar:

- Archivos de entrada inexistentes.
- JSON invalido.
- Prompts ambiguos.
- Cadenas vacias.
- Numeros grandes.
- Caracteres especiales.
- Funciones con multiples argumentos.
- Tipos incorrectos.

Objetivo del hito:

Detectar fallos antes de la evaluacion y poder explicar como se comprobo la solucion.

## Hito 9: README de defensa

Completar el `README.md` con la informacion exigida por el subject.

Debe incluir:

- Descripcion del proyecto.
- Instrucciones de instalacion y ejecucion.
- Explicacion del algoritmo de decodificacion restringida.
- Decisiones de diseno.
- Analisis de rendimiento.
- Retos encontrados.
- Estrategia de pruebas.
- Ejemplos de uso.
- Recursos utilizados, incluyendo el uso de IA.

Objetivo del hito:

Tener una documentacion clara para evaluacion entre pares y defensa tecnica.
