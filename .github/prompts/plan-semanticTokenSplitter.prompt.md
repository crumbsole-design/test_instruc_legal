# Plan: Tokenizer y Splitter Real para el Pipeline

**TL;DR:** Reemplazar el truncado por caracteres `doc_text[:ventana]` en `run_step()` por un splitter semántico basado en `SentenceSplitter`. Cada step procesará sus fragmentos según la estrategia configurada (map-reduce o merge-JSON), y el retry-by-halving se conserva como red de seguridad final.

---

## Fase 1 — Extensión del Config

1. Añadir 3 campos opcionales a `StepConfig` en `pipeline/config.py`:
   - `chunk_tokens: int | None` — tamaño de chunk en tokens (si es `None`, se usa `num_ctx` como fallback al comportamiento actual)
   - `chunk_strategy: str` — `"map_reduce"` (default) o `"merge_json"`
   - `reduce_prompt: str | None` — prompt del paso de reducción en map-reduce

2. Actualizar `pipeline_config.yaml` con estos valores por step:

   | Step | `chunk_tokens` | `chunk_strategy` | `reduce_prompt` |
   |---|---|---|---|
   | `resumen` | 2048 | `map_reduce` | *"Unifica los siguientes resúmenes parciales en uno coherente:"* |
   | `faq` | 2048 | `map_reduce` | *"Fusiona estas listas de preguntas, elimina duplicados:"* |
   | `entidades` | 2048 | `merge_json` | *(no aplica)* |

---

## Fase 2 — Utilidad de Splitting *(depende de Fase 1)*

3. Añadir la función `obtener_fragmentos_seguros(texto, max_tokens, overlap=200)` a `pipeline/steps.py`:
   - Reutiliza el mismo `SentenceSplitter` que ya existe en `pipeline/typesense_ops.py`
   - Crea un `Document` temporal, extrae nodos, devuelve `list[str]`

---

## Fase 3 — Refactor `run_step()` *(depende de Fase 2)*

4. Extraer la lógica actual de `run_step()` a una función privada `_run_single_fragment(step, llm, texto, file_name, debug)`. El retry-by-halving permanece **dentro** de esta función, intacto.

5. Añadir `_run_map_reduce(step, llm, fragmentos, file_name, debug)`:
   - **Map:** llamar `_run_single_fragment()` sobre cada fragmento → `partial_outputs: list[str]`
   - **Reduce:** concatenar parciales y llamar al LLM una vez más con `step.reduce_prompt` como prompt del usuario y los parciales como "documento"

6. Añadir `_run_merge_json(step, llm, fragmentos, file_name, debug)`:
   - **Map:** igual que en map-reduce → lista de JSON strings
   - **Reduce programático:** parsear cada JSON, hacer union de arrays campo a campo (`personas`, `acuerdos`, `proveedores`, etc.), devolver el JSON combinado como string — **sin llamada extra al LLM**

7. Actualizar el cuerpo de `run_step()`:
   - Si `step.chunk_tokens` es `None` → comportamiento actual sin cambios
   - Si está definido → `fragmentos = obtener_fragmentos_seguros(doc_text, step.chunk_tokens)`
   - Dispatch: `if len(fragmentos) == 1` → `_run_single_fragment`; else → dispatch por `step.chunk_strategy`

---

## Fase 4 — Tests *(puede ir en paralelo con Fase 3)*

8. Tests en `test/modelTest/`:
   - Test unitario de `obtener_fragmentos_seguros()`: verificar que un texto largo produce N chunks ≤ `max_tokens` cada uno
   - Test de `_run_map_reduce()` con LLM stubbeado (mock de `llm.chat`)
   - Test de `_run_merge_json()`: verificar merge correcto de arrays JSON con campos solapados

---

## Archivos a modificar

- `pipeline/config.py` — añadir `chunk_tokens`, `chunk_strategy`, `reduce_prompt` a `StepConfig`
- `pipeline/steps.py` — splitter utility + refactor `run_step()` + helpers privados
- `pipeline_config.yaml` — añadir los 3 nuevos campos por step

## Sin cambios necesarios

- `pipeline/runner.py` — la interfaz `run_step()` no cambia (sigue devolviendo `str`)
- `pipeline/typesense_ops.py` — no afectado

---

## Verificación

1. Correr `pytest test/modelTest/` — todos los tests nuevos y existentes deben pasar
2. Procesar manualmente un acta real con `--debug` y verificar en los logs que se loguea `"Fragmento N/M"` indicando el split
3. Verificar que el output de `entidades` sigue siendo JSON válido y con arrays correctamente combinados
4. Confirmar que si un acta pequeña cabe en un solo chunk, el comportamiento es idéntico al actual

---

## Decisiones registradas

- `entidades` → estrategia `merge_json` (reduce programático, sin llamada extra al LLM)
- `resumen` y `faq` → `map_reduce` (reduce con LLM usando `reduce_prompt`)
- `chunk_tokens` configurable en YAML; si ausente, el step funciona como hoy
- retry-by-halving se conserva dentro de `_run_single_fragment` como fallback de último recurso
