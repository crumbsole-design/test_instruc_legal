# Plan: Detección de actas y división en secciones

**TL;DR**: Añadir un step `identificacion` marcado con `role: detection` en `pipeline_config.yaml`. Antes del pipeline normal, `ingesta_actas.py` ejecuta ese step para cada documento y, si el modelo confirma que es un acta, un nuevo módulo `pipeline/acta_splitter.py` divide el texto en secciones (cabecera + asistentes + cada punto numerado del orden del día + ruegos y preguntas + opcionalmente cargos). Cada sección se convierte en un `Document` independiente que pasa por los steps habituales (resumen, faq, entidades), generando archivos por sección. Los no-actas pasan por el pipeline completo sin dividir.

---

## Fase 1 — Step de identificación en `pipeline_config.yaml`

1. Añadir al inicio de `steps:` el nuevo step con `role: detection`. Este flag es lo único que lo diferencia de un step normal: le indica al runner que lo omita en su bucle.

```yaml
- id: identificacion
  role: detection
  description: "Detecta si el documento es un acta de comunidad"
  model: "alia-extractorQ4_gpu"
  temperature: 0.0
  num_ctx: 4000
  output_dir: "./identificacion"
  mode: single
  inject_whitelist: null
  max_retries: 1
  prompt: |
    Analiza el siguiente documento. Nombre del archivo: {file_name}

    Evalúa EXACTAMENTE estos 4 criterios:
    1. El nombre del archivo contiene la palabra "acta" (ignora mayúsculas)
    2. El título o encabezado del documento contiene la palabra "acta"
    3. El documento tiene una sección de ASISTENTES (lista de personas presentes)
    4. El documento tiene una sección de ORDEN DEL DÍA (puntos a tratar)

    Devuelve SOLO este JSON, sin explicaciones ni markdown:
    {"es_acta": true, "criterios": {"nombre_contiene_acta": bool, "titulo_contiene_acta": bool, "tiene_asistentes": bool, "tiene_orden_del_dia": bool}, "razon": "motivo breve"}

    TEXTO DEL DOCUMENTO:
```

2. Añadir campo `role: str | None = None` al dataclass `StepConfig` en `pipeline/config.py`.

---

## Fase 2 — Soporte de `{file_name}` en el prompt (`pipeline/steps.py`)

3. En `run_step()`, antes de llamar `_build_messages()`, sustituir el placeholder sin mutar el objeto step:

```python
rendered_prompt = step.prompt.replace("{file_name}", file_name)
# pasar rendered_prompt a _build_messages en lugar de step.prompt
```

Los prompts existentes (resumen, faq, entidades) no tienen el placeholder y no se ven afectados.

---

## Fase 3 — Nuevo módulo `pipeline/acta_splitter.py`

4. Crear función `split_acta(text: str, filename: str) -> list[dict]`.

Cada dict: `{"section_id": str, "label": str, "content": str}` donde `content = header + cuerpo_de_sección`.

**Algoritmo de splitting:**

| Paso | Descripción |
|---|---|
| `header` | Texto antes de la primera ocurrencia de `\basistentes\b` (re.IGNORECASE) |
| `asistentes` | Desde "asistentes" hasta el primer punto numerado (`\d+[ºoº°]`) o "cargos" standalone |
| `cargos` (opcional) | Si aparece como título propio entre asistentes y los puntos numerados |
| Puntos del orden del día | Split por `(?im)^\s*\d+[ºo°]\s*[\.\-]` — cada match es un chunk independiente |
| IDs de sección | `punto_01`, `punto_02`, … Si el título contiene "RUEGOS Y PREGUNTAS" → `ruegos_y_preguntas` |

Si no se detectan secciones → retorna `[]` (fallback: doc completo al pipeline sin split).

**Ejemplo de archivos de salida para `030728 Acta.Junta.Constit.28-07-03`:**
```
resumen/030728 Acta.Junta.Constit.28-07-03__asistentes_resumen.md
resumen/030728 Acta.Junta.Constit.28-07-03__punto_01_resumen.md
resumen/030728 Acta.Junta.Constit.28-07-03__punto_02_resumen.md
resumen/030728 Acta.Junta.Constit.28-07-03__punto_03_resumen.md
resumen/030728 Acta.Junta.Constit.28-07-03__ruegos_y_preguntas_resumen.md
faq/030728 Acta.Junta.Constit.28-07-03__asistentes_faq.md
json/030728 Acta.Junta.Constit.28-07-03__asistentes_entidades.md
... (igual para todas las secciones y todos los steps)
```

---

## Fase 4 — Filtro en `pipeline/runner.py`

5. Cambio mínimo en `run_pipeline()`: filtrar el step de detección antes del bucle principal:

```python
process_steps = [s for s in config.steps if getattr(s, 'role', None) != 'detection']
for step in process_steps:
    ...  # lógica existente sin cambios
```

---

## Fase 5 — Orquestación en `ingesta_actas.py` *(depende de Fases 1–4)*

6. Nueva función `_detect_and_split(doc, detection_step, config) -> list[Document] | None`:
   - Llama `run_step()` con los primeros ~3 000 chars del documento (suficiente para detectar cabecera + asistentes + orden del día)
   - Parsea JSON de respuesta con fallback `re.search(r'\{.*\}', raw, re.DOTALL)`
   - Guarda output crudo en `identificacion/{base}_identificacion.md`
   - `es_acta == False` → retorna `None` (doc completo al pipeline sin split)
   - `es_acta == True` → llama `split_acta()` → si secciones vacías → retorna `None`
   - Para cada sección: crea `Document(text=header+content, metadata={"file_name": f"{base}__{section_id}", "seccion": section_id, ...})`

7. Bucle expandido en `main()`:

```python
detection_step = next((s for s in config.steps if getattr(s, 'role', None) == 'detection'), None)
expanded_docs = []
for doc in merged_documents:
    section_docs = _detect_and_split(doc, detection_step, config) if detection_step else None
    expanded_docs.extend(section_docs if section_docs else [doc])
run_pipeline(config, expanded_docs, storage_context, client)
```

---

## Archivos a modificar/crear

| Archivo | Tipo | Cambio |
|---|---|---|
| `pipeline_config.yaml` | Modificar | Añadir step `identificacion` con `role: detection` al inicio de `steps:` |
| `pipeline/config.py` | Modificar | Campo `role: str \| None = None` en `StepConfig` |
| `pipeline/steps.py` | Modificar | Sustituir placeholder `{file_name}` en `run_step()` sin mutar step |
| `pipeline/runner.py` | Modificar | Filtrar steps con `role == 'detection'` antes del bucle |
| `pipeline/acta_splitter.py` | **Nuevo** | Módulo de splitting por secciones |
| `ingesta_actas.py` | Modificar | Función `_detect_and_split` + bucle expandido en `main()` |

---

## Verificación

1. **Test unitario** de `split_acta()` con `log/mergedInitialDocuments/030728 Acta.Junta.Constit.28-07-03.pdf_merged.md`:
   - Esperar secciones: `asistentes`, `punto_01`, `punto_02`, `punto_03`, `ruegos_y_preguntas`
   - Verificar que todas las secciones incluyen el `header` (texto antes de "asistentes")
2. **Pipeline completo** sobre `mis_actas/`:
   - Verificar carpeta `identificacion/` con un JSON de detección por documento
   - Verificar archivos con sufijos de sección en `resumen/`, `faq/`, `json/`
3. **Documento no-acta**: colocar un PDF sin secciones de acta en `mis_actas/` y confirmar que genera un único conjunto de salidas sin sufijos de sección.

---

## Decisiones acordadas

- **No-acta**: procesar con pipeline completo sin dividir (resumen, faq, json del documento entero)
- **Naming**: sufijo doble guión bajo + section_id: `{base}__{section_id}_{step_id}.md`
- **Orden del día**: un chunk por cada punto numerado (1º, 2º, …), no todo junto
