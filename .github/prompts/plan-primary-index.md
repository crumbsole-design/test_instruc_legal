# Plan: Indexación Primaria del Documento Original

## TL;DR
Añadir en `runner.py`, **antes** del `for step in config.steps:`, un bloque explícito que indexe el texto fuente completo (`doc.text`) en Typesense con `step_id="documento_original"` usando la función `index_step_result()` ya existente (que internamente usa `SentenceSplitter`). Se integra con el sistema de caché existente para evitar re-indexaciones.

---

## Steps

### Fase única — Modificación de runner.py

1. Localizar en `runner.py` el punto exacto del bucle externo donde se construye `base_metadata` y antes del `for step in config.steps:`.

2. Insertar el bloque de indexación primaria usando las funciones ya existentes:
   - `get_step_hash(doc.text, "documento_original")` → hash de caché
   - `is_cached(cache, file_name, "documento_original", primary_hash)` → skip si ya indexado
   - `index_step_result(doc.text, "documento_original", file_name, base_metadata, storage_context)` → fragmenta con SentenceSplitter(1024/100) e indexa en Typesense
   - `save_cache(cache, file_name, "documento_original", primary_hash)` → marcar como completado
   - `logger.info(...)` → log del resultado

3. No se necesitan cambios en:
   - `typesense_ops.py` — `index_step_result()` ya hace SentenceSplitter + VectorStoreIndex
   - `config.py` — es un paso hardcoded, no configurable
   - `pipeline_config.yaml` — ídem
   - `schema.json` — `step_id` ya es un facet string que acepta cualquier valor

---

## Relevant files
- `pipeline/runner.py` — único archivo a modificar; insertar bloque antes del `for step in config.steps:`
- `pipeline/typesense_ops.py` — reutilizar `index_step_result()` sin cambios
- `pipeline/cache.py` — reutilizar `get_step_hash()`, `is_cached()`, `save_cache()` sin cambios

---

## Verification
1. Ejecutar el pipeline con un acta de prueba y verificar en Typesense que aparecen documentos con `step_id="documento_original"`.
2. Ejecutar de nuevo y verificar que el log muestra "ya indexado (cache)" para ese documento, confirmando que el sistema de caché funciona.
3. Verificar que los fragmentos indexados contienen el texto original completo (no solo los primeros 1000 chars del `contenido_original`).
4. Comprobar que los pasos posteriores (`resumen`, `faq`, `entidades`) siguen ejecutándose normalmente.

---

## Decisions
- Se reutiliza `index_step_result()` directamente — no hace falta código nuevo de fragmentación.
- El `base_metadata` en ese punto tiene `fecha` y `contenido_original` (primeros 1000 chars) — es suficiente para los metadatos de búsqueda del fragmento original.
- Se integra con caché para consistencia con el resto del pipeline.
- `step_id="documento_original"` es un valor hardcoded, no un step configurable en YAML.
