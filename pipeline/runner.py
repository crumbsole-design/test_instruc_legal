import json
import logging
import os
import re
from datetime import datetime

from pipeline.cache import get_step_hash, is_cached, save_cache
from pipeline.config import PipelineConfig
from pipeline.steps import run_step, save_step_output
from pipeline.typesense_ops import index_step_result, query_typesense

logger = logging.getLogger(__name__)


def _extract_fecha(nombre_archivo: str) -> int:
    """Extract 4-digit year from filename prefix (e.g. '030728' → 2003)."""
    match = re.search(r"(\d{2})\d{4}", nombre_archivo)
    if match:
        return 2000 + int(match.group(1))
    return 2000


def _enrich_metadata(step_id: str, raw_output: str, base_metadata: dict) -> dict:
    """
    For the 'entidades' step: parse the JSON output and add typed fields to metadata
    so Typesense facets (personas, acuerdos, president, …) are populated.
    For 'resumen' and 'faq': add the text to the matching schema field.
    """
    enriched = dict(base_metadata)

    if step_id == "resumen":
        enriched["resumen_ejecutivo"] = raw_output[:2000]

    elif step_id == "faq":
        enriched["faq"] = raw_output[:2000]

    elif step_id == "entidades":
        # Extract JSON block (model may add extra text around it)
        clean = raw_output.strip()
        if not clean.startswith("{"):
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                clean = m.group()
        # Remove trailing commas before closing braces/brackets (common model artifact)
        clean = re.sub(r",\s*([\]}])", r"\1", clean)
        try:
            data = json.loads(clean)
            for field in (
                "presidente", "vicepresidente", "administrador",
                "vocales", "personas", "ubicaciones",
                "acuerdos", "proveedores", "servicios",
            ):
                if field in data:
                    enriched[field] = data[field]
            if "resumen_ejecutivo" in data:
                enriched["resumen_ejecutivo"] = data["resumen_ejecutivo"]
            if "deudas" in data:
                enriched["deudas"] = json.dumps(data["deudas"], ensure_ascii=False)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("[entidades] No se pudo parsear JSON para enriquecer metadatos: %s", e)

    return enriched


def run_pipeline(config: PipelineConfig, documents, storage_context, client):
    """
    Main loop: for every document × every step — check cache, run, save to disk,
    index in Typesense, update cache.
    """
    cache_path = config.cache_file

    for doc in documents:
        nombre = doc.metadata.get("file_name", "archivo_desconocido")
        logger.info("=" * 60)
        logger.info("📄 Procesando: %s", nombre)

        base_metadata = {
            "fecha": _extract_fecha(nombre),
            "contenido_original": doc.text[:1000],
        }

        for step in config.steps:
            step_hash = get_step_hash(step)

            if is_cached(nombre, step.id, step_hash, cache_path):
                logger.debug("[%s] En caché — saltando.", step.id)
                continue

            logger.info("[%s] Ejecutando con modelo %s ...", step.id, step.model)
            t0 = datetime.now()

            try:
                output = run_step(step, doc.text, nombre, debug=config.debug)
                out_path = save_step_output(output, step.output_dir, nombre, step.id)
                elapsed_llm = datetime.now() - t0
                logger.info("[%s] LLM OK → %s | %s", step.id, out_path, elapsed_llm)
            except Exception as e:
                logger.error("[%s] LLM falló: %s", step.id, e, exc_info=True)
                continue

            try:
                enriched = _enrich_metadata(step.id, output, base_metadata)
                n_nodes = index_step_result(
                    output, step.id, nombre, enriched, storage_context
                )
                save_cache(nombre, step.id, step_hash, cache_path)
                elapsed = datetime.now() - t0
                logger.info("[%s] → %s | %d fragmentos | %s", step.id, out_path, n_nodes, elapsed)
            except Exception as e:
                logger.error("[%s] Indexado falló: %s", step.id, e, exc_info=True)


def run_synthesis_step(config: PipelineConfig, storage_context, client):
    """
    Special final step (--synthesis flag): query Typesense for context,
    feed it to the LLM, save output, and index it with step_id='sintesis'.
    """
    syn = config.synthesis_step
    if not syn:
        logger.error("No hay 'synthesis_step' definido en pipeline_config.yaml")
        return

    collection = client.collections.keys()[0] if hasattr(client.collections, "keys") else "actas_comunidad"

    logger.info("=" * 60)
    logger.info("🔬 Synthesis step — query: '%s'", syn.query)

    context_text = query_typesense(client, "actas_comunidad", syn.query, syn.num_results)
    if not context_text.strip():
        logger.error("La query no devolvió resultados de Typesense. Abortando synthesis.")
        return

    logger.info("%d fragmentos recuperados de Typesense", syn.num_results)

    try:
        output = run_step(syn, context_text, "sintesis_global", debug=config.debug)
        out_path = save_step_output(output, syn.output_dir, "sintesis_global", syn.id)

        base_meta = {"fecha": 0, "contenido_original": context_text[:500]}
        n_nodes = index_step_result(
            output, syn.id, "sintesis_global", base_meta, storage_context
        )
        logger.info("Synthesis → %s | %d fragmentos indexados", out_path, n_nodes)

    except Exception as e:
        logger.error("Synthesis step falló: %s", e, exc_info=True)
