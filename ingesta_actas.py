"""
ingesta_actas.py — Thin orchestrator for the modular step pipeline.

Usage:
    python ingesta_actas.py                          # Run all steps
    python ingesta_actas.py --synthesis              # Also run the synthesis step
    python ingesta_actas.py --config my_config.yaml  # Custom config file
    python ingesta_actas.py --actas-dir /path/actas  # Override actas directory
"""

import argparse
import json
import logging
import os
import re
import time
import urllib.request

import typesense
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.schema import Document as LIDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.typesense import TypesenseVectorStore

from pipeline.config import load_config, setup_logging
from pipeline.runner import run_pipeline, run_synthesis_step

logger = logging.getLogger("pipeline")


def _merge_pages(documents: list, debug: bool = False, output_dir: str = os.path.join("log", "mergedInitialDocuments")) -> list:
    """Merge per-page Documents from the same source file into one Document per file.

    SimpleDirectoryReader splits PDFs page-by-page; the pipeline expects one
    Document per acta so the LLM sees the full text in one call.

    If debug is enabled, the merged output is also written to a Markdown file
    under `log/mergedInitialDocuments/` for inspection.
    """
    from collections import OrderedDict
    groups: dict = OrderedDict()
    for doc in documents:
        key = doc.metadata.get("file_name") or doc.metadata.get("file_path", "unknown")
        if key not in groups:
            groups[key] = {"meta": dict(doc.metadata), "pages": []}
        groups[key]["pages"].append(doc.text)

    if debug:
        os.makedirs(output_dir, exist_ok=True)

    merged = []
    for key, data in groups.items():
        combined = "\n\n".join(
            f"--- Página {i + 1} ---\n{page}"
            for i, page in enumerate(data["pages"])
        )

        if debug:
            # Create a file name based on source document key
            filename = os.path.basename(str(key))
            safe_name = re.sub(r"[^\w\-\. ]", "_", filename)
            if not safe_name:
                safe_name = "documento"
            out_path = os.path.join(output_dir, f"{safe_name}_merged.md")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(combined)
                logger.debug("Merged document written to %s", out_path)
            except Exception:
                logger.warning("No se pudo escribir el documento mergeado en %s", out_path, exc_info=True)

        merged.append(LIDocument(text=combined, metadata=data["meta"]))
    return merged


def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingesta de actas por steps")
    parser.add_argument(
        "--config", default="pipeline_config.yaml",
        help="Ruta al YAML de configuración (default: pipeline_config.yaml)"
    )
    parser.add_argument(
        "--actas-dir", default=None,
        help="Override del directorio de actas definido en el YAML"
    )
    parser.add_argument(
        "--synthesis", action="store_true",
        help="Ejecutar el synthesis step final (query Typesense → LLM → indexar)"
    )
    args = parser.parse_args()

    # --- Config ---
    config = load_config(args.config)
    if args.actas_dir:
        config.actas_dir = args.actas_dir

    # --- Logging (se configura justo después de cargar el config) ---
    setup_logging(config.debug, config.log_file)

    # --- Embeddings (384 dims — must match schema.json vector_legal field) ---
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # --- Typesense ---
    with open("schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)

    typesense_host = os.environ.get("TYPESENSE_HOST", "typesense")
    typesense_api_key = os.environ.get("TYPESENSE_API_KEY", "clave_secreta_actas")

    client = typesense.Client({
        "nodes": [{"host": typesense_host, "port": "8108", "protocol": "http"}],
        "api_key": typesense_api_key,
        "connection_timeout_seconds": 10,
    })

    # --- Esperar a que Typesense esté listo (belt-and-suspenders con healthcheck) ---
    _ts_health = f"http://{typesense_host}:8108/health"
    for _attempt in range(1, 11):
        try:
            urllib.request.urlopen(_ts_health, timeout=3)
            logger.debug("Typesense listo.")
            break
        except Exception:
            if _attempt == 10:
                raise RuntimeError(f"Typesense no respondió en {_ts_health} tras 10 intentos.")
            logger.info("Esperando Typesense (%d/10)...", _attempt)
            time.sleep(3)

    # --- Invalidar caché si Typesense está vacío (p.ej. tras reinicio sin snapshot) ---
    try:
        _col_stats = client.collections[schema["name"]].retrieve()
        if _col_stats.get("num_documents", 0) == 0 and os.path.exists(config.cache_file):
            os.remove(config.cache_file)
            logger.info("Typesense vacío — caché eliminada para forzar reprocesado completo.")
    except Exception:
        if os.path.exists(config.cache_file):
            os.remove(config.cache_file)
            logger.info("Colección Typesense inexistente — caché eliminada para forzar reprocesado completo.")

    vector_store = TypesenseVectorStore(client=client, collection_name=schema["name"])
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # --- Load documents (PDF, DOCX, TXT, images/OCR via LlamaIndex readers) ---
    reader = SimpleDirectoryReader(config.actas_dir, recursive=True)
    documents = _merge_pages(reader.load_data(), debug=config.debug)
    logger.info("📚 %d documento(s) cargado(s) desde '%s'", len(documents), config.actas_dir)
    logger.info("📋 %d step(s) configurado(s): %s",
                len(config.steps), ', '.join(s.id for s in config.steps))

    # --- Run pipeline ---
    run_pipeline(config, documents, storage_context, client)

    # --- Optional synthesis step ---
    if args.synthesis:
        run_synthesis_step(config, storage_context, client)

    logger.info("🏁 Pipeline completado.")


if __name__ == "__main__":
    main()
