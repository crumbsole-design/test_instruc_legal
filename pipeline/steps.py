import json
import logging
import os
import time

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def obtener_fragmentos_seguros(texto: str, max_tokens: int, overlap: int = 200) -> list[str]:
    """Fragmenta texto con SentenceSplitter y devuelve lista de strings."""
    doc = Document(text=texto)
    splitter = SentenceSplitter(chunk_size=max_tokens, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents([doc])
    return [n.get_content() for n in nodes]


def _run_single_fragment(step, llm, texto: str, file_name: str, debug: bool = False) -> str:
    rendered_prompt = step.prompt.replace("{file_name}", file_name)

    extra_context = ""
    whitelist_path = getattr(step, "inject_whitelist", None)
    if whitelist_path:
        try:
            with open(whitelist_path, "r", encoding="utf-8") as f:
                wl = json.load(f)
            extra_context = (
                "\nLISTA DE PROVEEDORES CONOCIDOS (usa estos nombres oficiales en tu respuesta):\n"
                + json.dumps(wl, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.warning("No se pudo cargar whitelist '%s': %s", whitelist_path, e)

    max_retries = getattr(step, "max_retries", 2)
    ventana = step.num_ctx  # character budget (approx 1 char ≈ 1 token for safety)

    for intento in range(max_retries + 1):
        try:
            messages = _build_messages(step, texto[:ventana], rendered_prompt, extra_context)
            response = llm.chat(messages)
            output = response.message.content.strip()
            if debug:
                logger.debug("[%s] ▶ Respuesta (%s):\n%s", step.id, file_name, output)
            return output
        except Exception as e:
            if intento < max_retries:
                ventana //= 2
                logger.warning(
                    "[%s][%s] intento %d fallido: %s. Reduciendo ventana a %d chars...",
                    file_name, step.id, intento + 1, e, ventana,
                )
                time.sleep(1)
            else:
                raise RuntimeError(
                    f"[{file_name}][{step.id}] falló tras {max_retries + 1} intentos: {e}"
                ) from e


def _run_map_reduce(step, llm, fragmentos: list[str], file_name: str, debug: bool = False) -> str:
    """Map: procesa cada fragmento. Reduce: unifica salidas parciales con el LLM."""
    partial_outputs = []
    for i, frag in enumerate(fragmentos):
        logger.info("[%s] Fragmento %d/%d (map)", step.id, i + 1, len(fragmentos))
        partial_outputs.append(_run_single_fragment(step, llm, frag, file_name, debug))

    combined = "\n\n---\n\n".join(partial_outputs)
    # Reduce: usar reduce_prompt como prompt con los parciales como documento
    import copy
    reduce_step = copy.copy(step)
    reduce_step.prompt = step.reduce_prompt
    reduce_step.mode = "single"
    logger.info("[%s] Reduce sobre %d parciales", step.id, len(partial_outputs))
    return _run_single_fragment(reduce_step, llm, combined, file_name, debug)


def _run_merge_json(step, llm, fragmentos: list[str], file_name: str, debug: bool = False) -> str:
    """Map: extrae JSON por fragmento. Reduce: merge programático sin LLM."""
    import json, re

    _ARRAY_FIELDS = {"vocales", "personas", "ubicaciones", "acuerdos", "deudas", "proveedores", "servicios"}
    _STRING_FIELDS = {"presidente", "vicepresidente", "administrador", "resumen_ejecutivo"}

    parsed = []
    for i, frag in enumerate(fragmentos):
        logger.info("[%s] Fragmento %d/%d (map-json)", step.id, i + 1, len(fragmentos))
        raw = _run_single_fragment(step, llm, frag, file_name, debug)
        clean = raw.strip()
        if not clean.startswith("{"):
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                clean = m.group()
        clean = re.sub(r",(\s*[\]}])", r"\1", clean)
        try:
            parsed.append(json.loads(clean))
        except json.JSONDecodeError:
            logger.warning("[%s] Fragmento %d no produjo JSON válido", step.id, i + 1)

    if not parsed:
        return "{}"

    merged: dict = {}
    for obj in parsed:
        for field, value in obj.items():
            if field in _ARRAY_FIELDS:
                existing = merged.get(field, [])
                if isinstance(value, list):
                    for item in value:
                        if item not in existing:
                            existing.append(item)
                merged[field] = existing
            elif field in _STRING_FIELDS:
                # primer valor no-vacío y no "No consta" gana
                current = merged.get(field, "")
                if not current or current == "No consta":
                    if value and value != "No consta":
                        merged[field] = value

    return json.dumps(merged, ensure_ascii=False)


def _build_llm(step) -> Ollama:
    return Ollama(
        model=step.model,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
        request_timeout=600.0,
        temperature=step.temperature,
        context_window=step.num_ctx,
        additional_kwargs={
            "num_predict": 2048,
            "stop": ["[INST]", "[FIN INST]", "Fin_Resumen", "FIN_RESUMEN"],
        },
    )


def _build_messages(step, doc_text: str, prompt: str, extra_context: str = "") -> list:
    """Build the message list for the LLM call based on step mode."""
    if extra_context:
        prompt = f"{prompt}\n\n{extra_context}"

    if step.mode == "chat":
        intro = getattr(step, "msg_intro", "") or (
            "A continuación te proporciono el texto de un acta.\n"
            "Léelo detenidamente; a continuación te haré varias preguntas sobre él.\n\n"
            "TEXTO DEL ACTA:"
        )
        return [
            ChatMessage(role="user", content=f"{intro}\n{doc_text}"),
            ChatMessage(role="assistant", content="Entendido. He leído el acta. Dime qué necesitas."),
            ChatMessage(role="user", content=prompt),
        ]
    else:
        # single mode: prompt already ends with "TEXTO DEL ACTA:" marker
        return [
            ChatMessage(role="user", content=f"{prompt}\n{doc_text}"),
        ]


def run_step(step, doc_text: str, file_name: str, debug: bool = False) -> str:
    llm = _build_llm(step)

    if step.chunk_tokens is None:
        # comportamiento actual — sin fragmentación
        return _run_single_fragment(step, llm, doc_text, file_name, debug)

    fragmentos = obtener_fragmentos_seguros(doc_text, step.chunk_tokens)
    logger.info("[%s] %d fragmento(s) para procesamiento", step.id, len(fragmentos))

    if len(fragmentos) == 1:
        return _run_single_fragment(step, llm, fragmentos[0], file_name, debug)

    if step.chunk_strategy == "merge_json":
        return _run_merge_json(step, llm, fragmentos, file_name, debug)

    return _run_map_reduce(step, llm, fragmentos, file_name, debug)


def save_step_output(output: str, output_dir: str, file_name: str, step_id: str) -> str:
    """Write raw LLM output to disk for debugging. Returns the path written."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(file_name)[0]
    path = os.path.join(output_dir, f"{base}_{step_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(output)
    return path
