import json
import logging
import os
import time

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

logger = logging.getLogger(__name__)


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


def _build_messages(step, doc_text: str, extra_context: str = "") -> list:
    """Build the message list for the LLM call based on step mode."""
    prompt = step.prompt
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
    """Execute one step: call the LLM and return the raw string output."""
    llm = _build_llm(step)

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
            messages = _build_messages(step, doc_text[:ventana], extra_context)
            if debug:
                print(f"\n[{step.id}] ▶ Streaming ({file_name}) ...", flush=True)
                chunks = []
                for chunk in llm.stream_chat(messages):
                    delta = chunk.delta or ""
                    print(delta, end="", flush=True)
                    chunks.append(delta)
                print(flush=True)  # trailing newline
                return "".join(chunks).strip()
            else:
                response = llm.chat(messages)
                return response.message.content.strip()
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


def save_step_output(output: str, output_dir: str, file_name: str, step_id: str) -> str:
    """Write raw LLM output to disk for debugging. Returns the path written."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(file_name)[0]
    path = os.path.join(output_dir, f"{base}_{step_id}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(output)
    return path
