import os
import sys
from pathlib import Path

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage

MODEL = "alia-extractorQ4_gpu"
PROMPT_FILE = Path(__file__).parent / "testPromt1.txt"


def main():
    if not PROMPT_FILE.exists():
        print(f"ERROR: No se encontró el archivo {PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)

    prompt = PROMPT_FILE.read_text(encoding="utf-8")
    print(f"=== Modelo: {MODEL} ===")
    print(f"=== Prompt ({len(prompt)} chars) cargado desde {PROMPT_FILE.name} ===\n")

    llm = Ollama(
        model=MODEL,
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=600.0,
        temperature=0.1,
        context_window=12000,
        additional_kwargs={
            "num_predict": 2048,
            "stop": ["[INST]", "[FIN INST]", "Fin_Resumen", "FIN_RESUMEN"],
        },
    )

    # modo "single": un único mensaje de usuario con prompt + texto del acta
    messages = [
        ChatMessage(role="user", content=prompt),
    ]

    print("=== Respuesta (streaming) ===\n")
    for chunk in llm.stream_chat(messages):
        delta = chunk.delta or ""
        print(delta, end="", flush=True)
    print("\n\n=== Fin ===")


if __name__ == "__main__":
    main()
