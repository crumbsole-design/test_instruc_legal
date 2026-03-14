from dataclasses import dataclass, field
from typing import Optional, List
import logging
import os
import yaml


@dataclass
class StepConfig:
    id: str
    description: str
    model: str
    temperature: float
    num_ctx: int
    output_dir: str
    prompt: str
    mode: str = "single"          # "single" | "chat"
    msg_intro: str = (
        "A continuación te proporciono el texto de un acta.\n"
        "Léelo detenidamente; a continuación te haré varias preguntas sobre él.\n\n"
        "TEXTO DEL ACTA:"
    )
    inject_whitelist: Optional[str] = None
    max_retries: int = 2
    chunk_tokens: Optional[int] = None
    chunk_strategy: Optional[str] = None
    reduce_prompt: Optional[str] = None


@dataclass
class SynthesisStepConfig:
    id: str
    description: str
    query: str
    num_results: int
    model: str
    temperature: float
    num_ctx: int
    output_dir: str
    prompt: str
    mode: str = "single"
    inject_whitelist: Optional[str] = None
    max_retries: int = 2
    msg_intro: str = ""


@dataclass
class PipelineConfig:
    actas_dir: str
    cache_file: str
    steps: List[StepConfig]
    synthesis_step: Optional[SynthesisStepConfig]
    debug: bool = False
    log_file: str = "./log/pipeline.log"


def setup_logging(debug: bool, log_file: str) -> None:
    """Configure el logger 'pipeline' según el flag debug del YAML.

    - debug=False → nivel INFO  (consola + fichero).
    - debug=True  → nivel DEBUG (consola + fichero con toda la traza).
    El fichero se crea en log_file; el directorio se crea si no existe.
    Ver: https://docs.python.org/es/3/howto/logging.html
    """
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger("pipeline")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    # --- consola ---
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(ch)

    # --- fichero ---
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(fh)

    root.debug("Logging inicializado — nivel=%s  fichero=%s", logging.getLevelName(level), log_file or "(ninguno)")


def _make_step(data: dict) -> StepConfig:
    defaults = {
        "mode": "single",
        "inject_whitelist": None,
        "max_retries": 2,
        "msg_intro": (
            "A continuación te proporciono el texto de un acta.\n"
            "Léelo detenidamente; a continuación te haré varias preguntas sobre él.\n\n"
            "TEXTO DEL ACTA:"
        ),
    }
    merged = {**defaults, **data}
    known = {f for f in StepConfig.__dataclass_fields__}
    filtered = {k: v for k, v in merged.items() if k in known}
    return StepConfig(**filtered)


def _make_synthesis_step(data: dict) -> SynthesisStepConfig:
    defaults = {
        "mode": "single",
        "inject_whitelist": None,
        "max_retries": 2,
        "msg_intro": "",
    }
    merged = {**defaults, **data}
    known = {f for f in SynthesisStepConfig.__dataclass_fields__}
    filtered = {k: v for k, v in merged.items() if k in known}
    return SynthesisStepConfig(**filtered)


def load_config(path: str) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    s = data.get("settings", {})
    steps = [_make_step(item) for item in data.get("steps", [])]

    syn_data = data.get("synthesis_step")
    synthesis_step = _make_synthesis_step(syn_data) if syn_data else None

    return PipelineConfig(
        actas_dir=s.get("actas_dir", "/app/actas"),
        cache_file=s.get("cache_file", ".pipeline_cache.json"),
        steps=steps,
        synthesis_step=synthesis_step,
        debug=bool(s.get("debug", False)),
        log_file=s.get("log_file", "./log/pipeline.log"),
    )
