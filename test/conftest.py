"""
conftest.py — Shared pytest fixtures for pipeline integration tests.
"""
import pytest
from unittest.mock import MagicMock
from llama_index.core.schema import Document as LIDocument

from pipeline.config import PipelineConfig, StepConfig, SynthesisStepConfig


@pytest.fixture
def tmp_output_dirs(tmp_path):
    """Create and return a dict of temporary output directories per step."""
    dirs = {
        "resumen": tmp_path / "resumen",
        "faq":     tmp_path / "faq",
        "entidades": tmp_path / "json",
        "sintesis": tmp_path / "sintesis",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in dirs.items()}


@pytest.fixture
def cache_path(tmp_path):
    """Path to a temporary cache file (does not need to exist beforehand)."""
    return str(tmp_path / "cache.json")


@pytest.fixture
def test_config(tmp_output_dirs, cache_path):
    """PipelineConfig built from dataclasses — no YAML, no real paths."""
    steps = [
        StepConfig(
            id="resumen",
            description="Resumen ejecutivo",
            model="fake-model",
            temperature=0.1,
            num_ctx=8192,
            output_dir=tmp_output_dirs["resumen"],
            prompt="Genera un RESUMEN EJECUTIVO del acta.\n\nTEXTO DEL ACTA:",
            mode="single",
        ),
        StepConfig(
            id="faq",
            description="Preguntas frecuentes",
            model="fake-model",
            temperature=0.1,
            num_ctx=8192,
            output_dir=tmp_output_dirs["faq"],
            prompt="Genera una FAQ sobre el acta.",
            mode="chat",
        ),
        StepConfig(
            id="entidades",
            description="Extracción de entidades",
            model="fake-model",
            temperature=0.0,
            num_ctx=8192,
            output_dir=tmp_output_dirs["entidades"],
            prompt="Extrae entidades en JSON.",
            mode="chat",
            inject_whitelist=None,
        ),
    ]
    synthesis_step = SynthesisStepConfig(
        id="sintesis",
        description="Síntesis global",
        query="balance económico anual acuerdos",
        num_results=3,
        model="fake-model",
        temperature=0.1,
        num_ctx=8192,
        output_dir=tmp_output_dirs["sintesis"],
        prompt="Genera un análisis consolidado.",
        mode="single",
    )
    return PipelineConfig(
        actas_dir="/fake/actas",
        cache_file=cache_path,
        steps=steps,
        synthesis_step=synthesis_step,
        debug=False,
    )


@pytest.fixture
def fake_documents():
    """Two in-memory LlamaIndex Documents — no disk I/O required."""
    return [
        LIDocument(
            text="Texto del acta de constitución de la junta de propietarios del año 2003.",
            metadata={"file_name": "030101 Acta.Junta.Constit.pdf"},
        ),
        LIDocument(
            text="Texto de la segunda acta extraordinaria, aprobación de presupuesto 2004.",
            metadata={"file_name": "040202 Acta.Junta.Extr.pdf"},
        ),
    ]


@pytest.fixture
def mock_storage_context():
    return MagicMock()


@pytest.fixture
def mock_client():
    return MagicMock()
