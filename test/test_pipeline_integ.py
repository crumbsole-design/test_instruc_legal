"""
test_pipeline_integ.py — Integration test for run_pipeline.

Mocks:
  - pipeline.runner.run_step       → simula el LLM (Ollama)
  - pipeline.runner.index_step_result → simula el indexado en Typesense

Real components under test:
  - pipeline/runner.py  (_enrich_metadata, cache management, save_step_output)
  - pipeline/cache.py   (hash, is_cached, save_cache)
  - pipeline/steps.py   (save_step_output — writes real files to tmp_path)
"""
import json
import os
from unittest.mock import call, patch

import pytest

from pipeline.runner import run_pipeline


# ---------------------------------------------------------------------------
# Fake LLM responses — one per step_id
# ---------------------------------------------------------------------------

FAKE_RESUMEN = """\
## Fecha y Asistencia
Fecha: 01/01/2003. Asistentes: 10 propietarios.

## Estado de Cuentas
Saldo positivo de 500€.

## Acuerdos Principales
Aprobación del presupuesto anual.

FIN_RESUMEN
"""

FAKE_FAQ = """\
## FAQ

**¿Qué acuerdos se tomaron?**
Se aprobó el presupuesto y el nombramiento del presidente.

**¿Cuánto es la cuota mensual?**
La cuota queda fijada en 50€ al mes.
"""

FAKE_ENTIDADES = json.dumps({
    "presidente": "Juan García (Patio A)",
    "vicepresidente": "María López (Patio B)",
    "administrador": "Gestiones SA",
    "vocales": ["Pedro Martínez", "Ana Ruiz"],
    "personas": ["Juan García", "María López", "Pedro Martínez"],
    "ubicaciones": ["Patio A", "Patio B"],
    "resumen_ejecutivo": "Junta constitutiva, se aprobó presupuesto.",
    "acuerdos": ["Aprobación presupuesto", "Nombramiento presidente"],
    "deudas": [{"sujeto": "Vecino 3B", "importe": "120.00€", "concepto": "Cuotas atrasadas"}],
    "proveedores": ["Limpieza Pro SL"],
    "servicios": ["Limpieza portal"],
}, ensure_ascii=False)

_FAKE_RESPONSES = {
    "resumen":   FAKE_RESUMEN,
    "faq":       FAKE_FAQ,
    "entidades": FAKE_ENTIDADES,
}


def _fake_run_step(step, doc_text, file_name, debug=False):
    """side_effect for the run_step mock: returns canned output per step.id."""
    return _FAKE_RESPONSES[step.id]


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

class TestPipelineHappyPath:

    @patch("pipeline.runner.index_step_result", return_value=3)
    @patch("pipeline.runner.run_step", side_effect=_fake_run_step)
    def test_all_steps_called_for_all_documents(
        self,
        mock_run_step,
        mock_index,
        test_config,
        fake_documents,
        mock_storage_context,
        mock_client,
    ):
        """run_pipeline must call run_step once per document per step (2×3=6)."""
        run_pipeline(test_config, fake_documents, mock_storage_context, mock_client)

        n_docs = len(fake_documents)
        n_steps = len(test_config.steps)
        assert mock_run_step.call_count == n_docs * n_steps, (
            f"Expected {n_docs * n_steps} LLM calls, got {mock_run_step.call_count}"
        )

    @patch("pipeline.runner.index_step_result", return_value=3)
    @patch("pipeline.runner.run_step", side_effect=_fake_run_step)
    def test_step_ids_are_correct(
        self,
        mock_run_step,
        mock_index,
        test_config,
        fake_documents,
        mock_storage_context,
        mock_client,
    ):
        """Each call to run_step must use the configured step id."""
        run_pipeline(test_config, fake_documents, mock_storage_context, mock_client)

        called_ids = [c.args[0].id for c in mock_run_step.call_args_list]
        expected_ids = [s.id for s in test_config.steps] * len(fake_documents)
        assert called_ids == expected_ids, (
            f"Step id sequence mismatch.\nExpected: {expected_ids}\nGot:      {called_ids}"
        )

    @patch("pipeline.runner.index_step_result", return_value=3)
    @patch("pipeline.runner.run_step", side_effect=_fake_run_step)
    def test_output_files_created_on_disk(
        self,
        mock_run_step,
        mock_index,
        test_config,
        fake_documents,
        mock_storage_context,
        mock_client,
        tmp_output_dirs,
    ):
        """save_step_output must create one file per document × step."""
        run_pipeline(test_config, fake_documents, mock_storage_context, mock_client)

        step_dir_map = {s.id: s.output_dir for s in test_config.steps}
        for doc in fake_documents:
            base = os.path.splitext(doc.metadata["file_name"])[0]
            for step in test_config.steps:
                expected_file = os.path.join(step_dir_map[step.id], f"{base}_{step.id}.md")
                assert os.path.isfile(expected_file), (
                    f"Output file not found: {expected_file}"
                )

    @patch("pipeline.runner.index_step_result", return_value=3)
    @patch("pipeline.runner.run_step", side_effect=_fake_run_step)
    def test_cache_populated(
        self,
        mock_run_step,
        mock_index,
        test_config,
        fake_documents,
        mock_storage_context,
        mock_client,
    ):
        """Cache JSON must have entries for every document × step after a run."""
        run_pipeline(test_config, fake_documents, mock_storage_context, mock_client)

        assert os.path.isfile(test_config.cache_file), "Cache file was not created"

        with open(test_config.cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)

        for doc in fake_documents:
            doc_name = doc.metadata["file_name"]
            assert doc_name in cache, f"Document '{doc_name}' missing from cache"
            for step in test_config.steps:
                assert step.id in cache[doc_name], (
                    f"Step '{step.id}' missing from cache for '{doc_name}'"
                )

    @patch("pipeline.runner.index_step_result", return_value=3)
    @patch("pipeline.runner.run_step", side_effect=_fake_run_step)
    def test_index_called_for_every_step(
        self,
        mock_run_step,
        mock_index,
        test_config,
        fake_documents,
        mock_storage_context,
        mock_client,
    ):
        """index_step_result must be called once per document × step."""
        run_pipeline(test_config, fake_documents, mock_storage_context, mock_client)

        n_docs = len(fake_documents)
        n_steps = len(test_config.steps)
        assert mock_index.call_count == n_docs * n_steps, (
            f"Expected {n_docs * n_steps} index calls, got {mock_index.call_count}"
        )

    @patch("pipeline.runner.index_step_result", return_value=3)
    @patch("pipeline.runner.run_step", side_effect=_fake_run_step)
    def test_entidades_metadata_enriched(
        self,
        mock_run_step,
        mock_index,
        test_config,
        fake_documents,
        mock_storage_context,
        mock_client,
    ):
        """For 'entidades' steps, the JSON output must enrich base_metadata with
        typed fields (presidente, personas, acuerdos, deudas) before indexing."""
        run_pipeline(test_config, fake_documents, mock_storage_context, mock_client)

        # Collect every call to index_step_result where step_id == "entidades"
        entidades_calls = [
            c for c in mock_index.call_args_list if c.args[1] == "entidades"
        ]
        assert len(entidades_calls) == len(fake_documents), (
            "Expected one 'entidades' index call per document"
        )

        for c in entidades_calls:
            enriched_meta = c.args[3]  # positional: raw_output, step_id, file_name, base_metadata
            assert "presidente" in enriched_meta, "presidente missing from enriched metadata"
            assert "personas" in enriched_meta,   "personas missing from enriched metadata"
            assert "acuerdos" in enriched_meta,   "acuerdos missing from enriched metadata"
            assert "deudas" in enriched_meta,     "deudas missing from enriched metadata"
            # deudas should be JSON-serialised string (as per _enrich_metadata)
            assert isinstance(enriched_meta["deudas"], str), "deudas should be a JSON string"
            deudas = json.loads(enriched_meta["deudas"])
            assert isinstance(deudas, list) and len(deudas) > 0, "deudas list is empty"
