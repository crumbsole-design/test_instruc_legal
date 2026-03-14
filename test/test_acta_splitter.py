from pipeline.acta_splitter import split_acta


def test_split_acta_basic_sections():
    text = """ACTA DE LA JUNTA
Fecha: 01/01/2003

ASISTENTES
- Juan Pérez
- María López

CARGOS
Presidente: Juan Pérez

1º.- Aprobación del presupuesto
Se aprueba el presupuesto.

2º.- Ruegos y Preguntas
Se plantearon dudas sobre obras.
"""

    sections = split_acta(text, "acta_demo.md")
    ids = [s["section_id"] for s in sections]

    assert "asistentes" in ids
    assert "cargos" in ids
    assert "punto_01" in ids
    assert "ruegos_y_preguntas" in ids

    # Each section must include the header (text before "ASISTENTES")
    header = "ACTA DE LA JUNTA\nFecha: 01/01/2003"
    for s in sections:
        assert header in s["content"], f"Header missing in section {s['section_id']}"
