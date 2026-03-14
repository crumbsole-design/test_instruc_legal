"""
acta_splitter.py — Module for splitting acta text into logical sections.

This module provides functionality to divide the text of an acta (meeting minutes)
into sections such as attendees, agenda points, etc.
"""

import re
from typing import List, Dict


def split_acta(text: str, filename: str) -> List[Dict[str, str]]:
    """
    Divide el texto de un acta en secciones lógicas.

    Retorna lista de dicts:
        {"section_id": str, "label": str, "content": str}
    donde content = header + cuerpo de la sección.

    Si no se detectan secciones → retorna [] (el caller usará el doc completo).
    """
    # Paso 1 — Extraer header
    match = re.search(r'\basistentes\b', text, re.IGNORECASE)
    if not match:
        return []
    header = text[:match.start()].strip()
    remaining = text[match.start():]

    sections = []

    # Paso 2 — Extraer sección asistentes
    asistentes_end_pattern = re.compile(r'(?im)^\s*\d+[ºo°]\s*[\.\-]|\bcargos\b', re.IGNORECASE)
    match_end = asistentes_end_pattern.search(remaining)
    if match_end:
        asistentes_content = header + '\n' + remaining[:match_end.start()].strip()
        sections.append({
            "section_id": "asistentes",
            "label": "Asistentes",
            "content": asistentes_content
        })
        remaining = remaining[match_end.start():]
    else:
        asistentes_content = header + '\n' + remaining.strip()
        sections.append({
            "section_id": "asistentes",
            "label": "Asistentes",
            "content": asistentes_content
        })
        return sections

    # Paso 3 — Extraer sección cargos (opcional)
    if re.search(r'\bcargos\b', remaining[:200], re.IGNORECASE):
        cargos_end = re.search(r'(?im)^\s*\d+[ºo°]\s*[\.\-]', remaining)
        if cargos_end:
            cargos_content = header + '\n' + remaining[:cargos_end.start()].strip()
            sections.append({
                "section_id": "cargos",
                "label": "Elección de Cargos",
                "content": cargos_content
            })
            remaining = remaining[cargos_end.start():]

    # Paso 4 — Extraer puntos del Orden del Día
    point_pattern = re.compile(r'(?im)^\s*\d+[ºo°]\s*[\.\-]')
    point_matches = list(point_pattern.finditer(remaining))
    if point_matches:
        for i, match in enumerate(point_matches):
            start = match.start()
            if i < len(point_matches) - 1:
                end = point_matches[i + 1].start()
            else:
                end = len(remaining)
            point_text = remaining[start:end].strip()
            # Check for RUEGOS Y PREGUNTAS
            if re.search(r'RUEGOS Y PREGUNTAS', point_text, re.IGNORECASE):
                section_id = "ruegos_y_preguntas"
                label = "Ruegos y Preguntas"
            else:
                section_id = f"punto_{i+1:02d}"
                # Extract label from first line
                lines = point_text.split('\n', 1)
                label = lines[0].strip() if lines[0].strip() else f"Punto {i+1}"
            content = header + '\n' + point_text
            sections.append({
                "section_id": section_id,
                "label": label,
                "content": content
            })

    return sections