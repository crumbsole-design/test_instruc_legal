import json
import logging
import os
import re
import warnings
from typing import Dict, Generator, List, Optional, Tuple

import requests
import typesense
from sentence_transformers import SentenceTransformer

# Silence noisy library logs.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

COLLECTION_NAME = "actas_comunidad"
URL_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODELO_LLM = os.environ.get("OLLAMA_CHAT_MODEL", "ActasRAG")
MODELO_ANALISIS = os.environ.get("OLLAMA_ANALYSIS_MODEL", MODELO_LLM)


def crear_cliente_typesense() -> typesense.Client:
    return typesense.Client(
        {
            "nodes": [
                {
                    "host": os.environ.get("TYPESENSE_HOST", "localhost"),
                    "port": os.environ.get("TYPESENSE_PORT", "8108"),
                    "protocol": os.environ.get("TYPESENSE_PROTOCOL", "http"),
                }
            ],
            "api_key": os.environ.get("TYPESENSE_API_KEY", "clave_secreta_actas"),
            "connection_timeout_seconds": 5,
        }
    )


def cargar_modelo_embeddings() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def _normalizar_variantes(variantes: List[str], pregunta_original: str) -> List[str]:
    limpias = []
    vistos = set()
    for item in [pregunta_original] + variantes:
        txt = str(item).strip(" -*\"'\n\t")
        if len(txt) < 3:
            continue
        key = txt.lower()
        if key in vistos:
            continue
        vistos.add(key)
        limpias.append(txt)
    return limpias or [pregunta_original]


def _parsear_filtro_fecha_local(pregunta: str) -> str:
    texto = pregunta.lower()

    entre = re.search(r"entre\s+(20\d{2})\s+y\s+(20\d{2})", texto)
    if entre:
        inicio, fin = int(entre.group(1)), int(entre.group(2))
        if inicio > fin:
            inicio, fin = fin, inicio
        return f"fecha:>={inicio} && fecha:<={fin}"

    antes = re.search(r"antes\s+de\s+(20\d{2})", texto)
    if antes:
        return f"fecha:<{int(antes.group(1))}"

    despues = re.search(r"(despues|después)\s+de\s+(20\d{2})", texto)
    if despues:
        return f"fecha:>{int(despues.group(2))}"

    en = re.search(r"\ben\s+(20\d{2})\b", texto)
    if en:
        return f"fecha:={int(en.group(1))}"

    return ""


def analizar_pregunta_con_ia(pregunta_original: str) -> Tuple[List[str], str]:
    """
    Devuelve variantes de consulta y filtro por fecha en sintaxis Typesense.
    Solo trabaja con fecha para ser compatible con el schema actual.
    """
    prompt = f"""Eres un asistente de búsqueda para una base de actas.
Extrae:
1) question: la pregunta limpia, sin ruido
2) variantes: hasta 4 variantes cortas
3) filtro_fecha: solo sobre campo fecha (int), con sintaxis Typesense

Reglas de filtro_fecha:
- \"en 2012\" -> \"fecha:=2012\"
- \"antes de 2015\" -> \"fecha:<2015\"
- \"después de 2010\" -> \"fecha:>2010\"
- \"entre 2000 y 2008\" -> \"fecha:>=2000 && fecha:<=2008\"
- si no hay fecha -> \"\"

Devuelve SOLO JSON válido:
{{
  "question": "...",
  "variantes": ["..."],
  "filtro_fecha": "..."
}}

Consulta: "{pregunta_original}"
"""  # noqa: E501

    payload = {
        "model": MODELO_ANALISIS,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1},
    }

    try:
        respuesta = requests.post(URL_OLLAMA, json=payload, timeout=90)
        respuesta.raise_for_status()
        data = respuesta.json()
        parsed = json.loads(data.get("response", "{}"))

        pregunta_limpia = parsed.get("question", pregunta_original)
        variantes = parsed.get("variantes", [])
        filtro = str(parsed.get("filtro_fecha", "")).strip()

        if filtro and "fecha" not in filtro:
            filtro = ""

        lista_final = _normalizar_variantes(variantes, pregunta_limpia)
        if not filtro:
            filtro = _parsear_filtro_fecha_local(pregunta_original)

        return lista_final, filtro
    except Exception:
        return [pregunta_original], _parsear_filtro_fecha_local(pregunta_original)


def _combinar_filtros(*filtros: str) -> str:
    activos = [f.strip() for f in filtros if f and f.strip()]
    return " && ".join(activos)


def _format_filter_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).replace("`", "")
    return f"`{text}`"


def construir_filtro_facetas(filtros_faceta: Optional[Dict[str, List]]) -> str:
    if not filtros_faceta:
        return ""

    clauses = []
    for field_name, values in filtros_faceta.items():
        if not values:
            continue
        rendered = [_format_filter_value(v) for v in values]
        if len(rendered) == 1:
            clauses.append(f"{field_name}:={rendered[0]}")
        else:
            clauses.append(f"{field_name}:[{','.join(rendered)}]")

    return _combinar_filtros(*clauses)


def buscar_en_typesense(
    cliente_db: typesense.Client,
    modelo_emb: SentenceTransformer,
    variantes: List[str],
    filtro_fecha: str = "",
    step_ids: Optional[List[str]] = None,
    filtro_facetas: Optional[Dict[str, List]] = None,
    per_page: int = 5,
) -> List[Dict]:
    resultados_unicos: Dict[str, Dict] = {}
    peticiones = []

    filtro_step = ""
    if step_ids:
        quoted = ",".join([_format_filter_value(x) for x in step_ids])
        filtro_step = f"step_id:[{quoted}]"

    filtro_faceta = construir_filtro_facetas(filtro_facetas)
    filtro_total = _combinar_filtros(filtro_fecha, filtro_step, filtro_faceta)

    for variante in variantes:
        vector = modelo_emb.encode(variante).tolist()
        vector_str = ",".join(map(str, vector))

        req = {
            "collection": COLLECTION_NAME,
            "q": variante,
            "query_by": "contenido_original,resumen_ejecutivo,faq,acuerdos",
            "vector_query": f"vector_legal:([{vector_str}], k: 10, alpha: 0.6)",
            "per_page": per_page,
            "exclude_fields": "vector_legal",
        }

        if filtro_total:
            req["filter_by"] = filtro_total

        peticiones.append(req)

    resultados = cliente_db.multi_search.perform({"searches": peticiones}, {})

    for bloque in resultados.get("results", []):
        for hit in bloque.get("hits", []):
            doc = hit.get("document", {})
            doc_id = str(doc.get("id", ""))
            if not doc_id or doc_id in resultados_unicos:
                continue

            texto = (
                doc.get("resumen_ejecutivo")
                or doc.get("faq")
                or doc.get("contenido_original")
                or ""
            )

            resultados_unicos[doc_id] = {
                "id": doc_id,
                "file_name": doc.get("file_name", "Acta desconocida"),
                "fecha": doc.get("fecha", "Sin fecha"),
                "step_id": doc.get("step_id", "sin_step"),
                "texto": texto,
            }

    return list(resultados_unicos.values())


def obtener_facetas(
    cliente_db: typesense.Client,
    pregunta: str,
    filtro_fecha: str = "",
    campos_facet: Optional[List[str]] = None,
    filtro_facetas: Optional[Dict[str, List]] = None,
) -> Dict[str, List[Dict]]:
    campos = campos_facet or ["step_id", "fecha", "proveedores", "servicios"]

    payload = {
        "q": pregunta or "*",
        "query_by": "contenido_original,resumen_ejecutivo,faq,acuerdos",
        "facet_by": ",".join(campos),
        "max_facet_values": 10,
        "per_page": 0,
    }
    filtro_faceta = construir_filtro_facetas(filtro_facetas)
    filtro_total = _combinar_filtros(filtro_fecha, filtro_faceta)
    if filtro_total:
        payload["filter_by"] = filtro_total

    res = cliente_db.collections[COLLECTION_NAME].documents.search(payload)
    salida: Dict[str, List[Dict]] = {}
    for facet in res.get("facet_counts", []):
        salida[facet.get("field_name", "")] = facet.get("counts", [])
    return salida


def construir_contexto(contexto_lista: List[Dict]) -> str:
    bloques = []
    for ctx in contexto_lista:
        bloques.append(
            (
                f"[ID: {ctx.get('id')} | Step: {ctx.get('step_id')} | "
                f"Fuente: {ctx.get('file_name')} | Fecha: {ctx.get('fecha')}]\n"
                f"{ctx.get('texto', '')}"
            )
        )
    return "\n\n".join(bloques)


def generar_respuesta_stream(
    pregunta: str,
    contexto_lista: List[Dict],
) -> Generator[str, None, None]:
    if not contexto_lista:
        yield "No se encontraron fragmentos relevantes en Typesense para esa consulta."
        return

    contexto = construir_contexto(contexto_lista)

    prompt_final = f"""Eres un administrador de fincas experto.
Responde SOLO con datos presentes en el contexto.
Si hay ambiguedad, indicala de forma breve.
Responde en Markdown con puntos claros.

Contexto:
{contexto}

Pregunta:
{pregunta}
"""

    payload = {
        "model": MODELO_LLM,
        "prompt": prompt_final,
        "stream": True,
        "options": {"temperature": 0.1},
    }

    try:
        respuesta = requests.post(URL_OLLAMA, json=payload, stream=True, timeout=600)
        respuesta.raise_for_status()
        for linea in respuesta.iter_lines():
            if not linea:
                continue
            datos = json.loads(linea)
            chunk = datos.get("response", "")
            if chunk:
                yield chunk
    except Exception as e:
        yield f"\n\n[Error de conexion con Ollama: {e}]"


def responder_consulta(
    pregunta_usuario: str,
    cliente_db: typesense.Client,
    modelo_emb: SentenceTransformer,
    step_ids: Optional[List[str]] = None,
) -> Tuple[str, List[Dict], Dict[str, List[Dict]], str]:
    variantes, filtro_fecha = analizar_pregunta_con_ia(pregunta_usuario)
    fragmentos = buscar_en_typesense(
        cliente_db=cliente_db,
        modelo_emb=modelo_emb,
        variantes=variantes,
        filtro_fecha=filtro_fecha,
        step_ids=step_ids,
    )
    facetas = obtener_facetas(cliente_db, pregunta_usuario, filtro_fecha)

    texto_respuesta = ""
    for trozo in generar_respuesta_stream(pregunta_usuario, fragmentos):
        texto_respuesta += trozo

    return texto_respuesta, fragmentos, facetas, filtro_fecha


if __name__ == "__main__":
    cliente = crear_cliente_typesense()
    embeddings = cargar_modelo_embeddings()

    pregunta = "¿Que acuerdos hubo entre 2000 y 2008 sobre proveedores?"
    respuesta, docs, facets, filtro = responder_consulta(pregunta, cliente, embeddings)

    print(f"Filtro aplicado: {filtro}")
    print("\nFacetas:")
    print(json.dumps(facets, ensure_ascii=False, indent=2))
    print("\nRespuesta:")
    print(respuesta)
    print(f"\nFragmentos recuperados: {len(docs)}")
