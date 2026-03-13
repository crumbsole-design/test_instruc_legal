import os
import json
from datetime import datetime
import fitz  # PyMuPDF
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def dbg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: {msg}")

# Configuración
MODEL = "alia-extractorQ8_gpu"
PATH_ACTAS = "./actas"
PATH_RESUMEN = "./resumen"
PATH_FAQ = "./faq"
PATH_JSON = "./json"
PATH_ACTAS_PROCESADAS = "./actas_procesadas"

for p in [PATH_RESUMEN, PATH_FAQ, PATH_JSON, PATH_ACTAS_PROCESADAS]:
    os.makedirs(p, exist_ok=True)

llm = ChatOllama(model=MODEL, temperature=0.1)

def stream_prompt(prompt):
    """Envía un prompt con streaming y devuelve la respuesta completa."""
    respuesta = ""
    for chunk in llm.stream([HumanMessage(content=prompt)]):
        print(chunk.content, end="", flush=True)
        respuesta += chunk.content
    print()
    return respuesta

def procesar_actas():
    tiempo_inicio = datetime.now()
    archivos = sorted([f for f in os.listdir(PATH_ACTAS) if f.endswith('.pdf')])

    if not archivos:
        print("No se encontraron archivos PDF en ./actas")
        return

    dbg(f"Se encontraron {len(archivos)} archivos PDF para procesar con modelo {MODEL}")

    for archivo in archivos:
        t_archivo = datetime.now()
        print(f"\n{'='*60}\n--- Procesando: {archivo} ---")
        ruta_completa = os.path.join(PATH_ACTAS, archivo)

        # Extraer texto del PDF
        texto = ""
        with fitz.open(ruta_completa) as doc:
            for pagina in doc:
                texto += pagina.get_text()
        contexto = texto[:12000]
        nombre_base = os.path.splitext(archivo)[0]
        dbg(f"Texto extraído ({len(contexto)} chars)")

        # Guardar texto extraído en actas_procesadas/
        ruta_procesada = os.path.join(PATH_ACTAS_PROCESADAS, f"{nombre_base}.md")
        with open(ruta_procesada, "w", encoding="utf-8") as f:
            f.write(contexto)
        dbg(f"Texto guardado en: {ruta_procesada}")

# 1. GENERAR RESUMEN EJECUTIVO (.md)
        prompt_resumen = (
            "Genera un RESUMEN EJECUTIVO directo y profesional en Markdown del siguiente acta.\n"
            "Estructura obligatoria:\n"
            "## Fecha y Asistencia\n"
            "## Estado de Cuentas\n"
            "## Acuerdos Principales\n\n"
            "IMPORTANTE: Al terminar el punto de 'Acuerdos Principales', escribe exactamente la palabra 'FIN_RESUMEN' y no añadas ni una sola palabra más.\n\n"
            f"TEXTO DEL ACTA:\n{contexto}"
        )
        dbg(f"Generando resumen ejecutivo para {archivo}...")
        resumen = stream_prompt(prompt_resumen)
        with open(f"{PATH_RESUMEN}/{nombre_base}_resumen.md", "w", encoding="utf-8") as f:
            f.write(resumen)
        dbg(f"Resumen guardado en {PATH_RESUMEN}/{nombre_base}_resumen.md")

# 2. GENERAR FAQ (.md)
        prompt_faq = (
            "Usando el siguiente acta, genera una lista de 5 a 10 preguntas y respuestas frecuentes (FAQ) en Markdown.\n"
            "Las preguntas deben ser las que un vecino haría sobre sus deudas, obras o servicios.\n\n"
            f"TEXTO DEL ACTA:\n{contexto}"
        )
        dbg(f"Generando FAQ para {archivo}...")
        faq = stream_prompt(prompt_faq)
        with open(f"{PATH_FAQ}/{nombre_base}_faq.md", "w", encoding="utf-8") as f:
            f.write(faq)
        dbg(f"FAQ guardado en {PATH_FAQ}/{nombre_base}_faq.md")

# 3. EXTRACCIÓN DE ENTIDADES (JSON para Typesense)
        prompt_entidades = (
            "Usando el siguiente acta, extrae los datos en formato JSON puro.\n"
            "Usa estrictamente este esquema y devuelve SOLO el JSON, sin explicaciones:\n"
            "{\n"
            '  "personas": ["Nombre Completo"],\n'
            '  "ubicaciones": ["Patio/Puerta"],\n'
            '  "deudas": [{ "sujeto": "Nombre", "importe": "0.00€", "concepto": "motivo" }],\n'
            '  "acuerdos": ["Acuerdo 1", "Acuerdo 2"]\n'
            "}\n\n"
            f"TEXTO DEL ACTA:\n{contexto}"
        )
        dbg(f"Extrayendo entidades JSON para {archivo}...")
        respuesta_json = stream_prompt(prompt_entidades)

        try:
            datos_validados = json.loads(respuesta_json)
            with open(f"{PATH_JSON}/{nombre_base}.json", "w", encoding="utf-8") as f:
                json.dump(datos_validados, f, ensure_ascii=False, indent=2)
            dbg(f"JSON guardado en {PATH_JSON}/{nombre_base}.json")
            tiempo_archivo = datetime.now() - t_archivo
            print(f"✅ Proceso completo para {archivo} | ⏱️  {tiempo_archivo}")
        except json.JSONDecodeError:
            print(f"❌ Error: JSON inválido para {archivo}. Guardando salida en crudo.")
            with open(f"{PATH_JSON}/{nombre_base}_ERROR.txt", "w", encoding="utf-8") as f:
                f.write(respuesta_json)

    tiempo_total = datetime.now() - tiempo_inicio
    print(f"\n⏱️  Tiempo total: {tiempo_total} con modelo {MODEL}")

if __name__ == "__main__":
    procesar_actas()