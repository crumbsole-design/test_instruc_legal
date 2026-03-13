import os
import json
import fitz  # PyMuPDF
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def dbg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: {msg}")

PATH_ACTAS = "./actas"
PATH_JSON = "./json"
os.makedirs(PATH_JSON, exist_ok=True)

# Primer archivo PDF encontrado
archivos = sorted([f for f in os.listdir(PATH_ACTAS) if f.endswith('.pdf')])
if not archivos:
    print("No se encontraron archivos PDF en ./actas")
    exit(1)

archivo = archivos[0]
dbg(f"Archivo seleccionado: {archivo}")

# Extraer texto del PDF
ruta_completa = os.path.join(PATH_ACTAS, archivo)
texto = ""
with fitz.open(ruta_completa) as doc:
    for pagina in doc:
        texto += pagina.get_text()

contexto = texto[:12000]
dbg(f"Texto extraído ({len(contexto)} chars)")
dbg(f"--- Primeros 300 chars del texto ---\n{contexto[:300]}\n---")

# Inicializar modelo
llm = ChatOllama(model="alia-extractorQ8", temperature=0)
dbg("Modelo inicializado")

# Mensaje 1: enviar el documento
msg_documento = (
    "A continuación te proporciono el texto de un acta. "
    "Léelo detenidamente; a continuación te haré varias preguntas sobre él.\n\n"
    f"TEXTO DEL ACTA:\n{contexto}"
)

# Mensaje 2: tercer prompt — extracción de entidades JSON
prompt_entidades = (
    "Usando el mismo acta, extrae los datos en formato JSON puro.\n"
    "Usa estrictamente este esquema y devuelve SOLO el JSON, sin explicaciones:\n"
    "{\n"
    '  "personas": ["Nombre Completo"],\n'
    '  "ubicaciones": ["Patio/Puerta"],\n'
    '  "deudas": [{ "sujeto": "Nombre", "importe": "0.00€", "concepto": "motivo" }],\n'
    '  "acuerdos": ["Acuerdo 1", "Acuerdo 2"]\n'
    "}"
)

historial = [
    HumanMessage(content=msg_documento),
    HumanMessage(content=prompt_entidades),
]

dbg("Enviando al modelo (esto puede tardar)...")
respuesta = llm.invoke(historial).content
dbg("Respuesta recibida")

print("\n===== RESPUESTA DEL MODELO =====")
print(respuesta)
print("================================\n")

# Guardar en json/
nombre_base = os.path.splitext(archivo)[0]
try:
    datos_validados = json.loads(respuesta)
    ruta_salida = os.path.join(PATH_JSON, f"{nombre_base}.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(datos_validados, f, ensure_ascii=False, indent=2)
    dbg(f"Guardado en: {ruta_salida}")
    print(f"✅ JSON guardado en {ruta_salida}")
except json.JSONDecodeError as e:
    dbg(f"Error al parsear JSON: {e}")
    ruta_error = os.path.join(PATH_JSON, f"{nombre_base}_ERROR.txt")
    with open(ruta_error, "w", encoding="utf-8") as f:
        f.write(respuesta)
    print(f"❌ El modelo no devolvió un JSON válido. Salida cruda guardada en {ruta_error}")
