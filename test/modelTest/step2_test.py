import os
import fitz  # PyMuPDF
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def dbg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: {msg}")

PATH_ACTAS = "./actas"
PATH_FAQ = "./faq"
os.makedirs(PATH_FAQ, exist_ok=True)

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

# Mensaje 2: segundo prompt — FAQ
prompt_faq = (
    "Usando el mismo acta, genera una lista de 5 a 10 preguntas y respuestas frecuentes (FAQ) en Markdown.\n"
    "Las preguntas deben ser las que un vecino haría sobre sus deudas, obras o servicios."
)

historial = [
    HumanMessage(content=msg_documento),
    HumanMessage(content=prompt_faq),
]

dbg("Enviando al modelo (esto puede tardar)...")
respuesta = llm.invoke(historial).content
dbg("Respuesta recibida")

print("\n===== RESPUESTA DEL MODELO =====")
print(respuesta)
print("================================\n")

# Guardar en faq/
nombre_base = os.path.splitext(archivo)[0]
ruta_salida = os.path.join(PATH_FAQ, f"{nombre_base}_faq.md")
with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(respuesta)

dbg(f"Guardado en: {ruta_salida}")
print(f"✅ FAQ guardado en {ruta_salida}")
