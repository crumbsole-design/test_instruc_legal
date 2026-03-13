import os
import fitz  # PyMuPDF
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

def dbg(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: {msg}")

tiempo_inicio = datetime.now()

PATH_ACTAS = "./actas"
PATH_RESUMEN = "./resumen"
PATH_ACTAS_PROCESADAS = "./actas_procesadas"
MODEL = "alia-extractorQ4_cpu"  # Cambia a tu modelo optimizado
os.makedirs(PATH_RESUMEN, exist_ok=True)
os.makedirs(PATH_ACTAS_PROCESADAS, exist_ok=True)

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

# Guardar texto extraído en actas_procesadas/
nombre_base = os.path.splitext(archivo)[0]
ruta_procesada = os.path.join(PATH_ACTAS_PROCESADAS, f"{nombre_base}.md")
with open(ruta_procesada, "w", encoding="utf-8") as f:
    f.write(contexto)
dbg(f"Texto guardado en: {ruta_procesada}")

# Inicializar modelo
llm = ChatOllama(model=MODEL, temperature=0.1)
dbg("Modelo inicializado")

prompt_completo = (
    "Genera un RESUMEN EJECUTIVO directo y profesional en Markdown del siguiente acta.\n"
    "Estructura obligatoria:\n"
    "## Fecha y Asistencia\n"
    "## Estado de Cuentas\n"
    "## Acuerdos Principales\n\n"
    "IMPORTANTE: Al terminar el punto de 'Acuerdos Principales', escribe exactamente la palabra 'FIN_RESUMEN' y no añadas ni una sola palabra más.\n\n"
    f"TEXTO DEL ACTA:\n{contexto}"
)

historial = [
    HumanMessage(content=prompt_completo)
]

dbg("Enviando al modelo (Activando Streaming para ver el progreso)...")

print("\n===== RESPUESTA DEL MODELO =====")
respuesta_completa = ""

# Usamos .stream() en lugar de .invoke()
# Esto imprimirá cada palabra en cuanto Ollama la genere, igual que en la web
for chunk in llm.stream(historial):
    print(chunk.content, end="", flush=True)
    respuesta_completa += chunk.content

print("\n================================\n")
dbg("Respuesta finalizada")

# Guardar en resumen/
ruta_salida = os.path.join(PATH_RESUMEN, f"{nombre_base}_resumen.md")
with open(ruta_salida, "w", encoding="utf-8") as f:
    f.write(respuesta_completa)

dbg(f"Guardado en: {ruta_salida}")
print(f"✅ Resumen guardado en {ruta_salida}")

tiempo_total = datetime.now() - tiempo_inicio
print(f"\n⏱️  Tiempo total: {tiempo_total} con modelo {MODEL}")
