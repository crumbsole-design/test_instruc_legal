import sys
import typesense

# Configuración de Conexión a Typesense
cliente_db = typesense.Client({
  'nodes': [{'host': 'localhost', 'port': '8108', 'protocol': 'http'}],
  'api_key': 'clave_secreta_actas',
  'connection_timeout_seconds': 2
})

COLLECTION_NAME = 'actas_comunidad'

def obtener_contexto_ampliado(id_doc_encontrado):
    """
    Toma un ID (ej: 'hash_5') y recupera de Typesense los fragmentos 'hash_4', 'hash_5' y 'hash_6'.
    """
    print(f"\n[+] Conectando a Typesense para extraer el contexto del fragmento: {id_doc_encontrado}...\n")
    
    try:
        respuesta = cliente_db.collections[COLLECTION_NAME].documents.search(
            {
                'q': '*',
                'filter_by': f'id:={id_doc_encontrado}',
                'per_page': 1,
            }
        )
        
        documentos_recuperados = respuesta.get('hits', [])
        
        if not documentos_recuperados:
            return "❌ No se encontró ningún documento con ese ID en la base de datos."

        texto_ampliado = ""
        for hit in documentos_recuperados:
            doc = hit['document']
            texto = (
                doc.get('resumen_ejecutivo')
                or doc.get('faq')
                or doc.get('contenido_original')
                or ''
            )
            step_id = doc.get('step_id', 'sin_step')
            file_name = doc.get('file_name', 'Acta desconocida')
            fecha = doc.get('fecha', 'Sin fecha')

            texto_ampliado += (
                f"--- [FRAGMENTO (ID: {doc.get('id', 'sin_id')}) | Step: {step_id}] ---\n"
                f"Fuente: {file_name} | Fecha: {fecha}\n"
                f"{texto}\n\n"
            )
                
        return texto_ampliado

    except Exception as e:
        return f"❌ Error de conexión con la base de datos: {e}"

if __name__ == "__main__":
    # Verificamos que el usuario ha pasado el ID por consola
    if len(sys.argv) < 2:
        print("Uso correcto: python extractor_de_contexto.py <ID_DEL_DOCUMENTO>")
        print("Ejemplo: python extractor_de_contexto.py d490adf4da4bb09151_24")
        sys.exit(1)

    id_buscar = sys.argv[1]
    resultado = obtener_contexto_ampliado(id_buscar)
    print(resultado)