from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


def index_step_result(
    raw_output: str,
    step_id: str,
    file_name: str,
    base_metadata: dict,
    storage_context,
) -> int:
    """Fragment raw_output and index it in Typesense, tagging every node with step_id."""
    # Exclude large text fields from metadata: they are already in raw_output (the node text).
    # Keeping them in metadata inflates the metadata-size budget used by SentenceSplitter
    # and causes "Metadata length > chunk size" errors.
    _text_fields = {"faq", "resumen_ejecutivo", "contenido_original"}
    index_metadata = {
        k: v for k, v in {**base_metadata, "step_id": step_id, "file_name": file_name}.items()
        if k not in _text_fields
    }
    doc = Document(text=raw_output, metadata=index_metadata)

    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents([doc])

    VectorStoreIndex(nodes, storage_context=storage_context)
    return len(nodes)


def query_typesense(client, collection_name: str, query_text: str, num_results: int = 5) -> str:
    """Run a Typesense full-text search and return concatenated context fragments."""
    results = client.collections[collection_name].documents.search(
        {
            "q": query_text,
            "query_by": "contenido_original,resumen_ejecutivo,faq",
            "per_page": num_results,
        }
    )

    hits = results.get("hits", [])
    fragments = []
    for hit in hits:
        doc = hit.get("document", {})
        text = doc.get("resumen_ejecutivo") or doc.get("contenido_original", "")
        step = doc.get("step_id", "")
        fname = doc.get("file_name", "")
        fragments.append(f"[{fname} / {step}]\n{text}")

    return "\n\n---\n\n".join(fragments)
