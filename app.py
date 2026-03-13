import streamlit as st

from consultor.consulta_hibrida import (
    analizar_pregunta_con_ia,
    buscar_en_typesense,
    cargar_modelo_embeddings,
    crear_cliente_typesense,
    generar_respuesta_stream,
    obtener_facetas,
)


@st.cache_resource
def get_client():
    return crear_cliente_typesense()


@st.cache_resource
def get_embeddings_model():
    return cargar_modelo_embeddings()


def _toggle_facet(field_name: str, raw_value):
    selected = st.session_state.selected_facets
    current = selected.get(field_name, [])

    value = raw_value
    if field_name == "fecha":
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = raw_value

    if value in current:
        current = [x for x in current if x != value]
    else:
        current = current + [value]

    if current:
        selected[field_name] = current
    elif field_name in selected:
        selected.pop(field_name)

    st.session_state.selected_facets = selected
    st.rerun()


def _render_facets(facets: dict):
    with st.expander("Facetas detectadas (clic para filtrar)", expanded=True):
        if not facets:
            st.write("Sin facetas para esta consulta.")
            return

        for field_name, counts in facets.items():
            if not counts:
                continue

            st.markdown(f"**{field_name}**")
            cols = st.columns(2)

            for idx, item in enumerate(counts):
                value = item.get("value", "")
                count = item.get("count", 0)
                active = value in st.session_state.selected_facets.get(field_name, [])
                label = f"{'✓ ' if active else ''}{value} ({count})"
                key = f"facet_{field_name}_{idx}_{abs(hash(str(value))) % 100000}"
                col = cols[idx % 2]
                if col.button(label, key=key, use_container_width=True):
                    _toggle_facet(field_name, value)


def _ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_facets" not in st.session_state:
        st.session_state.selected_facets = {}


def main():
    st.set_page_config(page_title="Chat Actas RAG", layout="wide")
    st.title("Chat de Actas (Typesense + Ollama)")

    _ensure_session_state()
    cliente = get_client()
    modelo = get_embeddings_model()

    with st.sidebar:
        st.subheader("Filtros de recuperación")
        step_ids = st.multiselect(
            "step_id base",
            options=["resumen", "faq", "entidades", "sintesis"],
            default=["resumen", "faq", "entidades"],
            help="Filtro base por step_id. Además puedes afinar con facetas clicables.",
        )
        st.caption("El filtro de fecha se infiere automáticamente desde la pregunta.")

        st.markdown("**Facetas activas**")
        if st.session_state.selected_facets:
            st.json(st.session_state.selected_facets)
            if st.button("Limpiar facetas", use_container_width=True):
                st.session_state.selected_facets = {}
                st.rerun()
        else:
            st.caption("Sin facetas seleccionadas.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Pregunta sobre actas, acuerdos, proveedores o cargos...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        variantes, filtro_fecha = analizar_pregunta_con_ia(prompt)

        fragmentos = buscar_en_typesense(
            cliente_db=cliente,
            modelo_emb=modelo,
            variantes=variantes,
            filtro_fecha=filtro_fecha,
            step_ids=step_ids,
            filtro_facetas=st.session_state.selected_facets,
            per_page=5,
        )

        facetas = obtener_facetas(
            cliente,
            prompt,
            filtro_fecha,
            filtro_facetas=st.session_state.selected_facets,
        )

        if filtro_fecha:
            st.caption(f"Filtro fecha aplicado: {filtro_fecha}")
        if st.session_state.selected_facets:
            st.caption(f"Facetas activas: {st.session_state.selected_facets}")
        st.caption(f"Fragmentos recuperados: {len(fragmentos)}")

        placeholder = st.empty()
        acumulado = ""
        for chunk in generar_respuesta_stream(prompt, fragmentos):
            acumulado += chunk
            placeholder.markdown(acumulado)

        _render_facets(facetas)

    st.session_state.messages.append({"role": "assistant", "content": acumulado})


if __name__ == "__main__":
    main()
