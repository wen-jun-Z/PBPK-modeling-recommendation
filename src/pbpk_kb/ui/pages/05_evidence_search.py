from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Evidence Search")
page_title("Evidence Search")

mode = st.radio("Search mode", ["Structured", "Semantic", "Hybrid"], horizontal=True)

filter_columns = st.columns([2, 2, 2, 2])
compound = filter_columns[0].text_input("Compound", st.session_state.get("evidence_compound_filter", "midazolam"))
query = filter_columns[1].text_input("Parameter or text", st.session_state.get("evidence_query_filter", ""))
evidence_type = filter_columns[2].selectbox(
    "Evidence type",
    [""] + data.EVIDENCE_TYPES,
    index=([""] + data.EVIDENCE_TYPES).index(st.session_state.get("evidence_type_filter", ""))
    if st.session_state.get("evidence_type_filter", "") in [""] + data.EVIDENCE_TYPES
    else 0,
    format_func=lambda value: value or "All",
)
source_id = filter_columns[3].selectbox("Source", [""] + data.source_ids(), format_func=lambda value: value or "All")

filter_columns = st.columns([2, 2, 2, 2])
curation_status = filter_columns[0].selectbox(
    "Curation status",
    [""] + data.curation_statuses(),
    format_func=lambda value: value or "All",
)
source_family = filter_columns[1].selectbox("Source family", [""] + data.source_families(), format_func=lambda value: value or "All")
min_confidence = filter_columns[2].slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
limit = filter_columns[3].slider("Result limit", 5, 100, 25, 5)

context_columns = st.columns([2, 2, 2, 2])
species = context_columns[0].text_input("Species", "")
population = context_columns[1].text_input("Population", "")
route = context_columns[2].text_input("Route", "")
dose = context_columns[3].text_input("Dose", "")

detail_columns = st.columns([2, 2, 4])
enzyme_or_transporter = detail_columns[0].text_input("Enzyme / transporter", "")
source_date_from = detail_columns[1].date_input("Source date from", value=None)
source_date_to = detail_columns[2].date_input("Source date to", value=None)

st.session_state["evidence_compound_filter"] = compound
st.session_state["evidence_query_filter"] = query
st.session_state["evidence_type_filter"] = evidence_type

if mode in {"Structured", "Hybrid"}:
    st.subheader("Structured Evidence")
    structured_rows = data.evidence_search(
        compound=compound,
        query=query,
        evidence_type=evidence_type,
        source_id=source_id,
        source_family=source_family,
        species=species,
        population=population,
        route=route,
        dose=dose,
        enzyme_or_transporter=enzyme_or_transporter,
        source_date_from=source_date_from.isoformat() if source_date_from else "",
        source_date_to=source_date_to.isoformat() if source_date_to else "",
        curation_status=curation_status,
        min_confidence=min_confidence,
        limit=limit,
    )
    display_table(
        [
            {
                "layer": row.get("evidence_layer"),
                "type": row["evidence_type"],
                "compound": row["compound_name"],
                "role": row.get("compound_role"),
                "parameter": row.get("parameter_name"),
                "value": data.evidence_value(row),
                "context": row["context"],
                "species": row.get("species"),
                "population": row.get("population"),
                "route": row.get("route"),
                "dose": row.get("dose"),
                "enzyme_or_transporter": row.get("enzyme_or_transporter"),
                "source": row["source_id"],
                "source_family": row.get("source_family"),
                "location": row["source_location"],
                "method": row["extraction_method"],
                "confidence": row["confidence_score"],
                "attribution_confidence": row.get("attribution_confidence"),
                "status": row["curation_status"],
                "text": data.snippet(row["evidence_text"]),
            }
            for row in structured_rows
        ],
        height=420,
    )

if mode in {"Semantic", "Hybrid"}:
    semantic_query = query or compound or "midazolam pharmacokinetics"
    st.subheader("Vector Chunk Search")
    semantic_rows = data.semantic_search(semantic_query, source_id=source_id, limit=limit)
    display_table(
        [
            {
                "score": round(row["score"], 4) if row["score"] is not None else None,
                "source": row["source_id"],
                "document_type": row["document_type"],
                "title": row["title"],
                "section": row["section_title"],
                "paragraph": row["paragraph_number"],
                "location": row["source_location"],
                "embedding_model": row["embedding_model"],
                "text": data.snippet(row["chunk_text"], 360),
            }
            for row in semantic_rows
        ],
        height=420,
    )
    status_note("Vector search uses the post-final-evidence PubMedBERT index. Results retain the original source location and evidence layer.")
