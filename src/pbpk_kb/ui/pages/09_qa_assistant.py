from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import api_client, data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Q&A Assistant")
page_title("Q&A Assistant")

with st.form("qa-form"):
    question = st.text_area("Question", "What public clinical PK evidence exists for midazolam?", height=120)
    controls = st.columns([2, 1, 1, 1])
    compound = controls[0].text_input("Compound hint", "midazolam")
    evidence_limit = controls[1].slider("Evidence", 1, 25, 8)
    model_limit = controls[2].slider("Models", 0, 20, 5)
    chunk_limit = controls[3].slider("Chunks", 0, 20, 5)
    submitted = st.form_submit_button("Ask")

if submitted:
    try:
        result = api_client.ask_question(
            question=question,
            compound=compound,
            evidence_limit=evidence_limit,
            model_limit=model_limit,
            chunk_limit=chunk_limit,
        )
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["qa_result"] = result

result = st.session_state.get("qa_result")
if not result:
    status_note("Answers are generated only from retrieved KB evidence, public model records, graph relationships, and document chunks.")
    st.stop()

st.subheader("Answer")
st.markdown(result["answer"])

summary = result.get("retrieval_summary", {})
metric_columns = st.columns(5)
metric_columns[0].metric("Structured evidence", summary.get("structured_evidence_count", 0))
metric_columns[1].metric("Model records", summary.get("model_record_count", 0))
metric_columns[2].metric("Graph links", summary.get("graph_relationship_count", 0))
metric_columns[3].metric("RAG records", summary.get("document_chunk_count", 0))
metric_columns[4].metric("Citations", len(result.get("citations", [])))

tabs = st.tabs(["Key Evidence", "Public Models", "Graph Relationships", "RAG Context", "Citations", "Gaps"])

with tabs[0]:
    evidence_rows = [
        {
            "citation": row["citation_id"],
            "type": row["evidence_type"],
            "compound": row["compound_name"],
            "parameter": row["parameter_name"],
            "value": data.evidence_value(row),
            "source": row["source_id"],
            "location": row["source_location"],
            "status": row["curation_status"],
            "method": row["extraction_method"],
        }
        for row in result.get("key_evidence", [])
    ]
    display_table(evidence_rows, height=360)

with tabs[1]:
    model_rows = [
        {
            "citation": row["citation_id"],
            "model": row["model_name"],
            "compound": row["compound_name"],
            "software": row["software"],
            "source": row["source_id"],
            "location": row["source_location"],
            "validation": row["validation_status"],
            "status": row["curation_status"],
        }
        for row in result.get("model_records", [])
    ]
    display_table(model_rows, height=320)

with tabs[2]:
    graph_rows = [
        {
            "citation": row["citation_id"],
            "compound": row.get("compound_name"),
            "relationship": row.get("relationship_type"),
            "target": row.get("target_name"),
            "gene": row.get("gene_symbol"),
            "target_type": row.get("target_type"),
            "source": row.get("source_id"),
            "location": row.get("source_location"),
            "status": row.get("curation_status"),
        }
        for row in result.get("graph_relationships", [])
    ]
    display_table(graph_rows, height=320)

with tabs[3]:
    chunk_rows = [
        {
            "citation": row["citation_id"],
            "source": row["source_id"],
            "title": row["title"],
            "section": row["section_title"],
            "location": row["source_location"],
            "score": row["score"],
            "text": data.snippet(row["chunk_text"], 260),
            "embedding_model": row["embedding_model"],
        }
        for row in result.get("document_chunks", [])
    ]
    display_table(chunk_rows, height=360)

with tabs[4]:
    display_table(result.get("citations", []), height=360)

with tabs[5]:
    gaps = [{"gap": gap} for gap in result.get("uncertainties_and_gaps", [])]
    display_table(gaps, height=240)
