from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Home")
page_title("PBPK Stage 1 Public Knowledge Base", "Dashboard")

metrics = data.dashboard_metrics()

metric_columns = st.columns(4)
metric_columns[0].metric("Sources", metrics.get("source_count", 0))
metric_columns[1].metric("Compounds", metrics.get("compound_count", 0))
metric_columns[2].metric("Evidence records", metrics.get("evidence_count", 0))
metric_columns[3].metric(
    "Auto-validated",
    metrics.get("auto_validated_count", 0),
)

metric_columns = st.columns(4)
metric_columns[0].metric("Raw documents", metrics.get("raw_document_count", 0))
metric_columns[1].metric("Parsed documents", metrics.get("parsed_document_count", 0))
metric_columns[2].metric("RAG records", metrics.get("rag_record_count", 0))
metric_columns[3].metric("Public models", metrics.get("public_model_count", 0))

if metrics.get("evidence_count", 0) == 0:
    status_note("No indexed evidence is present. Run the Milestone 3 index command before using the UI.")
else:
    status_note(
        "Evidence counts use the final auto-validated layer after attribution, "
        "deduplication, conflict checks, repair, and independent validation."
    )

left, right = st.columns([1, 2])

with left:
    st.subheader("Curation Status")
    display_table(data.evidence_status_counts(), height=210)

with right:
    st.subheader("Source Coverage")
    source_rows = data.source_registry_rows()
    compact_sources = [
        {
            "source_id": row["source_id"],
            "family": row["source_family"],
            "raw": row["raw_document_count"],
            "parsed": row["parsed_document_count"],
            "chunks": row["chunk_count"],
            "evidence": row["evidence_count"],
            "models": row["model_count"],
        }
        for row in source_rows
    ]
    display_table(compact_sources, height=300)

st.subheader("Recent Operations")
operation_tabs = st.tabs(["Latest Ingestion Runs", "Recent Exports"])
with operation_tabs[0]:
    display_table(data.latest_ingestion_runs(limit=10), height=260)
with operation_tabs[1]:
    display_table(data.latest_export_jobs(limit=10), height=260)

st.subheader("Stage 1 Workflow")
flow_columns = st.columns(5)
if flow_columns[0].button("Compound Search", width="stretch"):
    st.switch_page("pages/03_compound_search.py")
if flow_columns[1].button("Evidence Search", width="stretch"):
    st.switch_page("pages/05_evidence_search.py")
if flow_columns[2].button("Source Registry", width="stretch"):
    st.switch_page("pages/02_source_registry.py")
if flow_columns[3].button("Model Registry", width="stretch"):
    st.switch_page("pages/06_model_registry.py")
if flow_columns[4].button("Review Queue", width="stretch"):
    st.switch_page("pages/07_review_queue.py")

flow_columns = st.columns(6)
if flow_columns[0].button("Knowledge Graph", width="stretch"):
    st.switch_page("pages/12_knowledge_graph.py")
if flow_columns[1].button("Q&A", width="stretch"):
    st.switch_page("pages/09_qa_assistant.py")
if flow_columns[2].button("Exports", width="stretch"):
    st.switch_page("pages/10_exports.py")
if flow_columns[3].button("Ingestion Monitor", width="stretch"):
    st.switch_page("pages/11_ingestion_monitor.py")
if flow_columns[4].button("Admin Settings", width="stretch"):
    st.switch_page("pages/13_admin_settings.py")
if flow_columns[5].button("PBPK Recommendation", width="stretch"):
    st.switch_page("pages/14_recommendation_agent.py")
