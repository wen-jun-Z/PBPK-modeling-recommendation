from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import data
from pbpk_kb.ui.layout import configure_page, display_table, page_title


configure_page("Source Registry")
page_title("Source Registry")

families = [""] + data.source_families()
filters = st.columns([2, 2, 3])
family = filters[0].selectbox("Source family", families, format_func=lambda value: value or "All")
enabled = filters[1].selectbox(
    "Enabled",
    ["all", "enabled", "disabled"],
    format_func=lambda value: {"all": "All", "enabled": "Enabled", "disabled": "Disabled"}[value],
)
text_filter = filters[2].text_input("Text filter", "")

rows = data.source_registry_rows(family=family, enabled=enabled)
if text_filter.strip():
    needle = text_filter.strip().lower()
    rows = [
        row
        for row in rows
        if needle in row["source_id"].lower()
        or needle in row["source_name"].lower()
        or needle in row["source_family"].lower()
    ]

display_rows = []
for row in rows:
    has_current_corpus = any(
        row.get(field, 0)
        for field in (
            "raw_document_count",
            "parsed_document_count",
            "chunk_count",
            "evidence_count",
            "model_count",
        )
    )
    display_rows.append(
        {
            "source_id": row["source_id"],
            "source_name": row["source_name"],
            "family": row["source_family"],
            "method": row["access_method"],
            "formats": data.format_list(row["expected_format"]),
            "entities": data.format_list(row["entities"]),
            "enabled": row["enabled"],
            "raw": row["raw_document_count"],
            "parsed": row["parsed_document_count"],
            "rag_records": row["chunk_count"],
            "evidence": row["evidence_count"],
            "models": row["model_count"],
            "corpus_status": "indexed" if has_current_corpus else "registered_only",
            "version": row["source_version"] or "no current snapshot",
            "checksum": row["checksum"] or "no current snapshot checksum",
            "latest_status": row.get("latest_ingestion_status"),
            "latest_completed": row.get("latest_ingestion_completed_at"),
        }
    )

display_table(display_rows, height=430)

if rows:
    st.subheader("Source Details")
    selected = st.selectbox("Source", [row["source_id"] for row in rows])
    source = next(row for row in rows if row["source_id"] == selected)
    detail_columns = st.columns(2)
    with detail_columns[0]:
        st.write("Name", source["source_name"])
        st.write("Family", source["source_family"])
        st.write("Access method", source["access_method"])
        st.write("Connector", source.get("connector") or "")
        st.write("Parser", source.get("parser") or "")
        st.write("Refresh frequency", source["refresh_frequency"])
        st.write("Entities", data.format_list(source["entities"]))
        st.write("License", source.get("license_notes") or "")
    with detail_columns[1]:
        st.write("Access URL", source["access_url"])
        st.write("API URL", source["api_url"] or "")
        st.write("Version", source["source_version"] or "No current corpus snapshot")
        st.write("Checksum", source["checksum"] or "No current corpus checksum")
        st.write("Registry updated", source["updated_at"])
        st.write("Latest ingestion status", source.get("latest_ingestion_status") or "")
        st.write("Latest ingestion completed", source.get("latest_ingestion_completed_at") or "")
        st.write("Latest ingestion log", source.get("latest_ingestion_log_path") or "")
