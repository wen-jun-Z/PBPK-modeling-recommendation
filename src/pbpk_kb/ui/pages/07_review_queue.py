from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import api_client, data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Review Queue")
page_title("Review Queue")

filter_columns = st.columns([2, 2, 2, 2])
status = filter_columns[0].selectbox(
    "Status",
    ["needs_review", "machine_extracted", "reviewed_accepted", "reviewed_rejected", "approved_for_modeling", "duplicate", "deprecated", "all"],
)
evidence_type = filter_columns[1].selectbox("Evidence type", [""] + data.EVIDENCE_TYPES, format_func=lambda value: value or "All")
source_id = filter_columns[2].selectbox("Source", [""] + data.source_ids(), format_func=lambda value: value or "All")
limit = filter_columns[3].slider("Limit", 10, 200, 50, 10)
detail_columns = st.columns([2, 2, 2, 2])
compound = detail_columns[0].text_input("Compound", "")
query = detail_columns[1].text_input("Queue text", "")
duplicate_risk = detail_columns[2].selectbox("Duplicate risk", ["all", "high", "low"])
min_confidence = detail_columns[3].slider("Min confidence", 0.0, 1.0, 0.0, 0.05)

quality_columns = st.columns([2, 2, 2, 2])
created_from = quality_columns[0].date_input("Created from", value=None)
created_to = quality_columns[1].date_input("Created to", value=None)
missing_unit_filter = quality_columns[2].selectbox("Missing unit", ["all", "yes", "no"])
missing_context_filter = quality_columns[3].selectbox("Missing context", ["all", "yes", "no"])

try:
    rows = api_client.review_queue(
        status=status,
        evidence_type=evidence_type,
        source_id=source_id,
        compound=compound,
        q=query,
        min_confidence=min_confidence if min_confidence > 0 else None,
        created_from=created_from.isoformat() if created_from else "",
        created_to=created_to.isoformat() if created_to else "",
        duplicate_risk=duplicate_risk,
        missing_unit={"all": None, "yes": True, "no": False}[missing_unit_filter],
        missing_context={"all": None, "yes": True, "no": False}[missing_context_filter],
        limit=limit,
    )
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

display_rows = [
    {
        "type": row["evidence_type"],
        "compound": row["compound_name"],
        "parameter": row["parameter_name"],
        "value": data.evidence_value(row),
        "source": row["source_id"],
        "method": row["extraction_method"],
        "confidence": row["confidence_score"],
        "status": row["curation_status"],
        "duplicate_risk": row["duplicate_risk"],
        "missing_unit": row["missing_unit"],
        "missing_context": row["missing_context"],
        "created_at": row["created_at"],
        "location": row["source_location"],
        "text": data.snippet(row["evidence_text"], 220),
    }
    for row in rows
]
display_table(display_rows, height=430)

if not rows:
    st.stop()

st.subheader("Open Evidence Detail")
selected_key = st.selectbox(
    "Evidence record",
    [f"{row['evidence_type']}::{row['evidence_id']}" for row in rows],
    format_func=lambda value: next(
        f"{row['evidence_type']} | {row['compound_name'] or ''} | {row['parameter_name']}"
        for row in rows
        if value == f"{row['evidence_type']}::{row['evidence_id']}"
    ),
)
selected_type, selected_id = selected_key.split("::", 1)
selected_row = next(row for row in rows if row["evidence_type"] == selected_type and row["evidence_id"] == selected_id)

summary_columns = st.columns(4)
summary_columns[0].metric("Type", selected_row["evidence_type"])
summary_columns[1].metric("Status", selected_row["curation_status"])
summary_columns[2].metric("Confidence", selected_row["confidence_score"] or 0)
summary_columns[3].metric("Duplicate risk", selected_row["duplicate_risk"])

if st.button("Open evidence detail", width="stretch"):
    st.session_state["selected_evidence_type"] = selected_type
    st.session_state["selected_evidence_id"] = selected_id
    st.switch_page("pages/08_evidence_detail.py")

status_note("Accept, reject, edit, duplicate, and approval actions are performed on the Evidence Detail page and recorded in curation_event.")
