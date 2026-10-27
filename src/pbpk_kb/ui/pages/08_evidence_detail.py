from __future__ import annotations

import json

import streamlit as st

from pbpk_kb.ui import api_client, data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Evidence Detail")
page_title("Evidence Detail")


def choose_record_from_queue() -> tuple[str, str] | None:
    rows = api_client.review_queue(status="needs_review", limit=100)
    if not rows:
        return None
    selected_key = st.selectbox(
        "Evidence record",
        [f"{row['evidence_type']}::{row['evidence_id']}" for row in rows],
        format_func=lambda value: next(
            f"{row['evidence_type']} | {row['compound_name'] or ''} | {row['parameter_name']}"
            for row in rows
            if value == f"{row['evidence_type']}::{row['evidence_id']}"
        ),
    )
    return selected_key.split("::", 1)


selected_type = st.session_state.get("selected_evidence_type")
selected_id = st.session_state.get("selected_evidence_id")

if not selected_type or not selected_id:
    st.subheader("Select Evidence")
    try:
        selected = choose_record_from_queue()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    if selected is None:
        st.info("No needs_review evidence records are available.")
        st.stop()
    selected_type, selected_id = selected
    st.session_state["selected_evidence_type"] = selected_type
    st.session_state["selected_evidence_id"] = selected_id

try:
    detail = api_client.evidence_detail(selected_type, selected_id)
except RuntimeError as exc:
    st.error(str(exc))
    if st.button("Back to review queue", width="stretch"):
        st.switch_page("pages/07_review_queue.py")
    st.stop()

summary = detail["summary"]
st.header(f"{summary['evidence_type']} | {summary['parameter_name']}")

metric_columns = st.columns(5)
metric_columns[0].metric("Compound", summary["compound_name"] or "")
metric_columns[1].metric("Value", data.evidence_value(summary))
metric_columns[2].metric("Status", summary["curation_status"])
metric_columns[3].metric("Method", summary["extraction_method"])
metric_columns[4].metric("Confidence", summary["confidence_score"] or 0)

tabs = st.tabs(["Extracted Fields", "Source Text", "Related Evidence", "Curation History", "Curate"])

with tabs[0]:
    extracted_rows = [{"field": key, "value": value} for key, value in detail["extracted_fields"].items()]
    display_table(extracted_rows, height=420)
    normalized_fields = detail.get("normalized_fields")
    if normalized_fields:
        st.subheader("Normalized Fields")
        normalized_rows = [{"field": key, "value": value} for key, value in normalized_fields.items()]
        display_table(normalized_rows, height=240)
    else:
        st.info("No normalized compound identity is linked to this evidence record.")

with tabs[1]:
    st.subheader("Evidence Text")
    st.text_area("Evidence text", summary.get("evidence_text") or "", height=220, disabled=True)
    source_document = detail.get("source_document")
    if source_document:
        st.subheader("Source Document")
        display_table([source_document], height=180)
    else:
        st.info("No parsed source document is linked to this evidence record.")

with tabs[2]:
    related_rows = [
        {
            "type": row["evidence_type"],
            "parameter": row["parameter_name"],
            "value": data.evidence_value(row),
            "source": row["source_id"],
            "status": row["curation_status"],
            "text": data.snippet(row["evidence_text"]),
        }
        for row in detail["related_evidence"]
    ]
    display_table(related_rows, height=360)

with tabs[3]:
    history_rows = [
        {
            "timestamp": row["curation_timestamp"],
            "previous": row["previous_status"],
            "new": row["new_status"],
            "curator": row["curator"],
            "comment": row["curator_comment"],
            "field_changes": row["field_changes"],
        }
        for row in detail["curation_history"]
    ]
    display_table(history_rows, height=360)

with tabs[4]:
    status_note("All curation actions are written to curation_event with previous status, new status, curator, comment, and field changes.")
    editable_fields = detail["editable_fields"]
    display_table([{"editable_field": field} for field in editable_fields], height=200)

    with st.form("curation-form"):
        action = st.selectbox(
            "Action",
            ["accept", "reject", "edit", "note", "approve_for_modeling", "mark_duplicate", "flag_for_review"],
        )
        curator = st.text_input("Curator", "stage1_curator")
        comment = st.text_area("Curator comment", "")
        field_updates_text = st.text_area("Field updates JSON", "{}", height=140)
        submitted = st.form_submit_button("Submit curation action")

    if submitted:
        try:
            field_updates = json.loads(field_updates_text or "{}")
            if not isinstance(field_updates, dict):
                raise ValueError("Field updates JSON must be an object")
            result = api_client.curate_evidence(
                evidence_type=selected_type,
                evidence_id=selected_id,
                action=action,
                curator=curator,
                comment=comment,
                field_updates=field_updates,
            )
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Curation recorded: {result['previous_status']} -> {result['new_status']}")
            st.json(result)
            st.session_state["selected_evidence_type"] = selected_type
            st.session_state["selected_evidence_id"] = selected_id

if st.button("Back to review queue", width="stretch"):
    st.switch_page("pages/07_review_queue.py")
