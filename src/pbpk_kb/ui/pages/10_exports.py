from __future__ import annotations

from pathlib import Path

import streamlit as st

from pbpk_kb.ui import api_client, data
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Exports")
page_title("Exports")


def show_export_result(result: dict) -> None:
    st.success(f"Export completed: {result['output_path']}")
    st.json(result["export_job"])
    output_path = Path(result["output_path"])
    if output_path.exists():
        st.download_button(
            "Download export",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="text/csv" if output_path.suffix == ".csv" else "text/markdown",
            width="stretch",
        )


tabs = st.tabs(["Evidence CSV", "Compound Profile", "Public Models", "Stage 1 Summary", "Export Jobs"])

with tabs[0]:
    status_note(
        "Exports contain only evidence from the final auto-validated "
        "quality layer."
    )
    with st.form("evidence-export-form"):
        columns = st.columns([2, 2, 2, 2, 2])
        compound = columns[0].text_input("Compound", "midazolam")
        evidence_type = columns[1].selectbox("Evidence type", [""] + data.EVIDENCE_TYPES, format_func=lambda value: value or "All")
        source_id = columns[2].selectbox("Source", [""] + data.source_ids(), format_func=lambda value: value or "All")
        curation_status = columns[3].selectbox("Curation status", [""] + data.curation_statuses(), format_func=lambda value: value or "All")
        evidence_layer = columns[4].selectbox(
            "Evidence layer",
            ["auto_validated"],
        )
        q = st.text_input("Text filter", "")
        requested_by = st.text_input("Requested by", "stage1_user")
        limit = st.slider("Limit", 1, 5000, 500, 50)
        submitted = st.form_submit_button("Export evidence CSV")
    if submitted:
        try:
            show_export_result(
                api_client.export_evidence(
                    compound=compound,
                    evidence_type=evidence_type,
                    source_id=source_id,
                    curation_status=curation_status,
                    evidence_layer=evidence_layer,
                    q=q,
                    requested_by=requested_by,
                    limit=limit,
                )
            )
        except RuntimeError as exc:
            st.error(str(exc))

with tabs[1]:
    with st.form("compound-profile-export-form"):
        compound = st.text_input("Compound", "midazolam", key="profile-compound")
        evidence_layer = st.selectbox(
            "Evidence layer",
            ["auto_validated"],
            key="profile-layer",
        )
        curation_status = st.selectbox("Curation status", [""] + data.curation_statuses(), format_func=lambda value: value or "All", key="profile-status")
        requested_by = st.text_input("Requested by", "stage1_user", key="profile-user")
        submitted = st.form_submit_button("Export compound profile")
    if submitted:
        try:
            show_export_result(
                api_client.export_compound_profile(
                    compound=compound,
                    curation_status=curation_status,
                    evidence_layer=evidence_layer,
                    requested_by=requested_by,
                )
            )
        except RuntimeError as exc:
            st.error(str(exc))

with tabs[2]:
    with st.form("model-registry-export-form"):
        columns = st.columns([2, 2, 2])
        compound = columns[0].text_input("Compound", "", key="model-compound")
        source_id = columns[1].selectbox("Source", [""] + data.source_ids(), format_func=lambda value: value or "All", key="model-source")
        validation_status = columns[2].selectbox("Validation", [""] + data.validation_statuses(), format_func=lambda value: value or "All")
        requested_by = st.text_input("Requested by", "stage1_user", key="model-user")
        submitted = st.form_submit_button("Export public model registry")
    if submitted:
        try:
            show_export_result(
                api_client.export_model_registry(
                    compound=compound,
                    source_id=source_id,
                    validation_status=validation_status,
                    requested_by=requested_by,
                )
            )
        except RuntimeError as exc:
            st.error(str(exc))

with tabs[3]:
    status_note("Stage 1 summary includes source versions, evidence status counts, curation event count, and export audit count.")
    requested_by = st.text_input("Requested by", "stage1_user", key="summary-user")
    if st.button("Export Stage 1 summary", width="stretch"):
        try:
            show_export_result(api_client.export_stage1_summary(requested_by=requested_by))
        except RuntimeError as exc:
            st.error(str(exc))

with tabs[4]:
    try:
        jobs = api_client.export_jobs(limit=25)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    display_table(jobs, height=420)
