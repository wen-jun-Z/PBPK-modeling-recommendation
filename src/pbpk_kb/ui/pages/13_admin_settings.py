from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import api_client
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Admin Settings")
page_title("Admin Settings", "Source controls, ingestion logs, and retry actions")


def _safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        st.error(f"{label}: {exc}")
        return None


sources = _safe_call("Load sources", api_client.list_sources) or []
source_ids = [source["source_id"] for source in sources]

st.subheader("Source Controls")
display_table(
    [
        {
            "source_id": source["source_id"],
            "source_name": source["source_name"],
            "family": source["source_family"],
            "enabled": source["enabled"],
            "owner": source.get("owner_curator"),
            "latest_status": source.get("latest_ingestion_status") or "never",
            "version": source.get("source_version") or "",
        }
        for source in sources
    ],
    height=300,
)

with st.form("source-update-form"):
    columns = st.columns([2, 1, 2, 1])
    selected_source = columns[0].selectbox("Source", source_ids or [""])
    enabled_choice = columns[1].selectbox("Enabled", ["No change", "Enabled", "Disabled"])
    owner_curator = columns[2].text_input("Owner curator", "")
    requested_by = columns[3].text_input("Requested by", "stage1_admin")
    submitted = st.form_submit_button("Update source")

if submitted and selected_source:
    enabled_value = None
    if enabled_choice == "Enabled":
        enabled_value = True
    elif enabled_choice == "Disabled":
        enabled_value = False
    result = _safe_call(
        "Update source",
        api_client.update_source,
        source_id=selected_source,
        enabled=enabled_value,
        owner_curator=owner_curator or None,
        requested_by=requested_by or "stage1_admin",
    )
    if result is not None:
        st.success(f"Updated {selected_source}.")
        st.json(result)

st.subheader("Ingestion Logs and Retry")
status_note("Retry uses the same connector path as manual ingestion. Network remains off unless explicitly enabled.")

run_filter_columns = st.columns([2, 2, 1])
run_source = run_filter_columns[0].selectbox("Run source", [""] + source_ids, format_func=lambda value: value or "All")
run_status = run_filter_columns[1].selectbox(
    "Run status",
    ["", "failed", "partially_succeeded", "succeeded", "running", "queued"],
    format_func=lambda value: value or "All",
)
run_limit = run_filter_columns[2].number_input("Runs", min_value=5, max_value=200, value=25, step=5)
runs = _safe_call(
    "Load ingestion runs",
    api_client.list_ingestion_runs,
    source_id=run_source,
    status=run_status,
    limit=int(run_limit),
) or []

display_table(
    [
        {
            "ingestion_run_id": run["ingestion_run_id"],
            "source_id": run["source_id"],
            "status": run["status"],
            "completed_at": run.get("completed_at"),
            "raw_records": run.get("raw_record_count"),
            "parsed_records": run.get("parsed_record_count"),
            "log_path": run.get("log_path"),
        }
        for run in runs
    ],
    height=320,
)

run_ids = [run["ingestion_run_id"] for run in runs]
if run_ids:
    selected_run = st.selectbox("Inspect run", run_ids)
    action_columns = st.columns([1, 1])
    if action_columns[0].button("Load log", width="stretch"):
        log_result = _safe_call("Load ingestion log", api_client.ingestion_run_log, selected_run)
        if log_result is not None:
            st.json(log_result)

    with action_columns[1].form("retry-run-form"):
        retry_query = st.text_input("Retry query", "midazolam")
        retry_retmax = st.number_input("Retry max records", min_value=1, max_value=20, value=1)
        retry_network = st.checkbox("Live network", value=False)
        retry_by = st.text_input("Retry requested by", "stage1_admin")
        retry_submitted = st.form_submit_button("Retry run")
    if retry_submitted:
        retry_result = _safe_call(
            "Retry ingestion",
            api_client.retry_ingestion_run,
            ingestion_run_id=selected_run,
            query=retry_query,
            retmax=int(retry_retmax),
            allow_network=retry_network,
            requested_by=retry_by or "stage1_admin",
        )
        if retry_result is not None:
            st.success(f"Retried run {selected_run}.")
            st.json(retry_result)
else:
    st.info("No ingestion runs match the current filters.")
