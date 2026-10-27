from __future__ import annotations

import streamlit as st

from pbpk_kb.ui import api_client
from pbpk_kb.ui.layout import configure_page, display_table, page_title, status_note


configure_page("Ingestion Monitor")
page_title("Ingestion Monitor", "Run history, logs, and manual source ingestion")


def _safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        st.error(f"{label}: {exc}")
        return None


sources = _safe_call("Load sources", api_client.list_sources) or []
source_ids = [source["source_id"] for source in sources]

st.subheader("Manual Ingestion")
with st.form("manual-ingestion-form"):
    columns = st.columns([2, 2, 1, 1, 2])
    ingest_source_id = columns[0].selectbox("Source", source_ids or [""])
    ingest_query = columns[1].text_input("Query", "midazolam")
    ingest_retmax = columns[2].number_input("Max records", min_value=1, max_value=20, value=1)
    ingest_network = columns[3].checkbox("Live network", value=False)
    requested_by = columns[4].text_input("Requested by", "stage1_admin")
    submitted = st.form_submit_button("Run ingestion")

if submitted and ingest_source_id:
    result = _safe_call(
        "Run ingestion",
        api_client.ingest_source,
        source_id=ingest_source_id,
        query=ingest_query,
        retmax=int(ingest_retmax),
        allow_network=ingest_network,
        requested_by=requested_by or "stage1_admin",
    )
    if result is not None:
        st.success(f"Started ingestion for {ingest_source_id}.")
        st.json(result)

st.subheader("Run History")
filter_columns = st.columns([2, 2, 1])
run_source = filter_columns[0].selectbox("Run source", [""] + source_ids, format_func=lambda value: value or "All")
run_status = filter_columns[1].selectbox(
    "Status",
    ["", "queued", "running", "succeeded", "partially_succeeded", "failed"],
    format_func=lambda value: value or "All",
)
run_limit = filter_columns[2].number_input("Rows", min_value=5, max_value=500, value=50, step=5)

runs = _safe_call(
    "Load ingestion runs",
    api_client.list_ingestion_runs,
    source_id=run_source,
    status=run_status,
    limit=int(run_limit),
) or []

status_counts: dict[str, int] = {}
for run in runs:
    status_counts[run["status"]] = status_counts.get(run["status"], 0) + 1

metric_columns = st.columns(5)
metric_columns[0].metric("Runs", len(runs))
metric_columns[1].metric("Succeeded", status_counts.get("succeeded", 0))
metric_columns[2].metric("Failed", status_counts.get("failed", 0))
metric_columns[3].metric("Raw records", sum(run.get("raw_record_count") or 0 for run in runs))
metric_columns[4].metric("Parsed records", sum(run.get("parsed_record_count") or 0 for run in runs))

display_table(
    [
        {
            "ingestion_run_id": run["ingestion_run_id"],
            "source_id": run["source_id"],
            "status": run["status"],
            "started_at": run.get("started_at"),
            "completed_at": run.get("completed_at"),
            "raw": run.get("raw_record_count"),
            "parsed": run.get("parsed_record_count"),
            "evidence": run.get("evidence_record_count"),
            "source_version": run.get("source_version"),
            "log_path": run.get("log_path"),
            "error": run.get("error_summary"),
        }
        for run in runs
    ],
    height=360,
)

if not runs:
    status_note("No ingestion runs match the selected filters.")
    st.stop()

st.subheader("Log Drill-Down and Retry")
selected_run = st.selectbox("Inspect run", [run["ingestion_run_id"] for run in runs])
left, right = st.columns([1, 1])

if left.button("Load log", width="stretch"):
    log_result = _safe_call("Load ingestion log", api_client.ingestion_run_log, selected_run)
    if log_result is not None:
        st.json(log_result)

with right.form("retry-ingestion-form"):
    retry_query = st.text_input("Retry query", "midazolam")
    retry_retmax = st.number_input("Retry max records", min_value=1, max_value=20, value=1)
    retry_network = st.checkbox("Retry with live network", value=False)
    retry_by = st.text_input("Retry requested by", "stage1_admin")
    retry_submitted = st.form_submit_button("Retry selected run")

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
        st.success(f"Retried {selected_run}.")
        st.json(retry_result)
