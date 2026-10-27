from __future__ import annotations

from typing import Any

import streamlit as st

from pbpk_kb.ui import api_client, data
from pbpk_kb.ui.layout import (
    configure_page,
    display_table,
    page_title,
    status_note,
)


configure_page("Public PBPK Model Registry")
page_title(
    "Public PBPK Model Registry",
    "Versioned model assets, applicability, simulations, and reproducibility evidence",
)


def safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        st.error(f"{label}: {exc}")
        return None


def compact_checksum(value: str | None) -> str:
    if not value:
        return ""
    return f"{value[:12]}...{value[-8:]}"


filters = st.columns([2, 2, 2, 2, 1])
compound = filters[0].text_input("Compound or model", "")
source_id = filters[1].selectbox(
    "Source",
    [""] + data.source_ids(),
    format_func=lambda value: value or "All",
)
software = filters[2].text_input("Software", "")
validation_status = filters[3].selectbox(
    "Validation status",
    [""] + data.validation_statuses(),
    format_func=lambda value: value or "All",
)
reusable_only = filters[4].checkbox("Reusable")
limit = st.slider("Result limit", 5, 100, 50, 5)

rows = safe_call(
    "Load public models",
    data.model_registry,
    compound=compound,
    source_id=source_id,
    software=software,
    validation_status=validation_status,
    reusable_only=reusable_only,
    limit=limit,
) or []

display_table(
    [
        {
            "model": row.get("model_name"),
            "compound": row.get("compound_name"),
            "release": row.get("release_tag"),
            "software": row.get("software"),
            "license": row.get("license_spdx_id"),
            "grade": row.get("run_label_grade"),
            "validation": row.get("validation_grade")
            or row.get("validation_status"),
            "reusable": row.get("reusable_candidate"),
            "simulations": row.get("simulation_count"),
            "source": row.get("source_id"),
        }
        for row in rows
    ],
    height=330,
)

unlinked_count = sum(1 for row in rows if not row.get("compound_name"))
if unlinked_count:
    status_note(
        f"{unlinked_count} model record(s) have no asserted compound linkage."
    )

if not rows:
    st.stop()

model_labels = {
    row["model_id"]: (
        f"{row['model_name']} | "
        f"{row.get('release_tag') or 'unversioned'} | "
        f"grade {row.get('run_label_grade') or 0}"
    )
    for row in rows
}
selected_model_id = st.selectbox(
    "Model detail",
    list(model_labels),
    format_func=model_labels.get,
)
detail = safe_call(
    "Load model detail",
    api_client.get_model_detail,
    selected_model_id,
)
if not detail:
    st.stop()

model = detail.get("model") or {}
versions = detail.get("versions") or []
assets = detail.get("assets") or []
simulations = detail.get("simulations") or []
runs = detail.get("validation_runs") or []
datasets = detail.get("validation_datasets") or []
metrics = detail.get("validation_metrics") or []
parameters = detail.get("parameters") or []

headline = st.columns(5)
headline[0].metric("Reuse decision", "Eligible" if model.get("reusable_candidate") else "Not eligible")
headline[1].metric("Validation grade", model.get("run_label_grade") or 0)
headline[2].metric("Simulations", len(simulations))
headline[3].metric("Validation datasets", len(datasets))
headline[4].metric("Validation metrics", len(metrics))

overview_tab, structure_tab, simulation_tab, validation_tab, asset_tab, parameter_tab = st.tabs(
    [
        "Overview",
        "Structure and Mechanisms",
        "Simulations",
        "Validation and Reproducibility",
        "Assets and Provenance",
        "Parameters",
    ]
)

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.write("Model", model.get("model_name") or "")
        st.write("Compound", model.get("compound_name") or "")
        st.write("Model type", model.get("model_type") or "")
        st.write("Software", model.get("software") or "")
        st.write("Species", model.get("species") or "")
        st.write("Routes", model.get("route") or "")
    with right:
        st.write("Release", model.get("release_tag") or "")
        st.write("License", model.get("license_spdx_id") or "")
        st.write("Validation", model.get("validation_grade") or "")
        st.write("Ranking label", model.get("ranking_label") or "")
        st.write("Curation", model.get("curation_status") or "")
        if model.get("repository_url"):
            st.link_button("Open official repository", model["repository_url"])
    display_table(
        [
            {
                "version": row.get("version_label"),
                "commit": row.get("repository_commit"),
                "project_checksum": row.get("project_sha256"),
                "source_checksum": row.get("source_snapshot_sha256"),
                "static_parse": row.get("static_parse_status"),
                "execution": row.get("execution_probe_status"),
                "PKML_parse": row.get("pkml_parse_status"),
                "reusable": row.get("reusable_candidate"),
            }
            for row in versions
        ],
        height=220,
    )

with structure_tab:
    for row in versions:
        mechanism_metrics = st.columns(4)
        mechanism_metrics[0].metric("Compartments", len(row.get("compartments") or []))
        mechanism_metrics[1].metric("Enzymes", len(row.get("enzymes") or []))
        mechanism_metrics[2].metric("Transporters", len(row.get("transporters") or []))
        mechanism_metrics[3].metric("Process types", len(row.get("process_types") or []))
        display_table(
            [
                {
                    "category": "Enzyme",
                    "name": value,
                }
                for value in row.get("enzymes") or []
            ]
            + [
                {
                    "category": "Transporter",
                    "name": value,
                }
                for value in row.get("transporters") or []
            ]
            + [
                {
                    "category": "Process",
                    "name": value,
                }
                for value in row.get("process_types") or []
            ],
            height=360,
        )
        with st.expander("Compartments and outputs"):
            st.write("Compartments", ", ".join(row.get("compartments") or []))
            st.write("Outputs", ", ".join(row.get("output_names") or []))

with simulation_tab:
    display_table(
        [
            {
                "sequence": row.get("protocol_sequence"),
                "simulation": row.get("simulation_name"),
                "role": row.get("protocol_role"),
                "route": ", ".join(row.get("routes") or []),
                "formulation": ", ".join(row.get("formulations") or []),
                "validation_dataset": row.get("validation_dataset_name"),
                "solver": row.get("solver"),
                "parameters": row.get("parameter_count"),
            }
            for row in simulations
        ],
        height=540,
    )

with validation_tab:
    if runs:
        run = runs[0]
        independence = run.get("validation_independence")
        validation_metrics = st.columns(5)
        validation_metrics[0].metric("Grade", run.get("run_label_grade") or 0)
        validation_metrics[1].metric("Observations", run.get("pooled_observation_count") or 0)
        validation_metrics[2].metric("GMFE", f"{float(run.get('pooled_gmfe') or 0):.3f}")
        validation_metrics[3].metric("Geometric bias", f"{float(run.get('pooled_geometric_bias') or 0):.3f}")
        validation_metrics[4].metric("Within two-fold", f"{float(run.get('pooled_two_fold_fraction') or 0):.1%}")
        if independence == "repository_preregistered_holdout":
            st.warning(
                "These results reproduce the public repository's frozen "
                "model-verification split under preregistered thresholds and "
                "an independent code audit. They are not an external "
                "replication by an unrelated laboratory."
            )
        elif independence == "externally_curated_independent":
            status_note(
                "The validation datasets were curated independently of the "
                "registered model repository."
            )
        status_note(
            "Reuse remains limited to requests that pass every compound, "
            "mechanism, population, regimen, endpoint, license, execution, "
            "and validation-scope constraint."
        )
    display_table(
        [
            {
                "sequence": row.get("simulation_sequence"),
                "dataset": row.get("dataset_name"),
                "observations": row.get("observation_count"),
                "GMFE": row.get("gmfe"),
                "bias": row.get("geometric_bias"),
                "two_fold_fraction": row.get("two_fold_fraction"),
                "AUC_ratio": row.get("auc0_last_ratio"),
                "Cmax_ratio": row.get("cmax_ratio"),
                "endpoint_passed": row.get("endpoint_passed"),
                "reference": row.get("reference"),
                "source_location": row.get("source_location"),
            }
            for row in datasets
        ],
        height=520,
    )
    with st.expander("All validation metric records"):
        display_table(metrics, height=480)

with asset_tab:
    checksum_summary = st.columns(3)
    checksum_summary[0].metric("Assets", len(assets))
    checksum_summary[1].metric("Project SHA256", compact_checksum(model.get("model_checksum")))
    checksum_summary[2].metric("Source SHA256", compact_checksum(model.get("source_snapshot_checksum")))
    display_table(
        [
            {
                "kind": row.get("asset_kind"),
                "path": row.get("relative_path"),
                "sha256": row.get("sha256"),
                "source_location": row.get("source_location"),
            }
            for row in assets
        ],
        height=520,
    )

with parameter_tab:
    scale = st.columns(3)
    scale[0].metric("Parameter facts", model.get("parameter_record_count") or 0)
    scale[1].metric("Component facts", model.get("component_record_count") or 0)
    scale[2].metric("Process facts", model.get("process_record_count") or 0)
    status_note(
        "Full parameter, component, and process facts remain in the immutable "
        "registered artifact and Model KG. This table contains only curated "
        "parameter evidence imported into the application database."
    )
    display_table(parameters, height=430)
