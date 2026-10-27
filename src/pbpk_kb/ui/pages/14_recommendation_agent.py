from __future__ import annotations

from typing import Any

import streamlit as st

from pbpk_kb.ui import api_client
from pbpk_kb.ui.blueprint_presenters import (
    artifact_rows,
    compartment_rows,
    parameter_rows,
    process_rows,
    process_summary,
    public_model_request_defaults,
    simulation_rows,
)
from pbpk_kb.ui.layout import (
    configure_page,
    display_table,
    page_title,
    status_note,
)


configure_page("PBPK Recommendation Agent")
page_title(
    "PBPK Recommendation Agent",
    "Inspect evidence-supported component candidates, rank relevant public "
    "models, and continue to an executable exact-compound blueprint",
)


CANDIDATE_LAYER_LABELS = {
    "validated_reuse": "Validated for reuse",
    "controlled_adaptation_candidate": "Controlled adaptation",
    "same_compound_qualification_candidate": (
        "Same-compound qualification"
    ),
    "mechanistic_reference": "Mechanistic reference",
    "blocked": "Blocked",
}

COMPONENT_LABELS = {
    "compound_identity": "Compound identity",
    "physicochemical": "Physicochemical properties",
    "absorption": "Absorption",
    "distribution": "Distribution",
    "metabolism": "Metabolism",
    "excretion": "Excretion and clearance",
    "ddi": "Drug-drug interaction",
    "special_population": "Special population",
    "public_model_template": "Public model template",
}

OBJECTIVE_LABELS = {
    "base_pk": "Base pharmacokinetics",
    "ddi_victim": "DDI victim",
    "ddi_perpetrator": "DDI perpetrator",
    "renal_impairment": "Renal impairment",
    "hepatic_impairment": "Hepatic impairment",
    "pediatric": "Pediatric",
    "geriatric": "Geriatric",
    "pregnancy": "Pregnancy",
    "public_model_reuse": "Public model reuse",
}


def safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except RuntimeError as exc:
        st.error(f"{label}: {exc}")
        return None


def reset_public_model_request() -> None:
    compound = str(
        st.session_state.get("public_model_target_compound") or ""
    )
    defaults = public_model_request_defaults(compound)
    for field, value in defaults.items():
        st.session_state[f"public_model_{field}"] = value
    for key in (
        "public_model_ranking_payload",
        "public_model_ranking_result",
        "blueprint_result",
        "blueprint_workspace",
        "blueprint_validation_report",
        "blueprint_compilation_job",
        "blueprint_simulation_job",
    ):
        st.session_state.pop(key, None)


def list_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value or "")


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def relevance_text(row: dict[str, Any]) -> str:
    if row.get("relevance_class") == "same_compound":
        return "Same compound"
    mechanisms = sorted(
        {
            str(basis.get("mechanism_name"))
            for basis in row.get("relevance_basis") or []
            if basis.get("mechanism_name")
        },
        key=str.casefold,
    )
    return ", ".join(mechanisms)


def candidate_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": row.get("rank"),
            "public model name": row.get("model_name"),
            "modeled compound(s)": list_text(
                row.get("supported_compounds") or []
            ),
            "relationship to target": (
                "Same target compound"
                if row.get("candidate_class")
                in {"exact_compound_model", "same_compound_adaptation"}
                else "Different-compound mechanistic reference"
                if row.get("candidate_class") == "mechanistic_template"
                else "Blocked or unrelated"
            ),
            "project": row.get("project_relative_path"),
            "layer": row.get("candidate_layer_label"),
            "class": row.get("candidate_class"),
            "relevance": relevance_text(row),
            "direct_reuse": row.get("direct_reuse_allowed"),
            "scenario_scope": row.get(
                "scenario_transferability_class"
            ),
            "nearest_validated_value": (
                (row.get("scenario_transferability") or {}).get(
                    "nearest_value"
                )
            ),
            "validated_unit": (
                (row.get("scenario_transferability") or {}).get(
                    "nearest_unit"
                )
            ),
            "relative_difference": (
                (row.get("scenario_transferability") or {}).get(
                    "relative_difference"
                )
            ),
            "compatibility": row.get("deterministic_compatibility"),
            "graph_match": row.get("matching_score"),
            "adaptation_cost": row.get("adaptation_cost"),
            "grade": row.get("run_label_grade"),
            "license": row.get("license_status"),
            "execution": row.get("execution_probe_status"),
            "supported_compounds": list_text(
                row.get("supported_compounds")
            ),
            "routes": list_text(row.get("routes")),
            "blockers": ", ".join(
                row.get("exclusion_reason_codes") or []
            ),
            "qualification_actions": ", ".join(
                row.get("qualification_actions") or []
            ),
        }
        for row in rows
    ]


def candidate_label(row: dict[str, Any]) -> str:
    return (
        f"#{row.get('rank')} {row.get('model_name')} | "
        f"{row.get('project_relative_path')}"
    )


def component_scenario_label(row: dict[str, Any]) -> str:
    request = row.get("request_json") or {}
    context = []
    for label, key in (
        ("interacting victim", "victim_drug"),
        ("interacting perpetrator", "perpetrator_drug"),
        ("population", "population"),
        ("route", "route"),
    ):
        if request.get(key):
            context.append(f"{label}: {request[key]}")
    suffix = str(row.get("recommendation_result_id") or "")[-8:]
    return " | ".join(context) if context else f"validated result {suffix}"


def component_scenario_option(row: dict[str, Any]) -> str:
    objective = str(row.get("objective") or "")
    target = str(row.get("compound_name") or "unknown")
    return (
        f"Target: {target} | {OBJECTIVE_LABELS.get(objective, objective)} | "
        f"{component_scenario_label(row)}"
    )


def component_recommendation_payload(
    detail: dict[str, Any],
) -> dict[str, Any]:
    recommendation = detail.get("recommendation_json")
    return recommendation if isinstance(recommendation, dict) else detail


def render_component_recommendation(detail: dict[str, Any]) -> None:
    recommendation = component_recommendation_payload(detail)
    validation = detail.get("validation_json") or {}
    model_run = detail.get("model_run") or recommendation.get("model_run") or {}
    components = recommendation.get("component_recommendations") or []
    recommended_components = [
        row
        for row in components
        if row.get("status") == "evidence_supported_candidate"
    ]
    citations = recommendation.get("citations") or []
    gaps = recommendation.get("evidence_gaps") or []

    metrics = st.columns(4)
    metrics[0].metric(
        "Evidence support score",
        f"{float(recommendation.get('overall_confidence') or 0):.1%}",
    )
    metrics[1].metric(
        "Evidence-supported candidates",
        len(recommended_components),
    )
    metrics[2].metric("Source citations", len(citations))
    metrics[3].metric("Evidence gaps", len(gaps))
    st.caption(
        "The evidence support score is a deterministic summary of required "
        "component coverage and source diversity. It is not a calibrated "
        "probability that a PBPK model will predict clinical PK correctly."
    )
    st.caption(
        "An evidence-supported candidate is a traceable model requirement, "
        "not a calibrated or simulation-qualified PBPK component."
    )

    if model_run.get("benchmark_level") == "silver":
        st.warning(
            "This result was generated by an engineering-accepted model that "
            "was evaluated on a silver benchmark. The output passed provenance "
            "validation, but it is not a gold clinical reference and does not "
            "establish cross-compound generalization."
        )


    component_tab, parameter_tab, citation_tab, graph_tab, audit_tab = st.tabs(
        [
            "Components",
            "Parameters and mechanisms",
            "Evidence citations",
            "KG retrieval",
            "Validation audit",
        ]
    )
    with component_tab:
        display_table(
            [
                {
                    "component": COMPONENT_LABELS.get(
                        str(row.get("component")),
                        row.get("component"),
                    ),
                    "status": row.get("status"),
                    "evidence_count": len(row.get("evidence_ids") or []),
                    "source_count": row.get("source_count") or 0,
                    "evidence_ids": row.get("evidence_ids") or [],
                }
                for row in components
            ],
            height=340,
        )
        structure = recommendation.get("recommended_model_structure") or {}
        if structure:
            st.json(structure)
        if gaps:
            st.warning("Unresolved evidence gaps remain for this scenario.")
            display_table(
                [{"evidence_gap": value} for value in gaps],
                height=220,
            )

    with parameter_tab:
        parameters = recommendation.get("parameter_candidates") or []
        rejected_parameters = recommendation.get("parameter_candidate_rejections") or []
        st.caption(
            "Only numeric bindings that passed target-compound attribution, "
            "parameter semantics, reported-to-canonical unit conversion, "
            "source-text association, and species/context checks are shown. "
            "They remain evidence candidates and are not calibrated PBPK inputs."
        )
        display_table(
            [
                {
                    "parameter": row.get("parameter"),
                    "reported_value": row.get("reported_value"),
                    "reported_low": row.get("reported_value_low"),
                    "reported_high": row.get("reported_value_high"),
                    "reported_unit": row.get("reported_unit"),
                    "canonical_value": row.get("value"),
                    "canonical_low": row.get("value_low"),
                    "canonical_high": row.get("value_high"),
                    "canonical_unit": row.get("canonical_unit"),
                    "conversion": row.get("unit_conversion_modes") or {},
                    "species_basis": row.get("species_context_basis"),
                    "study_context": row.get("study_context") or {},
                    "dose": row.get("dose"),
                    "route": row.get("route"),
                    "population": row.get("population"),
                    "formulation": row.get("formulation"),
                    "supporting_text": row.get("supporting_text"),
                    "source_id": row.get("source_id"),
                    "evidence_id": row.get("evidence_id"),
                    "source_location": row.get("source_location"),
                }
                for row in parameters
            ],
            height=380,
        )
        if rejected_parameters:
            with st.expander(
                f"Rejected numeric bindings ({len(rejected_parameters)})",
                expanded=False,
            ):
                st.caption(
                    "The evidence citation is retained for component support; "
                    "only the unsafe parameter-to-value binding is excluded."
                )
                display_table(
                    [
                        {
                            "reported_parameter": row.get("reported_parameter"),
                            "reported_value": row.get("reported_value"),
                            "reported_low": row.get("reported_value_low"),
                            "reported_high": row.get("reported_value_high"),
                            "reported_unit": row.get("reported_unit"),
                            "canonical_value": row.get("value"),
                            "canonical_low": row.get("value_low"),
                            "canonical_high": row.get("value_high"),
                            "canonical_unit": row.get("canonical_unit"),
                            "species_basis": row.get(
                                "species_context_basis"
                            ),
                            "study_context": row.get("study_context") or {},
                            "supporting_text": row.get("supporting_text"),
                            "reasons": row.get("reasons") or [],
                            "evidence_id": row.get("evidence_id"),
                            "source_location": row.get("source_location"),
                        }
                        for row in rejected_parameters
                    ],
                    height=300,
                )
        metabolism, transport = st.columns(2)
        with metabolism:
            st.markdown("**Metabolism pathways**")
            display_table(
                [
                    {
                        "name": row.get("preferred_name"),
                        "node_id": row.get("node_id"),
                    }
                    for row in recommendation.get("metabolism_pathways") or []
                ],
                height=250,
            )
        with transport:
            st.markdown("**Transport pathways**")
            display_table(
                [
                    {
                        "name": row.get("preferred_name"),
                        "node_id": row.get("node_id"),
                    }
                    for row in recommendation.get("transport_pathways") or []
                ],
                height=250,
            )

    with citation_tab:
        display_table(
            [
                {
                    "evidence_id": row.get("evidence_id"),
                    "type": row.get("evidence_type"),
                    "parameter": row.get("parameter"),
                    "source": row.get("source_id"),
                    "confidence": row.get("confidence"),
                    "target_compound": (
                        row.get("target_compound") or {}
                    ).get("preferred_name"),
                    "context_status": row.get("context_status"),
                    "structured_measurements": row.get(
                        "structured_measurements"
                    ) or [],
                    "supporting_text": row.get("supporting_text"),
                    "source_location": row.get("source_location"),
                }
                for row in citations
            ],
            height=520,
        )

    with graph_tab:
        display_table(
            [
                {
                    "rank": rank,
                    "subgraph_id": row.get("subgraph_id"),
                    "hop": row.get("hop"),
                    "score": row.get("score"),
                    "score_components": row.get("score_components") or {},
                }
                for rank, row in enumerate(
                    recommendation.get("retrieved_subgraphs") or [],
                    start=1,
                )
            ],
            height=420,
        )
        traversed = recommendation.get("traversed_path_ids") or []
        if traversed:
            with st.expander("Traversed KG paths"):
                display_table(
                    [{"path_id": value} for value in traversed],
                    height=300,
                )

    with audit_tab:
        st.json(
            {
                "result_id": detail.get("recommendation_result_id")
                or recommendation.get("recommendation_result_id"),
                "status": detail.get("status")
                or recommendation.get("status"),
                "generated_from_auto_validated_only": recommendation.get(
                    "generated_from_auto_validated_only"
                ),
                "parameter_value_gate_version": recommendation.get(
                    "parameter_value_gate_version"
                ),
                "accepted_parameter_count": len(
                    recommendation.get("parameter_candidates") or []
                ),
                "rejected_parameter_binding_count": len(
                    recommendation.get("parameter_candidate_rejections") or []
                ),
                "validation": validation,
                "model_run": model_run,
                "recommendation_sha256": detail.get(
                    "recommendation_sha256"
                ),
                "request_sha256": detail.get("request_sha256"),
            }
        )
def render_online_component_request(
    recommendation_summary: dict[str, Any],
    compound_names: list[str],
) -> None:
    supported_objectives = recommendation_summary.get(
        "supported_objectives"
    ) or ["base_pk"]
    normalized_names = [name.casefold() for name in compound_names]
    default_compound_index = (
        normalized_names.index("midazolam")
        if "midazolam" in normalized_names
        else 0
    )
    st.subheader("Generate a validated component recommendation")
    st.caption("Identical requests share generation; other new requests wait for the GPU. Do not submit repeatedly. Completed results are reused first.")
    with st.form("online_component_recommendation_request"):
        left, right = st.columns(2)
        compound = left.selectbox(
            "Component target compound",
            compound_names,
            index=default_compound_index,
        )
        objective = right.selectbox(
            "Component modeling objective",
            supported_objectives,
            format_func=lambda value: OBJECTIVE_LABELS.get(value, value),
        )
        species = left.text_input("Component target species", value="human")
        route = right.text_input("Component administration route")
        population = left.text_input("Component target population")
        victim_drug = right.text_input("DDI victim drug")
        perpetrator_drug = left.text_input("DDI perpetrator drug")
        enzymes = right.text_input("Required enzymes (component request)")
        transporters = left.text_input(
            "Required transporters (component request)"
        )
        constraints = right.text_area("Additional biological constraints")
        submitted = st.form_submit_button(
            "Generate validated component recommendation",
            type="primary",
            width="stretch",
        )

    if submitted:
        payload = {
            "compound": str(compound).strip(),
            "objective": objective,
            "species": species.strip() or "human",
            "route": route.strip() or None,
            "population": population.strip() or None,
            "victim_drug": victim_drug.strip() or None,
            "perpetrator_drug": perpetrator_drug.strip() or None,
            "required_enzymes": comma_values(enzymes),
            "required_transporters": comma_values(transporters),
            "constraints": constraints.strip() or None,
        }
        with st.spinner(
            "Waiting for the GPU or generating components; identical requests share the result. Keep this page open: results appear after retrieval, reranking, generation, and validation.",
            show_time=True,
        ):
            result = safe_call(
                "Generate validated component recommendation",
                api_client.recommend_pbpk_request,
                payload,
            )
        if result:
            st.session_state["online_component_recommendation"] = result

    if st.session_state.get("online_component_recommendation"):
        render_component_recommendation(
            st.session_state["online_component_recommendation"]
        )


def render_result(result: dict[str, Any]) -> None:
    decision = str(result.get("decision") or "abstain")
    summary = result.get("candidate_summary") or {}
    selected = result.get("selected_model") or {}
    top_candidate = (
        result.get("recommended_model_candidate")
        or result.get("top_same_compound_candidate")
        or {}
    )
    recommended_use = str(
        result.get("recommended_use")
        or ("direct_reuse" if decision == "reuse" else "none")
    )

    metrics = st.columns(4)
    metrics[0].metric(
        "Recommended use",
        {
            "direct_reuse": "Direct reuse",
            "controlled_adaptation": "Controlled adaptation",
            "qualification_required": "Qualification required",
            "no_same_compound_recommendation": "No same-compound model",
        }.get(recommended_use, "No recommendation"),
    )
    metrics[1].metric(
        "Catalog scanned",
        summary.get("catalog_scanned", summary.get("retrieved", 0)),
    )
    metrics[2].metric(
        "Relevant candidates",
        summary.get("relevant_candidates", summary.get("retrieved", 0)),
    )
    metrics[3].metric(
        "Unrelated excluded", summary.get("excluded_unrelated", 0)
    )

    layer_metrics = st.columns(4)
    layer_metrics[0].metric(
        "Same-compound", summary.get("same_compound", 0)
    )
    layer_metrics[1].metric(
        "Validated reuse", summary.get("validated_reuse", 0)
    )
    layer_metrics[2].metric(
        "Controlled adaptation",
        summary.get("controlled_adaptation", 0),
    )
    layer_metrics[3].metric(
        "Needs qualification",
        summary.get("same_compound_qualification", 0),
    )

    reference_metrics = st.columns(2)
    reference_metrics[0].metric(
        "Mechanistic references",
        summary.get("mechanistic_reference", 0),
    )
    reference_metrics[1].metric("Blocked", summary.get("blocked", 0))

    if decision == "reuse":
        st.success(
            "A public model passed every deterministic reuse gate for the "
            "registered request scope: "
            f"{selected.get('model_name') or ''}."
        )
        st.caption(
            "This is a scope-bound gate decision, not a calibrated probability "
            "of predictive success for a new clinical scenario."
        )
    elif top_candidate:
        if recommended_use == "controlled_adaptation":
            st.info(
                "Recommended same-compound model for controlled adaptation: "
                f"{top_candidate.get('model_name') or ''}. The model remains "
                "the recommendation; the changed scenario must be rerun and "
                "qualified before it is described as validated reuse."
            )
        else:
            st.warning(
                "A same-compound public model was ranked for qualification, "
                "but unresolved applicability or technical evidence prevents "
                "direct reuse."
            )
    elif summary.get("mechanistic_reference", 0):
        st.warning(
            "No downloaded same-compound public model was found. "
            "Only different-compound models with a provenance-bound shared "
            "disposition process are shown as mechanistic references."
        )
    else:
        st.warning(
            "No relevant public model was found. Catalog versions without "
            "same-compound or verified shared-mechanism evidence were excluded."
        )

    same_compound = result.get("same_compound_candidates") or []
    templates = result.get("mechanistic_templates") or []
    all_candidates = result.get("candidates") or (
        same_compound + templates
    )
    ranking_tab, matching_tab, constraint_tab, asset_tab, validation_tab, audit_tab = st.tabs(
        [
            "Layered Candidates",
            "Graph Compatibility",
            "Hard Constraints",
            "Assets and Provenance",
            "Validation",
            "Decision Audit",
        ]
    )

    with ranking_tab:
        available_layers = [
            layer
            for layer in CANDIDATE_LAYER_LABELS
            if any(
                row.get("candidate_layer") == layer
                for row in all_candidates
            )
        ]
        layer_options = ["all", *available_layers]
        preferred_layer = next(
            (
                str(row.get("candidate_layer"))
                for row in same_compound
                if row.get("candidate_layer") in available_layers
            ),
            "all",
        )
        st.caption(
            "Same-compound layers are the only public-model reuse or "
            "qualification candidates. Mechanistic references are "
            "different-compound process references and cannot substitute "
            "for a target-compound model. Names such as Erythromycin identify "
            "the compound modeled by an external public-model project; they "
            "do not add that compound to the Stage 1 target-compound set."
        )
        selected_layer = st.selectbox(
            "Candidate layer",
            layer_options,
            index=layer_options.index(preferred_layer),
            format_func=lambda value: (
                "All relevant candidates"
                if value == "all"
                else CANDIDATE_LAYER_LABELS[value]
            ),
        )
        visible_candidates = (
            all_candidates
            if selected_layer == "all"
            else [
                row
                for row in all_candidates
                if row.get("candidate_layer") == selected_layer
            ]
        )
        if visible_candidates:
            display_table(
                candidate_table(visible_candidates),
                height=520,
            )
        else:
            status_note("No model version is available in this layer.")

    inspected = same_compound[0] if same_compound else (
        templates[0] if templates else None
    )
    if all_candidates:
        labels = {
            str(row.get("external_model_version_id")): candidate_label(row)
            for row in all_candidates
        }
        selected_key = st.selectbox(
            "Inspect model version",
            list(labels),
            format_func=labels.get,
        )
        inspected = next(
            row
            for row in all_candidates
            if str(row.get("external_model_version_id")) == selected_key
        )

    with matching_tab:
        if inspected:
            matching = inspected.get("cross_graph_matching") or {}
            matching_metrics = st.columns(3)
            matching_metrics[0].metric(
                "Compatibility",
                f"{float(matching.get('score') or 0):.1%}",
            )
            matching_metrics[1].metric(
                "Adaptation cost",
                f"{float(matching.get('adaptation_cost') or 0):.1%}",
            )
            matching_metrics[2].metric(
                "Uncertainty",
                f"{float(matching.get('uncertainty') or 0):.1%}",
            )
            st.caption(
                f"Method: {matching.get('matching_method') or 'unknown'}; "
                f"benchmark: {matching.get('benchmark_level') or 'unknown'}; "
                f"serving scope: {matching.get('serving_scope') or 'unknown'}. "
                "Compatibility is not a probability of successful PBPK "
                "prediction, and hard constraints remain decisive."
            )
            if matching.get("serving_fallback_reason"):
                st.warning(
                    "The learned matcher was unavailable and this candidate "
                    "used the deterministic fallback: "
                    + str(matching["serving_fallback_reason"])
                )
            display_table(
                [
                    {"PBPK hierarchy": level, "coverage": score}
                    for level, score in (
                        matching.get("hierarchy_scores") or {}
                    ).items()
                ],
                height=230,
            )
            operations = matching.get("adaptation_operations") or []
            if operations:
                display_table(operations, height=320)
            else:
                status_note("No graph-level adaptation operation was identified.")
        else:
            status_note("No candidate is available for graph matching inspection.")

    with constraint_tab:
        if inspected:
            ranking_candidates = (
                (result.get("ranking") or {}).get("candidates") or []
            )
            constraint_candidate = next(
                (
                    row
                    for row in ranking_candidates
                    if row.get("source_model_version_key")
                    == inspected.get("external_model_version_id")
                ),
                {},
            )
            display_table(
                [
                    {
                        "constraint": row.get("constraint_id"),
                        "severity": row.get("severity"),
                        "result": row.get("result"),
                        "reason": row.get("reason_code"),
                        "observed": row.get("observed_value"),
                    }
                    for row in constraint_candidate.get(
                        "constraint_checks", []
                    )
                ],
                height=520,
            )
        else:
            status_note("No candidate is available for constraint inspection.")

    with asset_tab:
        if inspected:
            asset_metrics = st.columns(3)
            asset_metrics[0].metric(
                "Simulations", inspected.get("simulation_count") or 0
            )
            asset_metrics[1].metric(
                "Dataset references",
                inspected.get("dataset_reference_count") or 0,
            )
            asset_metrics[2].metric(
                "PKML parse",
                inspected.get("pkml_parse_status") or "unknown",
            )
            display_table(
                [
                    {
                        "project": inspected.get("project_relative_path"),
                        "project_sha256": inspected.get("project_sha256"),
                        "snapshot_sha256": inspected.get(
                            "source_snapshot_sha256"
                        ),
                        "source": inspected.get("source_id"),
                        "repository_commit": inspected.get(
                            "repository_commit"
                        ),
                    }
                ],
                height=150,
            )
            display_table(
                [
                    {"source_location": value}
                    for value in inspected.get("asset_locations") or []
                ],
                height=300,
            )
            if inspected.get("repository_url"):
                st.link_button(
                    "Open source repository",
                    inspected["repository_url"],
                )

    with validation_tab:
        if selected:
            run = selected.get("validation_run") or {}
            validation_metrics = st.columns(4)
            validation_metrics[0].metric(
                "Grade", run.get("run_label_grade") or 0
            )
            validation_metrics[1].metric(
                "Repository holdout datasets",
                run.get("verification_dataset_count") or 0,
            )
            validation_metrics[2].metric(
                "GMFE", f"{float(run.get('pooled_gmfe') or 0):.3f}"
            )
            validation_metrics[3].metric(
                "Within two-fold",
                f"{float(run.get('pooled_two_fold_fraction') or 0):.1%}",
            )
            display_table(
                selected.get("validation_references") or [],
                height=360,
            )
            independence = run.get("validation_independence")
            if independence == "repository_preregistered_holdout":
                st.warning(
                    "Validation used the public model repository's frozen "
                    "model-verification split with preregistered thresholds and "
                    "an independent code audit. It is not an external replication "
                    "performed by an unrelated laboratory."
                )
            scope = selected.get("validation_scope") or {}
            if scope:
                st.caption(
                    "Recommendation is separated from scenario qualification. "
                    "Exact or unit-normalized numerically equivalent inputs can "
                    "retain direct-reuse eligibility; interpolation and "
                    "extrapolation remain visible adaptation candidates."
                )
                display_table(
                    [
                        {
                            "routes": list_text(scope.get("routes")),
                            "formulations": list_text(
                                scope.get("formulations")
                            ),
                            "evaluated_endpoints": list_text(
                                scope.get("evaluated_endpoints")
                            ),
                            "datasets": len(
                                scope.get("dataset_names") or []
                            ),
                            "scenarios": len(scope.get("scenarios") or []),
                        }
                    ],
                    height=150,
                )
        elif inspected:
            st.warning(
                "This candidate has no Grade 3 independent validation run "
                "that is transferable to the request."
            )
            display_table(
                inspected.get("hard_blockers") or [],
                height=360,
            )

    with audit_tab:
        st.json(
            {
                "status": result.get("status"),
                "abstention_reason_codes": result.get(
                    "abstention_reason_codes"
                ),
                "blueprint_status": result.get("blueprint_status"),
                "candidate_summary": summary,
                "relevance_audit": result.get("relevance_audit"),
                "methodology": result.get("methodology"),
                "requirement_profile": result.get("requirement_profile"),
            }
        )
        excluded = result.get("excluded_unrelated_models") or []
        if excluded:
            with st.expander(
                f"Unrelated catalog exclusions ({len(excluded)})"
            ):
                display_table(
                    [
                        {
                            "model": row.get("model_name"),
                            "project": row.get("project_relative_path"),
                            "supported_compounds": list_text(
                                row.get("supported_compounds")
                            ),
                            "reason": list_text(
                                row.get("relevance_reason_codes")
                            ),
                        }
                        for row in excluded
                    ],
                    height=420,
                )


def load_blueprint_workspace(blueprint_id: str) -> dict[str, Any] | None:
    detail = safe_call(
        "Load blueprint detail",
        api_client.blueprint_detail,
        blueprint_id,
    )
    if detail is None:
        return None
    return {
        "detail": detail,
        "versions": safe_call(
            "Load blueprint versions",
            api_client.blueprint_versions,
            blueprint_id,
        )
        or [],
        "parameters": safe_call(
            "Load blueprint parameters",
            api_client.blueprint_parameters,
            blueprint_id,
        )
        or [],
        "processes": safe_call(
            "Load blueprint processes",
            api_client.blueprint_components,
            blueprint_id,
            "process",
        )
        or [],
        "compartments": safe_call(
            "Load blueprint compartments",
            api_client.blueprint_components,
            blueprint_id,
            "compartment",
        )
        or [],
        "artifacts": safe_call(
            "Load blueprint artifacts",
            api_client.blueprint_artifacts,
            blueprint_id,
        )
        or [],
        "simulation_runs": safe_call(
            "Load simulation runs",
            api_client.blueprint_simulation_runs,
            blueprint_id,
        )
        or [],
    }


def render_blueprint_workspace(workspace: dict[str, Any]) -> None:
    detail = workspace["detail"]
    blueprint_id = str(detail["blueprint_id"])
    counts = detail.get("counts") or {}
    versions = workspace.get("versions") or []
    parameters = workspace.get("parameters") or []
    processes = workspace.get("processes") or []
    compartments = workspace.get("compartments") or []
    artifacts = workspace.get("artifacts") or []
    runs = workspace.get("simulation_runs") or []

    st.subheader("Executable blueprint workspace")
    status_metrics = st.columns(4)
    status_metrics[0].metric(
        "Engineering status", detail.get("current_status") or "unknown"
    )
    status_metrics[1].metric(
        "Scientific status", detail.get("scientific_status") or "unknown"
    )
    status_metrics[2].metric("Current parameters", counts.get("parameters", 0))
    status_metrics[3].metric("Simulation runs", counts.get("simulation_runs", 0))

    if detail.get("registration_allowed"):
        st.success("The blueprint is eligible for scientific registration.")
    else:
        blocker_codes = detail.get("blocker_codes") or []
        blocker_text = ", ".join(str(code) for code in blocker_codes)
        st.error(
            "Engineering execution succeeded, but scientific registration is "
            "blocked by the registered scientific gates"
            + (f": {blocker_text}." if blocker_text else ".")
        )
    st.caption(
        "Executable means the locked artifact can be compiled and run in its "
        "registered environment. It does not mean calibrated, externally "
        "validated, or suitable for a new dosing decision."
    )

    action_columns = st.columns(4)
    if action_columns[0].button(
        "Validate blueprint contract",
        key=f"validate-blueprint-{blueprint_id}",
        width="stretch",
    ):
        report = safe_call(
            "Validate blueprint contract",
            api_client.validate_blueprint,
            blueprint_id,
        )
        if report:
            st.session_state["blueprint_validation_report"] = report
    if action_columns[1].button(
        "Load compilation audit",
        key=f"compile-blueprint-{blueprint_id}",
        width="stretch",
    ):
        report = safe_call(
            "Load compilation audit",
            api_client.compile_blueprint,
            blueprint_id,
        )
        if report:
            st.session_state["blueprint_compilation_job"] = report
    if action_columns[2].button(
        "Load simulation audit",
        key=f"run-blueprint-{blueprint_id}",
        width="stretch",
    ):
        report = safe_call(
            "Load simulation audit",
            api_client.run_blueprint,
            blueprint_id,
        )
        if report:
            st.session_state["blueprint_simulation_job"] = report
            workspace["simulation_runs"] = report.get("runs") or runs
            runs = workspace["simulation_runs"]
    package = detail.get("package") or {}
    if package.get("available"):
        package_data = safe_call(
            "Load decision package", api_client.blueprint_package_bytes, blueprint_id
        )
        if package_data is not None:
            action_columns[3].download_button(
                "Download decision package", data=package_data,
                file_name=f"pbpk-blueprint-{blueprint_id}.zip",
                mime="application/zip", width="stretch",
            )
    else:
        action_columns[3].button(
            "Package unavailable",
            disabled=True,
            key=f"package-unavailable-{blueprint_id}",
            width="stretch",
        )

    overview_tab, structure_tab, parameter_tab, simulation_tab, asset_tab, audit_tab = st.tabs(
        [
            "Overview",
            "Structure",
            "Parameters",
            "Simulation Runs",
            "Assets",
            "Audit",
        ]
    )

    with overview_tab:
        count_metrics = st.columns(4)
        count_metrics[0].metric("Compartments", counts.get("compartments", 0))
        count_metrics[1].metric("Processes", counts.get("processes", 0))
        count_metrics[2].metric("Artifacts", counts.get("artifacts", 0))
        count_metrics[3].metric("Outputs", counts.get("outputs", 0))
        display_table(
            [
                {
                    "compound": detail.get("compound_name"),
                    "objective": detail.get("objective"),
                    "template": detail.get("external_template_id"),
                    "template_version": detail.get("template_version"),
                    "runtime_adapter": detail.get("target_runtime_adapter"),
                    "runtime_version": detail.get("runtime_version"),
                }
            ],
            height=150,
        )
        blockers = detail.get("blocker_codes") or []
        if blockers:
            display_table(
                [{"scientific_blocker": value} for value in blockers],
                height=190,
            )

    with structure_tab:
        st.markdown("#### Process inventory")
        display_table(process_summary(processes), height=250)
        with st.expander("Process-level provenance"):
            display_table(process_rows(processes), height=520)
        st.markdown("#### Compartment inventory")
        display_table(compartment_rows(compartments), height=420)

    with parameter_tab:
        fixed_count = sum(row.get("value") is not None for row in parameters)
        estimate_count = sum(
            row.get("initial_value") is not None for row in parameters
        )
        missing_count = sum(
            row.get("value") is None and row.get("initial_value") is None
            for row in parameters
        )
        parameter_metrics = st.columns(3)
        parameter_metrics[0].metric("Fixed or derived", fixed_count)
        parameter_metrics[1].metric("Initial estimates", estimate_count)
        parameter_metrics[2].metric("Missing", missing_count)
        display_table(parameter_rows(parameters), height=520)

    with simulation_tab:
        succeeded = sum(row.get("status") == "succeeded" for row in runs)
        simulation_metrics = st.columns(3)
        simulation_metrics[0].metric("Registered runs", len(runs))
        simulation_metrics[1].metric("Succeeded", succeeded)
        simulation_metrics[2].metric(
            "Output files",
            sum(int(row.get("output_count") or 0) for row in runs),
        )
        display_table(simulation_rows(runs), height=520)
        st.warning(
            "Successful execution proves reproducibility of the locked runtime "
            "asset. It does not prove calibration or predictive validity."
        )

    with asset_tab:
        package_metrics = st.columns(3)
        package_metrics[0].metric("Registered assets", len(artifacts))
        package_metrics[1].metric(
            "Package bytes", package.get("byte_size") or 0
        )
        package_metrics[2].metric(
            "Package checksum", (package.get("sha256") or "missing")[:12]
        )
        display_table(artifact_rows(artifacts), height=540)

    with audit_tab:
        display_table(
            [
                {
                    "version": row.get("blueprint_version"),
                    "status": row.get("status"),
                    "parent_checksum": row.get("parent_blueprint_checksum"),
                    "checksum": row.get("blueprint_checksum"),
                    "source_location": row.get("source_location"),
                    "created_at": row.get("created_at"),
                }
                for row in versions
            ],
            height=300,
        )
        st.json(
            {
                "blueprint_id": blueprint_id,
                "blueprint_checksum": detail.get("blueprint_checksum"),
                "template_checksum": detail.get("template_checksum"),
                "profile_checksum": detail.get("profile_checksum"),
                "package": package,
                "validation_report": detail.get("validation_report_json"),
                "compilation_report": detail.get("compilation_report_json"),
                "validation_action": st.session_state.get(
                    "blueprint_validation_report"
                ),
                "compilation_action": st.session_state.get(
                    "blueprint_compilation_job"
                ),
                "simulation_action": st.session_state.get(
                    "blueprint_simulation_job"
                ),
            }
        )


def render_blueprint_result(result: dict[str, Any]) -> None:
    status = str(result.get("status") or "unknown")
    if status == "model_reuse_selected":
        st.info(
            "A reusable public model was selected, so blueprint generation "
            "was not started."
        )
        return
    if status == "abstain_no_executable_template":
        st.warning(
            "No exact-compound, checksum-locked executable template is "
            "registered. The system abstained instead of assembling a model "
            "from a different compound."
        )
        return
    if status == "abstain_template_incompatible":
        st.warning("The exact-compound template does not satisfy this request.")
        display_table(
            [{"hard_blocker": value} for value in result.get("reason_codes") or []],
            height=220,
        )
        with st.expander("Compatibility audit"):
            st.json(result.get("compatibility_audit") or {})
        return
    if status != "executable_template_available":
        st.error(f"Unexpected blueprint decision status: {status}")
        return
    st.info(
        "A previously registered, checksum-locked exact-compound blueprint "
        "matches the request contract. No new biological parameters or model "
        "structure were generated by this action."
    )
    if result.get("scientific_decision") == "abstain":
        st.warning(
            "Scientific use is blocked. The returned asset is an engineering "
            "execution baseline only."
        )
        display_table(
            [
                {"scientific_blocker": code}
                for code in result.get("scientific_blocker_codes") or []
            ],
            height=180,
        )
    workspace = st.session_state.get("blueprint_workspace")
    if workspace:
        render_blueprint_workspace(workspace)


st.header("PBPK Component Recommendation")
recommendation_summary = safe_call(
    "Load component recommendation run",
    api_client.recommendation_graph_summary,
) or {}
knowledge_base_compounds = safe_call(
    "Load knowledge-base recommendation coverage",
    api_client.recommendation_compounds,
) or []
validated_rows = safe_call(
    "Load validated component recommendations",
    api_client.validated_recommendations,
    limit=500,
) or []

component_metrics = st.columns(4)
component_metrics[0].metric(
    "Validated scenarios",
    recommendation_summary.get(
        "validated_recommendation_count",
        len(validated_rows),
    ),
)
component_metrics[1].metric(
    "KB compounds covered",
    (
        f"{recommendation_summary.get('compound_count', 0)} / "
        f"{recommendation_summary.get('knowledge_base_compound_count', 0)}"
    ),
)
component_metrics[2].metric(
    "Benchmark level",
    str(recommendation_summary.get("benchmark_level") or "unknown"),
)
component_metrics[3].metric(
    "Online inference",
    (
        "Enabled"
        if recommendation_summary.get("online_inference_enabled")
        else "Disabled"
    ),
)

if recommendation_summary.get("serving_mode") == "validated_results_only":
    status_note(
        "The server is in validated-results-only mode. You can inspect Gate 6 "
        "results below; arbitrary new component requests remain disabled until "
        "the server-side LLM/GNN inference runtime and validator are enabled."
    )

status_note(
    "Only recommendations whose compound node matches a compound_identity "
    "record are shown here. External benchmark compounds remain in the model "
    "evaluation audit and are excluded from the knowledge-base workspace."
)

with st.expander("Knowledge-base component recommendation coverage"):
    display_table(
        [
            {
                "compound": row.get("compound_name"),
                "pubchem_cid": row.get("pubchem_cid"),
                "auto_validated_evidence": row.get(
                    "auto_validated_evidence_count", 0
                ),
                "validated_scenarios": row.get(
                    "validated_scenario_count", 0
                ),
                "validated_objectives": list_text(
                    row.get("validated_objectives")
                ),
                "recommendation_status": (
                    "validated_results_available"
                    if row.get("validated_scenario_count")
                    else "not_generated_or_validated"
                ),
            }
            for row in knowledge_base_compounds
        ],
        height=390,
    )

component_compounds = sorted(
    {str(row["compound_name"]) for row in knowledge_base_compounds},
    key=str.casefold,
)

if recommendation_summary.get("online_inference_enabled"):
    render_online_component_request(
        recommendation_summary,
        component_compounds,
    )
    st.divider()

if component_compounds:
    normalized_component_names = [
        name.casefold() for name in component_compounds
    ]
    default_component_index = (
        normalized_component_names.index("midazolam")
        if "midazolam" in normalized_component_names
        else 0
    )
    component_compound = st.selectbox(
        "Component recommendation compound",
        component_compounds,
        index=default_component_index,
    )
    compound_scenarios = [
        row
        for row in validated_rows
        if str(row["compound_name"]) == component_compound
    ]
    coverage_by_name = {
        str(row["compound_name"]).casefold(): row
        for row in knowledge_base_compounds
    }
    selected_coverage = coverage_by_name.get(
        component_compound.casefold(),
        {},
    )
    st.caption(
        f"Validated scenarios for {component_compound}: "
        f"{selected_coverage.get('validated_scenario_count', 0)}"
    )
    if not compound_scenarios:
        st.warning(
            "This knowledge-base compound has auto-validated evidence, but no "
            "component recommendation has completed the accepted Gate 6 "
            "generation and validation pipeline. No substitute compound or "
            "benchmark result will be shown."
        )
    else:
        scenario_keys = [
            str(row["recommendation_result_id"])
            for row in compound_scenarios
        ]
        scenario_by_key = {
            str(row["recommendation_result_id"]): row
            for row in compound_scenarios
        }
        selected_scenario_id = st.selectbox(
            "Validated modeling scenario",
            scenario_keys,
            format_func=lambda value: component_scenario_option(
                scenario_by_key[value]
            ),
        )
        detail_key = "component_recommendation_detail"
        selected_key = "component_recommendation_selected_id"
        if st.session_state.get(selected_key) != selected_scenario_id:
            detail = safe_call(
                "Load validated component recommendation",
                api_client.validated_recommendation_detail,
                selected_scenario_id,
            )
            if detail:
                st.session_state[selected_key] = selected_scenario_id
                st.session_state[detail_key] = detail
        if (
            st.session_state.get(selected_key) == selected_scenario_id
            and st.session_state.get(detail_key)
        ):
            render_component_recommendation(st.session_state[detail_key])
else:
    st.warning(
        "No compound is currently indexed in the Stage 1 knowledge base."
    )

st.divider()
st.header("Public Model Ranking and Executable Blueprint")
catalog = safe_call(
    "Load public model catalog",
    api_client.public_model_catalog_summary,
) or {}
catalog_metrics = st.columns(4)
catalog_metrics[0].metric("Model families", catalog.get("model_count", 0))
catalog_metrics[1].metric(
    "Model versions", catalog.get("model_version_count", 0)
)
catalog_metrics[2].metric(
    "Runtime verified",
    catalog.get("runtime_verified_version_count", 0),
)
catalog_metrics[3].metric(
    "Reuse ready", catalog.get("reusable_version_count", 0)
)

matching_evaluation = safe_call(
    "Load model-matching evaluation",
    api_client.public_model_matching_evaluation,
) or {}
if matching_evaluation.get("status") == "available":
    with st.expander("Current model-ranking experiment and label audit"):
        st.warning(
            matching_evaluation.get("evaluation_notice")
            or "Labels are derived from evidence and model matching rules. Ranking scores "
            "measure agreement with these labels, not independently validated model applicability."
        )
        experiment_metrics = st.columns(4)
        experiment_metrics[0].metric("Evidence compounds", matching_evaluation.get("evidence_compound_count"))
        experiment_metrics[1].metric("Model candidates", matching_evaluation.get("model_count"))
        experiment_metrics[2].metric("Recommendation scenarios", matching_evaluation.get("query_count"))
        experiment_metrics[3].metric("Labeled request-model pairs", matching_evaluation.get("label_count"))
        training_status = matching_evaluation.get("training_status") or {}
        training_configuration = training_status.get("configuration") or {}
        active_methods = training_configuration.get("neural_variants") or []
        if training_status:
            st.caption(
                f"Experiment status: {training_status.get('status', 'unknown')}. "
                f"Selected neural methods: {', '.join(active_methods) or 'none'}."
            )
        results = matching_evaluation.get("results") or {}
        methods = results.get("methods") or []
        if methods:
            st.markdown("**Completed ranking results against rule-derived labels**")
            display_table(
                [
                    {
                        "method": row.get("method"),
                        "completed runs": row.get("runs"),
                        "planned runs": row.get("planned_runs"),
                        "NDCG@10": row.get("ndcg@10_mean"),
                        "MRR": row.get("mrr_mean"),
                        "MAP": row.get("map_mean"),
                    }
                    for row in methods
                ]
            )
        else:
            st.info("The experiment summary is not available yet.")
        if matching_evaluation.get("audit_available"):
            st.caption("The label-generation and held-out split audit is available for this experiment.")
        serving = matching_evaluation.get("serving") or {}
        if serving.get("learned") is False:
            st.caption("Live recommendations currently use the deterministic matching policy.")

if catalog.get("blueprint_generation_enabled"):
    status_note(
        f"{catalog.get('executable_blueprint_template_count', 0)} "
        "exact-compound executable template is registered for strict "
        "post-abstention blueprint lookup."
    )

coverage = catalog.get("compounds") or []
if coverage:
    with st.expander("Indexed compound model coverage"):
        display_table(coverage, height=320)

compound_names = list(component_compounds)
normalized_compound_names = [name.casefold() for name in compound_names]
default_index = (
    normalized_compound_names.index("rosuvastatin")
    if "rosuvastatin" in normalized_compound_names
    else 0
)

st.subheader("Recommendation request")
if compound_names:
    compound = st.selectbox(
        "Target compound",
        compound_names,
        index=default_index,
        key="public_model_target_compound",
        on_change=reset_public_model_request,
    )
else:
    compound = st.text_input(
        "Target compound",
        value="Rosuvastatin",
        key="public_model_target_compound",
    )
request_defaults = public_model_request_defaults(str(compound))
for field, value in request_defaults.items():
    st.session_state.setdefault(f"public_model_{field}", value)
with st.form("public_model_ranking_request"):
    left, right = st.columns(2)
    species = right.text_input(
        "Target species",
        key="public_model_species",
    )
    route = left.text_input(
        "Required route",
        key="public_model_route",
    )
    formulation = right.text_input(
        "Required formulation",
        key="public_model_formulation",
    )
    population = left.text_input(
        "Target population",
        key="public_model_population",
    )
    dose = left.number_input(
        "Required dose",
        min_value=0.0,
        key="public_model_dose",
        help=(
            "A positive dose is required for direct reuse. Zero keeps the "
            "request in candidate-browsing and abstention mode."
        ),
    )
    dose_unit = right.text_input(
        "Dose unit",
        key="public_model_dose_unit",
    )
    frequency = right.text_input(
        "Required frequency",
        key="public_model_frequency",
    )
    regimen_modes = ["unspecified", "single", "multiple"]
    regimen_mode = left.selectbox(
        "Regimen mode",
        regimen_modes,
        key="public_model_regimen_mode",
    )
    model_structures = ["unspecified", "whole_body", "reduced"]
    model_structure = left.selectbox(
        "Required model structure",
        model_structures,
        key="public_model_model_structure",
    )
    enzymes = left.text_input(
        "Required enzymes (comma separated)",
        key="public_model_enzymes",
    )
    transporters = right.text_input(
        "Required transporters (comma separated)",
        key="public_model_transporters",
    )
    endpoints = right.multiselect(
        "Required endpoints",
        [
            "AUC",
            "Cmax",
            "concentration_time_profile",
            "tissue_concentration",
            "fraction_absorbed",
            "renal_excretion",
        ],
        key="public_model_endpoints",
    )
    submitted = st.form_submit_button(
        "Rank downloaded public models",
        type="primary",
        width="stretch",
    )

if submitted:
    payload = {
        "compound": str(compound).strip(),
        "objective": "public_model_reuse",
        "species": species.strip() or "human",
        "route": route.strip() or None,
        "formulation": formulation.strip() or None,
        "population": population.strip() or None,
        "dose": float(dose) if dose > 0 else None,
        "dose_unit": dose_unit.strip() if dose > 0 else None,
        "frequency": frequency.strip() or None,
        "single_or_multiple": (
            regimen_mode if regimen_mode != "unspecified" else None
        ),
        "whole_body_or_reduced_pbpk": (
            model_structure if model_structure != "unspecified" else None
        ),
        "required_enzymes": [
            value.strip() for value in enzymes.split(",") if value.strip()
        ],
        "required_transporters": [
            value.strip()
            for value in transporters.split(",")
            if value.strip()
        ],
        "prediction_endpoints": endpoints,
    }
    result = safe_call(
        "Rank downloaded public models",
        api_client.rank_public_model,
        payload,
    )
    if result:
        st.session_state["public_model_ranking_payload"] = payload
        st.session_state["public_model_ranking_result"] = result
        for key in (
            "blueprint_result",
            "blueprint_workspace",
            "blueprint_validation_report",
            "blueprint_compilation_job",
            "blueprint_simulation_job",
        ):
            st.session_state.pop(key, None)

if st.session_state.get("public_model_ranking_result"):
    st.divider()
    ranking_result = st.session_state["public_model_ranking_result"]
    render_result(ranking_result)
    if ranking_result.get("decision") != "reuse":
        st.divider()
        st.subheader("Exact-compound executable blueprint")
        if st.button(
            "Check registered executable blueprint",
            type="primary",
            width="stretch",
        ):
            request_payload = st.session_state.get(
                "public_model_ranking_payload"
            )
            blueprint_result = safe_call(
                "Check registered executable blueprint",
                api_client.create_blueprint,
                request_payload,
            )
            if blueprint_result:
                st.session_state["blueprint_result"] = blueprint_result
                blueprint = blueprint_result.get("blueprint") or {}
                blueprint_id = blueprint.get("blueprint_id")
                if blueprint_id:
                    workspace = load_blueprint_workspace(str(blueprint_id))
                    if workspace:
                        st.session_state["blueprint_workspace"] = workspace
        if st.session_state.get("blueprint_result"):
            render_blueprint_result(st.session_state["blueprint_result"])
else:
    status_note(
        "Submit a model request to rank downloaded public models. After a "
        "strict abstention, the service checks only registered exact-compound "
        "executable blueprints. It does not synthesize a new model from a "
        "different compound."
    )
