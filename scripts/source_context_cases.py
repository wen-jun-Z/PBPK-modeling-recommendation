"""Frozen inputs for complete-source context adaptation and oral calibration.

No model execution, fitting, or candidate selection happens here. All data are
historical development data. The official sources already used many of these
studies in their own parameter identification; splitting our fitting studies
does not turn them into an independent validation set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "_source_context_legacy_cases", Path(__file__).with_name("source_adaptation_cases.py"))
_legacy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)
SCHEMA_VERSION = "source-context-development-cases-1.1"
ROLES = {"calibration", "selection", "evaluation", "sensitivity", "diagnostic"}

_NEW_CASES = (
    ("rifampicin_cal_chouchane_300", "Rifampicin", "calibration", "Rifampicin po 300 mg", (
        "Chouchane 1995 - Rifampicin Generic - Rifampicin - PO - 300 mg - Plasma - agg. (n=12)",
        "Chouchane 1995 - Rimactan - Rifampicin - PO - 300 mg - Plasma - agg. (n=12)")),
    ("rifampicin_cal_peloquin_600", "Rifampicin", "calibration", "Rifampicin po 600 mg", (
        "Peloquin 1997 - 600 mg - Rifampicin - PO - 600 mg - Plasma - agg. (n=24)",)),
    ("rifampicin_sel_blume_450", "Rifampicin", "selection", "Rifampicin po 450 mg", (
        "Blume 1989 - 450 mg REFERENZ - Rifampicin - PO - 450 mg - Plasma - agg. (n=8)",)),
    ("rifampicin_sel_blume_600", "Rifampicin", "selection", "Rifampicin po 600 mg", (
        "Blume 1989 - 600 mg REFERENZ - Rifampicin - PO - 600 mg - Plasma - agg. (n=12)",)),
    ("rifampicin_eval_baneyx_repeat", "Rifampicin", "evaluation", "Rifampicin po 600 mg MD OD (7 days)", (
        "Baneyx 2014 - Day 4 - 600 mg po MD - Rifampicin - PO - 600 mg - Plasma - agg. (n=12)",
        "Baneyx 2014 - Day 7 - 600 mg po MD - Rifampicin - PO - 600 mg - Plasma - agg. (n=12)")),
    ("rifampicin_eval_chattopadhyay_repeat", "Rifampicin", "evaluation", "Rifampicin po 600 mg Chattopadhyay 2018", (
        "Chattopadhyay 2018 - Rifampicin - Rifampicin - PO - 600 mg - Plasma - agg. (n=11)",)),
    ("midazolam_food_fasted", "Midazolam", "sensitivity", "po 15 mg (tablet)", (
        "Bornemann 1986 - fasting condition - Midazolam - PO - 15 mg - Plasma - agg. (n=18)",)),
    ("midazolam_food_before", "Midazolam", "sensitivity", "po 15 mg (tablet) - with 1 h before high-fat breakfast", (
        "Bornemann 1986 - 1 h before a meal - Midazolam - PO - 15 mg - Plasma - agg. (n=18)",)),
    ("midazolam_food_after", "Midazolam", "sensitivity", "po 15 mg (tablet) - with 1h after high-fat breakfast", (
        "Bornemann 1986 - 1 h after a meal - Midazolam - PO - 15 mg - Plasma - agg. (n=18)",)),
)


def _source_pi_membership(snapshot, observation_name):
    """Record exact official mappings, including duplicate and zero weights."""
    entries = []
    for identification in snapshot.get("ParameterIdentifications", []):
        for mapping in identification.get("OutputMappings", []):
            if mapping.get("ObservedData") == observation_name:
                entries.append({"identification_name": identification["Name"],
                                "weight": mapping.get("Weight", 1.0),
                                "weight_explicit": "Weight" in mapping,
                                "output_path": mapping.get("Path")})
    return {"registered_in_source_identification": bool(entries),
            "positive_weight_mapping_present": any(e["weight"] > 0 for e in entries),
            "mappings": entries,
            "final_fit_usage_proven_by_registration_alone": False}


def _new_case(definition, snapshots):
    case_id, name, role, simulation_name, observation_names = definition
    source = _legacy.SOURCES[name]
    snapshot = snapshots[name]
    simulation = _legacy._one(snapshot["Simulations"], simulation_name)
    target_output = f"Organism|PeripheralVenousBlood|{name}|Plasma (Peripheral Venous Blood)"
    if target_output not in simulation["OutputSelections"]:
        raise ValueError("Registered plasma output is absent")
    references = []
    for observation_name in observation_names:
        if observation_name not in simulation.get("ObservedData", []):
            raise ValueError("Observation is not registered against official simulation")
        mappings = [x for x in simulation.get("OutputMappings", [])
                    if x.get("ObservedData") == observation_name]
        if len(mappings) != 1 or mappings[0].get("Path") != simulation_name + "|" + target_output:
            raise ValueError("Official observation/output mapping changed")
        if any("offset" in key.lower() for key in mappings[0]):
            raise ValueError("Time offsets require an explicit frozen mapping")
        observed = _legacy._one(snapshot["ObservedData"], observation_name)
        references.append(_legacy.observation_metadata(
            observed, source, source["molecular_weight_g_per_mol"]))
    return {
        "case_id": case_id, "compound_id": source["compound_id"], "compound_name": name,
        "role": role, "tier": "reconstruction",
        "source": {"path": source["path"], "sha256": source["sha256"],
                   "simulation_name": simulation_name}, "request": {},
        "engineering_reference": {"path": source["path"], "sha256": source["sha256"],
                                  "simulation_name": simulation_name,
                                  "kind": "unmodified_official_target_simulation",
                                  "adapter_used_to_construct_reference": False},
        "target_output": target_output, "observation_refs": references,
        "native_expectations": _legacy._native_expectations(snapshot, simulation, simulation, {}),
        "reference_output_schema": deepcopy(simulation["OutputSchema"]),
        "selection_used_observation_values": False, "independent_blind_evaluation": False,
    }


def build_cases(project_root=PROJECT_ROOT):
    """Return 17 frozen metadata cases, retaining all nine prior pilot curves."""
    snapshots = {name: _legacy._read_source(source, project_root)
                 for name, source in _legacy.SOURCES.items()}
    cases = _legacy.build_cases(project_root)
    for case in cases:
        case["role"] = {"rifampicin_repeated_protocol_individual": "evaluation",
                        "midazolam_fed_source": "sensitivity"}.get(case["case_id"], "diagnostic")
        case["prior_pilot_curve"] = True
    cases.extend(_new_case(definition, snapshots) for definition in _NEW_CASES)
    for case in cases:
        snapshot = snapshots[case["compound_name"]]
        case["study_ids"] = sorted({x["study_id"] for x in case["observation_refs"]})
        for reference in case["observation_refs"]:
            observed = _legacy._one(snapshot["ObservedData"], reference["observation_name"])
            reference["required_times_min"] = [value * 60.0 for value in observed["BaseGrid"]["Values"]]
        case["required_times_min"] = sorted({time for reference in case["observation_refs"]
                                              for time in reference["required_times_min"]})
        case["source_fitting_audit"] = {
            x["observation_name"]: _source_pi_membership(snapshot, x["observation_name"])
            for x in case["observation_refs"]}
        case["scientific_status"] = "historical_development_not_independent_validation"
        case["clinical_metric_policy"] = "curve_log_rmse_observed_window_auc_sampled_cmax_separately"
        case["context_evidence"] = []
        case["context_request"] = {}
        case["sampling_type"] = "concentration_time_profile"
        case["endpoint_eligibility"] = {"auc_evaluable": True, "cmax_evaluable": True,
                                        "rule": "observed_window_auc_and_sampled_cmax"}
        case["calibration_eligible"] = True
        case["calibration_ineligibility_reason"] = None
        if case["case_id"] == "rifampicin_cal_peloquin_600":
            case["calibration_eligible"] = False
            case["calibration_ineligibility_reason"] = "initial_condition_conflict"
            case["initial_condition_conflict"] = {
                "time_h": 0.0, "observed_concentration_mg_per_l": 0.2,
                "source_has_prior_dose": False,
                "resolution": "retain_raw_curve_report_data_quality_failure_do_not_fit_or_silently_delete",
                "time_offset_evidence_available": False,
            }
        if case["role"] == "sensitivity":
            with_meal = case["case_id"] != "midazolam_food_fasted"
            case["allowed_comparisons"] = (["source", "context"] if with_meal else ["source"])
            case["solubility_sensitivity"] = {
                "group_name": "COMPOUND_SOLUBILITY", "alternatives": ["FaSSIF", "FeSSIF"],
                "selection_allowed": False,
                "interpretation": "paired_sensitivity_not_evidence_that_FeSSIF_is_correct",
                "fasted_arm_role": "unchanged_FaSSIF_control_not_exposed_to_FeSSIF",
                "apply_FeSSIF_hypothesis": with_meal,
                "meal_recipe_verified": False,
                "reference": "https://pubmed.ncbi.nlm.nih.gov/2936766/"}
            if with_meal:
                case["context_request"] = {"solubility_alternative": {
                    "value": "FeSSIF", "evidence": {
                        "path": case["source"]["path"], "sha256": case["source"]["sha256"],
                        "locator": "Compounds[Name=Midazolam].Solubility[Name=FeSSIF]; measured source alternative, hypothesis only"}}}
        else:
            case["allowed_comparisons"] = ["source", "context", "calibrated"]
        if case["case_id"] == "rifampicin_repeated_protocol_individual":
            case["context_evidence"] = [{
                "field": "Gender", "value": "FEMALE", "scope": "selected_source_individual",
                "source_value": "MALE", "strength": "official_model_report_study_description",
                "reference": "https://www.open-systems-pharmacology.org/OSP-PBPK-Model-Library/Rifampicin/",
                "description": "Acocella 1972a: female patients after cholecystectomy with external T-tube bile collection",
                "preserve": ["source EHC-off state", "full expression profiles", "all seven interactions"],
                "not_known": ["study mean age", "study mean weight", "study mean height"],
                "unknown_policy": "retain_explicitly_disclosed_source_defaults_not_imputed_study_measurements"}]
            report = (Path(case["source"]["path"]).parent /
                      "Evaluation/Input/Content/Section2.1_Modeling_Strategy.md")
            report_path = (Path(project_root).resolve() / report).resolve(strict=True)
            if not report_path.is_relative_to(Path(project_root).resolve()):
                raise ValueError("Context evidence must stay in project")
            case["context_request"] = {"human_sex": {"value": "FEMALE", "evidence": {
                "path": str(report), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "locator": "paragraph 4: female patients after cholecystectomy (Acocella 1972a)"}}}
        if case["case_id"] == "rifampicin_eval_chattopadhyay_repeat":
            # Methods explicitly identifies rifampicin samples as troughs.
            # Connecting these across 11 dosing days is not a PK exposure AUC,
            # and their maximum is not a sampled drug peak. This post-run
            # scientific correction does not alter or remove the poor pointwise
            # GMFE result, any observation, or the already executed prediction.
            case["sampling_type"] = "longitudinal_predose_trough_sequence"
            case["endpoint_eligibility"] = {
                "auc_evaluable": False, "cmax_evaluable": False,
                "rule": "trough_sequence_not_a_continuous_PK_profile",
                "pointwise_log_rmse_and_gmfe_evaluable": True,
                "reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6256003/",
                "locator": "Methods / Study assessments and variables: RIF trough concentrations",
                "correction_timing": "after_original_locked_run; endpoint semantics only, no refit or new predictions",
            }
            case["endpoint_review"] = {
                **deepcopy(case["endpoint_eligibility"]),
                "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6256003/",
                "reason": "Rifampicin observations are longitudinal trough concentrations, not a continuous profile for exposure AUC or peak Cmax",
            }
            case["context_evidence"].append({
                "field": "study_demographics", "status": "documented_source_context_mismatch_not_changed",
                "reference": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6256003/",
                "locator": "Results / Study subjects; Methods / Design and treatments",
                "study_population": "healthy postmenopausal Caucasian women",
                "study_age_years": {"range": [51, 64], "median": 57.0},
                "study_weight_kg": {"range": [55.0, 87.6], "median": 71.5},
                "demographic_summary_population": "12 enrolled women; PK analysis includes 11 completers",
                "source_gender": "MALE", "source_age_years": 30,
                "source_native_weight_kg_approx": 73.0,
                "rifampicin_meal_context": "30 min after a light meal; source has no meal event",
                "action": "retain_original_predictions_do_not_change_context_after_seeing_evaluation",
            })
    roles_by_study = {}
    for case in cases:
        for reference in case["observation_refs"]:
            study = reference["study_group_id"]
            previous = roles_by_study.setdefault(study, case["role"])
            if previous != case["role"]:
                raise ValueError(f"Study split leakage: {study}")
    return cases


def rifampicin_pint_spec(project_root=PROJECT_ROOT):
    source = _legacy.SOURCES["Rifampicin"]
    snapshot = _legacy._read_source(source, project_root)
    compound = _legacy._one(snapshot["Compounds"], "Rifampicin")
    alternative = _legacy._one(compound["IntestinalPermeability"], "Optimized")
    parameter_name = "Specific intestinal permeability (transcellular)"
    parameter = _legacy._one(alternative["Parameters"], parameter_name)
    if parameter.get("Unit") != "cm/min" or parameter["Value"] != 1.24e-5:
        raise ValueError("Registered rifampicin final source permeability changed")
    pi = _legacy._one(snapshot["ParameterIdentifications"], "PI OSP10")
    bounds = _legacy._one(pi["IdentificationParameters"], "P int")["Parameters"]
    low = _legacy._one(bounds, "MinValue")["Value"]
    high = _legacy._one(bounds, "MaxValue")["Value"]
    if (low, high) != (3.48e-7, .00348):
        raise ValueError("Registered source PI bounds changed")
    return {
        "compound_name": "Rifampicin", "group_name": "IntestinalPermeability",
        "alternative_name": "Optimized", "parameter_name": parameter_name,
        "unit": "cm/min", "source_value": parameter["Value"],
        "source_pi": "PI OSP10", "source_pi_bounds": [low, high],
        "search_bounds": [parameter["Value"] / 4, parameter["Value"] * 4],
        "search_scale": "log", "search_window_interpretation": "computational_window_not_physiological_confidence_interval",
        "free_parameter_count": 1, "prior_center": "final_source_value",
        "unchanged_modules": ["clearance", "partition", "enzyme_and_transporter_induction", "protocols", "interactions"],
        "source": deepcopy(source),
    }


def build_case_manifest(project_root=PROJECT_ROOT):
    cases = build_cases(project_root)
    manifest = {
        "schema_version": SCHEMA_VERSION, "sources": deepcopy(_legacy.SOURCES), "cases": cases,
        "rifampicin_pint_spec": rifampicin_pint_spec(project_root),
        "split_contract": {
            "case_count": len(cases), "historical_curve_count": sum(len(c["observation_refs"]) for c in cases),
            "case_count_by_role": dict(Counter(c["role"] for c in cases)),
            "curve_count_by_role": dict(Counter(c["role"] for c in cases for _ in c["observation_refs"])),
            "study_ids_by_role": {role: sorted({s for c in cases if c["role"] == role for s in c["study_ids"]})
                                  for role in sorted(ROLES)},
            "study_roles_disjoint": True,
            "aggregation": "points_within_curve_then_equal_curves_within_study_then_equal_studies",
            "observation_access": "calibration_and_selection_explicit_roles_evaluation_only_after_all_predictions_lock",
            "historical_source_overlap": True, "independent_blind_evaluation": False,
            "prior_pilot_outcomes_already_seen": True,
            "calibration_registered_curve_count": 3,
            "calibration_eligible_curve_count": 2,
            "calibration_data_quality_failure_curve_count": 1,
            "no_selection_on_bornemann": True, "no_tuning_on_evaluation": True,
            "solver_calls": 0, "weighted_composite": False,
            "failures": "retain_all_scenarios_and_validation_failures",
        },
    }
    manifest["manifest_sha256"] = _legacy._sha_json(manifest)
    return manifest


def build_reference_snapshot(case, project_root=PROJECT_ROOT):
    return _legacy.build_reference_snapshot(case, project_root)


def load_observations(case, project_root=PROJECT_ROOT, *, role, predictions_locked=False):
    """Role-isolated decoding; caller supplies the real persisted lock state.

    Metadata loaders never inspect concentration arrays. This function does;
    non-fitting roles require a prediction lock. The boolean is a guard against
    accidental access, not cryptographic proof of the runner's lock chronology.
    """
    if role not in ROLES or case.get("role") != role:
        raise ValueError("Observation role does not match the frozen case")
    if role not in {"calibration", "selection"} and predictions_locked is not True:
        raise ValueError("Non-fitting observation access requires all predictions locked")
    # The frozen older loader checks exact metadata equality; pass its original
    # schema, while the source hash and source_time_grid_sha256 still pin times.
    legacy_case = deepcopy(case)
    for reference in legacy_case["observation_refs"]:
        reference.pop("required_times_min", None)
    rows = _legacy.load_observations(legacy_case, project_root)
    for row in rows:
        row["dataset_role"] = role
        row["role"] = role if role in {"calibration", "selection"} else "heldout_evaluation"
        row["study_role"] = row["role"]
        row["source_fitting_audit"] = deepcopy(case["source_fitting_audit"][row["source_observation_name"]])
        row["scientific_status"] = "historical_development_not_independent_validation"
        row["calibration_eligible"] = case["calibration_eligible"]
        row["calibration_ineligibility_reason"] = case["calibration_ineligibility_reason"]
        row["sampling_type"] = case["sampling_type"]
        row["endpoint_protocol"] = deepcopy(case["endpoint_eligibility"])
        row["auc_evaluable"] = row["auc_evaluable"] and case["endpoint_eligibility"]["auc_evaluable"]
        row["cmax_evaluable"] = row["cmax_evaluable"] and case["endpoint_eligibility"]["cmax_evaluable"]
        if not case["calibration_eligible"]:
            row["data_quality_status"] = "initial_condition_conflict"
            row["initial_condition_conflict"] = deepcopy(case["initial_condition_conflict"])
        if any(point["time_min"] == 0 and point["concentration_umol_per_l"] > 0
               for point in row.get("observations", [])):
            row.setdefault("warnings", []).append("positive_concentration_at_time_zero_retained_without_deletion")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest = build_case_manifest(PROJECT_ROOT)
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        output = (PROJECT_ROOT / args.output).resolve()
        if not output.is_relative_to(PROJECT_ROOT) or output == PROJECT_ROOT:
            raise ValueError("Manifest output must remain inside the project")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        print(json.dumps({"path": str(output), "manifest_sha256": manifest["manifest_sha256"],
                          "case_count": len(manifest["cases"]), "solver_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
