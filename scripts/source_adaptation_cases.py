"""Frozen source-adaptation pilot inputs; no adapter, fitting, or ODE calls.

Case selection reads design metadata only. ``load_observations`` is a separate
evaluation-stage entry point and must only be called after predictions lock.
All observations come from the public source models: this is historical
reconstruction and engineering adaptation, not a new independent benchmark.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = "source-adaptation-pilot-cases-1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_COLLECTION = "artifacts/pbpk-model-assets/public_collection_20260909/"
SOURCES = {
    "Midazolam": {
        "path": _COLLECTION + "Midazolam-Model__cc916b56d5e3/source/Midazolam-Model.json",
        "sha256": "9d6c3a281176e9c52dcc3faa278cced2c251b2b7d28869fc1184fe956fd1b5f3",
        "compound_id": "pubchem:4192", "molecular_weight_g_per_mol": 325.78,
    },
    "Rifampicin": {
        "path": _COLLECTION + "Rifampicin-Model__1565559bf7b7/source/Rifampicin-Model.json",
        "sha256": "aa4516d170623073498bb661c3d231592b13bdeeb16d4d9bbe66af77e2fcd7d2",
        "compound_id": "pubchem:5381226", "molecular_weight_g_per_mol": 822.94,
    },
}

# Membership was fixed from protocol/module coverage, before concentration or
# prediction inspection. Multiple curves of a publication remain one study.
CASE_DEFINITIONS = (
    {
        "case_id": "midazolam_iv_source", "compound_name": "Midazolam",
        "source_simulation": "iv 1 mg (5 min)",
        "target_simulation": "iv 1 mg (5 min)", "request": {},
        "features": ["source_reproduction", "local_overrides", "process_bindings"],
        "observations": ["Hohmann 2015 - iv 1 mg - Midazolam - IV - 1 mg - Plasma - agg. (n=16)"],
    },
    {
        "case_id": "midazolam_iv_microdose", "compound_name": "Midazolam",
        "source_simulation": "iv 1 mg (5 min)",
        "target_simulation": "iv 0.001 mg (5 min)",
        "request": {"dose": {"value": 0.001, "unit": "mg"}},
        "features": ["absolute_dose", "local_overrides"],
        "observations": ["Hohmann 2015 - iv 0.001 mg - Midazolam - IV - 0.001 mg - Plasma - agg. (n=16)"],
    },
    {
        "case_id": "midazolam_source_individual_infusion", "compound_name": "Midazolam",
        "source_simulation": "iv 1 mg (5 min)",
        "target_simulation": "iv 1 mg (2 min)",
        "request": {"individual_name": "Korean (Yu 2004 study)", "infusion_minutes": 2.0},
        "features": ["source_individual_selection", "expression_profiles", "infusion_duration"],
        "observations": ["Yu 2004 - Control CYP3A5*3/*3 - Midazolam - IV - 1 mg - Plasma - agg. (n=7)"],
    },
    {
        "case_id": "midazolam_oral_formulation", "compound_name": "Midazolam",
        "source_simulation": "po 10 mg (solution)",
        "target_simulation": "po 10 mg (tablet)",
        "request": {"formulation_name": "Tablet (Dormicum)"},
        "features": ["source_formulation_selection", "oral_absorption", "override_path_rebinding"],
        "observations": ["Smith 1981 - oral tablet 10 mg - Midazolam - PO - 10 mg - Plasma - agg. (n=6)"],
    },
    {
        "case_id": "midazolam_fed_source", "compound_name": "Midazolam",
        "source_simulation": "po 15 mg (tablet) - with high-fat breakfast",
        "target_simulation": "po 15 mg (tablet) - with high-fat breakfast", "request": {},
        "features": ["source_reproduction", "meal_event_preservation", "local_overrides"],
        "observations": ["Bornemann 1986 - with a meal - Midazolam - PO - 15 mg - Plasma - agg. (n=18)"],
    },
    {
        "case_id": "rifampicin_iv_source", "compound_name": "Rifampicin",
        "source_simulation": "Rifampicin iv 300 mg (0.5 h)",
        "target_simulation": "Rifampicin iv 300 mg (0.5 h)", "request": {},
        "features": ["source_reproduction", "interaction_preservation", "process_bindings"],
        "observations": ["sanofi-aventis U.S. LLC. 2013 - 300 mg - Rifampicin - IV - 300 mg - Plasma - agg. (n=12)"],
    },
    {
        "case_id": "rifampicin_iv_dose_infusion", "compound_name": "Rifampicin",
        "source_simulation": "Rifampicin iv 300 mg (0.5 h)",
        "target_simulation": "Rifampicin iv 600 mg (3 h)",
        "request": {"dose_scale": 2.0, "infusion_minutes": 180.0},
        "features": ["dose_scaling", "infusion_duration", "interaction_preservation"],
        "observations": ["Nitti 1977 - 600 mg - Rifampicin - IV - 600 mg - Serum - agg. (n=6)"],
    },
    {
        "case_id": "rifampicin_repeated_protocol_individual", "compound_name": "Rifampicin",
        "source_simulation": "Rifampicin po 600 mg MD OD (7 days)",
        "target_simulation": "Rifampicin po 450 mg MD OD (7 days)",
        "request": {
            "protocol_name": "po 450 mg MD OD (7 days)",
            "individual_name": "European (P-gp modified, CYP3A4 36 h, EHC off)",
        },
        "features": ["source_protocol_selection", "source_individual_selection", "repeated_dosing", "interaction_preservation"],
        "observations": [
            "Acocella 1972a - day 1 - Rifampicin - PO - 450 mg - Serum - agg. (n=5)",
            "Acocella 1972a - day 7 - Rifampicin - PO - 450 mg - Serum - agg. (n=5)",
        ],
    },
)


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _one(items, name):
    matched = [item for item in items if item.get("Name") == name]
    if len(matched) != 1:
        raise ValueError(f"Expected exactly one source object: {name}")
    return matched[0]


def _read_source(reference, project_root):
    root = Path(project_root).resolve(strict=True)
    path = (root / reference["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or path == root or not path.is_file():
        raise ValueError("Source must be a regular file inside the project")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Source SHA256 changed")
    snapshot = json.loads(data)
    if snapshot.get("Version") != 80:
        raise ValueError("Pilot requires its registered v80 source")
    return snapshot


def _properties(observed):
    return {item["Name"]: item.get("Value") for item in observed.get("ExtendedProperties", [])}


def observation_metadata(observed, source, molecular_weight):
    """Inspect metadata/time coverage; never access concentration Columns.Values."""
    properties = _properties(observed)
    columns = [item for item in observed["Columns"]
               if item.get("DataInfo", {}).get("Origin") == "Observation"]
    if len(columns) != 1:
        raise ValueError("Expected one observed concentration column")
    column = columns[0]
    if observed["BaseGrid"].get("Unit") != "h" or column.get("Unit") != "mg/l":
        raise ValueError("Registered source observation units changed")
    if column.get("Dimension") != "Concentration (mass)":
        raise ValueError("Registered observation is not a mass concentration")
    times = observed["BaseGrid"]["Values"]
    if not times or any(isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0 for value in times):
        raise ValueError("Invalid registered observation time grid")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("Registered time grid must be strictly increasing")
    declared_serum = " - Serum - " in observed["Name"]
    return {
        "path": source["path"], "sha256": source["sha256"],
        "observation_name": observed["Name"], "column_name": column["Name"],
        "study_id": properties["Study Id"], "study_group_id": properties["Reference"],
        "reference": properties["Reference"], "metadata": deepcopy(properties),
        "time_unit": "h", "concentration_unit": "mg/l",
        "observation_count": len(times),
        "required_time_window_min": [times[0] * 60.0, times[-1] * 60.0],
        "source_time_grid_sha256": _sha_json(times),
        "molecular_weight_g_per_mol": molecular_weight,
        "tier": "reconstruction", "role": "heldout_evaluation",
        "historical_source_overlap": True, "independent_blind_evaluation": False,
        "source_fitting_membership": "not_individually_audited",
        "observation_values_inspected_for_case_selection": False,
        "time_mapping": "source_absolute_grid_no_time_offset_no_inferred_shift",
        "matrix_mapping": {
            "label": "Serum" if declared_serum else properties.get("Compartment"),
            "source_metadata": properties.get("Compartment"),
            "model_output": "Plasma",
            "status": "source_serum_to_plasma_mapping_assumption" if declared_serum else "matched_metadata",
            "strict_matrix_validation": not declared_serum,
        },
    }


def _parameter(parameters, name):
    return _one(parameters, name)


def _minutes(parameter):
    factors = {"h": 60.0, "min": 1.0, "s": 1.0 / 60.0}
    if parameter.get("Unit") not in factors:
        raise ValueError("Unsupported registered protocol time unit")
    return parameter["Value"] * factors[parameter["Unit"]]


def _applications(protocol):
    """Expand only source schema metadata, never construct a new regimen."""
    if "Schemas" not in protocol:
        if protocol.get("DosingInterval") != "Single":
            raise ValueError("Unregistered simple repeated protocol")
        return [{"route": protocol["ApplicationType"],
                 "parameters": deepcopy(protocol["Parameters"]),
                 "start_min": _minutes(_parameter(protocol["Parameters"], "Start time"))}]
    result = []
    for schema in protocol["Schemas"]:
        params = schema["Parameters"]
        repetitions = _parameter(params, "NumberOfRepetitions")["Value"]
        if isinstance(repetitions, bool) or not float(repetitions).is_integer() or not 1 <= repetitions <= 32:
            raise ValueError("Invalid registered repetition count")
        start = _minutes(_parameter(params, "Start time"))
        interval = _minutes(_parameter(params, "TimeBetweenRepetitions"))
        for repetition in range(int(repetitions)):
            for item in schema["SchemaItems"]:
                result.append({"route": item["ApplicationType"],
                               "parameters": deepcopy(item["Parameters"]),
                               "start_min": start + repetition * interval + _minutes(_parameter(item["Parameters"], "Start time"))})
    return sorted(result, key=lambda item: item["start_min"])


def _native_expectations(snapshot, source_simulation, target_simulation, request):
    """Audit against independent target protocol values at each arm's paths."""
    source_selection = source_simulation["Compounds"][0]["Protocol"]
    target_selection = target_simulation["Compounds"][0]["Protocol"]
    target_protocol = _one(snapshot["Protocols"], target_selection["Name"])
    apps = _applications(target_protocol)
    result = {}
    for arm in ("adapted", "official_reference"):
        protocol_name = (request.get("protocol_name", source_selection["Name"])
                         if arm == "adapted" else target_selection["Name"])
        formulations = target_selection.get("Formulations", [])
        expectations = []
        for index, application in enumerate(apps, 1):
            parts = ["Events", protocol_name]
            if application["route"] == "Oral":
                if len(formulations) != 1:
                    raise ValueError("Expected exactly one registered oral formulation")
                parts.append(formulations[0]["Name"])
            parts.extend([f"Application_{index}", "ProtocolSchemaItem"])
            prefix = "|".join(parts)
            dose = _parameter(application["parameters"], "InputDose")
            if dose["Unit"] != "mg":
                raise ValueError("Pilot expects registered absolute mg doses")
            expectations.extend([
                {"path": prefix + "|Dose", "value": dose["Value"], "unit": "mg"},
                {"path": prefix + "|Start time", "value": application["start_min"], "unit": "min"},
            ])
            if application["route"] == "Intravenous":
                duration = _minutes(_parameter(application["parameters"], "Infusion time"))
                expectations.append({"path": prefix + "|Infusion time", "value": duration, "unit": "min"})
        result[arm] = {"expectations": expectations, "expected_event_count": len(apps)}
    return result


def build_cases(project_root=PROJECT_ROOT):
    """Return eight source-pinned requests and independent reference selectors."""
    snapshots = {name: _read_source(source, project_root) for name, source in SOURCES.items()}
    cases = []
    for definition in CASE_DEFINITIONS:
        name = definition["compound_name"]
        source = SOURCES[name]
        snapshot = snapshots[name]
        source_sim = _one(snapshot["Simulations"], definition["source_simulation"])
        target_sim = _one(snapshot["Simulations"], definition["target_simulation"])
        compound = _one(snapshot["Compounds"], name)
        mw = _parameter(compound["Parameters"], "Molecular weight")
        if mw.get("Unit") != "g/mol" or mw["Value"] != source["molecular_weight_g_per_mol"]:
            raise ValueError("Registered molecular weight changed")
        request = deepcopy(definition["request"])
        target_output = f"Organism|PeripheralVenousBlood|{name}|Plasma (Peripheral Venous Blood)"
        if target_output not in target_sim["OutputSelections"] or target_output not in source_sim["OutputSelections"]:
            raise ValueError("Registered plasma output is absent")
        observation_refs = []
        for observation_name in definition["observations"]:
            if observation_name not in target_sim.get("ObservedData", []):
                raise ValueError("Observation is not registered against official target")
            observed = _one(snapshot["ObservedData"], observation_name)
            mappings = [item for item in target_sim.get("OutputMappings", [])
                        if item.get("ObservedData") == observation_name]
            if len(mappings) != 1 or mappings[0].get("Path") != target_sim["Name"] + "|" + target_output:
                raise ValueError("Official observation/output mapping changed")
            if any("offset" in key.lower() for key in mappings[0]):
                raise ValueError("Source time offset needs explicit registration before scoring")
            observation_refs.append(observation_metadata(observed, source, mw["Value"]))
        cases.append({
            "case_id": definition["case_id"], "compound_id": source["compound_id"],
            "compound_name": name, "tier": "reconstruction",
            "source": {"path": source["path"], "sha256": source["sha256"],
                       "simulation_name": source_sim["Name"]},
            "request": request,
            "engineering_reference": {"path": source["path"], "sha256": source["sha256"],
                                      "simulation_name": target_sim["Name"],
                                      "kind": "unmodified_official_target_simulation",
                                      "adapter_used_to_construct_reference": False},
            "target_output": target_output, "features": deepcopy(definition["features"]),
            "observation_refs": observation_refs,
            "native_expectations": _native_expectations(snapshot, source_sim, target_sim, request),
            "preservation_expectations": {
                "source_simulation_parameter_count": len(source_sim.get("Parameters", [])),
                "source_selected_process_count": len(source_sim["Compounds"][0].get("Processes", [])),
                "source_defined_process_count": len(compound.get("Processes", [])),
                "source_interaction_count": len(source_sim.get("Interactions", [])),
                "source_nondosing_event_count": len(source_sim.get("Events", [])),
            },
            "reference_output_schema": deepcopy(target_sim["OutputSchema"]),
            "clinical_metric_policy": "curve_log_rmse_observed_window_auc_sampled_cmax_separately",
            "selection_used_observation_values": False, "independent_blind_evaluation": False,
        })
    return cases


def build_case_manifest(project_root=PROJECT_ROOT):
    cases = build_cases(project_root)
    manifest = {
        "schema_version": SCHEMA_VERSION, "sources": deepcopy(SOURCES), "cases": cases,
        "evaluation_contract": {
            "case_count": len(cases), "compound_count": 2,
            "official_target_simulation_count": len({(c["compound_name"], c["engineering_reference"]["simulation_name"]) for c in cases}),
            "historical_curve_count": sum(len(c["observation_refs"]) for c in cases),
            "historical_study_count": len({o["study_group_id"] for c in cases for o in c["observation_refs"]}),
            "case_selection": "source_protocol_and_module_coverage_before_outcome_inspection",
            "observation_access": "only_after_all_predictions_are_locked",
            "engineering_reference": "direct_official_source_not_adapter_output",
            "scientific_tier": "historical_reconstruction_not_independent_blind_test",
            "failures": "retain_all_cases_and_all_observation_validation_failures",
            "endpoints": ["curve_log_rmse", "observed_window_auc_log_error", "sampled_cmax_log_error"],
            "weighted_composite": False, "solver_calls": 0,
        },
    }
    manifest["manifest_sha256"] = _sha_json(manifest)
    return manifest


def build_reference_snapshot(case, project_root=PROJECT_ROOT):
    """Select the untouched official target; only remove analysis attachments.

    This deliberately does not import or call source-model adaptation code.
    Model, solver, output schema, all building blocks, binding, expression,
    interactions, events and simulation parameter overrides are retained.
    """
    reference = case["engineering_reference"]
    snapshot = _read_source(reference, project_root)
    simulation = deepcopy(_one(snapshot["Simulations"], reference["simulation_name"]))
    for key in ("ObservedData", "OutputMappings", "IndividualAnalyses", "HasResults"):
        simulation.pop(key, None)
    snapshot["Simulations"] = [simulation]
    for key in ("ObservedData", "ObservedDataClassifications", "SimulationClassifications", "ParameterIdentifications"):
        snapshot.pop(key, None)
    return snapshot


def load_observations(case, project_root=PROJECT_ROOT):
    """Evaluation only: caller must lock ALL predictions before invoking this.

    No point deletion or inferred day-7 time shift is allowed. Every registered
    observation yields a scenario, including malformed/unsupported curves.
    """
    from pbpk_kb.blueprints.multidrug_cases import _external_observations

    snapshot = _read_source(case["source"], project_root)
    results = []
    for index, reference in enumerate(case["observation_refs"], 1):
        scenario = {
            "scenario_id": f"{case['case_id']}__observation_{index}",
            "compound_id": case["compound_id"], "study_group_id": reference["study_group_id"],
            "source_study_id": reference["study_id"],
            "source_observation_name": reference["observation_name"],
            "role": "heldout_evaluation", "tier": "reconstruction",
            "target_output": case["target_output"], "time_unit": "min", "unit": "umol/l",
            "matrix_mapping": deepcopy(reference["matrix_mapping"]),
            "source": deepcopy(reference), "historical_source_overlap": True,
            "independent_blind_evaluation": False, "observations": [],
            "auc_evaluable": False, "cmax_evaluable": False,
        }
        try:
            observed = _one(snapshot["ObservedData"], reference["observation_name"])
            current = observation_metadata(observed, case["source"], reference["molecular_weight_g_per_mol"])
            if current != reference:
                raise ValueError("Observation metadata differs from frozen case")
            points = _external_observations(observed, reference["molecular_weight_g_per_mol"])
            for point_index, point in enumerate(points):
                point["observation_id"] = f"{scenario['scenario_id']}:{point_index}"
            if len(points) < 2:
                raise ValueError("At least two observations are required for registered endpoints")
            scenario.update(status="ready", observations=points, auc_evaluable=True, cmax_evaluable=True)
        except (KeyError, TypeError, ValueError) as error:
            scenario.update(status="observation_validation_failed", errors=[str(error)])
        results.append(scenario)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New project-local manifest JSON; no execution")
    args = parser.parse_args(argv)
    manifest = build_case_manifest(PROJECT_ROOT)
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
        return 0
    output = (PROJECT_ROOT / args.output).resolve()
    if not output.is_relative_to(PROJECT_ROOT) or output == PROJECT_ROOT:
        raise ValueError("Manifest output must stay inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"path": str(output), "manifest_sha256": manifest["manifest_sha256"],
                      "case_count": len(manifest["cases"]), "solver_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
