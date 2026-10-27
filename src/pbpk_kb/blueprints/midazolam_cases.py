"""Source-locked Midazolam study preparation; no native/GPU execution.

This is historical reconstruction, not an independent-main benchmark. Existing
native MM capacities stay fixed: current typed shared-factor edits cannot couple
different enzymes. Only the source PI lipophilicity parameter is estimated.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .cohort_cases import _formula_counts, _normalise_protocol, _output_schema, _slug, qualify_sampled_endpoints
from .development_case import DevelopmentCaseError, _one, _plain, _reference, _verify
from .design_contracts import BlueprintDesignRequest
from .human_model import load_standard_human_model
from .human_model_snapshot import build_human_model_snapshot
from .multidrug_cases import _external_observations, _metadata, _write_once

SOURCE = ("artifacts/pbpk-model-assets/public_collection_20260909/Midazolam-Model__cc916b56d5e3/source/Midazolam-Model.json",
          "9d6c3a281176e9c52dcc3faa278cced2c251b2b7d28869fc1184fe956fd1b5f3")
IDENTITY = ("experiments/blueprint_design/development_inputs/midazolam_reconstruction_20260920/source_identity.json",
            "96b4016d5c0d10cec328f8290966bd583543c38058765299f64787aaa8a96417")
LITERATURE = ("artifacts/pbpk-model-assets/public_collection_20260909/Midazolam-Model__cc916b56d5e3/source/Evaluation/Input/Content/Section2.2_Data.md",
              "e84c69a93065ab5075c30dbbae8ac5fc801cea87981b810a8a0012bdce785e0b")
INVENTORY = ("experiments/blueprint_design/development_inputs/cohort_screening_20260918/source_inventory.json",
             "e50f3921f188929cfe19148a1fe4a12310536c8d977de667b1f436bfd766a8c1")
PROFILE_CATEGORY = "European (P-gp modified, CYP3A4 36 h)"
PI_NAME = "PI Hohmann iv+po, Hyland feUr MDZG, Thummel feUr unchanged - Pint"
STUDIES = {"Hohmann 2015": ("calibration", "25588320"), "Olkkola 1996": ("calibration", "8623953"),
           "Olkkola 1993": ("selection", "8453848"), "Phimmasone 2001": ("selection", "11753266"),
           "Allonen 1981": ("external", "6117393"), "Swart 2002": ("external", "11851636"),
           "Schwagmeier 1998": ("external", "9764959"), "Smith 1981": ("external", "6116606")}
ZERO_TIME_STUDIES = {"Saari 2006", "Kharasch 2011", "Darwish 2008"}


def _lipophilicity_spec(snapshot):
    pi = _one(snapshot["ParameterIdentifications"], "Name", PI_NAME)
    parameter = _one(pi["IdentificationParameters"], "Name", "Lipophilicity")
    values = [_one(parameter["Parameters"], "Name", k) for k in ("Start value", "MinValue", "MaxValue")]
    if (tuple(p["Value"] for p in values) != (3.1, 2.8, 4.0)
            or any(p.get("Unit") != "Log Units" for p in values)
            or parameter.get("UseAsFactor", False)
            or "iv 1 mg (5 min)|Midazolam|Lipophilicity" not in parameter["LinkedParameters"]):
        raise DevelopmentCaseError("Source Midazolam lipophilicity PI changed.")
    return {"parameter_id": "lipophilicity", "initial_value": 3.1, "lower_bound": 2.8,
            "upper_bound": 4.0, "unit": "Log Units", "search_scale": "linear", "sensitivity_scale": "bounded_linear",
            "runtime_mappings": [{"runtime_path": "Midazolam|Lipophilicity", "multiplier": 1.0}],
            "bound_semantics": "historical_numerical_search_guards", "anchor_may_be_historically_fitted": True,
            "source": _reference(*SOURCE, f"ParameterIdentifications[{PI_NAME}].Lipophilicity; historical start3.1 bounds2.8..4.0, not optimized final value")}


def prepare_midazolam_inputs(project_root: Path, output_dir: Path) -> dict:
    """Prepare 8 publication groups and 9 curves without executing a solver.

    Enrollment follows source protocol/timing/type, never prediction error.
    Publication IDs are global: cross-drug reuse of Phimmasone2001 must not
    become a new independent study merely because the drug prefix changes.
    """
    root = Path(project_root).resolve()
    output = (root / output_dir).resolve()
    if output == root or not output.is_relative_to(root):
        raise DevelopmentCaseError("Midazolam input output must stay inside the project.")
    snapshot = _verify(root, *SOURCE, parse=True)
    inventory = _verify(root, *INVENTORY, parse=True)
    _verify(root, *LITERATURE)
    identity = _verify(root, *IDENTITY, parse=True)["response"]["PropertyTable"]["Properties"][0]
    if identity["CID"] != 4192 or identity["MolecularFormula"] != "C18H13ClFN3":
        raise DevelopmentCaseError("Midazolam chemical identity changed.")
    definition = load_standard_human_model(root)
    spec = _lipophilicity_spec(snapshot)
    compound = _plain(_one(snapshot["Compounds"], "Name", "Midazolam"))
    compound.pop("Permeability", None)
    compound.pop("IntestinalPermeability", None)
    compound["Description"] = "Historical IV reconstruction; source nonlinear capacities fixed, source PI lipophilicity recalibrated; standard human regenerated."
    compound["CalculationMethods"] = ["Cellular partition coefficient method - PK-Sim Standard", "Cellular permeability - PK-Sim Standard"]
    compound["Solubility"] = [_one(compound["Solubility"], "Name", "FaSSIF")]
    _one(compound["Lipophilicity"][0]["Parameters"], "Name", "Lipophilicity")["Value"] = spec["initial_value"]
    for p in compound["Processes"]:
        p.setdefault("Species", "Human")
    mw = _one(compound["Parameters"], "Name", "Molecular weight")
    compound["Parameters"] = [mw, *({"Name": atom, "Value": count} for atom, count in _formula_counts(identity["MolecularFormula"]).items())]
    sources = {k: _reference(*SOURCE, f"Compounds[Midazolam].{k}; historical input, fixed fitted process capacities retained")
               for k in compound if k not in {"Name", "Description", "IsSmallMolecule"}}
    sources["Lipophilicity"] = copy.deepcopy(spec["source"])
    sources["Parameters"] = _reference(*IDENTITY, "PubChem4192 formula-derived halogens; molecular weight325.78 retained from source public model rather than rounded325.8")
    source_individual = _one(snapshot["Individuals"], "Name", PROFILE_CATEGORY)
    individual = {k: _plain(source_individual[k]) for k in ("Name", "Seed", "OriginData")}
    individual["OriginData"].pop("CalculationMethods", None)
    individual["Name"] = "IND_Midazolam_Reference"
    proteins = ("CYP3A4", "UGT1A4", "GABRG2")
    profiles = [_plain(_one([p for p in snapshot["ExpressionProfiles"] if p["Molecule"] == name], "Category", PROFILE_CATEGORY)) for name in proteins]
    profile_ids = [name + "|Human|" + PROFILE_CATEGORY for name in proteins]
    individual["ExpressionProfiles"] = profile_ids
    entry = _one(inventory["entries"], "path", SOURCE[0])
    selected, screening = [], []
    for item in entry["eligible_scenarios"]:
        study = item["study_id"]
        reason = ("outside_prespecified_single_IV_scope" if item["route"] not in {"Intravenous", "IntravenousBolus"}
                  else "unresolved_source_zero_time_observation_semantics" if study in ZERO_TIME_STUDIES
                  else "outside_prespecified_initial_study_subset_pending_separate_audit" if study not in STUDIES else None)
        screening.append({"source_study_id": study, "simulation": item["simulation"], "source_observation_name": item["observed_name"],
                          "status": "selected" if reason is None else "not_selected", "reason": reason,
                          "prediction_errors_inspected": False})
        if reason is None:
            selected.append(item)
    if len(selected) != 9 or {s["study_id"] for s in selected} != set(STUDIES):
        raise DevelopmentCaseError("Frozen source study membership changed.")
    scenarios, external, references = [], [], []
    for item in selected:
        study = item["study_id"]
        role, pmid = STUDIES[study]
        simulation = _one(snapshot["Simulations"], "Name", item["simulation"])
        observed = _one(snapshot["ObservedData"], "Name", item["observed_name"])
        meta = _metadata(observed)
        if (meta.get("Study Id") != study or meta.get("Species") != "Human" or meta.get("Route") != "IV"
                or meta.get("Molecule") != "Midazolam" or meta.get("Compartment") != "Plasma"
                or meta.get("Data type") != "aggregated" or not str(meta.get("Reference", "")).rstrip("/").endswith("/" + pmid)):
            raise DevelopmentCaseError("Source study identity, actual observation type, or plasma/IV metadata changed.")
        if simulation["Individual"] != PROFILE_CATEGORY:
            raise DevelopmentCaseError("Do not silently substitute a genotype-adjusted source individual.")
        group = "pmid:" + pmid
        identifier = "midazolam_" + _slug(study + "_" + item["simulation"])
        protocol = _normalise_protocol(_one(snapshot["Protocols"], "Name", item["protocol"]), identifier)
        # Exact unit conversion, not a time-origin correction.
        for parameter in protocol["Parameters"]:
            if parameter["Name"] == "Infusion time" and parameter.get("Unit") == "s":
                parameter.update(Value=parameter["Value"] / 60, Unit="min")
        if observed["BaseGrid"]["Unit"] != "h":
            raise DevelopmentCaseError("Expected registered hour-based source grid.")
        times = [x * 60 for x in observed["BaseGrid"]["Values"]]
        sampling = qualify_sampled_endpoints(times, source=_reference(*SOURCE, f"ObservedData[{observed['Name']}].BaseGrid; no time shifting or point deletion"))
        schema = _output_schema(simulation)
        ends = [_one(interval["Parameters"], "Name", "End time")["Value"] * 60 for interval in schema]
        if times[-1] > max(ends):
            raise DevelopmentCaseError("Source native output window does not cover all registered samples.")
        alternatives = simulation["Compounds"][0].get("Alternatives", [])
        if not any(a.get("GroupName") == "COMPOUND_SOLUBILITY" and a.get("AlternativeName") == "FaSSIF" for a in alternatives):
            raise DevelopmentCaseError("Selected source solubility alternative differs.")
        dose = _one(protocol["Parameters"], "Name", "InputDose")
        target = "Organism|PeripheralVenousBlood|Midazolam|Plasma (Peripheral Venous Blood)"
        requirement = {"compound_semantic_id": "pubchem:4192", "name": "Midazolam", "population": "healthy_adult",
            "route": "intravenous", "dose": dose["Value"], "dose_unit": dose["Unit"], "formulation": None,
            "targets": [{"target_id": "midazolam_parent_plasma", "analyte": "parent", "matrix": "plasma", "quantity": "concentration", "binding": "total", "required": True}]}
        instance = {"population_context": "healthy_adult", "individual": copy.deepcopy(individual),
            "generated_demographics": ["Weight", "Height"], "compound": copy.deepcopy(compound), "protocol": protocol,
            "snapshot_version": 80, "simulation_name": identifier,
            "human_source": _reference(*SOURCE, "Individuals[" + PROFILE_CATEGORY + "].OriginData; generic male age30 is a historical assumption, not each trial's observed mean"),
            "compound_sources": copy.deepcopy(sources), "protocol_source": _reference(*SOURCE, f"Protocols[{item['protocol']}]; unchanged dose/time, seconds explicitly converted to minutes"),
            "formulation": None, "formulation_source": None, "expression_profiles": copy.deepcopy(profiles),
            "expression_profile_sources": {i: _reference(*SOURCE, "ExpressionProfiles[" + i + "]; historical reference") for i in profile_ids},
            "output_schema": schema, "output_selections": [target]}
        # Validate all model fields before decoding this study's concentrations.
        checked = BlueprintDesignRequest.model_validate({"requirement": requirement,
            "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
            "instance": instance, "base": None, "modules": [], "evidence": [], "studies": []}).instance.model_dump(mode="json", exclude_none=True)
        checked.pop("population_context")
        build_human_model_snapshot(**checked, expected_compound_name="Midazolam")
        points = _external_observations(observed, mw["Value"])
        for index, point in enumerate(points):
            point["observation_id"] = identifier + f":row_{index}"
        document = {"schema_version": "midazolam-study-1.0", "scenario_id": identifier, "compound_id": "pubchem:4192",
            "study_group_id": group, "source_study_id": study, "global_study_id": group, "role": role, "tier": "reconstruction",
            "metadata": meta, "unit": "umol/l", "time_unit": "min", "observations": points,
            "source": _reference(*SOURCE, f"ObservedData[{observed['Name']}]"),
            "source_governance": {"historical_public_model_data": True, "newly_blinded_independent_test": False,
                "cross_publication_subject_deduplication": "unverified", "independence_unit": "PMID_publication_group_not_file_or_drug"}}
        path = output / ("external_observations" if role == "external" else "study_inputs") / (identifier + ".json")
        ref = {"path": path.relative_to(root).as_posix(), "sha256": _write_once(path, document)}
        study_ref = {"study_id": group, "source_id": ref["path"], "source_checksum": ref["sha256"],
            "role": role, "cohort_ids": [group], "observed_target_ids": ["midazolam_parent_plasma"]}
        request = {"requirement": requirement, "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
            "instance": instance, "base": None, "modules": [], "evidence": [], "studies": [] if role == "external" else [study_ref]}
        scenario = {"scenario_id": identifier, "compound_id": "pubchem:4192", "study_group_id": group, "global_study_id": group,
            "role": role, "tier": "reconstruction", "design_request": request, "target_output": target, "unit": "umol/l", "time_unit": "min",
            "observation_count": len(times), "source_study_id": study, "source_observation_name": observed["Name"],
            "source_audit": {"source": ref, "original_container": SOURCE[0], "original_container_sha256": SOURCE[1], "historical_model_data": True}, **sampling}
        if role == "external":
            scenario["holdout_reference"] = ref
            external.append(scenario)
        else:
            scenario["observations"] = points
            scenarios.append(scenario)
            references.append(ref)
    case = {"case_id": "midazolam_multistudy_iv_reconstruction", "compound_id": "pubchem:4192", "compound_name": "Midazolam",
        "tier": "reconstruction", "purpose": "historical_development_reconstruction", "strict_main_eligible": False,
        "design_request": copy.deepcopy(scenarios[0]["design_request"]), "requirement": copy.deepcopy(scenarios[0]["design_request"]["requirement"]),
        "scenarios": scenarios, "forbidden_study_group_ids": sorted({s["study_group_id"] for s in external}),
        "parameter_search": {"parameters": [spec], "initial_values_are_optimal": False, "anchor_may_be_historically_fitted": True,
            "dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "calibration"],
            "selection_dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "selection"],
            "objective": "study_curve_equal_log_rmse", "aggregation": "sqrt(mean_study(mean_curve(mean_point(log_error_squared))))",
            "identifiability_verified": False, "runtime_units_must_be_verified": True,
            "fixed_mm_capacity_reason": "Existing single-process application does not support MM kcat; shared factor cannot couple different enzymes. No frozen adapter is modified."},
        "study_counts": {role: len({s["study_group_id"] for s in scenarios + external if s["role"] == role}) for role in ("calibration", "selection", "external")},
        "source_audit": [{"path": p, "sha256": h} for p, h in (SOURCE, IDENTITY, LITERATURE, INVENTORY)] + references,
        "limitations": ["Historical public-model reconstruction; prior model-building and verification sources are reused, not strictly independent test.",
            "CYP3A4 and UGT1A4 MM capacities, GFR fraction, binding kinetics, expression and fu remain disclosed historical parameters.",
            "Only lipophilicity is calibrated from source PI start/bounds; this is not a two-enzyme joint capacity estimation experiment.",
            "Hohmann2015 two IV doses stay in one calibration group. Source source-container sharing does not create independence.",
            "Global PMID group IDs allow cross-drug deduplication; overlap of subjects across different publications is unverified.",
            "Generic male age30 reference physiology is regenerated; neither study population variance nor individual fitting is represented.",
            "Zero-time-ambiguous IV study groups are excluded before performance inspection with their reasons retained.",
            "Only registered single-IV parent-plasma tasks; no oral, DDI, metabolite or clinical-accuracy claim.",
            "No point deletion, time shifting, fitted source PKML, source organ override or source simulated curve is copied.",
            "Sample-window AUC and sampled Cmax are separate endpoints; they are not total AUC or a continuous peak."]}
    screening_path = output / "source_qualification.json"
    screening_ref = {"path": screening_path.relative_to(root).as_posix(), "sha256": _write_once(screening_path,
        {"schema_version": "midazolam-source-qualification-1.0", "source": {"path": SOURCE[0], "sha256": SOURCE[1]},
         "study_role_policy": STUDIES, "screening": screening, "selection_performance_based": False,
         "native_execution_performed": False, "newly_added_to_running_171_jobs": False})}
    bundle = {"schema_version": "midazolam-reconstruction-input-1.0", "case": case, "external_scenarios": external,
              "source_qualification": screening_ref, "solver_calls": 0, "scientific_accepted": False}
    path = output / "input_package.json"
    checksum = _write_once(path, bundle)
    return {"case_id": case["case_id"], "compound_id": case["compound_id"], "compound_name": "Midazolam", "tier": "reconstruction",
        "strict_main_eligible": False, "bundle_path": path.relative_to(root).as_posix(), "bundle_sha256": checksum,
        "study_counts": case["study_counts"], "status": "static_qualified_lipophilicity_only_native_not_run", "solver_calls": 0}
