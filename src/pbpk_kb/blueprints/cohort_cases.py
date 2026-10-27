"""Source-locked multi-study reconstruction cohort, without model execution.

This adapter is data preparation, not a designer or a held-out evaluator. Source
publication roles are assigned before concentration conversion. External values
are exported separately and never returned in a design bundle. All inputs remain
historical reconstructions: prior fitting, incomplete subject deduplication and
population assumptions preclude independent-main-benchmark claims. Native human
physiology is rebuilt; source simulation structures/results are never copied.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .development_case import DevelopmentCaseError, _one, _plain, _reference, _verify
from .human_model import load_standard_human_model
from .multidrug_cases import _external_observations, _metadata, _write_once
from .native_edits import reviewed_process_parameter_path

_COLLECTION = "artifacts/pbpk-model-assets/public_collection_20260909/"
_INVENTORY = ("experiments/blueprint_design/development_inputs/cohort_screening_20260918/source_inventory.json",
              "e50f3921f188929cfe19148a1fe4a12310536c8d977de667b1f436bfd766a8c1")
_IDENTITY = ("experiments/blueprint_design/development_inputs/cohort_sources/pubchem_identity_facts_20260918.json",
             "28994f3147eaf1a11074b975b24ac11ff67f8b3bfac2cf4c4798713552e19671")
_CACHE = "artifacts/pbpk_raw_runs/compound_runs/run_20260909T083120Z/compounds/"
_POLICY = {"policy_id": "reconstruction-numerical-guards-v1", "frozen_date": "2026-09-18",
           "priority": "source PI start and bounds where available",
           "otherwise_clearance": "positive source anchor times [0.1,10]",
           "otherwise_lipophilicity": "source Log Units anchor plus [-1,1], linear coordinate",
           "biological_confidence_interval": False, "prior_fitted_anchors_allowed": True,
           "scope": "historical reconstruction only; same guards for every method"}

_REGISTRY = {
    "sufentanil": {
        "name": "Sufentanil", "source": (_COLLECTION + "Sufentanil-Model__04dcc64cae6e/source/Sufentanil.json",
                                           "7948ed8d4549219856ca5a7b3674254b8f936042b3f2e17e6270b1b6fb5214e6"),
        "identity": _IDENTITY, "identity_key": "sufentanil", "pi": None,
        "clearance": ("CYP3A4", "Zhou et al. 2017", "Intrinsic clearance"),
        "studies": {"Bovill 1984": "calibration", "Willsie 2015": "selection", "Taverne 1992": "external"},
        "limitations": [
            "Bovill1984 and Taverne1992 concern surgical patients; Willsie2015 concerns healthy volunteers. Population shift is explicit, not a healthy-only cohort.",
            "Observed Bovill name says 5 mg/kg, but source protocol and Evaluation Section3 state 5 microgram/kg; the explicit protocol is retained with this label conflict disclosed.",
            "Willsie2014 simulation and Willsie2015 publication identify the same epub2014/final2015 study.",
            "Published fitted logP and clearance anchor the numerical search; no independent-prior claim is made.",
            "Source report describes uncertainty extracting the Taverne mean profile; it is retained as historical external reference, not certified truth."]},
    "fluconazole": {
        "name": "Fluconazole", "source": (_COLLECTION + "Fluconazole-Model__fa23aa5a5865/source/Fluconazole-Model.json",
                                            "2bed78364a32fb8245b74e50dbc1441d52044a426ae74d2317ecd90d7f8181ae"),
        "identity": (_CACHE + "fluconazole/compound_api/pubchem/properties_by_name_16eaed226286.bin",
                     "16eaed22628628d05c81439603c56d18afe697f827414143d977d5638c8f85fe"),
        "pi": "Parameter Identification 1 - IV", "clearance": ("UGT2B7", "Watt2018", "Specific clearance"),
        "studies": {"Ripa 1993": "calibration", "Shiba 1990": "calibration", "Brammer 1990": "selection", "Yeates 1994": "external"},
        "limitations": [
            "Both Shiba1990 IV doses are calibration: the historical source's train/test dose split is not treated as independent studies.",
            "The original CYP3A4 noncompetitive inhibitor and its expression profile remain explicit alongside UGT2B7 parent metabolism; no DDI prediction is claimed.",
            "Ahonen1997 is excluded because its source protocol explicitly identifies a midazolam DDI experiment; that exclusion precedes error inspection.",
            "GFR fraction remains a disclosed historical fitted value. Generic age30 male physiology is a model assumption.",
            "Molecular weight 306.328 remains the locked public model value; cached PubChem306.27 differs slightly. Formula alone supplies halogen counts."]},
    "moxifloxacin": {
        "name": "Moxifloxacin", "source": (_COLLECTION + "Moxifloxacin-Model__51038550e450/source/Moxifloxacin.json",
                                             "17afa15db903a1ce1abbecb42b810c151e252a20835eed9e21aae34dcfaa694f"),
        "identity": (_CACHE + "moxifloxacin/compound_api/pubchem/properties_by_name_b81f5eb9b397.bin",
                     "b81f5eb9b3972140040b5f0c68b761dd111d5d4c612985aede4c381ad91b2bd4"),
        "pi": "IV", "clearance": ("SULT2A1", "Fit", "Specific clearance"),
        "studies": {"Stass 1999": "calibration", "Siefert 1999": "selection", "PH 27517/0139": "external"},
        "limitations": [
            "PH27517/0139 is a source study identifier; independent participant identity and full public bibliographic linkage are not established.",
            "UGT1A1 downstream CLspec/[Enzyme] remains a historical fitted value; the source recombinant input alone is not presented as the effective clearance.",
            "Source individual/simulation parameter overrides, including potential enterohepatic adaptations, are not transferred into the regenerated standard model.",
            "Source IV protocols named400/100mg actually specify368/92mg InputDose. These exact source doses remain; no unverified salt correction is applied.",
            "Protocol End time10days is omitted only for DosingInterval Single: locked PKSim SimpleProtocolToSchemaMapper returns after one dose before consulting repetition end time. OutputSchema remains source-defined.",
            "The source PI allows zero specific clearance. Its zero lower bound is retained with linear search and bounded-linear sensitivity."]},
}

_BLOCKS = {
    "Alfentanil": "already_executed_reference; unresolved original t0 positive-concentration timing; not resubmitted as a new drug",
    "Alprazolam": "native Michaelis-Menten and source alternative selection not yet audited",
    "Dapagliflozin": "explicit named metabolites missing from the single-parent compound container; linked metabolite handling unaudited",
    "Digoxin": "active transport and specific binding unsupported in current audited adapter",
    "Ethinylestradiol": "mixed nonlinear metabolism and self-inhibition require mechanism-preserving adapter",
    "Fluvoxamine": "nonlinear metabolism and self-inhibition require mechanism-preserving adapter",
    "Furosemide": "nonlinear active transport and metabolism not implemented in audited native adapter",
    "Isoniazid": "nonlinear acetylation alternatives and phenotype grouping not yet audited",
    "Mexiletine": "self-inhibition plus hepatic/systemic clearance and clinical populations require joint mechanism and population audit",
    "Midazolam": "nonlinear metabolism and specific binding need mechanism-preserving adapter",
    "Moclobemide": "self-inhibition and nonlinear CYP metabolism unsupported",
    "Moxidectin": "source labels tablets as dissolved formulation; actual solution studies and genotype/sex arms require study-level verification",
    "Paclitaxel": "oncology/co-treatment populations and observed infusion mapping require explicit task qualification",
    "Rifampicin": "self-induction and nonlinear transport/metabolism unsupported",
    "S-Mephenytoin": "one source simulation reused across distinct genotype studies; dose/genotype matching unresolved",
    "Sildenafil": "recombinant Michaelis-Menten metabolism not implemented in audited adapter",
}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def qualify_sampled_endpoints(times_min: list[float], *, source: dict) -> dict:
    """Qualify only sampling-window summaries, without reading concentrations.

    This deliberately does not qualify AUC to infinity or a continuous Cmax.
    Missing/invalid concentrations and failed predictions remain evaluator
    failures; metadata eligibility must never silently remove those cases.
    """
    if (len(times_min) < 2 or any(isinstance(t, bool) or not isinstance(t, (int, float))
            or not math.isfinite(t) or t <= 0 for t in times_min)
            or any(b <= a for a, b in zip(times_min, times_min[1:]))):
        raise DevelopmentCaseError("Sampling qualification requires distinct, positive, increasing times; zero-time semantics need separate source review.")
    if not source.get("source_checksum") or not source.get("source_location"):
        raise DevelopmentCaseError("Sampling qualification needs a source-locked time-grid reference.")
    return {"auc_evaluable": True, "cmax_evaluable": True,
            "sampling_qualification": {
                "policy_id": "observed-grid-endpoints-v1", "based_on_concentration_or_error": False,
                "source": copy.deepcopy(source), "observation_count": len(times_min),
                "time_unit": "min", "sampling_times_min": list(times_min),
                "observed_window_auc": {"start_min": times_min[0], "end_min": times_min[-1],
                    "rule": "linear_trapezoid_at_observed_times", "extrapolated": False},
                "sampled_cmax": {"definition": "maximum_at_registered_observation_times",
                    "continuous_peak_estimated": False},
                "total_auc_qualified": False, "continuous_cmax_qualified": False,
                "limitations": ["Sparse samples may miss peaks and between-sample curvature.",
                    "Both model and observations use the identical registered sampling grid."]}}


def prepare_fluconazole_endpoint_inputs(project_root: Path, input_bundle: Path, output_dir: Path) -> dict:
    """Add declared endpoint metadata to an immutable historical input package.

    Only source time grids are inspected. Concentrations, study roles, protocols,
    parameters, and prior files are unchanged. This is a new endpoint analysis
    registration after the historical development run, not prospective validation.
    """
    root = Path(project_root).resolve()
    source_path = (root / input_bundle).resolve(strict=True)
    output = (root / output_dir).resolve()
    if not source_path.is_relative_to(root) or not output.is_relative_to(root) or output == root:
        raise DevelopmentCaseError("Endpoint resources must remain inside the project.")
    original = source_path.read_bytes()
    bundle = json.loads(original)
    if bundle["case"].get("case_id") != "fluconazole_multistudy_iv_reconstruction" or bundle["case"].get("strict_main_eligible") is not False:
        raise DevelopmentCaseError("This registration covers the existing Fluconazole historical reconstruction only.")
    registration = _REGISTRY["fluconazole"]
    snapshot = _verify(root, *registration["source"], parse=True)
    for scenario in bundle["case"]["scenarios"] + bundle["external_scenarios"]:
        observed = _one(snapshot["ObservedData"], "Name", scenario["source_observation_name"])
        grid = observed["BaseGrid"]
        factors = {"h": 60.0, "min": 1.0}
        if grid.get("Unit") not in factors:
            raise DevelopmentCaseError("Unsupported source sampling time unit.")
        times = [t * factors[grid["Unit"]] for t in grid["Values"]]
        if len(times) != scenario["observation_count"]:
            raise DevelopmentCaseError("Sampling grid does not match the registered observation count.")
        scenario.update(qualify_sampled_endpoints(times, source=_reference(
            *registration["source"], f"ObservedData[{observed['Name']}].BaseGrid; timing metadata only")))
    bundle["endpoint_registration"] = {
        "policy_id": "observed-grid-endpoints-v1", "date": "2026-09-20",
        "original_bundle_path": source_path.relative_to(root).as_posix(),
        "original_bundle_sha256": hashlib.sha256(original).hexdigest(),
        "historical_results_already_available": True, "prospective_validation": False,
        "observations_times_roles_parameters_and_protocols_changed": False,
        "selection_objective_changed": False}
    path = output / "input_package.json"
    checksum = _write_once(path, bundle)
    return {"bundle_path": path.relative_to(root).as_posix(), "bundle_sha256": checksum,
            "scenarios_qualified": len(bundle["case"]["scenarios"]) + len(bundle["external_scenarios"])}


def _identity(root: Path, registration: dict) -> dict:
    value = _verify(root, *registration["identity"], parse=True)
    return value["compounds"][registration["identity_key"]] if "identity_key" in registration else value["PropertyTable"]["Properties"][0]


def _formula_counts(formula: str) -> dict:
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    if not tokens or "".join(a + n for a, n in tokens) != formula:
        raise DevelopmentCaseError("Unsupported source molecular formula notation.")
    counts = {}
    for atom, count in tokens:
        counts[atom] = counts.get(atom, 0) + int(count or 1)
    return {atom: counts.get(atom, 0) for atom in ("F", "Cl", "Br", "I")}


def _parameter_specs(snapshot: dict, compound: dict, registration: dict, policy_source: tuple) -> list[dict]:
    enzyme, data_source, parameter_name = registration["clearance"]
    process = _one([p for p in compound["Processes"] if p.get("Molecule") == enzyme], "DataSource", data_source)
    native = {"internal_name": process["InternalName"], "molecule": enzyme,
              "data_source": data_source, "parameter_name": parameter_name}
    process_parameter = _one(process["Parameters"], "Name", parameter_name)
    logp = _one(compound["Lipophilicity"][0]["Parameters"], "Name", "Lipophilicity")
    pi = _one(snapshot.get("ParameterIdentifications", []), "Name", registration["pi"]) if registration["pi"] else None
    specs = []
    for identifier, parameter, runtime_path in (
        ("single_process_clearance", process_parameter, reviewed_process_parameter_path(compound["Name"], native)),
        ("lipophilicity", logp, compound["Name"] + "|Lipophilicity"),
    ):
        previous = [p for p in pi.get("IdentificationParameters", []) if p["Name"] == parameter["Name"]] if pi else []
        if len(previous) > 1:
            raise DevelopmentCaseError("Ambiguous historical parameter identification.")
        if previous:
            previous = previous[0]
            if not any(p.endswith("|" + runtime_path) for p in previous["LinkedParameters"]):
                raise DevelopmentCaseError("Source PI is not linked to the declared native parameter.")
            values = [_one(previous["Parameters"], "Name", n) for n in ("Start value", "MinValue", "MaxValue")]
            if any(p.get("Unit") != parameter.get("Unit") for p in values):
                raise DevelopmentCaseError("PI unit drift.")
            initial, lower, upper = [p["Value"] for p in values]
            source = _reference(*registration["source"], f"ParameterIdentifications[{pi['Name']}].{parameter['Name']}; historical numerical start/bounds")
            policy_id = "source_pi_numerical_guards"
        else:
            initial = parameter["Value"]
            if identifier == "lipophilicity":
                lower, upper = initial - 1, initial + 1
            else:
                if initial <= 0:
                    raise DevelopmentCaseError("Derived clearance policy needs a positive source anchor.")
                lower, upper = .1 * initial, 10 * initial
            source = _reference(*policy_source, f"{identifier}; anchor from {registration['source'][0]} SHA256={registration['source'][1]}; may be historically fitted")
            policy_id = _POLICY["policy_id"]
        linear = identifier == "lipophilicity" or lower == 0
        spec = {"parameter_id": identifier, "initial_value": initial, "lower_bound": lower, "upper_bound": upper,
                "unit": parameter["Unit"], "source": source, "policy_id": policy_id,
                "bound_semantics": "historical_numerical_search_guards", "anchor_may_be_historically_fitted": not bool(previous),
                "search_scale": "linear" if linear else "log", "sensitivity_scale": "bounded_linear" if linear else "log",
                "runtime_mappings": [{"runtime_path": runtime_path, "multiplier": 1.0}]}
        if identifier == "single_process_clearance":
            spec["native_process"] = native
        specs.append(spec)
        parameter["Value"] = initial
    return specs


def _normalise_protocol(protocol: dict, scenario_id: str) -> dict:
    protocol = _plain(protocol)
    protocol["Name"] = "PROTO_" + scenario_id
    if protocol.get("DosingInterval") != "Single" or protocol.get("ApplicationType") not in {"Intravenous", "IntravenousBolus"}:
        raise DevelopmentCaseError("This frozen cohort adapter admits sourced single-dose IV protocols only.")
    # PK-Sim's locked SimpleProtocolToSchemaMapper returns in its Single branch
    # before consulting EndTimeParameter; this is not an observation time edit.
    protocol["Parameters"] = [p for p in protocol["Parameters"] if p["Name"] != "End time"]
    dose = _one(protocol["Parameters"], "Name", "InputDose")
    unit = dose["Unit"].replace("µ", "u").replace("μ", "u")
    if unit in {"ug", "ug/kg"}:
        dose.update(Value=dose["Value"] / 1000, Unit=unit.replace("ug", "mg"))
    if dose["Unit"] not in {"mg", "mg/kg"}:
        raise DevelopmentCaseError("Unsupported source dose unit.")
    for p in protocol["Parameters"]:
        if p["Name"] == "Start time" and p.get("Unit") == "day(s)":
            p.update(Value=p["Value"] * 24, Unit="h")
    return protocol


def _output_schema(simulation: dict) -> list[dict]:
    result = _plain(simulation["OutputSchema"])
    for interval in result:
        for parameter in interval["Parameters"]:
            if parameter.get("Unit") == "day(s)":
                parameter.update(Value=parameter["Value"] * 24, Unit="h")
    return result


def _make_bundle(root: Path, output: Path, key: str, entry: dict, definition: dict, policy_source: tuple) -> dict:
    registration = _REGISTRY[key]
    source = registration["source"]
    snapshot = _verify(root, *source, parse=True)
    identity = _identity(root, registration)
    name, compound_id = registration["name"], f"pubchem:{identity['CID']}"
    compound = _plain(_one(snapshot["Compounds"], "Name", name))
    for field in ("Permeability", "IntestinalPermeability"):
        compound.pop(field, None)
    compound["CalculationMethods"] = ["Cellular partition coefficient method - PK-Sim Standard", "Cellular permeability - PK-Sim Standard"]
    compound["Description"] = "Historical reconstruction input; standard human model regenerated, no simulation structure inherited."
    mw = _one(compound["Parameters"], "Name", "Molecular weight")
    compound["Parameters"] = [mw, *({"Name": atom, "Value": value} for atom, value in _formula_counts(identity["MolecularFormula"]).items())]
    solubility = _one(compound["Solubility"][0]["Parameters"], "Name", "Solubility at reference pH")
    if solubility["Unit"] == "mg/l":
        solubility.update(Value=solubility["Value"] / 1000, Unit="mg/ml")
    if solubility["Unit"] != "mg/ml":
        raise DevelopmentCaseError("Unsupported solubility unit.")
    fu = _one(compound["FractionUnbound"][0]["Parameters"], "Name", "Fraction unbound (plasma, reference value)")
    if fu.get("Unit") == "%":
        fu["Value"] /= 100
        fu.pop("Unit")
    # Missing process species is resolved only by the explicit Human profile and
    # Human-only single-parent simulations, and disclosed in the source audit.
    for process in compound["Processes"]:
        process.setdefault("Species", "Human")
    specs = _parameter_specs(snapshot, compound, registration, policy_source)
    sources = {field: _reference(*source, f"Compounds[{name}].{field}; historical reconstruction; explicit unit conversions")
               for field in compound if field not in {"Name", "Description", "IsSmallMolecule"}}
    sources["Parameters"] = _reference(*registration["identity"], f"formula-derived halogens; MW retained from source {source[0]} SHA256={source[1]}")
    sources["Lipophilicity"] = copy.deepcopy(specs[1]["source"])
    enzyme_names = {p["Molecule"] for p in compound["Processes"] if "Molecule" in p}
    scenarios, external_scenarios, study_audit = [], [], []
    selected_entries = [s for s in entry["eligible_scenarios"] if s["study_id"] in registration["studies"]]
    for selected in selected_entries:
        study = selected["study_id"]
        role = registration["studies"][study]
        simulation = _one(snapshot["Simulations"], "Name", selected["simulation"])
        original_individual = _one(snapshot["Individuals"], "Name", simulation["Individual"])
        # Inherited organ/individual overrides are intentionally excluded.
        individual = {"Name": "IND_" + _slug(key + "_" + original_individual["Name"]), "Seed": original_individual["Seed"], "OriginData": _plain(original_individual["OriginData"])}
        individual["OriginData"].pop("CalculationMethods", None)
        if individual["OriginData"]["Species"] != "Human" or individual["OriginData"]["Age"]["Value"] < 18:
            raise DevelopmentCaseError("Only explicit adult Human individuals are admitted.")
        profiles, profile_sources, profile_ids = [], {}, []
        for enzyme in sorted(enzyme_names):
            ids = [p for p in original_individual.get("ExpressionProfiles", []) if p.startswith(enzyme + "|Human|")]
            if len(ids) != 1:
                raise DevelopmentCaseError("Exactly one selected source Human profile per enzyme is required.")
            profile_id = ids[0]
            profile = _one([p for p in snapshot.get("ExpressionProfiles", []) if p.get("Molecule") == enzyme and p.get("Species") == "Human"], "Category", profile_id.split("|", 2)[2])
            profiles.append(_plain(profile)); profile_ids.append(profile_id)
            profile_sources[profile_id] = _reference(*source, "ExpressionProfiles[" + profile_id + "]; source values may reflect historical development")
        if profile_ids:
            individual["ExpressionProfiles"] = profile_ids
        scenario_id = key + "_" + _slug(selected["simulation"])
        study_id = key + "_" + _slug(study)
        protocol_name = simulation["Compounds"][0]["Protocol"]["Name"]
        protocol = _normalise_protocol(_one(snapshot["Protocols"], "Name", protocol_name), scenario_id)
        dose = _one(protocol["Parameters"], "Name", "InputDose")
        population = "adult_surgical_patients" if key == "sufentanil" and study != "Willsie 2015" else "healthy_adult"
        target = f"Organism|PeripheralVenousBlood|{name}|Plasma (Peripheral Venous Blood)"
        requirement = {"compound_semantic_id": compound_id, "name": name, "population": population, "route": "intravenous",
                       "dose": dose["Value"], "dose_unit": dose["Unit"], "formulation": None,
                       "targets": [{"target_id": key + "_parent_plasma", "analyte": "parent", "matrix": "plasma", "quantity": "concentration", "binding": "total", "required": True}]}
        instance = {"population_context": population, "individual": individual,
                    "generated_demographics": [n for n in ("Weight", "Height") if n not in individual["OriginData"]],
                    "compound": copy.deepcopy(compound), "protocol": protocol, "snapshot_version": 80, "simulation_name": scenario_id,
                    "human_source": _reference(*source, f"Individuals[{original_individual['Name']}].OriginData; unreported demographics generated; disease physiology not inferred"),
                    "compound_sources": copy.deepcopy(sources), "protocol_source": _reference(*source, f"Protocols[{protocol_name}]; exact dose/time with explicit unit conversion"),
                    "formulation": None, "formulation_source": None, "expression_profiles": profiles, "expression_profile_sources": profile_sources,
                    "output_schema": _output_schema(simulation), "output_selections": [target]}
        observed = _one(snapshot["ObservedData"], "Name", selected["observed_name"])
        meta = _metadata(observed)
        if meta.get("Study Id") != study or meta.get("Species") != "Human" or meta.get("Molecule") != name:
            raise DevelopmentCaseError("Observed source study/species/compound mismatch.")
        observations = _external_observations(observed, mw["Value"])
        document = {"schema_version": "cohort-study-observations-1.0", "scenario_id": scenario_id, "compound_id": compound_id,
                    "study_group_id": study_id, "source_study_id": study, "role": role, "tier": "reconstruction",
                    "unit": "umol/l", "time_unit": "min", "metadata": meta, "source": _reference(*source, f"ObservedData[{observed['Name']}]"),
                    "observations": observations, "source_governance": {"original_container_shared": True, "independence_unit": "publication_and_study_group_not_file_checksum",
                        "historical_public_model_data": True, "newly_blinded_independent_test": False, "cross_publication_subject_deduplication": "unverified"}}
        path = output / ("external_observations" if role == "external" else "study_inputs") / (scenario_id + ".json")
        reference = {"path": path.relative_to(root).as_posix(), "sha256": _write_once(path, document)}
        study_reference = {"study_id": study_id, "source_id": reference["path"], "source_checksum": reference["sha256"],
                           "role": role, "cohort_ids": [study_id], "observed_target_ids": [key + "_parent_plasma"]}
        request = {"requirement": requirement, "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
                   "instance": instance, "base": None, "modules": [], "evidence": [], "studies": [] if role == "external" else [study_reference]}
        scenario = {"scenario_id": scenario_id, "study_group_id": study_id, "role": role, "design_request": request,
                    "target_output": target, "unit": "umol/l", "time_unit": "min", "observation_count": len(observations),
                    "source_study_id": study, "source_observation_name": observed["Name"],
                    "source_audit": {"source": reference, "original_container": source[0], "original_container_sha256": source[1], "historical_model_data": True}}
        if role == "external":
            scenario["holdout_reference"] = reference
            external_scenarios.append(scenario)
        else:
            scenario["observations"] = observations
            scenarios.append(scenario)
            study_audit.append(reference)
    if {s["role"] for s in scenarios} != {"calibration", "selection"} or not external_scenarios:
        raise DevelopmentCaseError("A full-method case requires separate calibration, selection and external source groups.")
    groups = {role: {s["study_group_id"] for s in scenarios + external_scenarios if s["role"] == role} for role in ("calibration", "selection", "external")}
    if any(groups[a] & groups[b] for a, b in (("calibration", "selection"), ("calibration", "external"), ("selection", "external"))):
        raise DevelopmentCaseError("Study-level role leakage.")
    audit = [{"path": path, "sha256": sha} for path, sha in (source, registration["identity"], policy_source)]
    case = {"case_id": key + "_multistudy_iv_reconstruction", "compound_id": compound_id, "compound_name": name,
            "tier": "reconstruction", "purpose": "historical_development_reconstruction", "strict_main_eligible": False,
            "design_request": copy.deepcopy(scenarios[0]["design_request"]), "requirement": copy.deepcopy(scenarios[0]["design_request"]["requirement"]),
            "scenarios": scenarios, "forbidden_study_group_ids": sorted(groups["external"]),
            "parameter_search": {"parameters": specs, "initial_values_are_optimal": False, "anchor_may_be_historically_fitted": True,
                "dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "calibration"],
                "selection_dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "selection"],
                "objective": "study_curve_equal_log_rmse", "aggregation": "sqrt(mean_study(mean_curve(mean_point(log_error_squared))))",
                "identifiability_verified": False, "runtime_units_must_be_verified": True},
            "study_counts": {role: len(ids) for role, ids in groups.items()}, "source_audit": audit + study_audit,
            "limitations": registration["limitations"] + [
                "All data and several parameters informed a prior public model; no main_holdout or freshly blinded status is claimed.",
                "Source study identifiers are grouped consistently, but cross-publication subject-level independence is not established.",
                "Numerical search guards are frozen across methods; they are not physiological confidence intervals.",
                "Source organism parameters, Kp, ModelProperties, Solver and simulated results are not transplanted; only sourced drug/person/protocol/profile inputs are used.",
                "External values are evaluator-only after prediction locking; no time shift, point deletion or error-driven case selection is performed."]}
    return {"schema_version": "cohort-reconstruction-input-1.0", "case": case, "external_scenarios": external_scenarios,
            "source_audit": audit, "solver_calls": 0, "scientific_accepted": False}


def prepare_cohort_inputs(project_root: Path, output_dir: Path, *, compounds: list[str] | None = None) -> dict:
    """Create immutable case bundles and a full-library screening manifest.

    Optional subsets permit early submission while further cases are prepared in
    separate directories. Existing changed files are never overwritten. This
    does not call a solver, fit parameters, inspect prediction errors or load any
    external values into a designer. Source count is not independent-drug count.
    """
    root = Path(project_root).resolve()
    output = Path(output_dir)
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(root) or output == root:
        raise DevelopmentCaseError("Cohort output must be a project-contained directory.")
    inventory = _verify(root, *_INVENTORY, parse=True)
    keys = list(_REGISTRY) if compounds is None else compounds
    if not keys or len(keys) != len(set(keys)) or any(k not in _REGISTRY for k in keys):
        raise DevelopmentCaseError("Provide unique registered cohort compound keys.")
    policy_path = output / "numerical_search_policy.json"
    policy_source = (policy_path.relative_to(root).as_posix(), _write_once(policy_path, _POLICY))
    definition = load_standard_human_model(root)
    cases = []
    for key in keys:
        registration = _REGISTRY[key]
        entry = _one(inventory["entries"], "path", registration["source"][0])
        if entry["sha256"] != registration["source"][1]:
            raise DevelopmentCaseError("Screening source hash differs from the registered model.")
        bundle = _make_bundle(root, output / key, key, entry, definition, policy_source)
        path = output / key / "input_package.json"
        checksum = _write_once(path, bundle)
        cases.append({"case_id": bundle["case"]["case_id"], "compound_id": bundle["case"]["compound_id"], "compound_name": registration["name"],
                      "tier": "reconstruction", "strict_main_eligible": False, "bundle_path": path.relative_to(root).as_posix(), "bundle_sha256": checksum,
                      "status": "ready", "blocked_reason": None, "source": {"path": registration["source"][0], "sha256": registration["source"][1]},
                      "study_counts": bundle["case"]["study_counts"], "execution_mode": "full_methods"})
    ready_sources = {c["source"]["path"] for c in cases}
    screening = []
    for e in inventory["entries"]:
        names = e["compound_names"]
        name = names[0] if len(names) == 1 else None
        reason = (None if e["path"] in ready_sources else _BLOCKS.get(name)
                  or ("registered_in_separate_submission" if name and name.lower() in _REGISTRY
                      else "fewer_than_three_eligible_study_labels_or_missing_single_parent_adult_IV_inputs"))
        screening.append({"source_path": e["path"], "source_sha256": e["sha256"], "compound_names": names,
                          "eligible_study_labels": e["eligible_study_count"], "multi_study_screening_denominator": e["eligible_study_count"] >= 3,
                          "status": "ready" if reason is None else "blocked", "blocked_reason": reason,
                          "mechanisms": e["processes"], "study_labels_are_verified_subject_independence": False})
    manifest = {"schema_version": "cohort-execution-manifest-1.0", "scope": "frozen_multidrug_historical_reconstruction_benchmark",
                "methods": [*[f"M{i}" for i in range(7)], "A1", "A2", "A3", "E0"], "seeds": [17, 43, 101],
                "cases": cases, "screening": screening, "source_inventory": {"path": _INVENTORY[0], "sha256": _INVENTORY[1]},
                "summary": {"source_snapshots": len(screening), "multi_study_label_candidates": sum(s["multi_study_screening_denominator"] for s in screening),
                            "ready_distinct_drugs": len({c["compound_id"] for c in cases}), "strict_main_eligible_drugs": 0},
                "solver_calls": 0, "observations_in_design_manifest": False}
    _write_once(output / "cohort_manifest.json", manifest)
    return manifest
