"""Immutable historical Alprazolam reconstruction inputs, with evaluator-only curves.

The native human model is regenerated from sourced inputs. No fitted PKML,
simulation organ overrides, or observed external concentrations enter a designer.
The original shared kcat PI is a factor on historical branch anchors, not two
identical turnover rates. Its historical information leakage is explicit.
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

SOURCE = ("artifacts/pbpk-model-assets/public_collection_20260909/Alprazolam-Model__73359a50af33/source/Alprazolam-Model.json",
          "da8460baaaf2ecf058d1a328b4c6bb9b2ea77311fe1b03414361aef228de60c0")
IDENTITY = ("experiments/blueprint_design/development_inputs/cohort_sources/alprazolam_identity_20260920.json",
            "c601e0cb4205095cbb44a9a39c99447608dfc2792cdd8ade5fb99137ac0af0fb")
BINDING = ("experiments/blueprint_design/development_inputs/cohort_sources/alprazolam_binding_20260920.json",
           "0459468fb825bb4bff50b564063241e059ca1847c8aaf2e873f2fee835ba3f32")
LITERATURE = ("artifacts/pbpk-model-assets/public_collection_20260909/Alprazolam-Model__73359a50af33/source/Evaluation/Input/Content/Section2.2_Data.md",
              "9cc13c5abfc93cd5c266c06fa261474ea555653c471b260aaf761a59b648e99d")
INVENTORY = ("experiments/blueprint_design/development_inputs/cohort_screening_20260918/source_inventory.json",
             "e50f3921f188929cfe19148a1fe4a12310536c8d977de667b1f436bfd766a8c1")
STUDY_ROLES = {"Kroboth 1988": "calibration", "Fleishaker 1989": "calibration",
               "Smith 1984": "selection", "Lin 1988": "external",
               "Bertz 1997": "external", "Venkatakrishnan 2005": "external"}
BRANCHES = ("alpha-OH pathway", "4-OH pathway")
ANCHORS = (0.8066945978, 13.7322820855)


def _specifications(snapshot: dict, compound: dict) -> list[dict]:
    pi = _one(snapshot["ParameterIdentifications"], "Name", "Parameter Identification")
    factor = _one(pi["IdentificationParameters"], "Name", "kcat")
    if factor.get("UseAsFactor") is not True or factor.get("Scaling") != "Log":
        raise DevelopmentCaseError("Historical kcat PI must remain a shared logarithmic factor.")
    bounds = [_one(factor["Parameters"], "Name", key) for key in ("Start value", "MinValue", "MaxValue")]
    if tuple(p["Value"] for p in bounds) != (1.0, 0.1, 10.0) or any(p.get("Unit") for p in bounds):
        raise DevelopmentCaseError("Source shared-factor bounds or units changed.")
    identities, mappings = [], []
    for branch, anchor in zip(BRANCHES, ANCHORS, strict=True):
        process = _one(compound["Processes"], "DataSource", branch)
        kcat = _one(process["Parameters"], "Name", "kcat")
        if process["InternalName"] != "MetabolizationLiverMicrosomes_MM" or process["Molecule"] != "CYP3A4" or kcat != {"Name": "kcat", "Value": anchor, "Unit": "1/min"}:
            raise DevelopmentCaseError("Source nonlinear branch identity/anchor changed.")
        path = "Alprazolam-CYP3A4-" + branch + "|kcat"
        if not all(any(link == s + "|" + path for link in factor["LinkedParameters"])
                   for s in ("Kroboth1988_IV_0.5mg", "Smith1984_IV_1mg", "Bertz1997_IV_2mg")):
            raise DevelopmentCaseError("Source PI does not bind both branches across study roles.")
        identities.append({"internal_name": process["InternalName"], "molecule": "CYP3A4",
                           "data_source": branch, "parameter_name": "kcat"})
        mappings.append({"runtime_path": path, "multiplier": anchor, "native_unit": "1/min"})
    shared = {"parameter_id": "shared_process_parameter", "parameterization": "shared_factor",
              "initial_value": 1.0, "lower_bound": 0.1, "upper_bound": 10.0, "unit": "dimensionless",
              "native_processes": identities, "source_anchor_values": list(ANCHORS), "runtime_mappings": mappings,
              "search_scale": "log", "sensitivity_scale": "log",
              "bound_semantics": "historical_numerical_search_guards",
              "anchor_may_be_historically_fitted": True,
              "mapping_rule": "shared_dimensionless_factor_times_historical_branch_kcat; preserve_unequal_branches",
              "source": _reference(*SOURCE, "ParameterIdentifications[Parameter Identification].kcat UseAsFactor=true, Start=1 Min=.1 Max=10; each Compounds[Alprazolam].Processes branch kcat anchor was historically fitted")}
    logp = _one(pi["IdentificationParameters"], "Name", "Lipophilicity")
    values = [_one(logp["Parameters"], "Name", key) for key in ("Start value", "MinValue", "MaxValue")]
    if tuple(p["Value"] for p in values) != (2.0756837827, 1.26, 2.19) or any(p.get("Unit") != "Log Units" for p in values):
        raise DevelopmentCaseError("Source lipophilicity start/bounds changed.")
    lipophilicity = {"parameter_id": "lipophilicity", "initial_value": values[0]["Value"],
        "lower_bound": values[1]["Value"], "upper_bound": values[2]["Value"], "unit": "Log Units",
        "runtime_mappings": [{"runtime_path": "Alprazolam|Lipophilicity", "multiplier": 1.0}],
        "search_scale": "linear", "sensitivity_scale": "bounded_linear",
        "bound_semantics": "historical_numerical_search_guards", "anchor_may_be_historically_fitted": True,
        "source": _reference(*SOURCE, "ParameterIdentifications[Parameter Identification].Lipophilicity Start value/MinValue/MaxValue; historical input")}
    return [shared, lipophilicity]


def prepare_alprazolam_inputs(project_root: Path, output_dir: Path) -> dict:
    """Prepare 6 publication groups/7 IV curves; no solver is invoked."""
    root = Path(project_root).resolve()
    output = (root / output_dir).resolve()
    if not output.is_relative_to(root) or output == root:
        raise DevelopmentCaseError("Alprazolam output must stay in this project.")
    snapshot = _verify(root, *SOURCE, parse=True)
    inventory = _verify(root, *INVENTORY, parse=True)
    _verify(root, *LITERATURE)
    _verify(root, *BINDING)
    identity = _verify(root, *IDENTITY, parse=True)["response"]["PropertyTable"]["Properties"][0]
    if identity["CID"] != 2118 or identity["MolecularFormula"] != "C17H13ClN4":
        raise DevelopmentCaseError("Alprazolam identity changed.")
    definition = load_standard_human_model(root)
    compound = _plain(_one(snapshot["Compounds"], "Name", "Alprazolam"))
    specs = _specifications(snapshot, compound)
    compound.pop("IntestinalPermeability", None)
    compound.pop("Permeability", None)
    compound["Description"] = "Historical reconstruction with two preserved nonlinear CYP3A4 branches; regenerated standard human anatomy."
    compound["PlasmaProteinBindingPartner"] = "Albumin"
    compound["CalculationMethods"] = ["Cellular partition coefficient method - PK-Sim Standard", "Cellular permeability - PK-Sim Standard"]
    _one(compound["Lipophilicity"][0]["Parameters"], "Name", "Lipophilicity")["Value"] = specs[1]["initial_value"]
    for p in compound["Processes"]:
        p.setdefault("Species", "Human")
    sol = _one(compound["Solubility"][0]["Parameters"], "Name", "Solubility at reference pH")
    if sol != {"Name": "Solubility at reference pH", "Value": 40.0, "Unit": "mg/l"}:
        raise DevelopmentCaseError("Alprazolam solubility changed.")
    sol.update(Value=0.04, Unit="mg/ml")
    mw = _one(compound["Parameters"], "Name", "Molecular weight")
    compound["Parameters"] = [mw, *({"Name": atom, "Value": n} for atom, n in _formula_counts(identity["MolecularFormula"]).items())]
    sources = {field: _reference(*SOURCE, f"Compounds[Alprazolam].{field}; disclosed historical reconstruction")
               for field in compound if field not in {"Name", "Description", "IsSmallMolecule"}}
    sources["Parameters"] = _reference(*IDENTITY, "PubChem2118 formula C17H13ClN4 supplies halogens; molecular weight308.765 retained from locked public model rather than rounded PubChem308.8")
    sources["Lipophilicity"] = copy.deepcopy(specs[1]["source"])
    sources["PlasmaProteinBindingPartner"] = _reference(*BINDING, "FDA XANAX label2021 distribution: predominant serum albumin binding; fu kept from independent public-model input")
    profile_id = "CYP3A4|Human|European (P-gp modified, CYP3A4 36 h)"
    profile = _plain(_one([p for p in snapshot["ExpressionProfiles"] if p["Molecule"] == "CYP3A4"], "Category", profile_id.split("|", 2)[2]))
    entry = _one(inventory["entries"], "path", SOURCE[0])
    selected = [s for s in entry["eligible_scenarios"] if s["study_id"] in STUDY_ROLES]
    if len(selected) != 7:
        raise DevelopmentCaseError("Frozen Alprazolam screening membership changed.")
    scenarios, external, audit = [], [], []
    for item in selected:
        study, role = item["study_id"], STUDY_ROLES[item["study_id"]]
        simulation = _one(snapshot["Simulations"], "Name", item["simulation"])
        observed = _one(snapshot["ObservedData"], "Name", item["observed_name"])
        meta = _metadata(observed)
        if meta.get("Study Id") != study or meta.get("Species") != "Human" or meta.get("Molecule") != "Alprazolam" or meta.get("Route") != "IV" or "agg." not in observed["Name"]:
            raise DevelopmentCaseError("Only predeclared aggregate adult-human parent-plasma IV groups are admitted.")
        identifier = "alprazolam_" + _slug(item["simulation"])
        group = "alprazolam_" + _slug(study)
        protocol = _normalise_protocol(_one(snapshot["Protocols"], "Name", item["protocol"]), identifier)
        if protocol["ApplicationType"] != "Intravenous":
            raise DevelopmentCaseError("This registered cohort contains short infusions only.")
        original_individual = _one(snapshot["Individuals"], "Name", simulation["Individual"])
        if original_individual["Name"] != "European (P-gp modified, CYP3A4 36 h)":
            raise DevelopmentCaseError("The selected source arm must use the declared Caucasian human reference.")
        individual = {k: _plain(original_individual[k]) for k in ("Name", "Seed", "OriginData")}
        individual["OriginData"].pop("CalculationMethods", None)
        individual["ExpressionProfiles"] = [profile_id]
        individual["Name"] = "IND_Alprazolam_Reference"
        grid = observed["BaseGrid"]
        if grid["Unit"] != "h":
            raise DevelopmentCaseError("Frozen Alprazolam source grid must be hours.")
        times = [t * 60 for t in grid["Values"]]
        schema = _output_schema(simulation)
        ends = [_one(interval["Parameters"], "Name", "End time")["Value"] * 60 for interval in schema]
        if times[-1] > max(ends):
            raise DevelopmentCaseError("Observation extends beyond the native source output horizon.")
        sampling = qualify_sampled_endpoints(times, source=_reference(*SOURCE, f"ObservedData[{observed['Name']}].BaseGrid"))
        dose = _one(protocol["Parameters"], "Name", "InputDose")
        target = "Organism|PeripheralVenousBlood|Alprazolam|Plasma (Peripheral Venous Blood)"
        requirement = {"compound_semantic_id": "pubchem:2118", "name": "Alprazolam", "population": "healthy_adult",
            "route": "intravenous", "dose": dose["Value"], "dose_unit": dose["Unit"], "formulation": None,
            "targets": [{"target_id": "alprazolam_parent_plasma", "analyte": "parent", "matrix": "plasma", "quantity": "concentration", "binding": "total", "required": True}]}
        instance = {"population_context": "healthy_adult", "individual": individual,
            "generated_demographics": ["Weight", "Height"], "compound": copy.deepcopy(compound), "protocol": protocol,
            "snapshot_version": 80, "simulation_name": identifier,
            "human_source": _reference(*SOURCE, "Individuals[European (P-gp modified, CYP3A4 36 h)].OriginData; generic male age30, not measured study mean; no source organ overrides"),
            "compound_sources": copy.deepcopy(sources), "protocol_source": _reference(*SOURCE, f"Protocols[{item['protocol']}]; exact source dose and infusion time"),
            "formulation": None, "formulation_source": None, "expression_profiles": [copy.deepcopy(profile)],
            "expression_profile_sources": {profile_id: _reference(*SOURCE, "ExpressionProfiles[" + profile_id + "]; historical reference")},
            "output_schema": schema, "output_selections": [target]}
        # Role assignment above precedes decoding. External values are emitted
        # only into a separate evaluator file; they never enter the case bundle.
        observations = _external_observations(observed, mw["Value"])
        for index, obs in enumerate(observations):
            obs["observation_id"] = identifier + f":row_{index}"
        document = {"schema_version": "alprazolam-study-1.0", "scenario_id": identifier, "compound_id": "pubchem:2118",
            "study_group_id": group, "source_study_id": study, "role": role, "tier": "reconstruction",
            "metadata": meta, "unit": "umol/l", "time_unit": "min", "observations": observations,
            "source": _reference(*SOURCE, f"ObservedData[{observed['Name']}]"),
            "source_governance": {"historical_public_model_data": True, "newly_blinded_independent_test": False,
                "cross_publication_subject_deduplication": "unverified", "independence_unit": "publication_group_not_file"}}
        path = output / ("external_observations" if role == "external" else "study_inputs") / (identifier + ".json")
        ref = {"path": path.relative_to(root).as_posix(), "sha256": _write_once(path, document)}
        study_ref = {"study_id": group, "source_id": ref["path"], "source_checksum": ref["sha256"],
            "role": role, "cohort_ids": [group], "observed_target_ids": ["alprazolam_parent_plasma"]}
        request = {"requirement": requirement, "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
            "instance": instance, "base": None, "modules": [], "evidence": [], "studies": [] if role == "external" else [study_ref]}
        checked = BlueprintDesignRequest.model_validate(request).instance.model_dump(mode="json", exclude_none=True)
        checked.pop("population_context")
        build_human_model_snapshot(**checked, expected_compound_name="Alprazolam")
        scenario = {"scenario_id": identifier, "compound_id": "pubchem:2118", "study_group_id": group, "role": role,
            "tier": "reconstruction", "design_request": request, "target_output": target, "unit": "umol/l", "time_unit": "min",
            "observation_count": len(times), "source_study_id": study, "source_observation_name": observed["Name"],
            "source_audit": {"source": ref, "original_container": SOURCE[0], "original_container_sha256": SOURCE[1],
                "historical_model_data": True}, **sampling}
        if role == "external":
            scenario["holdout_reference"] = ref
            external.append(scenario)
        else:
            scenario["observations"] = observations
            scenarios.append(scenario)
            audit.append(ref)
    case = {"case_id": "alprazolam_multistudy_iv_reconstruction", "compound_id": "pubchem:2118", "compound_name": "Alprazolam",
        "tier": "reconstruction", "purpose": "historical_development_reconstruction", "strict_main_eligible": False,
        "design_request": copy.deepcopy(scenarios[0]["design_request"]), "requirement": copy.deepcopy(scenarios[0]["design_request"]["requirement"]),
        "scenarios": scenarios, "forbidden_study_group_ids": sorted({s["study_group_id"] for s in external}),
        "parameter_search": {"parameters": specs, "initial_values_are_optimal": False, "anchor_may_be_historically_fitted": True,
            "dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "calibration"],
            "selection_dataset_ids": [s["scenario_id"] for s in scenarios if s["role"] == "selection"],
            "objective": "study_curve_equal_log_rmse", "aggregation": "sqrt(mean_study(mean_curve(mean_point(log_error_squared))))",
            "identifiability_verified": False, "runtime_units_must_be_verified": True},
        "study_counts": {role: len({s["study_group_id"] for s in scenarios + external if s["role"] == role}) for role in ("calibration", "selection", "external")},
        "source_audit": [{"path": p, "sha256": h} for p, h in (SOURCE, IDENTITY, BINDING, LITERATURE, INVENTORY)] + audit,
        "limitations": ["All selected publication groups informed the prior public model; no independent-main claim is made.",
            "Source kcat factor anchors and GFR fraction are historical fits; this exercise cannot demonstrate independent de novo prior selection.",
            "The source kcat PI UseAsFactor couples two unequal CYP3A4 branches; Km and in-vitro Vmax are retained, not linearized.",
            "Generic source male age30 physiology is regenerated; measured cohort mean physiology and cross-publication participant deduplication are unavailable.",
            "Kroboth1988 two doses remain one calibration study; the continuous-infusion arm and Adams1984 individual profiles were excluded by protocol/data-type qualification before scoring.",
            "Only short single IV infusion parent-plasma endpoints are assessed; no oral, DDI or metabolite accuracy is claimed.",
            "Source simulation organ parameters, solved structures, fitted cellular/intestinal permeability and output results are not copied.",
            "Numerical bounds are historical search guards, not biological confidence intervals.",
            "AUC is restricted to observed-window linear trapezoids; Cmax is sampled maximum, not continuous peak."]}
    bundle = {"schema_version": "alprazolam-reconstruction-input-1.0", "case": case, "external_scenarios": external,
              "solver_calls": 0, "scientific_accepted": False}
    path = output / "input_package.json"
    checksum = _write_once(path, bundle)
    return {"case_id": case["case_id"], "compound_id": case["compound_id"], "compound_name": "Alprazolam",
        "tier": "reconstruction", "strict_main_eligible": False, "bundle_path": path.relative_to(root).as_posix(),
        "bundle_sha256": checksum, "study_counts": case["study_counts"], "source": {"path": SOURCE[0], "sha256": SOURCE[1]},
        "status": "pending_native_contract_and_execution", "solver_calls": 0}
