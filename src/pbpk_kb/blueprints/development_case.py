"""Verified, historical caffeine inputs for developing the standard-model adapter.

This is a reconstruction case, not an independent evaluation set or a newly
validated PBPK model. It imports explicit drug, expression, individual and dose
inputs, never a previous simulation's structure, equations, PKML or predictions.
The calibration-only configuration is read instead of the mixed observation
dataset or locked validation package. Source containers are hashed without
decoding their other observations. No file is written and no solver is invoked.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .human_model import load_standard_human_model


_ASSETS = "artifacts/pbpk-model-assets/"
_CAFFEINE = _ASSETS + "caffeine_selfbuild/"
_DATASET = "blanchard_1983_oral_5mgkg"
_STUDY = "blanchard_1983_absolute_bioavailability"
_SIMULATION = "CAL_BLANCHARD_PO"
_TARGET = "caffeine_parent_plasma"
_PROFILE = "CYP1A2|Human|PBPK-KB-Caffeine-Adult-v1"
_INCHIKEY = "RYYVLZVUVIJVGH-UHFFFAOYSA-N"
# These locks identify the inspected historical inputs, not newly trusted files
# at client-supplied paths. Additional primary sources are pinned by these files.
_INPUTS = {
    "snapshot": (_CAFFEINE + "builds/caffeine_study_scenarios_v5/study_snapshot.json",
                 "f7f5f0eec5b2d31676014e9e6802f5877fac636fa9e8e8dd6c33fe5aca8ef017"),
    "calibration_config": (_CAFFEINE + "calibration/caffeine_fit_validation_v4_inputs/identification_config.json",
                           "3f3ea2db3760d2b58220294e05a394d672ccb80c5eb8ccf40f26a041eb7ab605"),
    "parameter_sources": (_CAFFEINE + "parameters/caffeine_parameter_manifest_v1/caffeine_parameter_manifest.json",
                          "40eb6d90444beedd204f8f269afab3ce00bfbab04860124c428fb4ac863fc86c"),
    "parameters": (_CAFFEINE + "parameters/caffeine_parameter_manifest_v3/caffeine_parameter_manifest.json",
                   "aa6fc9627142aa4f17de870ae85b0e51491077852c29957b90acd34d5a3b9a27"),
    "dataset_manifest": (_CAFFEINE + "datasets/caffeine_dataset_v3/dataset_split_manifest.json",
                         "f45cfe13e4f83790077f8a85f7b880fc61463fbdd84f305d1bc1ed654b6d1b61"),
    "scenario_manifest": (_CAFFEINE + "builds/caffeine_study_scenarios_v5/study_scenario_manifest.json",
                          "987d5787639012995f5afe72d0cf73c5991e7d25dd478f00bbce4619e44d933d"),
}


class DevelopmentCaseError(ValueError):
    """A missing, changed or incompatible development input cannot be used."""


def _one(values: list[dict], key: str, expected: str) -> dict:
    found = [value for value in values if value.get(key) == expected]
    if len(found) != 1:
        raise DevelopmentCaseError(f"Expected one {key}={expected!r}.")
    return found[0]


def _canonical_checksum(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _verify(root: Path, relative: str, checksum: str, *, parse: bool = False) -> Any:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise DevelopmentCaseError("Source path is not project-relative.")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise DevelopmentCaseError(f"Source missing or outside project: {relative}")
    # Fixed case inputs are small; the original observation container is ~3 MB.
    if resolved.stat().st_size > 8 * 1024 * 1024:
        raise DevelopmentCaseError(f"Unexpectedly large development source: {relative}")
    digest, chunks = hashlib.sha256(), []
    with resolved.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
            if parse:
                chunks.append(chunk)
    if digest.hexdigest() != checksum:
        raise DevelopmentCaseError(f"Source checksum mismatch: {relative}")
    if parse:
        return json.loads(b"".join(chunks))
    return None


def _reference(path: str, checksum: str, location: str) -> dict:
    # Model-asset provenance intentionally does not upgrade a historical model
    # assumption, search start or transformed value to validated evidence.
    return {
        "provenance_id": str(uuid5(NAMESPACE_URL, path + "#" + location)),
        "provenance_kind": "model_asset", "source_id": path,
        "source_location": location, "source_checksum": checksum,
    }


def _plain(value: Any) -> Any:
    """Keep source semantics in references; strip unsupported inline origins."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items() if key != "ValueOrigin"}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def make_caffeine_development_case(project_root: Path) -> dict[str, Any]:
    """Return a source-checked request, 11 PO calibration points and fit bounds.

    Only the Blanchard oral study is exposed to a caller. Study-level selection
    and test sets are absent. Historical bounds and initials are engineering
    search inputs, not posterior estimates or independent biological limits.
    Population weight/height are explicitly unobserved and delegated to the
    locked PK-Sim generator. Paths in old files are never executed or followed.
    """
    root = Path(project_root).resolve()
    inputs = {key: _verify(root, *entry, parse=True) for key, entry in _INPUTS.items()}
    audit = [{"artifact_id": key, "path": path, "sha256": checksum,
              "verification": "passed", "operation": "read_json"}
             for key, (path, checksum) in _INPUTS.items()]

    def primary(manifest: dict, artifact_id: str, *, parse: bool = False) -> tuple[dict, Any]:
        artifact = _one(manifest["source_artifacts"], "artifact_id", artifact_id)
        relative = _ASSETS + artifact["source_location"]
        value = _verify(root, relative, artifact["sha256"], parse=parse)
        audit.append({"artifact_id": artifact_id, "path": relative, "sha256": artifact["sha256"],
                      "verification": "passed", "operation": "read_json" if parse else "hash_only",
                      "license": artifact.get("license"), "redistribution": artifact.get("redistribution")})
        return {**artifact, "path": relative}, value

    properties_source, properties_json = primary(inputs["parameter_sources"], "pubchem_2519_properties_json", parse=True)
    properties = _one(properties_json["PropertyTable"]["Properties"], "InChIKey", _INCHIKEY)
    if properties["MolecularFormula"] != "C8H10N4O2":
        raise DevelopmentCaseError("Caffeine formula does not support zero halogen counts.")
    for source in ("pubchem_2519_pka_json", "pubchem_2519_solubility_json", "lelo_1986_pdf", "birkett_1991_pdf"):
        primary(inputs["parameter_sources"], source)
    observation_source, _ = primary(inputs["dataset_manifest"], "pmc8302243_caffeine_pksim5")
    primary(inputs["dataset_manifest"], "pmc8302243_supinfo_s2_docx")

    config, snapshot = inputs["calibration_config"], inputs["snapshot"]
    if any(scenario.get("split") != "calibration" for scenario in config["calibration_scenarios"]):
        raise DevelopmentCaseError("Historical configuration contains non-calibration scenarios.")
    if (config["parameter_manifest_sha256"] != _INPUTS["parameters"][1]
            or config["dataset_manifest_sha256"] != _INPUTS["dataset_manifest"][1]
            or inputs["scenario_manifest"]["transformed_snapshot_sha256"] != _INPUTS["snapshot"][1]):
        raise DevelopmentCaseError("Historical source lineage does not match the pinned artifacts.")
    scenario = _one(config["calibration_scenarios"], "dataset_id", _DATASET)
    dataset = _one(inputs["dataset_manifest"]["datasets"], "dataset_id", _DATASET)
    if dataset["study_id"] != _STUDY or dataset["split"] != "calibration":
        raise DevelopmentCaseError("Oral calibration study identity drifted.")
    observations = []
    for item in scenario["observations"]:
        if (item["dataset_id"] != _DATASET or item["split"] != "calibration"
                or item["source_artifact_id"] != observation_source["artifact_id"]
                or not item["use_in_objective"] or item.get("exclusion_reason") is not None
                or "DATA_REPOSITORIES.Id=HzJXqJlvoU2p5p0GiaE-8g;CONTENTS.Id=274;" not in item["source_location"]):
            raise DevelopmentCaseError("Observation is not an included Blanchard oral calibration point.")
        time, concentration = item["time_min"], item["value_umol_per_l"]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
               for v in (time, concentration)):
            raise DevelopmentCaseError("Calibration points must be positive finite measurements.")
        observations.append({
            "observation_id": item["observation_id"], "time_min": time,
            "concentration_umol_per_l": concentration,
            "provenance": [
                _reference(*_INPUTS["calibration_config"], "calibration_scenarios[" + _DATASET + "].observations[" + item["observation_id"] + "]"),
                _reference(observation_source["path"], observation_source["sha256"], item["source_location"]),
            ],
        })
    if (len(observations) != 11 or {x["observation_id"] for x in observations} != set(dataset["observation_ids"])
            or any(a["time_min"] >= b["time_min"] for a, b in zip(observations, observations[1:]))):
        raise DevelopmentCaseError("Expected the 11 uniquely identified, increasing oral observations.")

    compound = _plain(_one(snapshot["Compounds"], "Name", "Caffeine"))
    mw = _one(compound["Parameters"], "Name", "Molecular weight")
    logp = _one(compound["Lipophilicity"][0]["Parameters"], "Name", "Lipophilicity")
    if mw["Value"] != float(properties["MolecularWeight"]) or logp["Value"] != properties["XLogP"]:
        raise DevelopmentCaseError("Snapshot compound identity or lipophilicity conflicts with PubChem.")
    if {p["Name"] for p in compound["Parameters"]} != {"Molecular weight"}:
        raise DevelopmentCaseError("Unexpected inherited compound parameters.")
    compound["Parameters"].extend({"Name": atom, "Value": 0} for atom in ("F", "Cl", "Br", "I"))
    solubility = _one(compound["Solubility"][0]["Parameters"], "Name", "Solubility at reference pH")
    permeability = _one(compound["IntestinalPermeability"][0]["Parameters"], "Name", "Specific intestinal permeability (transcellular)")
    if solubility["Unit"] != "mg/l" or permeability["Unit"] != "dm/min":
        raise DevelopmentCaseError("Historical source units changed; conversion requires review.")
    solubility.update(Value=solubility["Value"] / 1000, Unit="mg/ml")
    permeability.update(Value=permeability["Value"] / 6, Unit="cm/s")
    # Explicit choice: the historical simulation selected this nondefault value.
    compound["IntestinalPermeability"][0]["IsDefault"] = True
    compound["Description"] = "Historical development inputs; permeability and CYP1A2 clearance are search initial values, not validated estimates."

    individual = _plain(_one(snapshot["Individuals"], "Name", "IND_CAL_BLANCHARD_PO"))
    individual["OriginData"].pop("CalculationMethods", None)
    demographics = _one(inputs["scenario_manifest"]["inherited_individuals"], "name", individual["Name"])
    if (set(demographics["system_generated_fields"]) != {"weight", "height"}
            or "Weight" in individual["OriginData"] or "Height" in individual["OriginData"]
            or individual["ExpressionProfiles"] != [_PROFILE]):
        raise DevelopmentCaseError("Historical generated-demographics or expression binding changed.")
    profile = _plain(_one(snapshot["ExpressionProfiles"], "Molecule", "CYP1A2"))
    protocol = _plain(_one(snapshot["Protocols"], "Name", "PROTO_CAL_BLANCHARD_PO"))
    formulation = _plain(_one(snapshot["Formulations"], "Name", "FORM_BLANCHARD_AQUEOUS_DISSOLVED"))
    formulation["Parameters"] = []  # Dissolved native formulation has no release parameters.
    old_simulation = _one(snapshot["Simulations"], "Name", _SIMULATION)
    selected = _one(old_simulation["Compounds"][0]["Alternatives"], "GroupName", "COMPOUND_INTESTINAL_PERMEABILITY")
    if selected["AlternativeName"] != compound["IntestinalPermeability"][0]["Name"]:
        raise DevelopmentCaseError("Permeability was not the explicitly selected historical alternative.")

    source_path, source_sha = _INPUTS["snapshot"]
    def source(location: str) -> dict:
        return _reference(source_path, source_sha, location)

    parameter_specs = []
    for name in ("total_cyp1a2_specific_clearance", "specific_intestinal_permeability_transcellular"):
        spec = _one(config["parameters"], "parameter_id", name)
        if spec != _one(inputs["parameters"]["parameters"], "parameter_id", name):
            raise DevelopmentCaseError("Parameter search configuration differs from its source manifest.")
        parameter_specs.append({
            "parameter_id": name, "initial_value": spec["value"], "lower_bound": spec["lower_bound"],
            "upper_bound": spec["upper_bound"], "unit": spec["unit"],
            "runtime_mappings": copy.deepcopy(spec["runtime_mappings"]),
            "search_scale": "linear_nonnegative" if "clearance" in name else "log10",
            "source": _reference(*_INPUTS["calibration_config"], "parameters[" + name + "]"),
            "bound_semantics": ("historical_development_search_bounds" if "clearance" in name
                                else "historical_numerical_guards_not_biological_confidence_limits"),
        })
    clearance = parameter_specs[0]
    if not math.isclose(permeability["Value"], parameter_specs[1]["initial_value"] / 6, rel_tol=1e-12):
        raise DevelopmentCaseError("Permeability input differs from its declared search start.")
    if not math.isclose(sum(m["multiplier"] for m in clearance["runtime_mappings"]), 1.0):
        raise DevelopmentCaseError("CYP1A2 branch fractions do not sum to one.")
    for mapping in clearance["runtime_mappings"]:
        data_source = mapping["runtime_path"].removeprefix("Caffeine-CYP1A2-").removesuffix("|Specific clearance")
        process = _one(compound["Processes"], "DataSource", data_source)
        initial = _one(process["Parameters"], "Name", "Specific clearance")["Value"]
        if process["InternalName"] != "MetabolizationIntrinsic_FirstOrder" or not math.isclose(
            initial, clearance["initial_value"] * mapping["multiplier"], rel_tol=1e-12
        ):
            raise DevelopmentCaseError("Native CYP1A2 process differs from its declared search start.")

    definition = load_standard_human_model(root)
    instance = {
        "population_context": "healthy_adult", "individual": individual,
        "generated_demographics": ["Weight", "Height"],
        "compound": compound, "protocol": protocol, "snapshot_version": snapshot["Version"],
        "simulation_name": "Caffeine_Blanchard_PO_Development",
        "human_source": source("Individuals[IND_CAL_BLANCHARD_PO]; unreported Weight/Height use locked PK-Sim generation"),
        "compound_sources": {key: source("Compounds[Caffeine]." + key + "; historical development input")
                             for key in compound if key not in {"Name", "Description", "IsSmallMolecule"}},
        "protocol_source": source("Protocols[PROTO_CAL_BLANCHARD_PO]; water volume is a model assumption"),
        "formulation": formulation, "formulation_source": source("Formulations[FORM_BLANCHARD_AQUEOUS_DISSOLVED]"),
        "expression_profiles": [profile], "expression_profile_sources": {_PROFILE: source("ExpressionProfiles[" + _PROFILE + "]; database defaults and normalization rules")},
        "output_schema": _plain(old_simulation["OutputSchema"]),
        "output_selections": copy.deepcopy(old_simulation["OutputSelections"]),
    }
    instance["compound_sources"]["Parameters"] = _reference(properties_source["path"], properties_source["sha256"],
                                                            "PropertyTable.Properties[CID=2519]; MW and formula-derived zero F/Cl/Br/I")
    request = {
        "requirement": {"compound_semantic_id": "inchikey:" + _INCHIKEY, "name": "Caffeine",
                        "population": "healthy_adult", "route": "oral", "dose": 5.0, "dose_unit": "mg/kg",
                        "formulation": formulation["Name"],
                        "targets": [{"target_id": _TARGET, "analyte": "parent", "matrix": "plasma",
                                     "quantity": "concentration", "binding": "total", "required": True}]},
        "human_model": {"registry_id": definition["registry_id"], "definition_checksum": definition["checksum"]},
        "instance": instance, "base": None, "modules": [], "evidence": [],
        "studies": [{"study_id": _STUDY, "source_id": observation_source["path"],
                     "source_checksum": observation_source["sha256"], "role": "calibration",
                     "cohort_ids": [_STUDY], "observed_target_ids": [_TARGET]}],
    }
    result = {
        "schema_version": "caffeine-development-case-1.0.0", "case_id": "caffeine_blanchard_po_development",
        "purpose": "historical_development_reconstruction", "design_request": request,
        "calibration": {"dataset_id": _DATASET, "study_id": _STUDY, "role": "calibration",
                        "matrix": "plasma", "target_id": _TARGET, "observation_count": len(observations),
                        "observations": observations, "checksum": _canonical_checksum(observations)},
        "parameter_search": {
            "parameters": parameter_specs, "objective": "mean((log(Cpred)-log(Cobs))**2)",
            "dataset_ids": [_DATASET], "clearance_starts": copy.deepcopy(config["clearance_starts"]),
            "permeability_candidates": copy.deepcopy(config["permeability_candidates"]),
            "initial_values_are_optimal": False, "runtime_units_must_be_verified": True,
            "identifiability_verified": False,
        },
        "source_audit": audit,
        "limitations": [
            "Historical development reconstruction; not an independent paper benchmark or validated new model.",
            "Only the 11 Blanchard oral calibration observations are included; no selection or test observations are loaded.",
            "Original observation container is checksum-verified only; independent studies in that container are not decoded.",
            "Weight and height were not observed; PK-Sim must generate them from the explicitly sourced age, sex, population and seed.",
            "Albumin binding partner, reference pH 7 and water volume 3.5 ml/kg are historical model assumptions.",
            "XLogP is computed; expression concentrations/turnover and relative expression include PK-Sim defaults and normalization rules.",
            "CYP1A2 capacity and intestinal permeability are search starts. Oral-only fitting does not establish separate absorption and clearance identifiability.",
            "Historical permeability bounds are numerical guards; the prior absorption structure failed and external validation was not run.",
            "No clinical/source claim is upgraded to validated evidence merely because file hashes match.",
            "Observe recorded source license and redistribution restrictions; verification does not grant redistribution rights.",
        ],
        "compile_status": "not_run", "simulation_status": "not_run", "scientific_status": "not_evaluated",
    }
    # Reject accidental non-finite data and detach all returned nested objects.
    return json.loads(json.dumps(result, allow_nan=False))
