"""An explicit original-source execution mapping; no parameter-variant aliasing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VERSION = "original-source-design-execution-mapping-1"
SOURCE_SHA256 = "77d4db0edcb3811789c0289534b585ee286d3071c21f8a71cc92dd71b0816467"
AMENDMENT = "experiments/model_recommendation/execution_validation_20261001/engineering_amendments/json_identity_followup_20261002/posaconazole_source_design_amendment.json"
AMENDMENT_SHA256 = "6a76605d75bdaa0cd2ea0ce058d99177ff9018fa309b8f885ab60b3fbb3fda83"


def _require(ok, reason):
    if not ok: raise ValueError("SOURCE_DESIGN_AMENDMENT_" + reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _checked(ref, root):
    path = (root / ref["path"]).resolve(strict=True)
    _require(path.is_relative_to(root) and not any("sealed" in p.lower() for p in path.parts), "PROJECT_METADATA_PATH_REQUIRED")
    _require(hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"], "EVIDENCE_CHANGED")
    return path


def _one(rows, name):
    matches = [row for row in rows if row.get("Name") == name]
    _require(len(matches) == 1, "UNIQUE_ORIGINAL_ENTITY_REQUIRED:" + name)
    return matches[0]


def _source_selection(document, amendment):
    simulation = _one(document.get("Simulations", []), amendment["source_simulation"])
    semantics = {k: simulation.get(k) for k in ["Name", "Model", "Individual", "Compounds", "Parameters",
        "Interactions", "ObserverSets", "OutputSelections", "OutputSchema", "Solver"]}
    _require(_digest(semantics) == amendment["source_simulation_semantics_sha256"], "ORIGINAL_SIMULATION_SEMANTICS_CHANGED")
    administered = [c for c in simulation["Compounds"] if c.get("Protocol")]
    _require(len(administered) == 1 and administered[0]["Name"] == amendment["native_name"], "EXACT_ORIGINAL_SINGLE_PARENT_REQUIRED")
    selection = administered[0]
    parent = _one(document["Compounds"], amendment["native_name"])
    protocol = _one(document["Protocols"], selection["Protocol"]["Name"])
    _require(len(selection["Protocol"].get("Formulations", [])) == 1, "EXACT_ORIGINAL_FORMULATION_REQUIRED")
    formulation = _one(document["Formulations"], selection["Protocol"]["Formulations"][0]["Name"])
    individual = _one(document["Individuals"], simulation["Individual"])
    checks = {"compound_definition_sha256": parent, "source_selection_sha256": selection,
        "source_protocol_sha256": protocol, "source_formulation_sha256": formulation,
        "source_individual_sha256": individual, "source_parameters_sha256": simulation["Parameters"],
        "source_compound_definitions_sha256": document["Compounds"],
        "source_expression_profiles_sha256": document.get("ExpressionProfiles")}
    _require(all(_digest(value) == amendment[key] for key, value in checks.items()), "ORIGINAL_PARAMETERS_OR_MECHANISMS_CHANGED")
    _require(individual.get("OriginData", {}).get("Species") == "Human"
        and simulation.get("Model") == "4Comp" and not simulation.get("Population"), "ORIGINAL_HUMAN_INDIVIDUAL_REQUIRED")
    _require(protocol.get("ApplicationType") == "Oral" and protocol.get("DosingInterval") == "Single"
        and "Schemas" not in protocol, "ORIGINAL_SINGLE_ORAL_PROTOCOL_REQUIRED")
    dose = _one(protocol["Parameters"], "InputDose"); start = _one(protocol["Parameters"], "Start time")
    _require(dose.get("Value") == 100.0 and dose.get("Unit") == "mg"
        and start.get("Value") == 0.0 and start.get("Unit") == "h", "ORIGINAL_DOSE_OR_TIME_CHANGED")
    _require(formulation.get("FormulationType") == "Formulation_Tablet_Weibull", "ORIGINAL_TABLET_FORMULATION_REQUIRED")
    _require(not any(p.get("Path", "").split("|")[-1] in {"Dose", "InputDose", "Start time", "Infusion time"}
        and p.get("Path", "").startswith(("Events|", "Applications|")) for p in simulation.get("Parameters", [])), "ORIGINAL_ADMINISTRATION_OVERRIDE_NOT_SUPPORTED")
    return selection


def prepare_source_design_amendment(provenance, clinical_contract, *, project_root):
    """Return a source-specific mapping or None. The frozen scenario is untouched.

    The shared bridge must restrict its already verified source/project/simulation
    IDs to source_simulation below before normal route/regimen/output validation.
    """
    if provenance.get("source_sha256") != SOURCE_SHA256 or clinical_contract.get("compound") != "posaconazole":
        return None
    root = Path(project_root).resolve(strict=True)
    amendment_ref = {"path": AMENDMENT, "sha256": AMENDMENT_SHA256}
    amendment = json.loads(_checked(amendment_ref, root).read_text())
    _require(_digest(clinical_contract) == amendment["clinical_contract_sha256"], "EXACT_FROZEN_CLINICAL_CONTRACT_REQUIRED")
    _require(clinical_contract.get("route") == "Oral" and clinical_contract.get("dose_mg") == 100.0
        and clinical_contract.get("administration_times_min") == [0.0], "FROZEN_ROUTE_REGIMEN_REQUIRED")
    source = _checked(amendment["source"], root)
    _require(source == (root / provenance["source_path"]).resolve(strict=True), "ORIGINAL_SOURCE_PATH_DIFFERS")
    for key in ("author_document", "scenarios", "source_receipt", "verified_snapshot"):
        _checked(amendment[key], root)
    from .snapshot_repair import _verify_receipt
    receipt = _verify_receipt(root / amendment["source_receipt"]["path"], provenance, root)
    _require({k: receipt["snapshot"][k] for k in ("path", "sha256")} == amendment["verified_snapshot"], "VERIFIED_SNAPSHOT_DIFFERS")
    selection = _source_selection(json.loads(source.read_text()), amendment)
    return {"version": VERSION, "kind": "explicit_source_design_execution_mapping", "amendment": amendment_ref,
        "native_name": amendment["native_name"], "clinical_parent": clinical_contract["compound"],
        "source_simulation": amendment["source_simulation"], "source": deepcopy(amendment["source"]),
        "source_sha256": SOURCE_SHA256, "clinical_contract_sha256": amendment["clinical_contract_sha256"],
        "original_frozen_native_hint": deepcopy(amendment["original_native_hint"]),
        "source_selection": deepcopy(selection), "source_selection_sha256": amendment["source_selection_sha256"],
        "compound_definition_sha256": amendment["compound_definition_sha256"],
        "source_parameters_sha256": amendment["source_parameters_sha256"],
        "basis": amendment["reason"], "selection_rule": amendment["selection_rule"],
        "compatibility": deepcopy(amendment["compatibility"]),
        "evidence_refs": [amendment_ref] + [deepcopy(amendment[k]) for k in
            ("source", "author_document", "scenarios", "source_receipt", "verified_snapshot")],
        "parameter_variant_alias_equivalence_claimed": False, "original_coefficients_changed": False,
        "frozen_scenario_or_candidate_modified": False, "native_execution_verified": False}
