"""Bind synthetic native inputs to a tiny provider; never load or execute PK-Sim."""
from copy import deepcopy
from pathlib import Path

import pytest

from pbpk_kb.blueprints import human_model, human_model_snapshot
from pbpk_kb.blueprints.design import design_blueprint
from pbpk_kb.blueprints.design_contracts import BlueprintDesignRequest
from test_human_model_snapshot import _request as _native_fixture

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFINITION_CHECKSUM = "d" * 64


def _payload(*, oral=False):
    instance = _native_fixture(oral=oral)
    name = instance.pop("expected_compound_name")
    instance["population_context"] = "healthy_adult"
    return {
        "human_model": {"definition_checksum": DEFINITION_CHECKSUM},
        "requirement": {
            "compound_semantic_id": "compound:synthetic", "name": name,
            "route": "oral" if oral else "intravenous", "dose": 10, "dose_unit": "mg",
            "population": "healthy_adult",
            "targets": [{"target_id": "plasma", "matrix": "plasma", "quantity": "concentration"}],
        },
        "instance": instance,
    }


@pytest.fixture
def provider(monkeypatch):
    definition = {
        "checksum": DEFINITION_CHECKSUM,
        "model": {"model_id": "4Comp", "species": "Human", "configuration_id": "4Comp|Human",
                  "source_version": "12.3.173"},
        "resources": [{"path": "synthetic/provider.db", "sha256": "b" * 64}],
        "reference_structure": {"containers": {"rows": [
            {"container_id": 0, "parent_container_id": None, "container_name": "ROOT", "container_type": "ROOT", "parent_container_name": "ROOT"},
            {"container_id": 1, "parent_container_id": 0, "container_name": "Organism", "container_type": "ORGANISM", "parent_container_name": "ROOT"},
            {"container_id": 2, "parent_container_id": 1, "container_name": "Liver", "container_type": "ORGAN", "parent_container_name": "Organism"},
            {"container_id": 3, "parent_container_id": 1, "container_name": "PeripheralVenousBlood", "container_type": "COMPARTMENT", "parent_container_name": "Organism"},
        ]}},
    }
    calls = {"load": 0, "audit": 0, "build": 0}
    real_builder = human_model_snapshot.build_human_model_snapshot

    def load(root):
        assert root == PROJECT_ROOT
        calls["load"] += 1
        return deepcopy(definition)

    def validate(candidate, root):
        assert root == PROJECT_ROOT and candidate == definition
        calls["audit"] += 1
        return {"accepted": True, "errors": []}

    def build(**values):
        calls["build"] += 1
        return real_builder(**values)

    monkeypatch.setattr(human_model, "load_standard_human_model", load)
    monkeypatch.setattr(human_model, "validate_human_model_resources", validate)
    monkeypatch.setattr(human_model_snapshot, "build_human_model_snapshot", build)
    return definition, calls


def _design(payload):
    return design_blueprint(BlueprintDesignRequest.model_validate(payload), project_root=PROJECT_ROOT)


def _codes(result):
    return {gap["code"] for candidate in result["candidates"] for gap in candidate["gaps"]}


def _module(payload, module_id="extension", **fields):
    return {
        "module_id": module_id, "version": "synthetic-1", "mechanism": "synthetic-extension",
        "requires_evidence": False, "provenance": [payload["instance"]["human_source"]],
        **fields,
    }


def test_unbound_request_remains_incomplete_and_never_loads_provider(provider):
    payload = _payload()
    del payload["human_model"]
    result = _design(payload)
    assert "BASE_MODEL_UNBOUND" in _codes(result)
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])
    assert provider[1] == {"load": 0, "audit": 0, "build": 0}
    assert result["human_model_binding"]["inherited_roles"] == []


def test_definition_checksum_mismatch_stops_before_instance_build(provider):
    payload = _payload()
    payload["human_model"]["definition_checksum"] = "e" * 64
    result = _design(payload)
    assert "BASE_MODEL_DEFINITION_CHECKSUM_MISMATCH" in _codes(result)
    assert provider[1]["build"] == 0
    assert result["human_model_binding"]["model_reference"] is None


@pytest.mark.parametrize("failure", ["unavailable", "audit_failed"])
def test_provider_failure_cannot_fall_back_to_fabricated_base(provider, monkeypatch, failure):
    if failure == "unavailable":
        def unavailable(root):
            raise OSError("synthetic resource absent")
        monkeypatch.setattr(human_model, "load_standard_human_model", unavailable)
        expected = "BASE_MODEL_RESOURCE_UNAVAILABLE"
    else:
        monkeypatch.setattr(human_model, "validate_human_model_resources",
                            lambda definition, root: {"accepted": False, "errors": ["SYNTHETIC_CHECKSUM_MISMATCH"]})
        expected = "BASE_MODEL_RESOURCE_CHECK_FAILED"
    result = _design(_payload())
    assert expected in _codes(result)
    assert provider[1]["build"] == 0
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])


@pytest.mark.parametrize("oral", [False, True])
def test_native_base_inherits_roles_and_needs_no_additional_modules(provider, oral):
    payload = _payload(oral=oral)
    before = deepcopy(payload)
    result = _design(payload)
    assert payload == before and "base" not in payload
    assert result["human_base"] is None
    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["status"] == "draft_assembled"
    assert set(candidate["inherited_roles"]) == {"distribution", "input", "elimination"}
    assert candidate["inherited_targets"] == ["plasma"]
    assert candidate["target_mapping"] == {"plasma": ["@base:4Comp|Human"]}
    assert candidate["module_ids"] == candidate["edits"] == candidate["parameters"] == []
    binding = result["human_model_binding"]
    assert binding["status"] == "standard_model_snapshot_prepared"
    assert binding["instance"]["snapshot"]["Simulations"][0]["Model"] == "4Comp"
    assert binding["instance"]["manifest"]["normalized_administration"]["route"] == payload["requirement"]["route"]
    assert binding["instance"]["manifest"]["individual_seed"] == 17
    assert candidate["compile_status"] == candidate["simulation_status"] == "not_run"
    assert candidate["scientific_status"] == "not_evaluated"
    assert binding["instance"]["manifest"]["execution"] == {"solver_calls": 0, "llm_calls": 0, "gpu_jobs": 0}


@pytest.mark.parametrize("kind", ["module_id", "parameter"])
def test_optional_edits_cannot_mutate_registered_core(provider, kind):
    payload = _payload()
    module = _module(payload)
    if kind == "module_id":
        module["module_id"] = "@base:distribution"
    else:
        module["parameter_specs"] = [{"semantic_id": "physiology:volume", "dimension": "volume", "unit": "L"}]
    payload["modules"] = [module]
    result = _design(payload)
    assert {"module_id": module["module_id"], "reason": "PROTECTED_HUMAN_BASE_MUTATION"} in result["unavailable_modules"]
    assert all(module["module_id"] not in c["module_ids"] for c in result["candidates"])
    assert result["human_model_binding"]["instance"]["snapshot"]["Individuals"][0] == payload["instance"]["individual"]


@pytest.mark.parametrize("role", ["input", "distribution", "elimination", "metabolism"])
def test_generic_modules_cannot_duplicate_inherited_native_processes(provider, role):
    payload = _payload()
    payload["modules"] = [_module(payload, role=role)]
    result = _design(payload)
    assert {"module_id": "extension", "reason": "BASE_PROCESS_REPLACEMENT_ADAPTER_REQUIRED"} in result["unavailable_modules"]
    assert all(c["module_ids"] == [] for c in result["candidates"])


@pytest.mark.parametrize("field,value", [("route", "oral"), ("dose", 11), ("dose_unit", "mg/kg")])
def test_native_administration_must_match_requested_route_and_dose(provider, field, value):
    payload = _payload()
    payload["requirement"][field] = value
    result = _design(payload)
    assert "INSTANCE_REGIMEN_MISMATCH" in _codes(result)
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])
    assert result["human_model_binding"]["instance"] is None


def test_missing_native_instance_does_not_claim_a_complete_model(provider):
    payload = _payload()
    del payload["instance"]
    result = _design(payload)
    assert "HUMAN_DRUG_INSTANCE_NOT_PROVIDED" in _codes(result)
    assert result["human_model_binding"]["inherited_roles"] == ["distribution"]
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])


def test_missing_seed_returns_controlled_instance_gap(provider):
    payload = _payload()
    del payload["instance"]["individual"]["Seed"]
    result = _design(payload)
    assert "HUMAN_DRUG_INSTANCE_INVALID" in _codes(result)
    assert result["human_model_binding"]["instance"] is None


@pytest.mark.parametrize("age,population", [(17, "healthy_adult"), (30, "unverified_population")])
def test_population_scope_is_not_inferred_from_standard_individual(provider, age, population):
    payload = _payload()
    payload["instance"]["individual"]["OriginData"]["Age"]["Value"] = age
    payload["requirement"]["population"] = payload["instance"]["population_context"] = population
    result = _design(payload)
    assert "INSTANCE_POPULATION_UNSUPPORTED" in _codes(result)
    assert result["human_model_binding"]["instance"] is None


def test_zero_clearance_does_not_claim_inherited_elimination(provider):
    payload = _payload()
    payload["instance"]["compound"]["Processes"][0]["Parameters"][1]["Value"] = 0
    result = _design(payload)
    assert "elimination" not in result["human_model_binding"]["inherited_roles"]
    assert "REQUIRED_ROLE_MISSING" in _codes(result)
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])


def test_native_arrays_are_preserved_instead_of_sorted_as_metadata(provider):
    payload = _payload()
    instance = payload["instance"]
    instance["compound"]["Parameters"].reverse()
    instance["compound"]["CalculationMethods"].reverse()
    instance["protocol"]["Parameters"].reverse()
    first = deepcopy(instance["output_schema"][0])
    second = deepcopy(first)
    # Chronologically valid order would be inverted by the metadata JSON sorter.
    first["Parameters"][1]["Value"] = 2
    second["Parameters"][0]["Value"] = 2
    second["Parameters"][1]["Value"] = 12
    instance["output_schema"] = [first, second]
    result = _design(payload)
    assert "HUMAN_DRUG_INSTANCE_INVALID" not in _codes(result)
    snapshot = result["human_model_binding"]["instance"]["snapshot"]
    assert snapshot["Compounds"][0] == instance["compound"]
    assert snapshot["Protocols"][0] == instance["protocol"]
    assert snapshot["Simulations"][0]["OutputSchema"] == instance["output_schema"]


@pytest.mark.parametrize("missing", ["output_schema", "output_selections"])
def test_missing_native_output_configuration_remains_a_gap(provider, missing):
    payload = _payload()
    del payload["instance"][missing]
    result = _design(payload)
    assert "EXECUTION_CONFIGURATION_MISSING" in _codes(result)
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])
    assert missing in result["human_model_binding"]["instance"]["manifest"]["missing_execution_configuration"]


def test_native_observer_must_match_required_parent_plasma_target(provider):
    payload = _payload()
    payload["instance"]["output_selections"] = ["Organism|Liver|ToyDrug|Concentration"]
    result = _design(payload)
    assert "INSTANCE_OUTPUT_MAPPING_MISMATCH" in _codes(result)
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])


def test_bound_model_compartments_come_from_provider_not_handwritten_metadata(provider):
    payload = _payload()
    payload["modules"] = [_module(payload, required_compartments=["Organism|Liver"])]
    result = _design(payload)
    extension = [c for c in result["candidates"] if "extension" in c["module_ids"]]
    assert extension
    assert all("PHYSIOLOGY_COMPARTMENT_MISSING" not in {g["code"] for g in c["gaps"]} for c in extension)
    payload["modules"][0]["required_compartments"] = ["Organism|InventedOrgan"]
    result = _design(payload)
    assert any("PHYSIOLOGY_COMPARTMENT_MISSING" in {g["code"] for g in c["gaps"]}
               for c in result["candidates"] if "extension" in c["module_ids"])


def test_held_out_resource_checksum_prevents_provider_binding(provider):
    payload = _payload()
    payload["studies"] = [{"study_id": "holdout", "source_id": "held-out-alias", "source_checksum": "b" * 64,
                           "role": "test", "cohort_ids": ["holdout-cohort"], "observed_target_ids": ["plasma"]}]
    result = _design(payload)
    assert "BASE_MODEL_SOURCE_ISOLATION_FAILED" in _codes(result)
    assert provider[1]["build"] == 0


def test_neighborhood_and_event_rows_are_not_anatomical_compartments(provider):
    definition, _ = provider
    definition["reference_structure"]["containers"]["rows"].extend([
        {"container_id": 10, "parent_container_id": 1, "parent_container_name": "Organism",
         "container_name": "LiverBloodConnection", "container_type": "NEIGHBORHOOD"},
        {"container_id": 11, "parent_container_id": 1, "parent_container_name": "Organism",
         "container_name": "Events", "container_type": "EVENT_GROUP"},
    ])
    result = _design(_payload())
    assert result["human_model_binding"]["compartment_paths"] == [
        "Organism", "Organism|Liver", "Organism|PeripheralVenousBlood",
    ]


def test_nonroot_parent_gap_blocks_native_instance_build(provider):
    definition, calls = provider
    row = definition["reference_structure"]["containers"]["rows"][2]
    row.update(parent_container_id=999, parent_container_name="MissingParent")
    result = _design(_payload())
    assert "BASE_MODEL_COMPARTMENT_PATH_UNRESOLVED" in _codes(result)
    assert calls["build"] == 0
    assert all(c["status"] == "draft_with_gaps" for c in result["candidates"])
