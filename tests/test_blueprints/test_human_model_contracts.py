"""Tiny contracts only: these synthetic blocks are not executable PK-Sim models."""
from copy import deepcopy
import hashlib
from uuid import UUID

import pytest
from pydantic import ValidationError

from pbpk_kb.blueprints.design_contracts import (
    BlueprintDesignRequest,
    HumanModelInstanceInput,
    RegisteredHumanModelRef,
)


def _source(name):
    digest = hashlib.sha256(name.encode()).hexdigest()
    return {
        "provenance_id": str(UUID(digest[:32])), "provenance_kind": "model_asset",
        "source_id": name, "source_location": "synthetic-fixture",
        "source_checksum": digest,
    }


def _instance():
    return {
        "population_context": "healthy_adult",
        "individual": {"Name": "Fixture individual", "OriginData": {"Species": "Human"}},
        "compound": {"Name": "Fixture compound"},
        "protocol": {"Name": "Fixture protocol"},
        "simulation_name": "Fixture design",
        "human_source": _source("human"),
        "compound_sources": {"Name": _source("compound")},
        "protocol_source": _source("protocol"),
    }


def _request():
    return {
        "requirement": {
            "compound_semantic_id": "compound:fixture", "name": "Fixture compound",
            "route": "oral", "dose": 1, "dose_unit": "mg",
            "targets": [{"target_id": "plasma", "matrix": "plasma", "quantity": "concentration"}],
        },
        "base": {
            "base_id": "fixture-base", "version": "fixture-1",
            "supported_populations": ["healthy_adult"], "provenance": [_source("base")],
            "compartments": [{"compartment_id": "central", "semantic_id": "physiology:central",
                              "name": "Fixture central", "compartment_type": "vascular",
                              "provenance": [_source("base")]}],
        },
        "instance": _instance(),
        "human_model": {"definition_checksum": "a" * 64},
    }


def test_registry_reference_is_pinned_but_does_not_assert_runtime_verification():
    reference = RegisteredHumanModelRef(definition_checksum="a" * 64)
    assert reference.registry_id == "pksim-human-standard-small-molecule"
    assert reference.model_dump() == {
        "registry_id": "pksim-human-standard-small-molecule", "definition_checksum": "a" * 64,
    }
    with pytest.raises(ValidationError):
        RegisteredHumanModelRef(registry_id="unregistered-model", definition_checksum="a" * 64)
    with pytest.raises(ValidationError):
        RegisteredHumanModelRef(definition_checksum="invalid")


def test_legacy_request_is_parseable_without_a_model_binding():
    payload = _request()
    del payload["instance"], payload["human_model"]
    request = BlueprintDesignRequest.model_validate(payload)
    assert request.human_model is None and request.instance is None


def test_explicit_instance_defaults_only_snapshot_version():
    instance = HumanModelInstanceInput.model_validate(_instance())
    assert instance.snapshot_version == 80
    assert instance.compound == {"Name": "Fixture compound"}
    assert instance.formulation is None and instance.formulation_source is None
    assert BlueprintDesignRequest.model_validate(_request()).instance is not None


@pytest.mark.parametrize("species", ["Mouse", "human", None])
def test_exact_human_snapshot_species_is_required(species):
    instance = _instance()
    instance["individual"]["OriginData"]["Species"] = species
    with pytest.raises(ValidationError, match="OriginData.Species"):
        HumanModelInstanceInput.model_validate(instance)


def test_compound_identity_cannot_change_while_reusing_request_name():
    payload = _request()
    payload["instance"]["compound"]["Name"] = "Another compound"
    with pytest.raises(ValidationError, match="exactly match"):
        BlueprintDesignRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["formulation", "formulation_source"])
def test_optional_formulation_requires_its_explicit_source(field):
    instance = _instance()
    instance[field] = {"Name": "Fixture tablet"} if field == "formulation" else _source("formulation")
    with pytest.raises(ValidationError, match="supplied together"):
        HumanModelInstanceInput.model_validate(instance)


@pytest.mark.parametrize("role", ["selection", "test"])
@pytest.mark.parametrize("entry", ["human_source", "compound_sources", "protocol_source", "formulation_source"])
@pytest.mark.parametrize("alias", [False, True])
def test_all_instance_sources_respect_held_out_roles_even_without_study_ids(role, entry, alias):
    payload = _request()
    source = _source("held-out")
    payload["studies"] = [{
        "study_id": "held-out", "source_id": source["source_id"],
        "source_checksum": source["source_checksum"], "role": role,
        "cohort_ids": ["held-out-cohort"], "observed_target_ids": ["plasma"],
    }]
    if alias:
        source = dict(source, source_id="source-alias")
    if entry == "compound_sources":
        payload["instance"][entry] = {"Name": source}
    else:
        payload["instance"][entry] = source
    if entry == "formulation_source":
        payload["instance"]["formulation"] = {"Name": "Fixture tablet"}
    with pytest.raises(ValidationError, match="selection, forbidden or test source"):
        BlueprintDesignRequest.model_validate(payload)


@pytest.mark.parametrize("block", ["individual", "compound", "protocol", "formulation"])
def test_nested_instance_nan_is_rejected(block):
    instance = _instance()
    if block == "formulation":
        instance["formulation"] = {"Name": "Fixture tablet"}
        instance["formulation_source"] = _source("formulation")
    instance[block]["Parameters"] = [{"Value": float("nan")}]
    with pytest.raises(ValidationError, match="NaN or infinity"):
        HumanModelInstanceInput.model_validate(instance)


def test_source_checksum_conflict_is_not_hidden_inside_instance_block():
    payload = _request()
    contradictory = deepcopy(payload["instance"]["human_source"])
    contradictory["source_checksum"] = "b" * 64
    payload["instance"]["protocol_source"] = contradictory
    with pytest.raises(ValidationError, match="checksum conflicts"):
        BlueprintDesignRequest.model_validate(payload)


def test_native_instance_does_not_require_handwritten_physiology_metadata():
    payload = _request()
    del payload["base"]
    request = BlueprintDesignRequest.model_validate(payload)
    assert request.base is None
    assert request.instance.population_context == request.requirement.population


def test_population_context_must_be_explicit_and_match_requirement():
    payload = _request()
    del payload["instance"]["population_context"]
    with pytest.raises(ValidationError, match="population_context"):
        BlueprintDesignRequest.model_validate(payload)
    payload["instance"]["population_context"] = "pediatric"
    with pytest.raises(ValidationError, match="exactly match the requested population"):
        BlueprintDesignRequest.model_validate(payload)


def test_output_configuration_is_preserved_without_reordering():
    payload = _request()
    payload["instance"]["output_schema"] = [{"StartTime": 4}, {"StartTime": 1}]
    payload["instance"]["output_selections"] = ["z-observer", "a-observer"]
    request = BlueprintDesignRequest.model_validate(payload)
    assert request.instance.output_schema == payload["instance"]["output_schema"]
    assert request.instance.output_selections == ["z-observer", "a-observer"]
