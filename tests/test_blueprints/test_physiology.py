"""CPU-only adapter checks using small in-memory JSON and anatomical fixtures."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from pbpk_kb.blueprints.physiology import (
    HumanPhysiologyExtractionError,
    from_snapshot,
)
from pbpk_kb.blueprints.schemas import CompartmentDefinition, ProvenanceRef


SOURCE_PATH = "fixture/human_snapshot.json"
SOURCE_POPULATION = "European_ICRP_2002"


@pytest.fixture
def snapshot() -> dict:
    return {
        "Version": 80,
        "Individuals": [{
            "Name": "Explicit individual",
            "OriginData": {
                "Species": "Human", "Population": SOURCE_POPULATION,
                "Age": {"Value": 35.0, "Unit": "year(s)"},
                "Weight": {"Value": 77.6, "Unit": "kg"},
                "Height": {"Value": 177.0, "Unit": "cm"},
            },
            "Parameters": [
                {"Path": "Organism|Liver|Volume", "Value": 1.4, "Unit": "l"},
                {"Path": "Organism|Kidney|Blood flow", "Value": 1.1, "Unit": "l/min"},
            ],
        }],
    }


@pytest.fixture
def compartments() -> list[CompartmentDefinition]:
    checksum = hashlib.sha256(b"explicit anatomy fixture").hexdigest()
    result = []
    for identifier, name, parent, kind in (
        ("body", "Organism", None, "other"),
        ("liver", "Liver", "body", "organ"),
        ("kidney", "Kidney", "body", "organ"),
    ):
        result.append(CompartmentDefinition(
            compartment_id=identifier, semantic_id=f"human:{identifier}",
            name=name, compartment_type=kind, parent_compartment_id=parent,
            provenance=[ProvenanceRef(
                provenance_id=uuid5(NAMESPACE_URL, f"anatomy/{identifier}"),
                provenance_kind="model_asset", source_id="explicit_anatomy",
                source_location=f"fixture/anatomy.json#/{identifier}",
                source_checksum=checksum,
            )],
        ))
    return result


def _import(snapshot: dict, compartments, **overrides):
    raw = json.dumps(snapshot, ensure_ascii=False).encode()
    arguments = dict(
        individual_name="Explicit individual", source_path=SOURCE_PATH,
        source_checksum=hashlib.sha256(raw).hexdigest(),
        base_id="fixture-human", version="1", compartment_manifest=compartments,
        source_bytes=raw,
    )
    arguments.update(overrides)
    return from_snapshot(snapshot, **arguments)


def test_import_preserves_values_units_scope_and_sources(snapshot, compartments, monkeypatch):
    def no_file_reads(*args, **kwargs):
        pytest.fail("An in-memory import must not open files")

    monkeypatch.setattr(Path, "open", no_file_reads)
    original = deepcopy(snapshot)
    base = _import(snapshot, compartments)
    assert snapshot == original
    assert base.supported_populations == [SOURCE_POPULATION]
    assert "healthy_adult" not in base.supported_populations
    values = {parameter.name: (parameter.value, parameter.canonical_unit) for parameter in base.parameters}
    assert values == {"Age": (35.0, "year(s)"), "Weight": (77.6, "kg"),
                      "Volume": (1.4, "l"), "Blood flow": (1.1, "l/min")}
    assert base.compartments == compartments
    assert base.compartments[0] is not compartments[0]
    source_sha = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False).encode()).hexdigest()
    for parameter in base.parameters:
        assert parameter.origin == "derived"
        assert parameter.context.species == "Human"
        assert parameter.context.population == SOURCE_POPULATION
        assert parameter.context.compound_semantic_id is None
        assert parameter.provenance[0].source_checksum == source_sha
        assert parameter.provenance[0].source_id == SOURCE_PATH
        assert parameter.provenance[0] in base.provenance
    for compartment in compartments:
        assert compartment.provenance[0] in base.provenance
    assert "processes" not in base.model_dump()
    assert "executable" not in base.model_dump()


def test_checksum_is_over_original_bytes(snapshot, compartments):
    with pytest.raises(HumanPhysiologyExtractionError, match="SOURCE_CHECKSUM_MISMATCH"):
        _import(snapshot, compartments, source_checksum="0" * 64)


def test_mutated_snapshot_cannot_borrow_original_checksum(snapshot, compartments):
    original = json.dumps(snapshot).encode()
    snapshot["Individuals"][0]["OriginData"]["Weight"]["Value"] = 120.0
    with pytest.raises(HumanPhysiologyExtractionError, match="SNAPSHOT_DOES_NOT_MATCH_SOURCE_BYTES"):
        _import(snapshot, compartments, source_bytes=original,
                source_checksum=hashlib.sha256(original).hexdigest())


@pytest.mark.parametrize("species", ["Mouse", "human", None])
def test_species_must_be_explicit_snapshot_human(snapshot, compartments, species):
    snapshot["Individuals"][0]["OriginData"]["Species"] = species
    with pytest.raises(HumanPhysiologyExtractionError, match="EXPLICIT_HUMAN_SPECIES_REQUIRED"):
        _import(snapshot, compartments)


def test_duplicate_selected_individual_is_ambiguous(snapshot, compartments):
    snapshot["Individuals"].append(deepcopy(snapshot["Individuals"][0]))
    with pytest.raises(HumanPhysiologyExtractionError, match="ONE_EXPLICIT_INDIVIDUAL_REQUIRED"):
        _import(snapshot, compartments)


def test_no_matching_individual_does_not_select_first(snapshot, compartments):
    with pytest.raises(HumanPhysiologyExtractionError, match="ONE_EXPLICIT_INDIVIDUAL_REQUIRED"):
        _import(snapshot, compartments, individual_name="Unknown")


@pytest.mark.parametrize("name,row", [
    ("Age", {"Value": 35.0}),
    ("Age", {"Value": 35.0, "Unit": "kg"}),
    ("Weight", {"Value": 77.6, "Unit": "display-only"}),
])
def test_missing_or_incompatible_unit_is_not_assumed(snapshot, compartments, name, row):
    snapshot["Individuals"][0]["OriginData"][name] = row
    with pytest.raises(HumanPhysiologyExtractionError, match="EXPLICIT_SUPPORTED_UNIT_REQUIRED"):
        _import(snapshot, compartments)


@pytest.mark.parametrize("value", [True, "77.6", None, -1.0, 0.0])
def test_invalid_body_weight_is_rejected(snapshot, compartments, value):
    snapshot["Individuals"][0]["OriginData"]["Weight"]["Value"] = value
    with pytest.raises(HumanPhysiologyExtractionError, match="FINITE_VALUE_REQUIRED|INVALID_PHYSIOLOGY_VALUE"):
        _import(snapshot, compartments)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_json_is_rejected(snapshot, compartments, value):
    snapshot["Individuals"][0]["OriginData"]["Age"]["Value"] = value
    with pytest.raises(HumanPhysiologyExtractionError, match="NONFINITE_SNAPSHOT_VALUE"):
        _import(snapshot, compartments)


def test_does_not_inherit_drug_or_dosing_values(snapshot, compartments):
    snapshot["Compounds"] = [{"Name": "Drug A", "Parameters": [
        {"Name": "Kp", "Value": 2.0},
        {"Name": "Fraction unbound", "Value": 0.5},
        {"Name": "Clearance", "Value": 0.3, "Unit": "l/min"},
        {"Name": "Volume", "Value": 9.0, "Unit": "l"},
    ]}]
    snapshot["Protocols"] = [{"InputDose": 200, "Unit": "mg"}]
    snapshot["Simulations"] = [{"OutputSelections": ["Organism|Brain|Drug A|Concentration"]}]
    snapshot["ObservedData"] = [{"Value": 500}]
    snapshot["Individuals"][0]["ExpressionProfiles"] = ["CYP3A4|Human|Drug A"]
    snapshot["Individuals"][0]["Parameters"].extend([
        {"Path": "Organism|Liver|Kp", "Value": 3.0},
        {"Path": "Organism|Liver|Fraction unbound", "Value": 0.2},
        {"Path": "Organism|Liver|Clearance", "Value": 10.0, "Unit": "l/min"},
        {"Path": "Organism|Liver|EHC continuous fraction", "Value": 1.0},
        {"Path": "Organism|Liver|Drug A|Volume", "Value": 900.0, "Unit": "l"},
    ])
    base = _import(snapshot, compartments)
    assert {parameter.name for parameter in base.parameters} == {"Age", "Weight", "Volume", "Blood flow"}
    assert {compartment.name for compartment in base.compartments} == {"Organism", "Liver", "Kidney"}
    assert all(parameter.context.compound_semantic_id is None for parameter in base.parameters)


def test_missing_age_weight_are_explicit_blockers(snapshot, compartments):
    origin = snapshot["Individuals"][0]["OriginData"]
    del origin["Age"]
    del origin["Weight"]
    base = _import(snapshot, compartments)
    missing = [parameter for parameter in base.parameters if parameter.origin == "missing"]
    assert {parameter.name for parameter in missing} == {"Age", "Weight"}
    assert all(parameter.blocking and parameter.value is None for parameter in missing)
    assert all(parameter.canonical_unit is None for parameter in missing)
    assert all(parameter.provenance[0].source_location.endswith("/OriginData") for parameter in missing)


def test_missing_manifest_is_not_inferred_from_model_outputs(snapshot):
    snapshot["Simulations"] = [{"OutputSelections": ["Organism|Liver|Drug A|Concentration"]}]
    with pytest.raises(HumanPhysiologyExtractionError, match="EXPLICIT_SOURCED_COMPARTMENT_MANIFEST_REQUIRED"):
        _import(snapshot, None)


def test_compartment_source_is_required(snapshot, compartments):
    manifest = [compartment.model_dump() for compartment in compartments]
    manifest[1]["provenance"] = []
    with pytest.raises(ValidationError, match="provenance"):
        _import(snapshot, manifest)


def test_physiology_path_must_resolve_to_supplied_anatomy(snapshot, compartments):
    with pytest.raises(HumanPhysiologyExtractionError, match="PHYSIOLOGY_PATH_NOT_IN_MANIFEST"):
        _import(snapshot, compartments[:2])


def test_no_source_population_requires_explicit_scope(snapshot, compartments):
    del snapshot["Individuals"][0]["OriginData"]["Population"]
    with pytest.raises(HumanPhysiologyExtractionError, match="EXPLICIT_POPULATION_SCOPE_REQUIRED"):
        _import(snapshot, compartments)
    base = _import(snapshot, compartments, supported_populations=["reviewed_population"])
    assert base.supported_populations == ["reviewed_population"]
    assert all(parameter.context.population is None for parameter in base.parameters)


def test_european_population_is_not_relabelled_healthy_adult(snapshot, compartments):
    with pytest.raises(HumanPhysiologyExtractionError, match="SOURCE_POPULATION_MUST_REMAIN_IN_SCOPE"):
        _import(snapshot, compartments, supported_populations=["healthy_adult"])
    base = _import(snapshot, compartments)
    assert "healthy_adult" not in base.supported_populations
    assert base.supported_populations == [SOURCE_POPULATION]


def test_duplicate_organ_parameter_is_not_silently_selected(snapshot, compartments):
    snapshot["Individuals"][0]["Parameters"].append(
        deepcopy(snapshot["Individuals"][0]["Parameters"][0]))
    with pytest.raises(HumanPhysiologyExtractionError, match="DUPLICATE_PHYSIOLOGY_PATH"):
        _import(snapshot, compartments)
