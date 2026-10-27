"""One static check of the registered sources; other checks use tiny local files.

No PK-Sim executable, compiler, solver, model service or GPU is invoked.
"""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest

from pbpk_kb.blueprints import human_model


def _seal(definition: dict) -> dict:
    body = {key: value for key, value in definition.items() if key != "checksum"}
    definition["checksum"] = hashlib.sha256(json.dumps(
        body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()
    return definition


def test_registered_standard_model_has_real_verified_sources():
    # This is the only test which opens actual project resources. load performs
    # their integrity validation once; no test copies or executes the real DB.
    root = Path(__file__).resolve().parents[2]
    definition = human_model.load_standard_human_model(root)
    assert definition["registry_id"] == "pksim-human-standard-small-molecule"
    assert definition["schema_version"] == human_model.SCHEMA_VERSION
    assert definition["model"]["configuration_id"] == "4Comp|Human"
    assert definition["model"]["model_id"] == "4Comp"
    assert definition["model"]["species"] == "Human"
    assert definition["model"]["source_version"] == "12.3.173"
    assert definition["status"] == "source_bound_standard_model_definition"
    assert definition["executable"] is False
    assert definition["reference_structure"]["population"] == "European_ICRP_2002"
    assert definition["reference_structure"]["containers"]["row_count"] > 0
    assert definition["reference_structure"]["neighborhoods"]["row_count"] > 0
    assert definition["formula_sources"]["rates"]["row_count"] > 0
    assert {resource["role"] for resource in definition["resources"]} >= {
        "database", "license", "individual_factory", "simulation_model_creator",
        "spatial_structure_factory", "neighborhood_finalizer", "formula_factory",
    }
    assert all(resource["byte_size"] > 0 and len(resource["sha256"]) == 64
               for resource in definition["resources"])
    assert _seal(deepcopy(definition))["checksum"] == definition["checksum"]


@pytest.fixture
def tiny_sources(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    resources = []
    for role, suffix in human_model._RESOURCES.items():
        relative = f"{human_model.SOURCE_ROOT}/{suffix}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"fixture {role}".encode()
        path.write_bytes(data)
        resources.append({"role": role, "path": relative,
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "byte_size": len(data)})
    expected = _seal({
        "schema_version": human_model.SCHEMA_VERSION,
        "registry_id": human_model.STANDARD_HUMAN_MODEL_ID,
        "model": {"model_id": "4Comp", "species": "Human", "configuration_id": "4Comp|Human"},
        "resources": resources,
        "reference_structure": {"resolved_neighborhood_paths": [{
            "neighborhood_id": 1, "first_neighbor_path": "Organism|ArterialBlood",
            "second_neighbor_path": "Organism|Liver",
        }]},
    })
    # Isolate resource guards and comparison from SQLite semantics. The real
    # original-table comparison is covered by the single integration check above.
    def collect_sources(project_root):
        assert project_root == root
        return deepcopy(expected)

    monkeypatch.setattr(human_model, "make_standard_human_model", collect_sources)
    return root, expected


def test_small_fixture_validates_without_claiming_execution(tiny_sources):
    root, definition = tiny_sources
    result = human_model.validate_human_model_resources(definition, root)
    assert result["accepted"]
    assert len(result["verified_resources"]) == len(definition["resources"])
    assert result["compiled"] is False
    assert result["simulated"] is False


@pytest.mark.parametrize("path", ["../outside.sqlite", "/outside.sqlite", "bad\\path.sqlite"])
def test_unsafe_resource_path_is_rejected_even_with_resealed_manifest(tiny_sources, path):
    root, original = tiny_sources
    changed = deepcopy(original)
    changed["resources"][0]["path"] = path
    result = human_model.validate_human_model_resources(_seal(changed), root)
    assert result["accepted"] is False
    assert result["verified_resources"] == []
    assert any("RESOURCE_PATH" in error for error in result["errors"])


def test_symlink_cannot_escape_project(tiny_sources):
    root, definition = tiny_sources
    resource = root / definition["resources"][0]["path"]
    outside = root.parent / "outside-fixture.sqlite"
    outside.write_bytes(resource.read_bytes())
    resource.unlink()
    resource.symlink_to(outside)
    result = human_model.validate_human_model_resources(definition, root)
    assert result["accepted"] is False
    assert result["errors"] == ["RESOURCE_PATH_OUTSIDE_PROJECT"]


def test_manifest_checksum_tampering_is_rejected(tiny_sources):
    root, definition = tiny_sources
    changed = deepcopy(definition)
    changed["model"]["species"] = "Mouse"
    result = human_model.validate_human_model_resources(changed, root)
    assert result["accepted"] is False
    assert result["errors"] == ["DEFINITION_CHECKSUM_MISMATCH"]


@pytest.mark.parametrize("change", ["species", "connection"])
def test_resealing_does_not_authorize_invented_model_content(tiny_sources, change):
    root, original = tiny_sources
    changed = deepcopy(original)
    if change == "species":
        changed["model"]["species"] = "Mouse"
    else:
        changed["reference_structure"]["resolved_neighborhood_paths"][0]["second_neighbor_path"] = "Organism|InventedOrgan"
    result = human_model.validate_human_model_resources(_seal(changed), root)
    assert result["accepted"] is False
    assert result["errors"] == ["MODEL_DEFINITION_DOES_NOT_MATCH_SOURCES"]


def test_same_size_source_edit_invalidates_hash_cache_even_with_restored_mtime(tiny_sources):
    root, definition = tiny_sources
    assert human_model.validate_human_model_resources(definition, root)["accepted"]
    path = root / definition["resources"][0]["path"]
    before = path.stat()
    data = path.read_bytes()
    path.write_bytes(b"F" + data[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    result = human_model.validate_human_model_resources(definition, root)
    assert result["accepted"] is False
    assert result["errors"] == ["MODEL_RESOURCE_CHECKSUM_MISMATCH: database"]


def test_missing_resource_rejected(tiny_sources):
    root, definition = tiny_sources
    (root / definition["resources"][0]["path"]).unlink()
    result = human_model.validate_human_model_resources(definition, root)
    assert result["accepted"] is False
    assert result["errors"] == ["MODEL_RESOURCE_CHECKSUM_MISMATCH: database"]


def test_load_does_not_register_or_write_missing_definition(tmp_path, monkeypatch):
    def registration_forbidden(*args, **kwargs):
        pytest.fail("load must not generate or register a model definition")

    monkeypatch.setattr(human_model, "register_standard_human_model", registration_forbidden)
    monkeypatch.setattr(human_model, "make_standard_human_model", registration_forbidden)
    with pytest.raises(FileNotFoundError):
        human_model.load_standard_human_model(tmp_path)
    assert list(tmp_path.iterdir()) == []
