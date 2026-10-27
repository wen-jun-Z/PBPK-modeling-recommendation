"""Schmitt mapping preserves native evidence and reconstructs immutable caches."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from pbpk_kb.blueprint_policy import perfusion_export as native
from pbpk_kb.blueprint_policy import schmitt_physiology as schmitt

ROOT = Path(__file__).resolve().parents[1]


def _write(path, value):
    path.write_text(json.dumps(value))


def _fixture():
    specs, values = {}, {}

    def add(path, value, unit):
        specs[path] = {"path": path, "unit": unit}
        values[path] = value
        return path

    global_paths = {}
    for key, leaf, value, unit in (
        ("blood_to_plasma_ratio", "Blood/Plasma concentration ratio", 8., "fraction"),
        ("rbc_kp_plasma", "Partition coefficient (blood cells/plasma)", 2., "fraction"),
        ("fraction_unbound_plasma", "Fraction unbound (plasma)", .1, "fraction"),
        ("molecular_weight_g_mol", "Molecular weight", 300., "g/mol"),
        ("logp", "Lipophilicity", 2., "Log Units"),
    ):
        global_paths[key] = add("Drug|" + leaf, value, unit)
    global_paths["weight_kg"] = add("Organism|Weight", 70., "kg")

    def aggregate(name, blood=False):
        components = []
        kinds = (("Plasma", .6), ("BloodCells", .4)) if blood else (
            ("Plasma", .2), ("BloodCells", .1), ("Interstitial", .3), ("Intracellular", .4))
        for kind, volume in kinds:
            stem = "Organism|" + name + "|" + kind
            kp = (None if kind == "Plasma" else global_paths["rbc_kp_plasma"] if kind == "BloodCells"
                  else add(stem + "|Kp", .5 if kind == "Interstitial" else 3., "fraction"))
            components.append({"path": stem, "region": "Organism|" + name, "kind": kind,
                               "volume_path": add(stem + "|Volume", volume, "l"), "kp_path": kp})
        result = {"name": name, "components": components}
        if not blood:
            result["flow_path"] = add("Organism|" + name + "|Blood flow rate", 14. if name == "Lung" else 1., "l/min")
        return result

    plan = {"global_paths": global_paths, "organs": [aggregate(name) for name in native.ORGANS],
            "blood": {key: aggregate(name, True) for key, name in native.BLOOD.items()},
            "portal_flow_path": add("Organism|PortalVein|Blood flow rate", 5., "l/min"),
            "peripheral_weight_paths": {name: add("Organism|" + name + "|Peripheral blood flow fraction",
                                                  1. if name == "Skin" else 0., "fraction")
                                        for name in native.WEIGHTS}, "excluded_organ_containers": []}
    stem = "Drug|LiverClearance"
    for leaf, value, unit in (
        ("Plasma clearance", .1, "ml/min/kg"), ("Body weight", 70., "kg"),
        ("Fraction unbound (experiment)", .1, "fraction"),
        ("Specific clearance", .007 * 6 / (.1 * (6 - .007 / 8)), "1/min"),
        ("Blood flow rate (liver)", 1., "l/min"), ("Blood flow rate (portal vein)", 5., "l/min"),
        ("Blood/Plasma concentration ratio", 8., "fraction"), ("Volume (liver)", 2., "l"),
        ("Fraction intracellular (liver)", .5, "fraction"),
    ):
        add(stem + "|" + leaf, value, unit)
    plan["parameters"] = list(specs.values())
    rows = [{**spec, "value": values[path], "native_value": values[path], "native_unit": spec["unit"]}
            for path, spec in specs.items()]
    processes = [{"organ": "hepatic", "stem": stem, "clearance_unit": "ml/min/kg"}]
    return plan, rows, processes


def _assemble(plan, rows, processes):
    return schmitt._assemble_export(plan, rows, processes,
        method_source={"calculation_methods": [schmitt.SCHMITT_METHOD, schmitt.PERMEABILITY_METHOD]},
        code_pins={"unit_test_source": "a" * 64})


def test_derived_bp_preserves_all_other_native_quantities_and_inputs():
    plan, rows, processes = _fixture()
    frozen = deepcopy((plan, rows, processes))
    result = _assemble(plan, rows, processes)
    assert result["blood_to_plasma_ratio"] == pytest.approx(1.4)
    assert result["native_blood_to_plasma_ratio"] == 8.
    assert result["hematocrit"] == .4
    assert result["blood_partition_mapping"]["basis"] == "derived_not_measured"
    assert result["blood_partition_mapping"]["clipping_applied"] is False
    consistent = deepcopy(rows)
    next(row for row in consistent if row["path"] == plan["global_paths"]["blood_to_plasma_ratio"])["value"] = 1.4
    expected = native.assemble_export(plan, consistent, processes)
    assert all(result[key] == value for key, value in expected.items())
    assert (plan, rows, processes) == frozen
    assert result["clearance"]["processes"][0]["native_inversion_audit"]["accepted"] is True


def test_native_clearance_audit_uses_raw_bp_even_if_global_path_aliases_process():
    plan, rows, processes = _fixture()
    plan["global_paths"]["blood_to_plasma_ratio"] = processes[0]["stem"] + "|Blood/Plasma concentration ratio"
    raw_values = {row["path"]: row["value"] for row in rows}
    expected = native.clearance._capacity(processes[0], raw_values)
    result = _assemble(plan, rows, processes)
    actual = result["clearance"]["processes"][0]["native_inversion_audit"]
    assert actual == expected
    assert actual["plasma_clearance_capacity_l_min"] == 48.
    assert actual["accepted"] is True
    assert result["blood_partition_mapping"]["native_clearance_inversion_uses_original_readback"] is True


@pytest.mark.parametrize("path,value", [
    ("Drug|Blood/Plasma concentration ratio", 0.),
    ("Drug|Partition coefficient (blood cells/plasma)", -1.),
    ("Drug|Partition coefficient (blood cells/plasma)", float("nan")),
    ("Drug|Partition coefficient (blood cells/plasma)", True),
    ("Organism|VenousBlood|Plasma|Volume", 0.),
    ("Organism|VenousBlood|BloodCells|Volume", -.1),
])
def test_invalid_native_quantities_are_rejected(path, value):
    plan, rows, processes = _fixture()
    next(row for row in rows if row["path"] == path)["value"] = value
    with pytest.raises(ValueError):
        _assemble(plan, rows, processes)


def test_wrong_units_are_not_silently_converted():
    plan, rows, processes = _fixture()
    next(row for row in rows if row["path"] == "Organism|VenousBlood|Plasma|Volume")["unit"] = "ml"
    with pytest.raises(ValueError, match="noncanonical"):
        _assemble(plan, rows, processes)


def test_non_bp_native_errors_and_ordered_coverage_remain_enforced():
    plan, rows, processes = _fixture()
    with pytest.raises(ValueError, match="ordered plan"):
        _assemble(plan, rows[::-1], processes)
    next(row for row in rows if row["path"] == plan["portal_flow_path"])["value"] = 6.
    with pytest.raises(ValueError, match="Portal tributary"):
        _assemble(plan, rows, processes)


@pytest.fixture
def bound_export(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(ROOT / "runtime")
    plan, rows, processes = _fixture()
    folder = tmp_path / "source"
    folder.mkdir()
    snapshot = {"Version": 80,
        "Compounds": [{"Name": "Drug", "Lipophilicity": [{"Name": "EvidenceLogP"}],
            "FractionUnbound": [{"Name": "EvidenceFu"}], "Processes": [{"DataSource": "EvidenceAssembly"}],
            "CalculationMethods": [schmitt.SCHMITT_METHOD, schmitt.PERMEABILITY_METHOD]}],
        "Simulations": [{"Compounds": [{"Name": "Drug", "CalculationMethods":
                         [schmitt.SCHMITT_METHOD, schmitt.PERMEABILITY_METHOD]}]}],
        "Protocols": [{"ApplicationType": "Intravenous", "DosingInterval": "Single"}]}
    snapshot_path, pkml, manifest = (folder / name for name in ("snapshot.json", "source.pkml", "execution_manifest.json"))
    _write(snapshot_path, snapshot)
    pkml.write_text("<test-native-pkml-unchanged/>")

    def bind():
        _write(manifest, {"engineering_accepted": True, "source_unchanged": True,
            "source_sha256": native.source._sha(snapshot_path),
            "pkml_outputs": [{"path": pkml.name, "sha256": native.source._sha(pkml)}]})
        return {"execution_manifest_path": manifest, "source_manifest_sha256": native.source._sha(manifest),
                "output_dir": tmp_path / "physiology", "project_root": ROOT}

    calls = []
    def run_bounded_r(**kwargs):
        calls.append(kwargs)
        request_path = Path(kwargs["args"][0]); request = json.loads(request_path.read_text())
        assert kwargs["cpu_threads"] == 2 and kwargs["timeout_seconds"] == 180
        _write(Path(request["result_path"]), {"version": native.VERSION, "solver_calls": 0,
            "model_modified": False, "pkml_sha256": request["pkml_sha256"],
            "request_sha256": native.source._sha(request_path), "parameters": deepcopy(rows)})
        return {"status": "completed", "returncode": 0}

    from pbpk_kb.blueprints import design_runtime
    monkeypatch.setattr(design_runtime, "run_bounded_r", run_bounded_r)
    monkeypatch.setattr(native, "discover_plan", lambda path, compound: deepcopy(plan))
    monkeypatch.setattr(native.clearance, "_plan", lambda snapshot: ([], deepcopy(processes)))
    return {"kwargs": bind(), "bind": bind, "snapshot": snapshot, "snapshot_path": snapshot_path,
            "pkml": pkml, "rows": rows, "calls": calls}


def test_export_is_read_only_versioned_and_cache_reconstructs(bound_export):
    fixture = bound_export
    before = fixture["pkml"].read_bytes()
    original_assembler = native.assemble_export
    result = schmitt.export_physiology(**fixture["kwargs"])
    assert native.assemble_export is original_assembler
    assert result["version"] == native.VERSION
    assert result["mapping_version"] == schmitt.VERSION
    assert result["solver_calls"] == 0 and result["observations_used"] is False
    adapter_path = str(Path(schmitt.__file__).resolve().relative_to(ROOT))
    assert result["mapping_code_pins"][adapter_path] == native.source._sha(ROOT / adapter_path)
    raw_path = fixture["kwargs"]["output_dir"] / "native_readback.json"
    raw_bytes = raw_path.read_bytes()
    assert json.loads(raw_bytes)["parameters"] == fixture["rows"]
    again = schmitt.export_physiology(**fixture["kwargs"])
    assert again == result and len(fixture["calls"]) == 1
    assert fixture["pkml"].read_bytes() == before and raw_path.read_bytes() == raw_bytes


@pytest.mark.parametrize("location", ["compound", "simulation"])
def test_bound_wrong_formula_is_rejected_before_native_export(bound_export, location):
    f = bound_export
    target = f["snapshot"]["Compounds"][0] if location == "compound" else f["snapshot"]["Simulations"][0]["Compounds"][0]
    target["CalculationMethods"][0] = "Cellular partition coefficient method - PK-Sim Standard"
    _write(f["snapshot_path"], f["snapshot"])
    kwargs = f["bind"]()
    with pytest.raises(ValueError, match="must be native Schmitt"):
        schmitt.export_physiology(**kwargs)
    assert f["calls"] == [] and not kwargs["output_dir"].exists()


def test_unbound_snapshot_or_execution_is_rejected(bound_export):
    f = bound_export
    with pytest.raises(ValueError, match="manifest hash"):
        schmitt.export_physiology(**{**f["kwargs"], "source_manifest_sha256": "0" * 64})
    f["snapshot_path"].write_text(f["snapshot_path"].read_text() + " ")
    with pytest.raises(ValueError, match="snapshot hash"):
        schmitt.export_physiology(**f["kwargs"])
    assert not f["calls"]


@pytest.mark.parametrize("field", ["blood_to_plasma_ratio", "mapping_code_pins", "native_clearance_audit"])
def test_cache_rejects_derived_or_native_audit_tampering(bound_export, field):
    f = bound_export
    result = schmitt.export_physiology(**f["kwargs"])
    path = ROOT / result["manifest"]["path"]
    value = json.loads(path.read_text())
    if field == "native_clearance_audit":
        value["clearance"]["processes"][0]["native_inversion_audit"]["plasma_clearance_capacity_l_min"] = 999.
    else:
        value[field] = 999. if field == "blood_to_plasma_ratio" else {}
    _write(path, value)
    with pytest.raises(ValueError, match="reconstruction"):
        schmitt.export_physiology(**f["kwargs"])
    assert len(f["calls"]) == 1


def test_cache_rejects_changed_raw_readback(bound_export):
    f = bound_export
    schmitt.export_physiology(**f["kwargs"])
    path = f["kwargs"]["output_dir"] / "native_readback.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="artifact changed"):
        schmitt.export_physiology(**f["kwargs"])
    assert len(f["calls"]) == 1
