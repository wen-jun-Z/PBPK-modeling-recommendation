"""Read-only Schmitt export with an explicit reduced-model blood partition.

PK-Sim's global B/P can retain its standard calculation although Schmitt changes
K_RBC/plasma. The reduced equilibrium model instead uses the native Schmitt K_RBC
and venous hematocrit. This is a derived mapping, not a native readback, measured
B/P, clinical fit, or correction to the source PKML. Native clearance inversion
is always audited with the original native readback values.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import json
import math
from pathlib import Path
from types import FunctionType, SimpleNamespace

from . import perfusion_export as native

VERSION = "schmitt-equilibrium-physiology-1"
SCHMITT_METHOD = "Cellular partition coefficient method - Schmitt"
PERMEABILITY_METHOD = "Cellular permeability - PK-Sim Standard"
FORMULA = "B/P_reduced = (1 - Hct_venous) + Hct_venous * K_RBC/plasma_native_Schmitt"


def _clone(function, **replacements):
    """Replace call bindings in a private namespace; never mutate old modules."""
    namespace = {**function.__globals__, **replacements}
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = deepcopy(function.__kwdefaults__)
    return clone


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _selection(snapshot, compound):
    compounds, simulations = snapshot.get("Compounds", []), snapshot.get("Simulations", [])
    if len(compounds) != 1 or len(simulations) != 1:
        raise ValueError("Schmitt mapping requires exactly one compound and simulation")
    selected = simulations[0].get("Compounds", [])
    if len(selected) != 1 or compounds[0].get("Name") != compound or selected[0].get("Name") != compound:
        raise ValueError("Schmitt snapshot compound selection differs")
    expected = [SCHMITT_METHOD, PERMEABILITY_METHOD]
    if any(row.get("CalculationMethods") != expected for row in (compounds[0], selected[0])):
        raise ValueError("Both bound snapshot formula selections must be native Schmitt")
    return {"compound": compound, "calculation_methods": expected,
            "verified_locations": ["/Compounds/0/CalculationMethods",
                                   "/Simulations/0/Compounds/0/CalculationMethods"]}


def _assemble_export(plan, rows, processes, *, method_source, code_pins):
    if method_source.get("calculation_methods") != [SCHMITT_METHOD, PERMEABILITY_METHOD]:
        raise ValueError("Bound Schmitt method source required")
    specs = plan["parameters"]
    expected = [spec["path"] for spec in specs]
    if (len(expected) != len(set(expected)) or not isinstance(rows, list)
            or [row.get("path") for row in rows] != expected):
        raise ValueError("Native Schmitt readback must cover the exact unique ordered plan")
    for spec, row in zip(specs, rows):
        value = row.get("value")
        if (row.get("unit") != spec["unit"] or isinstance(value, bool)
                or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("Nonfinite or noncanonical native Schmitt readback")
    values = {row["path"]: row["value"] for row in rows}
    units = {row["path"]: row["unit"] for row in rows}
    bp_path, rbc_path = (plan["global_paths"][key] for key in ("blood_to_plasma_ratio", "rbc_kp_plasma"))
    if bp_path == rbc_path or any(units[path] != "fraction" for path in (bp_path, rbc_path)):
        raise ValueError("Native B/P and Schmitt RBC partition require distinct dimensionless paths")
    original_bp, rbc_kp = values[bp_path], values[rbc_path]
    if original_bp <= 0 or rbc_kp <= 0:
        raise ValueError("Native B/P and Schmitt RBC partition must be positive")
    components = plan["blood"]["venous"]["components"]
    if not components or {row["kind"] for row in components} != {"Plasma", "BloodCells"}:
        raise ValueError("Venous hematocrit requires plasma and RBC volumes only")
    paths = [row["volume_path"] for row in components]
    if len(paths) != len(set(paths)) or any(units[path] != "l" or values[path] < 0 for path in paths):
        raise ValueError("Venous volumes must be unique, nonnegative and in liters")
    plasma = math.fsum(values[row["volume_path"]] for row in components if row["kind"] == "Plasma")
    rbc = math.fsum(values[row["volume_path"]] for row in components if row["kind"] == "BloodCells")
    if plasma <= 0 or rbc <= 0:
        raise ValueError("Positive plasma and RBC volumes are required for Schmitt equilibrium mapping")
    hct = rbc / (plasma + rbc)
    derived = (1. - hct) + hct * rbc_kp
    if not math.isfinite(derived) or derived <= 0 or not 0 < hct < 1:
        raise ValueError("Invalid derived Schmitt blood partition")

    # Only this in-memory view changes. The worker output, request and PKML do
    # not change. Even an aliased clearance BP path must use the raw values.
    mapped_rows = deepcopy(rows)
    next(row for row in mapped_rows if row["path"] == bp_path)["value"] = derived
    raw_audits = {process["stem"]: native.clearance._capacity(process, values) for process in processes}
    if len(raw_audits) != len(processes):
        raise ValueError("Duplicate native clearance processes")
    original_values_auditor = SimpleNamespace(
        _capacity=lambda process, unused: deepcopy(raw_audits[process["stem"]]))
    assemble = _clone(native.assemble_export, clearance=original_values_auditor)
    result = assemble(plan, mapped_rows, processes)
    if not math.isclose(result["hematocrit"], hct, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Native and mapped venous volume accounting differs")
    result.update(
        mapping_version=VERSION,
        mapping_code_pins=deepcopy(code_pins),
        native_blood_to_plasma_ratio=original_bp,
        blood_partition_mapping={
            "version": VERSION, "method_source": deepcopy(method_source),
            "basis": "derived_not_measured", "formula": FORMULA,
            "hematocrit": hct, "venous_plasma_volume_l": plasma, "venous_rbc_volume_l": rbc,
            "venous_volume_inputs": [{"path": row["volume_path"], "kind": row["kind"],
                "value": values[row["volume_path"]], "unit": "l"} for row in components],
            "native_rbc_partition": {"path": rbc_path, "value": rbc_kp, "unit": "fraction"},
            "modified_reduced_inputs": [{"path": bp_path, "native_value": original_bp,
                "derived_value": derived, "unit": "fraction", "scope": "reduced_model_only"}],
            "native_readback_parameters_sha256": _digest(rows),
            "reduced_readback_view_sha256": _digest(mapped_rows),
            "native_clearance_inversion_uses_original_readback": True,
            "native_model_modified": False, "clinical_observations_used": False,
            "clipping_applied": False,
            "protected": ["native_PKML", "native_readback", "compartment_volumes", "blood_flows",
                          "tissue_partition_coefficients", "RBC_partition_coefficient", "fraction_unbound",
                          "clearance_inputs", "native_clearance_inversion_audit"],
            "limitations": ["Equilibrium reduction of native Schmitt compartments, not a native transient B/P measurement",
                "Native global B/P is retained separately and is not silently relabeled as the derived value",
                "No guarantee of feasible Vss anchoring or clinical prediction accuracy"]})
    return result


def export_physiology(*, execution_manifest_path, source_manifest_sha256, output_dir, project_root):
    """Compatible native exporter with an audited Schmitt-only reduced mapping.

    Reuses the frozen worker, path discovery, source checks, artifact hashes and
    cache reconstruction. The physiology schema version remains compatible;
    ``mapping_version`` and ``mapping_code_pins`` identify this added mapping.
    """
    root = Path(project_root).resolve(strict=True)
    manifest, snapshot_path, pkml, compound = native.source._bind(
        execution_manifest_path, source_manifest_sha256, root)
    snapshot = json.loads(snapshot_path.read_text())
    method_source = {**_selection(snapshot, compound),
        "snapshot": {"path": str(snapshot_path.relative_to(root)), "sha256": native.source._sha(snapshot_path)},
        "execution_manifest": {"path": str(manifest.relative_to(root)), "sha256": source_manifest_sha256},
        "pkml": {"path": str(pkml.relative_to(root)), "sha256": native.source._sha(pkml)}}
    paths = [native.source._inside(path, root) for path in (Path(__file__).resolve(), Path(native.__file__).resolve())]
    code_pins = {str(path.relative_to(root)): native.source._sha(path) for path in paths}
    assembly = partial(_assemble_export, method_source=method_source, code_pins=code_pins)
    exporter = _clone(native.export_physiology, assemble_export=assembly)
    result = exporter(execution_manifest_path=execution_manifest_path,
        source_manifest_sha256=source_manifest_sha256, output_dir=output_dir, project_root=root)
    if any(native.source._sha(root / path) != digest for path, digest in code_pins.items()):
        raise ValueError("Schmitt mapping code changed during export")
    if result.get("mapping_version") != VERSION or result.get("mapping_code_pins") != code_pins:
        raise ValueError("Schmitt mapping version or code binding differs")
    return result
