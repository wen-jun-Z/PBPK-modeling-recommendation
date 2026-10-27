"""Evidence-guarded RBC/plasma calibration on the frozen exchange equations.

PK-Sim usually predicts red-cell partition from physicochemical properties;
measured B:P is a distinct input (not another tunable clinical fit target):
https://docs.open-systems-pharmacology.org/working-with-pk-sim/pk-sim-documentation/pk-sim-compounds-definition-and-work-flow

Only an in-memory derived physiology is changed. Native readback, anatomical
volumes, flows, fu, and plasma-clearance inputs remain unchanged. This is not
a new native PK-Sim run or proof of measured RBC binding. No file/clinical
observation access or parameter estimation takes place here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re

import numpy as np

from . import permeability_model as exchange
from . import perfusion_model as perfusion
from .perfusion_calibration import clearance_inputs

VERSION = "evidence-guarded-blood-partition-calibration-1"
PARAMETERS = ("clearance_scale", "partition_scale", "bidirectional_exchange_scale", "rbc_partition_scale")
_KINDS = {"Plasma": "plasma", "BloodCells": "rbc", "Interstitial": "interstitial", "Intracellular": "intracellular"}
_BP = {"bloodtoplasmaratio", "bloodplasmaratio", "bloodtoplasmaconcentrationratio",
       "bloodplasmaconcentrationratio", "bloodtoplasmapartitioncoefficient", "bpratio", "bpr"}
_RBC = {"rbckpplasma", "rbctoplasmaratio", "redbloodcelltoplasmaratio", "rbcplasmapartitioncoefficient",
        "partitioncoefficientbloodcellsplasma", "krbc", "bloodcellstoplasmaratio"}
_number = perfusion._number


def _close(value, expected, label):
    value = _number(value, label, nonnegative=True)
    if not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-10):
        raise ValueError(f"Inconsistent {label}: {value} versus {expected}")


def _native_source(physiology):
    if (not isinstance(physiology, Mapping) or physiology.get("version") != perfusion.PHYSIOLOGY_VERSION
            or physiology.get("source_kind") != "fresh_evidence_native_model"
            or physiology.get("observations_used") is not False
            or type(physiology.get("solver_calls")) is not int or physiology["solver_calls"] != 0
            or physiology.get("derived_not_native_readback") or "blood_partition_audit" in physiology):
        raise ValueError("RBC calibration requires untouched fresh evidence native physiology without observations")
    if any(key in physiology for key in ("blood_partition_source", "blood_partition_basis",
                                        "blood_to_plasma_ratio_source", "rbc_kp_plasma_source",
                                        "rbc_partition_evidence", "blood_partition_evidence", "evidence")):
        raise ValueError("Explicit or mixed blood-partition provenance is unsupported")
    digest = physiology.get("source_execution_manifest_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Native source manifest identity is missing")
    # Actual files are verified by the calling runner; this pure module verifies
    # that the supplied export retains the source/readback provenance contract.
    for label in ("source_pins", "artifact_pins"):
        pins = physiology.get(label)
        if (not isinstance(pins, Mapping) or not pins
                or any(not isinstance(path, str) or not path or not isinstance(sha, str)
                       or re.fullmatch(r"[0-9a-f]{64}", sha) is None for path, sha in pins.items())):
            raise ValueError("Native source/readback pins are missing or invalid")


def _evidence_policy(physiology, request, log_scale):
    if not isinstance(request, Mapping) or not isinstance(request.get("compound_id"), str) or not request["compound_id"]:
        raise ValueError("An identified compound request is required")
    rows = request.get("evidence")
    if not isinstance(rows, (list, tuple)) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("Explicit evidence ledger is required")
    protected = []
    for row in rows:
        name = re.sub(r"[^a-z0-9]", "", str(row.get("parameter", "")).lower())
        if name not in _BP | _RBC:
            continue
        source = row.get("source")
        context = row.get("context")
        if (row.get("basis") != "measured" or row.get("compound_id") != request.get("compound_id")
                or row.get("unit") not in {"1", "fraction", "dimensionless", "L/L", "l/l"}
                or not isinstance(context, Mapping) or context.get("species") != "human"
                or not isinstance(source, Mapping) or not source.get("path")
                or not isinstance(source.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None):
            raise ValueError("Unknown, conflicting or non-measured explicit blood-partition evidence")
        key = "blood_to_plasma_ratio" if name in _BP else "rbc_kp_plasma"
        _close(_number(row.get("value"), key, positive=True), physiology[key], "measured " + key)
        protected.append({"parameter": row["parameter"], "value": row["value"], "source": deepcopy(source)})
    if protected and log_scale != 0.:
        raise ValueError("Measured blood/RBC partition is protected from calibration")
    return {"rule": "preserve_measured_partition" if protected else "calibrate_native_predicted_partition",
            "explicit_measured_records": protected, "clinical_reference_used": False}


def _validate_components(region, rbc_kp, *, blood=False, hematocrit=None):
    name = region.get("name", "unnamed")
    parts = region.get("components")
    if not isinstance(parts, (list, tuple)) or not parts:
        raise ValueError(f"Missing component readback for {name}")
    volumes = {key: 0. for key in _KINDS}
    capacities = {key: 0. for key in _KINDS}
    paths = set()
    for part in parts:
        kind = part.get("kind")
        if kind not in _KINDS or not part.get("path") or part["path"] in paths:
            raise ValueError(f"Unknown or duplicate component in {name}")
        paths.add(part["path"])
        volume = _number(part.get("volume_l"), name + ".component volume", nonnegative=True)
        kp = _number(part.get("kp_plasma"), name + ".component kp", positive=True)
        _close(part.get("capacity_l"), volume * kp, name + ".component capacity")
        if kind == "BloodCells":
            _close(kp, rbc_kp, name + ".RBC partition")
        elif kind == "Plasma":
            _close(kp, 1., name + ".plasma partition")
        volumes[kind] += volume
        capacities[kind] += volume * kp
    for kind, stem in _KINDS.items():
        _close(region.get(stem + "_volume_l"), volumes[kind], name + "." + stem + " volume")
    volume = _number(region.get("volume_l"), name + ".volume", positive=True)
    vascular_volume = volumes["Plasma"] + volumes["BloodCells"]
    extra_volume = volumes["Interstitial"] + volumes["Intracellular"]
    vascular = capacities["Plasma"] + capacities["BloodCells"]
    extra = capacities["Interstitial"] + capacities["Intracellular"]
    _close(volume, vascular_volume + extra_volume, name + ".total volume")
    _close(region.get("vascular_volume_l"), vascular_volume, name + ".vascular volume")
    _close(region.get("extravascular_volume_l"), extra_volume, name + ".extravascular volume")
    _close(region.get("vascular_capacity_l"), vascular, name + ".vascular capacity")
    _close(region.get("extravascular_capacity_l"), extra, name + ".extravascular capacity")
    _close(region.get("kp_total_plasma"), (vascular + extra) / volume, name + ".total partition")
    if extra_volume > 0:
        _close(region.get("kp_extravascular_plasma"), extra / extra_volume, name + ".extravascular partition")
    elif region.get("kp_extravascular_plasma") is not None:
        raise ValueError("Blood-only compartment cannot have an extravascular partition")
    if vascular_volume <= 0:
        raise ValueError("Explicit positive vascular volume is required")
    if blood:
        if extra_volume != 0 or extra != 0:
            raise ValueError("Blood pool contains unsupported extravascular components")
        _close(volumes["BloodCells"] / volume, hematocrit, name + ".hematocrit")


def derive_physiology(physiology, request, log_rbc_scale):
    """Return a derived copy; never mutate or relabel native readback as fitted."""
    _native_source(physiology)
    log_scale = _number(log_rbc_scale, "log_rbc_scale")
    try:
        scale = _number(math.exp(log_scale), "rbc_partition_scale", positive=True)
    except OverflowError as exc:
        raise ValueError("RBC partition multiplier overflow") from exc
    kp = _number(physiology.get("rbc_kp_plasma"), "rbc_kp_plasma", positive=True)
    bp = _number(physiology.get("blood_to_plasma_ratio"), "blood_to_plasma_ratio", positive=True)
    hct = _number(physiology.get("hematocrit"), "hematocrit")
    if not 0 < hct < 1:
        raise ValueError("Hematocrit must be strictly between zero and one")
    _close(bp, 1 - hct + hct * kp, "native B/P from hematocrit and RBC partition")
    _close(physiology.get("equilibrium_blood_to_plasma_ratio_check"), bp, "native B/P check")
    blood = physiology.get("blood")
    organs = physiology.get("organs")
    if not isinstance(blood, Mapping) or set(blood) != {"arterial", "venous", "portal"}:
        raise ValueError("Require all three native blood pools")
    if (not isinstance(organs, (list, tuple))
            or {row.get("name") for row in organs} != perfusion.FAST_TISSUES | perfusion.SLOW_TISSUES
            or len(organs) != 15):
        raise ValueError("Require the fifteen native organs")
    for row in blood.values():
        _validate_components(row, kp, blood=True, hematocrit=hct)
    for row in organs:
        _validate_components(row, kp)
    policy = _evidence_policy(physiology, request, log_scale)
    new_kp = _number(kp * scale, "derived RBC partition", positive=True)
    new_bp = _number((1 - hct) + hct * new_kp, "derived B/P", positive=True)
    derived = deepcopy(physiology)
    # Preserve numeric identity exactly at scale 1, avoiding needless readback
    # rounding differences in regression comparisons with the frozen solver.
    if log_scale == 0.:
        new_kp, new_bp = kp, bp
    derived.update(rbc_kp_plasma=new_kp, blood_to_plasma_ratio=new_bp,
                   equilibrium_blood_to_plasma_ratio_check=new_bp)
    changed = []
    for old, row in zip([*blood.values(), *organs], [*derived["blood"].values(), *derived["organs"]]):
        if log_scale != 0.:
            row["vascular_capacity_l"] = row["plasma_volume_l"] + row["rbc_volume_l"] * new_kp
            row["kp_total_plasma"] = (row["vascular_capacity_l"] + row["extravascular_capacity_l"]) / row["volume_l"]
            for part in row["components"]:
                if part["kind"] == "BloodCells":
                    part["kp_plasma"] = new_kp
                    part["capacity_l"] = _number(part["volume_l"] * new_kp, "derived RBC capacity", nonnegative=True)
            _number(row["vascular_capacity_l"], "derived vascular capacity", positive=True)
            _number(row["kp_total_plasma"], "derived total partition", positive=True)
        changed.append({"name": row["name"], "source_vascular_capacity_l": old["vascular_capacity_l"],
                        "derived_vascular_capacity_l": row["vascular_capacity_l"]})
    audit = {"version": VERSION, "derived_not_native_readback": True,
             "source_kind": physiology["source_kind"],
             "source_execution_manifest_sha256": physiology["source_execution_manifest_sha256"],
             "source_rbc_kp_plasma": kp, "derived_rbc_kp_plasma": new_kp,
             "source_blood_to_plasma_ratio": bp, "derived_blood_to_plasma_ratio": new_bp,
             "hematocrit": hct, "rbc_partition_scale": scale, "policy": policy,
             "capacities": changed, "native_files_modified": False,
             "anatomical_volumes_flows_fu_clearance_unchanged": True,
             "calibration_target_access": False,
             "interpretation": "Effective equilibrium RBC/plasma partition; not independently measured red-cell binding"}
    derived.update(source_kind="derived_blood_partition_model", derived_not_native_readback=True,
                   blood_partition_audit=audit)
    return derived


def simulate(physiology, request, times_h, log_parameters):
    """Four log scales [CL, tissue K, exchange PS, RBC K]; original units."""
    raw = np.asarray(log_parameters, dtype=object)
    if raw.shape != (4,) or any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError("Require four finite numeric log scales")
    logs = np.asarray(log_parameters)
    if (logs.shape != (4,) or logs.dtype.kind not in "iuf" or not np.isfinite(logs).all()):
        raise ValueError("Require four finite numeric log scales")
    if request.get("dosing", {}).get("route") != "intravenous" or request["dosing"].get("start_h", 0.) != 0.:
        raise ValueError("Blood-partition calibration supports IV administration at time zero")
    derived = derive_physiology(physiology, request, float(logs[3]))
    cl = clearance_inputs(request)
    result = exchange.simulate(derived, times_h=times_h, dose_mg=request["dosing"]["dose_mg"],
        infusion_duration_h=request["dosing"]["infusion_minutes"] / 60.,
        total_clearance_l_h=cl["total_clearance_l_h"], renal_clearance_l_h=cl["renal_clearance_l_h"],
        log_parameters=dict(zip(PARAMETERS[:3], logs[:3].astype(float).tolist())), observation="auto")
    result["blood_partition_audit"] = derived["blood_partition_audit"]
    result["calibration_version"] = VERSION
    return result
