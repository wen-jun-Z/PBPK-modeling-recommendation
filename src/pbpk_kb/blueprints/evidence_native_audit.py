"""Read back evidence-built drug inputs and native clearance inversion limits.

This performs no ODE and does not establish mass balance or predictive accuracy.
Only the current systemic hepatic/renal clearance assembly is supported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3

from .native_edits import reviewed_process_parameter_path
from .source_workflow import inside, sha256

VERSION = "evidence-native-input-capacity-audit-v1"
_DATABASE = "artifacts/pbpk-model-assets/template_sources/pksim_v12.3.173/src/Db/PKSimDB.sqlite"
_DATABASE_SHA = "aec61433ddddf39ec5c832bda90a6977a25ea5427e77edc08f2d97a751fe1469"
_FORMULAS = {"hepatic": "PARAM_SpecificHepaticClearance", "renal": "PARAM_CL_spec_from_CL_pls_kid"}


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _run_bounded_r(**kwargs):
    from .design_runtime import run_bounded_r
    return run_bounded_r(**kwargs)


def _plan(snapshot):
    if snapshot.get("Version") != 80 or len(snapshot.get("Compounds", [])) != 1 or len(snapshot.get("Simulations", [])) != 1:
        raise ValueError("Native evidence audit requires a single Version80 drug simulation")
    compound = snapshot["Compounds"][0]
    name = compound["Name"]
    parameters, processes = [], []
    def add(path, unit, expected=None, tolerance=1e-8):
        row = {"path": path, "unit": unit}
        if expected is not None:
            if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected):
                raise ValueError("Native expected parameter values must be finite numbers")
            row.update(expected=expected, relative_tolerance=tolerance)
        parameters.append(row)
    for p in compound["Parameters"]:
        add(name + "|" + p["Name"], p.get("Unit", "fraction"), p["Value"])
    for group in ("Lipophilicity", "FractionUnbound", "Solubility", "IntestinalPermeability"):
        if group in compound:
            if len(compound[group]) != 1:
                raise ValueError("Evidence audit requires one explicit alternative per property")
            for p in compound[group][0]["Parameters"]:
                add(name + "|" + p["Name"], p.get("Unit", "fraction"), p["Value"])
    binding = {"Albumin": 1, "Glycoprotein": 0, "Unknown": 2}
    add(name + "|Plasma protein binding partner", "fraction", binding[compound["PlasmaProteinBindingPartner"]])
    for index in range(3):
        pka = compound["PkaTypes"][index] if index < len(compound["PkaTypes"]) else {"Type": "Neutral", "Pka": 0.}
        add(name + "|Compound type " + str(index), "fraction", {"Acid": -1, "Base": 1, "Neutral": 0}[pka["Type"]])
        add(name + "|pKa value " + str(index), "fraction", pka["Pka"])
    for process in compound["Processes"]:
        internal = process["InternalName"]
        if internal not in {"LiverClearance", "KidneyClearance"}:
            raise ValueError("Native capacity audit supports explicit hepatic/renal plasma-clearance processes only")
        path = reviewed_process_parameter_path(name, {"internal_name": internal, "molecule": None,
            "data_source": process["DataSource"], "parameter_name": "Plasma clearance"})
        stem = path.rsplit("|", 1)[0]
        supplied = {p["Name"]: p for p in process["Parameters"]}
        if ("Specific clearance" in supplied or "Plasma clearance" not in supplied
                or "Fraction unbound (experiment)" not in supplied):
            raise ValueError("Audit expects native inversion from plasma CL, without an overriding specific clearance")
        for p in process["Parameters"]:
            add(stem + "|" + p["Name"], p.get("Unit", "fraction"), p["Value"])
        organ = "hepatic" if internal == "LiverClearance" else "renal"
        fields = {"Specific clearance": "1/min", "Body weight": "kg"}
        fields.update({"Blood flow rate (liver)": "l/min", "Blood flow rate (portal vein)": "l/min",
                       "Blood/Plasma concentration ratio": "fraction", "Volume (liver)": "l",
                       "Fraction intracellular (liver)": "fraction"} if organ == "hepatic" else
                      {"Blood flow rate (kidney)": "l/min", "Hematocrit": "fraction", "Volume (kidney)": "l"})
        for field, unit in fields.items():
            add(stem + "|" + field, unit)
        processes.append({"organ": organ, "stem": stem, "clearance_unit": supplied["Plasma clearance"]["Unit"]})
    if not processes or len({p["organ"] for p in processes}) != len(processes):
        raise ValueError("Require one or two distinct systemic clearance processes")
    paths = [r["path"] for r in parameters]
    if len(paths) != len(set(paths)) or len(paths) > 128:
        raise ValueError("Native parameter readback plan is duplicate or excessive")
    return parameters, processes


def _capacity(process, values):
    stem, organ = process["stem"], process["organ"]
    def value(field):
        return values[stem + "|" + field]
    unit = process["clearance_unit"]
    factor = {"ml/min/kg": .001, "l/h/kg": 1/60., "ml/h/kg": .001/60.}
    if unit not in factor:
        raise ValueError("Unreviewed plasma clearance unit")
    cl = value("Plasma clearance") * factor[unit] * value("Body weight")
    fu = value("Fraction unbound (experiment)")
    specific = value("Specific clearance")
    if organ == "hepatic":
        flow = value("Blood flow rate (liver)") + value("Blood flow rate (portal vein)")
        b2p = value("Blood/Plasma concentration ratio")
        volume = value("Volume (liver)") * value("Fraction intracellular (liver)")
        denominator = flow - cl / b2p if b2p > 0 else None
        capacity = flow * b2p
        fractions_valid = 0 < value("Fraction intracellular (liver)") <= 1
    else:
        flow = value("Blood flow rate (kidney)") * (1. - value("Hematocrit"))
        volume = value("Volume (kidney)")
        denominator, capacity = flow - cl, flow
        fractions_valid = 0 <= value("Hematocrit") < 1
    physical = (denominator is not None and denominator > 0 and capacity > 0 and volume > 0
                and 0 < fu <= 1 and cl > 0 and value("Body weight") > 0 and fractions_valid)
    predicted = cl * flow / (fu * denominator * volume) if physical else None
    no_fallback = 0 < specific < 1000000.
    formula_matches = predicted is not None and math.isfinite(predicted) and math.isclose(predicted, specific, rel_tol=1e-6, abs_tol=1e-10)
    return {"organ": organ, "accepted": bool(physical and no_fallback and formula_matches),
            "plasma_clearance_l_min": cl, "plasma_clearance_capacity_l_min": capacity,
            "capacity_denominator_l_min": denominator, "within_flow_capacity": physical,
            "native_specific_clearance_per_min": specific, "recomputed_specific_clearance_per_min": predicted,
            "native_fallback_1000000_absent": no_fallback, "native_formula_matches": formula_matches,
            "formula_name": _FORMULAS[organ], "interpretation": "native inversion check, not proof of the biological organ allocation"}


def audit_evidence_execution(snapshot, execution_manifest, output_dir, project_root):
    """Verify native inputs and clearance capacity without new simulation.

    Returns accepted, parameter_checks, clearance_checks, reasons and limitations.
    ``accepted`` covers these readback checks only; mass_balance_checked=False.
    """
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if output.exists():
        raise FileExistsError("Use a new native evidence audit directory")
    manifest_path = inside(execution_manifest["manifest_path"] if isinstance(execution_manifest, dict) else execution_manifest, root, exists=True)
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256(manifest_path)
    if isinstance(execution_manifest, dict) and execution_manifest != manifest:
        raise ValueError("Execution manifest differs from persisted native receipt")
    if not manifest.get("engineering_accepted") or not manifest.get("source_unchanged"):
        raise ValueError("A successful unchanged native execution is required")
    source = inside(manifest_path.parent / "snapshot.json", root, exists=True)
    if sha256(source) != manifest.get("source_sha256") or json.loads(source.read_text()) != snapshot:
        raise ValueError("Snapshot does not match its native execution")
    if len(manifest.get("pkml_outputs", [])) != 1:
        raise ValueError("Exactly one exported model is required")
    artifact = manifest["pkml_outputs"][0]
    pkml = inside(manifest_path.parent / artifact["path"], root, exists=True)
    if not pkml.is_relative_to(manifest_path.parent) or sha256(pkml) != artifact["sha256"]:
        raise ValueError("Native model path or checksum changed")
    database = inside(_DATABASE, root, exists=True)
    if sha256(database) != _DATABASE_SHA:
        raise ValueError("Native clearance formula database differs from reviewed source")
    with sqlite3.connect("file:" + str(database) + "?mode=ro", uri=True) as connection:
        formulas = {organ: connection.execute("SELECT formula FROM tab_calculation_method_rates WHERE calculation_method='LinksCommon' AND formula_rate=?", (name,)).fetchone()[0] for organ, name in _FORMULAS.items()}
    plan, processes = _plan(snapshot)
    output.mkdir(parents=True)
    result_path = output / "native_values.json"
    request = {"version": VERSION, "pkml_path": str(pkml), "pkml_sha256": sha256(pkml),
               "parameters": plan, "result_path": str(result_path)}
    request_path = output / "request.json"
    _write(request_path, request)
    request_sha = sha256(request_path)
    runtime = _run_bounded_r(script=root / "src/pbpk_kb/model_runtime/R/evidence_native_audit.R",
        args=[str(request_path), request_sha], output_dir=output, project_root=root,
        timeout_seconds=180, cpu_threads=2, max_output_bytes=32 * 1024**2)
    reasons, checks, capacities = [], [], []
    if runtime.get("status") != "completed" or runtime.get("returncode") != 0:
        reasons.append("native_parameter_read_failed")
    else:
        native = json.loads(result_path.read_text())
        if native.get("request_sha256") != request_sha or native.get("pkml_sha256") != sha256(pkml):
            raise ValueError("Native readback binding changed")
        if native.get("solver_calls") != 0 or native.get("model_modified") is not False:
            raise ValueError("Native audit must not execute or modify the model")
        rows = native.get("parameters", [])
        if len(rows) != len(plan) or [r.get("path") for r in rows] != [r["path"] for r in plan]:
            raise ValueError("Native parameter mapping differs from the requested paths")
        values = {}
        for requested, actual in zip(plan, rows):
            value = actual.get("value")
            valid = (actual.get("status") == "evaluated" and actual.get("unit") == requested["unit"]
                     and not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value))
            passed = valid and ("expected" not in requested or math.isclose(value, requested["expected"], rel_tol=requested["relative_tolerance"], abs_tol=1e-12))
            checks.append({**requested, "actual_value": value, "accepted": bool(passed), "error": actual.get("error")})
            if passed:
                values[requested["path"]] = value
            else:
                reasons.append("input_readback_mismatch:" + requested["path"])
        if not reasons:
            capacities = [_capacity(p, values) for p in processes]
            reasons.extend("clearance_capacity_or_formula_failed:" + r["organ"] for r in capacities if not r["accepted"])
    if (sha256(pkml) != artifact["sha256"] or sha256(source) != manifest["source_sha256"]
            or sha256(manifest_path) != manifest_sha or sha256(request_path) != request_sha
            or sha256(database) != _DATABASE_SHA):
        raise ValueError("Native audit inputs changed during readback")
    result = {"version": VERSION, "accepted": not reasons, "status": "passed" if not reasons else "failed",
              "reasons": reasons, "parameter_checks": checks, "clearance_checks": capacities,
              "source_sha256": manifest["source_sha256"], "pkml_sha256": artifact["sha256"],
              "execution_manifest_sha256": manifest_sha, "formula_database_sha256": _DATABASE_SHA,
              "official_formulas": formulas, "runtime": runtime, "solver_calls": 0,
              "model_modified": False, "mass_balance_checked": False, "scientific_accepted": False,
              "limitations": ["administration_audit_is_a_separate_check", "no_mass_balance_or_prediction_accuracy_claim",
                              "organ_allocation_evidence_not_proven_by_readback", "population_and_demographic_validity_not_assessed",
                              "native_partition_method_selection_not_independently_reconstructed"]}
    _write(output / "audit.json", result)
    return result
