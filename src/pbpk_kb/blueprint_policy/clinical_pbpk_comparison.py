"""Comparable clinical PBPK trajectories without clinical outcome access.

The native comparators use supplied, common molecular/ML inputs and the real
PK-Sim partition formulas. They are literature-inspired controlled baselines,
not reproductions of the original authors' parameter predictors or engine
versions. No Vss mapping, curve fitting or target-dependent formula selection
is performed for a native comparator. The separate ``run_ours`` entry retains
the frozen vascular/extravascular exchange construction and input-only search.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from pbpk_kb.blueprints.evidence_blueprint import assemble_evidence_blueprint
from pbpk_kb.blueprints.design_runtime import execute_snapshot
from pbpk_kb.blueprints.evidence_native_audit import audit_evidence_execution
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprints.native_edits import _method_source
from pbpk_kb.blueprints.source_adaptation_audit import audit_execution, derive_administration_expectations
from pbpk_kb.blueprints.source_workflow import inside, read_curve, save, sha256
from .axes_calibration import extended_snapshot
from .perfusion_export import export_physiology
from .learned_exchange_design import simulate_learned_design

VERSION = "clinical-pbpk-comparison-1"
PARTITION_METHODS = {
    "PKSimStandard": "Cellular partition coefficient method - PK-Sim Standard",
    "RodgersRowland": "Cellular partition coefficient method - Rodgers and Rowland",
    "Berezhkovskiy": "Cellular partition coefficient method - Berezhkovskiy",
    "PoulinTheil": "Cellular partition coefficient method - Poulin and Theil",
    "Schmitt": "Cellular partition coefficient method - Schmitt",
}
PERMEABILITY_METHOD = "Cellular permeability - PK-Sim Standard"
HORIZON_H = 168.
_OUTCOME_ROLES = {"target", "evaluation", "test", "validation", "heldout"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _ref(path, root):
    path = inside(path, root, exists=True)
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def _read_ref(reference, root):
    path = inside(reference["path"], root, exists=True)
    if sha256(path) != reference["sha256"]:
        raise ValueError("Comparison artifact checksum changed: " + reference["path"])
    return json.loads(path.read_text())


def _check_request(request):
    # Source identity checks precede reading an evidence file or native build.
    if not isinstance(request, dict):
        raise ValueError("An explicit construction request is required")
    dosing, target = request.get("dosing", {}), request.get("target", {})
    if (dosing.get("route") != "intravenous" or dosing.get("start_h", 0.) != 0
            or dosing.get("single_dose", True) is not True):
        raise ValueError("Comparison supports single IV administration at time zero")
    end = target.get("window_end_h")
    if isinstance(end, bool) or not isinstance(end, (int, float)) or not math.isfinite(end) or not 0 < end <= HORIZON_H:
        raise ValueError("Requested prediction window must end within 168 h")
    forbidden = request.get("forbidden_sources", [])
    paths, hashes = ({r.get(key) for r in forbidden} for key in ("path", "sha256"))
    for row in request.get("evidence", []):
        context, source = row.get("context", {}), row.get("source", {})
        if (context.get("study_role") in _OUTCOME_ROLES or source.get("path") in paths
                or source.get("sha256") in hashes):
            raise ValueError("Target outcome evidence is forbidden: " + str(row.get("parameter")))


def _check_sidecars(request, records, root):
    if not isinstance(records, list):
        raise ValueError("Independent parameter inputs must be sidecar lists")
    paths = {r.get("path") for r in request.get("forbidden_sources", [])}
    hashes = {r.get("sha256") for r in request.get("forbidden_sources", [])}
    refs = []
    for row in records:
        source, context = row.get("source", {}), row.get("context", {})
        if (context.get("study_role") in _OUTCOME_ROLES or context.get("independent_of_target") is not True
                or source.get("path") in paths or source.get("sha256") in hashes):
            raise ValueError("Target or non-independent sidecar is forbidden")
        path = inside(source["path"], root, exists=True)
        if not path.is_file() or path.stat().st_size > 64 * 1024**2 or sha256(path) != source["sha256"]:
            raise ValueError("Independent parameter source checksum or size mismatch")
        refs.append(_ref(path, root))
    return refs


def _partition_snapshot(snapshot, partition_method, root):
    if partition_method not in PARTITION_METHODS:
        raise ValueError("Unsupported native partition method")
    method = PARTITION_METHODS[partition_method]
    source, definition = _method_source(root, method)
    database = next(r for r in definition["resources"] if r["role"] == "database")
    path = inside(database["path"], root, exists=True)
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT m.calculation_method FROM tab_model_calculation_methods m "
            "JOIN tab_species_calculation_methods s USING(calculation_method) "
            "WHERE m.model='4Comp' AND s.species='Human' AND m.calculation_method=?", (method,)).fetchall()
    if rows != [(method,)]:
        raise ValueError("Partition formula is not bound to the standard human model")
    changed = deepcopy(snapshot)
    if len(changed.get("Compounds", [])) != 1 or len(changed.get("Simulations", [])) != 1:
        raise ValueError("One compound and simulation are required")
    selection = changed["Simulations"][0]["Compounds"]
    if len(selection) != 1 or selection[0]["Name"] != changed["Compounds"][0]["Name"]:
        raise ValueError("Simulation compound selection mismatch")
    before = [PARTITION_METHODS["PKSimStandard"], PERMEABILITY_METHOD]
    for target in (changed["Compounds"][0], selection[0]):
        if target.get("CalculationMethods") != before:
            raise ValueError("Only a freshly assembled standard partition selection can be replaced")
        target["CalculationMethods"] = [method, PERMEABILITY_METHOD]
    # Only these two selections change; no drug, physiology or clearance edits.
    restored = deepcopy(changed)
    restored["Compounds"][0]["CalculationMethods"] = before
    restored["Simulations"][0]["Compounds"][0]["CalculationMethods"] = before
    if restored != snapshot:
        raise ValueError("Unexpected mutation outside the two formula selections")
    for process in changed["Compounds"][0]["Processes"]:
        if process["InternalName"] not in {"LiverClearance", "KidneyClearance"}:
            raise ValueError("Systemic plasma CL comparator must not add GFR or metabolism")
    return changed, source


def _complete(folder, root, result):
    save(folder / "result.json", result)
    return {**result, "result_ref": _ref(folder / "result.json", root)}


def _extended_snapshot_168(snapshot):
    # The frozen helper currently extends to 1344 h. Reuse its interval policy
    # but enforce this comparison's separately declared 168 h protocol cap.
    native = extended_snapshot(snapshot)
    intervals = []
    for interval in native["Simulations"][0]["OutputSchema"]:
        values = {p["Name"]: p for p in interval["Parameters"]}
        if any(values[k].get("Unit") != "h" for k in ("Start time", "End time")):
            raise ValueError("Expected an explicitly hourly native output schema")
        if values["Start time"]["Value"] >= HORIZON_H:
            continue
        values["End time"]["Value"] = min(values["End time"]["Value"], HORIZON_H)
        intervals.append(interval)
    if not intervals or next(p["Value"] for p in intervals[-1]["Parameters"] if p["Name"] == "End time") != HORIZON_H:
        raise ValueError("Output schema does not cover the frozen 168 h horizon")
    native["Simulations"][0]["OutputSchema"] = intervals
    return native


def run_native_partition(request, output_dir, project_root, partition_method):
    """Run one actual PK-Sim formula; persist outputs, readbacks and one receipt.

    Callers coordinate at most fourteen simultaneous native jobs. Every solve
    uses two CPU threads, a 180 s timeout and a 128 MiB output cap. Existing
    folders are never overwritten; runner-owned receipts govern resumption.
    """
    _check_request(request)
    root = Path(project_root).resolve(strict=True)
    folder = inside(output_dir, root)
    if partition_method not in PARTITION_METHODS:
        raise ValueError("Unsupported native partition method")
    build_request = deepcopy(request)
    build_request["partition_method"] = "PKSimStandard"
    assembled = assemble_evidence_blueprint(build_request, root)
    if assembled["status"] != "assembled":
        raise ValueError("Incomplete native construction evidence: " + str(assembled.get("gaps")))
    native, formula_source = _partition_snapshot(_extended_snapshot_168(assembled["snapshot"]), partition_method, root)
    folder.mkdir(parents=True, exist_ok=False)
    result = {"version": VERSION, "status": "failed", "method": "native_" + partition_method,
        "partition_method": partition_method, "native_formula": PARTITION_METHODS[partition_method],
        "request_sha256": _digest(request), "observations_used": False, "curve": None, "endpoints": None,
        "native_solver_calls": 1, "cpu_threads": 2, "timeout_seconds": 180,
        "prediction_horizon_h": HORIZON_H, "formula_source": formula_source,
        "input_sources": assembled["manifest"]["sources"], "adapter_sha256": sha256(Path(__file__)),
        "interpretation": "Literature-inspired common-input native PBPK; not original QSAR/engine reproduction",
        "limitations": ["Native plasma-clearance input is not a demonstrated systemic dose/AUC identity",
                        "Declared organ allocation remains a construction assumption",
                        "Clinical accuracy is assessed only after trajectories are locked"]}
    execution = execute_snapshot(snapshot=native, output_dir=folder / "native", project_root=root,
        timeout_seconds=180, cpu_threads=2, max_output_bytes=128 * 1024**2)
    result["execution_ref"] = _ref(execution["manifest_path"], root)
    if not execution.get("engineering_accepted"):
        result["reason"] = "native_execution_failed"
        return _complete(folder, root, result)
    administration = audit_execution(snapshot=native, execution_manifest=execution,
        output_dir=folder / "administration_audit", project_root=root,
        **derive_administration_expectations(native))
    parameters = audit_evidence_execution(native, execution, folder / "parameter_audit", root)
    result["readback_checks"] = {"administration": bool(administration.get("accepted")),
                                  "parameters_and_clearance_capacity": bool(parameters.get("accepted"))}
    result["readback_diagnostics"] = {"administration": administration.get("reasons", []),
                                       "parameters": parameters.get("reasons", [])}
    result["audit_refs"] = {"parameters": _ref(folder / "parameter_audit/audit.json", root),
                              "administration": _ref(folder / "administration_audit/audit.json", root)}
    if not all(result["readback_checks"].values()):
        result["reason"] = "native_readback_failed"
        return _complete(folder, root, result)
    target = native["Simulations"][0]["OutputSelections"][0]
    curve = read_curve(Path(execution["manifest_path"]).parent, execution, target)
    result.update(status="predicted", curve=curve, endpoints=summarize_extended_prediction(curve, request),
                  runtime_binding=execution.get("runtime_binding"))
    return _complete(folder, root, result)


def run_ours(request, vss_evidence, half_evidence, output_dir, project_root, *, native_result=None):
    """Run the frozen learned-exchange design on the same native physiology/grid.

    ``native_result`` may reuse a receipt returned by ``run_native_partition``
    for this exact request and PKSimStandard. Sidecar values are independent
    predicted inputs, never the evaluation observations.
    """
    _check_request(request)
    root = Path(project_root).resolve(strict=True)
    folder = inside(output_dir, root)
    sources = _check_sidecars(request, vss_evidence, root) + _check_sidecars(request, half_evidence, root)
    if native_result is not None:
        persisted = _read_ref(native_result["result_ref"], root)
        if persisted != {k: v for k, v in native_result.items() if k != "result_ref"}:
            raise ValueError("Supplied native result differs from its persisted receipt")
        if (native_result.get("request_sha256") != _digest(request)
                or native_result.get("partition_method") != "PKSimStandard"
                or native_result.get("adapter_sha256") != sha256(Path(__file__))):
            raise ValueError("Reusable native result belongs to another request, formula or adapter")
    folder.mkdir(parents=True, exist_ok=False)
    native = native_result or run_native_partition(request, folder / "source", root, "PKSimStandard")
    result = {"version": VERSION, "status": "failed", "method": "ours_learned_exchange",
        "request_sha256": _digest(request), "observations_used": False, "curve": None, "endpoints": None,
        "native_source_ref": native["result_ref"], "execution_ref": native["execution_ref"],
        "sidecar_sources": sources, "native_solver_calls": 0 if native_result is not None else 1,
        "prediction_horizon_h": HORIZON_H,
        "interpretation": "Native human physiology plus vascular/extravascular exchange; independently predicted Vss/half inputs"}
    if native["status"] != "predicted":
        result["reason"] = "native_source_failed"
        return _complete(folder, root, result)
    execution = _read_ref(native["execution_ref"], root)
    physiology = export_physiology(execution_manifest_path=root / native["execution_ref"]["path"],
        source_manifest_sha256=native["execution_ref"]["sha256"], output_dir=folder / "physiology", project_root=root)
    result["physiology_ref"] = physiology["manifest"]
    model_request = deepcopy(request)
    canonicalized = []
    for row in model_request["evidence"]:
        if row["parameter"] in {"total_clearance", "hepatic_clearance", "renal_clearance"} and row.get("context", {}).get("route") == "iv":
            row["context"]["route"] = "intravenous"
            canonicalized.append(row["parameter"])
    result["iv_route_alias_canonicalized"] = canonicalized
    prediction = simulate_learned_design(physiology, model_request, deepcopy(vss_evidence), deepcopy(half_evidence),
        ps_bounds=(.01, 100.), fallback=True, target_basis="sampled_nca", max_trajectory_calls=33,
        times_h=np.asarray(native["curve"]["time_min"], dtype=float) / 60.)
    result["design_audit"] = prediction.get("learned_exchange_audit", {})
    result["linear_trajectory_calls"] = result["design_audit"].get("linear_trajectory_calls", 0)
    if prediction["status"] != "predicted":
        result["reason"] = prediction.get("reason", "learned_exchange_failed")
        return _complete(folder, root, result)
    full = prediction["full_trajectory"]
    mw = next(r["value"] for r in request["evidence"] if r["parameter"] == "molecular_weight")
    curve = {"time_min": [float(t) * 60. for t in full["time_h"]],
        "concentration": [float(c) * 1000. / mw for c in full["concentration_mg_l"]],
        "time_unit": "min", "unit": "umol/l"}
    result.update(status="predicted", curve=curve, endpoints=summarize_extended_prediction(curve, request),
        mass_balance=prediction["mass_balance"], construction_decision=prediction.get("decision"),
        observer=prediction["observer"], exact_moments_used_to_fill_NCA=False,
        runtime_binding=execution.get("runtime_binding"))
    return _complete(folder, root, result)


def mean_native_curves(curves, *, target_times_min=None):
    """Arithmetic curve mean before NCA; never average endpoints or select drugs.

    Equal grids are mandatory unless the caller supplies an explicit frozen
    interpolation grid. Interpolation is linear and never extrapolates.
    A missing/failed formula is an error, never silently dropped.
    """
    if not isinstance(curves, (list, tuple)) or len(curves) < 2:
        raise ValueError("At least two complete native curves are required")
    checked = []
    for curve in curves:
        if not isinstance(curve, dict) or curve.get("time_unit") != "min" or curve.get("unit") not in {"umol/l", "µmol/l", "μmol/l"}:
            raise ValueError("Every ensemble input must be a standard molar/minute curve")
        times, values = np.asarray(curve["time_min"], dtype=float), np.asarray(curve["concentration"], dtype=float)
        if (times.ndim != 1 or values.shape != times.shape or len(times) < 2
                or not np.isfinite(times).all() or not np.isfinite(values).all()
                or np.any(times < 0) or np.any(np.diff(times) <= 0) or np.any(values < 0)):
            raise ValueError("Invalid or missing ensemble trajectory")
        checked.append((times, values))
    if target_times_min is None:
        grid = checked[0][0]
        if any(not np.array_equal(grid, times) for times, _ in checked):
            raise ValueError("Different native grids require an explicitly declared interpolation grid")
        values = [v for _, v in checked]
        policy = "identical_native_grids_no_interpolation"
    else:
        grid = np.asarray(target_times_min, dtype=float)
        if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
                or np.any(np.diff(grid) <= 0) or grid[0] < 0):
            raise ValueError("Invalid predeclared interpolation grid")
        if any(grid[0] < t[0] or grid[-1] > t[-1] for t, _ in checked):
            raise ValueError("Ensemble interpolation cannot extrapolate")
        values = [np.interp(grid, t, v) for t, v in checked]
        policy = "linear_interpolation_on_explicit_grid_without_extrapolation"
    mean = np.sum([v / len(checked) for v in values], axis=0)
    if not np.isfinite(mean).all():
        raise ValueError("Nonfinite ensemble mean")
    return {"time_min": grid.tolist(), "concentration": mean.tolist(), "time_unit": "min", "unit": "umol/l",
        "ensemble_audit": {"member_count": len(checked), "weights": [1. / len(checked)] * len(checked),
            "combination": "arithmetic_mean_of_concentrations_before_endpoint_recalculation",
            "grid_policy": policy, "observations_used": False, "failed_members_dropped": False,
            "member_curve_sha256": [_digest(c) for c in curves]}}
