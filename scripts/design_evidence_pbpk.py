#!/usr/bin/env python3
"""Build and inspect a purpose-constrained PBPK design, without target outcomes.

prepare --input request.json --output experiments/blueprint_design/purpose_NAME
run/status --output experiments/blueprint_design/purpose_NAME

Input keys: request, fallback_evidence, ranges, endpoint_limits, optional purpose
and calibration_mapping ({path, sha256}). Existing fit mappings are validated;
without one all three multipliers are one. No fitting or model training occurs.
Range scenarios are sensitivity probes, not observations or confidence intervals.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, Evidence, _digest
from pbpk_kb.blueprints.evidence_preserving_calibration import _exclusive
from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
from pbpk_kb.blueprint_learning.native_calibration import _embedded_pins, _verify
from pbpk_kb.blueprint_policy import transport_calibration as engine
from pbpk_kb.blueprint_policy.purpose_design import plan_design, assess_design

VERSION = "purpose-pbpk-runner-1"
LIMIT_BYTES = 1024**3
SCALAR_FIELDS = {
    "auc": ("value_mg_h_per_l", "mg*h/l"),
    "cmax": ("value_mg_per_l", "mg/l"),
    "tmax": ("value_h", "h"), "c_last": ("value_mg_per_l", "mg/l"),
    "terminal_half_life": ("value_h", "h"),
    "auc_inf": ("value_mg_h_per_l", "mg*h/l"),
    "clearance": ("value_l_per_h", "l/h"),
    "vz": ("value_l", "l"), "mrt": ("value_h", "h"), "vss": ("value_l", "l"),
}


def read(path):
    path = inside(path, ROOT, exists=True)
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise ValueError("Expected a bounded project JSON document")
    return json.loads(path.read_text())


def ref(path):
    path = inside(path, ROOT, exists=True)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def run_dir(path):
    path = inside(path, ROOT)
    parent = ROOT / "experiments/blueprint_design"
    if path.parent != parent or not path.name.startswith("purpose_"):
        raise ValueError("Use an immediate experiments/blueprint_design/purpose_ directory")
    return path


def checked_input(document):
    allowed = {"request", "fallback_evidence", "ranges", "endpoint_limits", "purpose", "calibration_mapping"}
    if not isinstance(document, dict) or set(document) - allowed:
        raise ValueError("Unknown design input keys; outcomes are not design inputs")
    required = {"request", "fallback_evidence", "ranges", "endpoint_limits"}
    if not required <= set(document):
        raise ValueError("Missing design input keys")
    # Verify explicit source references, including rejected evidence, before
    # deriving a new ledger. Hashes establish integrity, not extraction truth.
    pins = _embedded_pins(document, ROOT)
    _verify(pins, ROOT)
    request = document["request"]
    forbidden = request.get("forbidden_sources", [])
    bad_paths = {str(inside(x["path"], ROOT).relative_to(ROOT)) for x in forbidden}
    bad_hashes = {x["sha256"] for x in forbidden}
    used = _embedded_pins({k: v for k, v in document.items() if k != "request"} |
                          {"request": {k: v for k, v in request.items() if k != "forbidden_sources"}}, ROOT)
    if bad_paths & used.keys() or bad_hashes & set(used.values()):
        raise ValueError("A design input refers to a forbidden outcome source")
    ledger_ref = request["evidence_ledger"]
    ledger = read(ledger_ref["path"])
    rows = ledger.get("evidence") if isinstance(ledger, dict) else ledger
    if not isinstance(rows, list):
        raise ValueError("Original ledger has no evidence records")
    canonical = lambda records: sorted((_digest(Evidence.model_validate(x).model_dump(mode="json")) for x in records))
    if canonical(rows) != canonical(request["evidence"]):
        raise ValueError("Request evidence differs from its original source ledger")
    if request.get("dosing", {}).get("start_h", 0) != 0:
        raise ValueError("The current exchange solver requires dose time zero")
    return pins


def reject_forbidden(value, request):
    """Apply the current request's exclusions to dependencies of an old fit."""
    forbidden = request.get("forbidden_sources", [])
    paths = {str(inside(x["path"], ROOT).relative_to(ROOT)) for x in forbidden}
    hashes = {x["sha256"] for x in forbidden}
    refs = _embedded_pins(value, ROOT)
    if paths & refs.keys() or hashes & set(refs.values()):
        raise ValueError("Calibration depends on a currently forbidden outcome source")


def mapping_parameters(document, selected_request):
    mapping_ref = document.get("calibration_mapping")
    if not mapping_ref:
        return {"log_parameters": [0., 0., 0.], "mode": "uncalibrated",
                "development_curve_used": False}
    path = inside(mapping_ref["path"], ROOT, exists=True)
    if sha256(path) != mapping_ref["sha256"]:
        raise ValueError("Calibration mapping changed")
    mapping = read(path)
    engine._validated(mapping, path, ROOT)
    if not mapping.get("eligible") or mapping["variant"] != "flow_exchange":
        raise ValueError("Only an eligible frozen paper flow_exchange mapping is supported")
    original = read(path.parent / "input.json")["request"]
    # A formerly valid development fit can be invalid for a new request whose
    # evaluation source was used by that fit. Check before decoding training.
    training_source = mapping["training_source"]
    reject_forbidden({"training_source": training_source,
                      "request": {k: v for k, v in original.items() if k != "forbidden_sources"}}, selected_request)
    reject_forbidden(read(training_source["path"]), selected_request)
    def drug(req):
        return {"compound_id": req["compound_id"], "compound_name": req["compound_name"],
                "partition_method": req["partition_method"],
                "evidence": sorted([Evidence.model_validate(x).model_dump(mode="json") for x in req["evidence"]],
                                   key=lambda x: x["parameter"])}
    if drug(original) != drug(selected_request):
        raise ValueError("Mapping nominal drug inputs differ; do not reuse a complete-evidence fit after changing inputs")
    return {"log_parameters": mapping["decision"]["log_parameters"],
            "mode": "locked_development_calibration", "mapping": mapping_ref,
            "development_curve_used": True, "scenario_refitting": False}


def prepare(input_path, output):
    output = run_dir(output)
    if output.exists():
        entries = list(output.iterdir())
        if (inside(input_path, ROOT, exists=True) != output / "request.json"
                or any(p.is_symlink() or not p.is_file() or p.name not in {"request.json", "range_source.json"}
                       for p in entries)):
            raise FileExistsError("Preserve an existing design; use run/status")
    document = read(input_path)
    pins = checked_input(document)
    decision = plan_design(document["request"], document["fallback_evidence"], document["ranges"],
                           endpoint_limits=document["endpoint_limits"], purpose=document.get("purpose"))
    selected = deepcopy(decision["selected_request"])
    calibration = mapping_parameters(document, selected) if decision["scenarios"] else None
    code = engine._code_pins(ROOT)
    for path in (Path(__file__), ROOT / "src/pbpk_kb/blueprint_policy/purpose_design.py",
                 ROOT / "src/pbpk_kb/blueprint_policy/evidence_decision.py"):
        code[str(path.relative_to(ROOT))] = sha256(path)
    protocol = {"version": VERSION, "input": document, "input_ref": ref(input_path),
                "decision": decision, "calibration": calibration, "source_pins": pins, "code_pins": code,
                "native_threads": 2, "native_timeout_seconds": 180, "directory_limit_bytes": LIMIT_BYTES,
                "scientific_status": "not_clinically_validated",
                "scope": "adult single IV parent plasma; curated evidence, no complete drug model, no target outcomes"}
    output.mkdir(parents=True, exist_ok=True)
    save(output / "protocol.json", protocol)
    save(output / "state.json", {"version": VERSION, "protocol": ref(output / "protocol.json"),
                                 "status": "prepared", "predictions": []})
    return status(output)


def load(output):
    output = run_dir(output)
    protocol, state = read(output / "protocol.json"), read(output / "state.json")
    if protocol["version"] != VERSION or state["protocol"] != ref(output / "protocol.json"):
        raise ValueError("Design protocol changed")
    if protocol["input_ref"] != ref(protocol["input_ref"]["path"]):
        raise ValueError("Design input changed")
    _verify(protocol["source_pins"], ROOT)
    _verify(protocol["code_pins"], ROOT)
    validate_calibration(protocol)
    for row in state["predictions"]:
        _verify(row.get("artifacts", {}), ROOT)
        stored = output / "models" / row["scenario_id"] / "prediction.json"
        if read(stored) != {k: v for k, v in row.items() if k != "artifacts"}:
            raise ValueError("Checkpoint prediction differs from its locked scenario output")
    return output, protocol, state


def validate_calibration(protocol):
    if protocol["decision"]["scenarios"]:
        actual = mapping_parameters(protocol["input"], protocol["decision"]["selected_request"])
        if actual != protocol["calibration"]:
            raise ValueError("Locked development calibration changed")


def selected_scenario(protocol, scenario, folder):
    request = deepcopy(protocol["decision"]["selected_request"])
    bounds = {r["parameter"]: r for r in protocol["input"]["ranges"]}
    folder.mkdir(parents=True, exist_ok=True)
    for row in request["evidence"]:
        name = row["parameter"]
        value = scenario["values"].get(name, row["value"])
        if value != row["value"]:
            if row["basis"] != "predicted" or name not in bounds:
                raise ValueError("Sensitivity scenarios may only change explicitly ranged predictions")
            original = deepcopy(row)
            source = {"kind": "sensitivity_scenario_not_observed", "parameter": name,
                      "nominal_evidence": original, "range": bounds[name], "value": value,
                      "range_is_probability_interval": False}
            path = folder / (name + "_scenario.json")
            save(path, source)
            row["source"] = {**ref(path), "location": "value (sensitivity scenario, not a new measurement)"}
            row["value"] = value
            row["notes"] = "Scenario derived from the declared range; original prediction retained in nominal_evidence."
    ledger = folder / "selected_evidence.json"
    save(ledger, {"evidence": request["evidence"]})
    request["evidence_ledger"] = {**ref(ledger), "location": "evidence"}
    return EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)


def simulate_scenario(protocol, scenario, folder, guard):
    request = selected_scenario(protocol, scenario, folder)
    physiology = engine._physiology(request, folder, ROOT, guard)
    times = engine.prediction_grid(request)
    prediction = engine._simulate(physiology, request, times, "flow_exchange",
                                  engine.PARAMETERS["flow_exchange"], protocol["calibration"]["log_parameters"])
    mw = next(x["value"] for x in request["evidence"] if x["parameter"] == "molecular_weight")
    curve = {"time_min": (times * 60).tolist(),
             "concentration": (np.asarray(prediction["concentration_mg_l"]) * 1000 / mw).tolist(),
             "time_unit": "min", "unit": "umol/l", "observation": prediction["observation"]}
    summary = summarize_extended_prediction(curve, request)
    endpoints = {}
    for name in protocol["decision"]["requested_endpoints"]:
        if name not in SCALAR_FIELDS:
            continue
        key, unit = SCALAR_FIELDS[name]
        result = summary["endpoints"][name]
        endpoints[name] = {"value": result.get(key) if result.get("status") == "completed" else None,
                           "unit": unit, "status": result.get("status"), "reason": result.get("reason")}
    start, end = request["target"]["window_start_h"], request["target"]["window_end_h"]
    in_window = np.unique(np.r_[start, times[(times > start) & (times < end)], end])
    sampled = np.interp(in_window, times, prediction["concentration_mg_l"])
    return {"scenario_id": scenario["scenario_id"], "status": "predicted", "endpoints": endpoints,
            "curve": {"time_h": in_window.tolist(), "concentration_mg_l": sampled.tolist()},
            "mass_balance": prediction["mass_balance"], "observer": prediction["observation"],
            "terminal_diagnostics": summary.get("terminal_diagnostics"),
            "effective_total_clearance_l_h": prediction["effective_total_clearance_l_h"],
            "endpoint_estimator_uses_observations": False}


def execute(output):
    output = run_dir(output)
    with _exclusive(output):
        output, protocol, state = load(output)
        if state["status"] == "complete":
            return status(output)
        if state.get("incomplete_scenario"):
            raise ValueError("Interrupted native scenario retained; inspect it before starting a new design")
        def guard():
            try:
                _verify(protocol["source_pins"], ROOT)
                _verify(protocol["code_pins"], ROOT)
                validate_calibration(protocol)
            except ValueError as exc:
                raise RuntimeError("Source integrity changed; native submissions paused") from exc
            size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            if size > LIMIT_BYTES:
                raise RuntimeError("Design directory exceeded its 1 GiB limit; submissions paused")
        state["status"] = "running"
        save(output / "state.json", state)
        started = time.monotonic()
        for scenario in protocol["decision"]["scenarios"]:
            if any(row["scenario_id"] == scenario["scenario_id"] for row in state["predictions"]):
                continue
            try:
                guard()
            except RuntimeError as exc:
                state.update(status="paused", infrastructure_error=str(exc))
                save(output / "state.json", state)
                raise
            state["incomplete_scenario"] = scenario["scenario_id"]
            save(output / "state.json", state)
            folder = output / "models" / scenario["scenario_id"]
            try:
                row = simulate_scenario(protocol, scenario, folder, guard)
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                row = {"scenario_id": scenario["scenario_id"], "status": "failed", "reason": str(exc)}
            except Exception as exc:
                state.update(status="paused", infrastructure_error=str(exc))
                save(output / "state.json", state)
                raise
            folder.mkdir(parents=True, exist_ok=True)
            save(folder / "prediction.json", row)
            row["artifacts"] = {str(p.relative_to(ROOT)): sha256(p)
                                for p in folder.rglob("*") if p.is_file() and not p.name.startswith(".")}
            state["predictions"].append(row)
            state.pop("incomplete_scenario", None)
            save(output / "state.json", state)
        try:
            guard()
        except RuntimeError as exc:
            state.update(status="paused", infrastructure_error=str(exc))
            save(output / "state.json", state)
            raise
        assessment = assess_design(protocol["decision"], state["predictions"])
        save(output / "result.json", {"version": VERSION, "protocol": ref(output / "protocol.json"),
                                      "assessment": assessment, "calibration": protocol["calibration"],
                                      "predictions": state["predictions"],
                                      "target_observations_used": False,
                                      "scientific_status": "not_clinically_validated",
                                      "elapsed_seconds_this_invocation": time.monotonic() - started})
        state.update(status="complete", result=ref(output / "result.json"))
        save(output / "state.json", state)
    return status(output)


def status(output):
    output, protocol, state = load(output)
    if state.get("result") and state["result"] != ref(output / "result.json"):
        raise ValueError("Design result changed")
    return {"status": state["status"], "completed": len(state["predictions"]),
            "planned": len(protocol["decision"]["scenarios"]),
            "failed": sum(x["status"] != "predicted" for x in state["predictions"]),
            "result": state.get("result"), "scientific_status": "not_clinically_validated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and args.input is None:
        parser.error("prepare requires --input")
    result = prepare(args.input, args.output) if args.command == "prepare" else (
        execute(args.output) if args.command == "run" else status(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
