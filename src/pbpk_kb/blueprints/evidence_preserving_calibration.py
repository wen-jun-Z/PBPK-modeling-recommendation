"""Restartable development calibration with a fixed net renal input.

Only the selected development rows enter the unchanged two-parameter loss.
Keeping a native renal-clearance input fixed does not establish the observed
systemic renal clearance or a validated renal prediction.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
from numbers import Integral
from pathlib import Path
import time

from .curve_pattern_blueprint import optimize_pattern_curve, ALGORITHM_VERSION as OPTIMIZER_VERSION
from .development_curve_blueprint import _training_document, _run_development, _annotate
from .evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, _digest
from .pk_anchored_blueprint import mapped_snapshot, _drug_signature, _run
from .source_workflow import inside, save, sha256

VERSION = "evidence-preserving-calibration-1.0"
ALGORITHM_VERSION = "evidence-preserving-pattern-1.0"
CLEARANCE_POLICY = "only_nonrenal_when_net_renal_evidence"
INFRASTRUCTURE_RETRIES_PER_FIT = 1


class InfrastructurePause(RuntimeError):
    """Stop the fit without converting an infrastructure interruption to loss."""


class _AbortFit(BaseException):
    """The legacy numerical optimizer catches Exception; halt must escape it."""


def _read(path):
    return json.loads(path.read_text())


def _ref(path, root):
    path = inside(path, root, exists=True)
    return {"path": str(path.relative_to(root)), "sha256": sha256(path)}


def _verify(pins, root):
    for name, checksum in pins.items():
        if sha256(inside(name, root, exists=True)) != checksum:
            raise ValueError("Frozen source or artifact changed: " + name)


def _source_pins(value, root):
    pins = {}
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = inside(value["path"], root, exists=True)
            pins[str(path.relative_to(root))] = value["sha256"]
        for child in value.values():
            pins.update(_source_pins(child, root))
    elif isinstance(value, list):
        for child in value:
            pins.update(_source_pins(child, root))
    return pins


def _code_pins():
    # These project-local Python dependencies include the native/R adapter
    # generator and physical readback gates. No environment or credentials read.
    package = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in sorted(package.glob("*.py"))}


def _check_code(expected):
    if _code_pins() != expected:
        raise ValueError("Calibration code/dependency hashes changed")


@contextmanager
def _exclusive(directory):
    # A kernel lock disappears on a crash. Each native invocation nevertheless
    # gets a fresh directory, so an orphaned R process cannot corrupt its retry.
    with (directory / ".calibration.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InfrastructurePause("Another invocation owns this calibration directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def snapshot_for_mapping(blueprint, lipophilicity, clearance_scale):
    """Use the old mapping exactly, then restore any net renal input verbatim."""
    native = mapped_snapshot(blueprint, lipophilicity, clearance_scale)
    original = blueprint["snapshot"]["Compounds"][0]["Processes"]
    changed = native["Compounds"][0]["Processes"]
    for before, after in zip(original, changed):
        if before["InternalName"] != "KidneyClearance":
            continue
        values = [p for p in before["Parameters"] if p["Name"] == "Plasma clearance"]
        if len(values) != 1 or isinstance(values[0]["Value"], bool) or not isinstance(values[0]["Value"], (int, float)) or not math.isfinite(values[0]["Value"]) or values[0]["Value"] <= 0:
            raise ValueError("Net renal process requires one positive finite plasma-clearance input")
        for parameter in after["Parameters"]:
            if parameter["Name"] == "Plasma clearance":
                parameter.clear(); parameter.update(deepcopy(values[0]))
    return native


def _signature(request):
    return _digest({"version": VERSION, "clearance_policy": CLEARANCE_POLICY,
                    "algorithm_version": ALGORITHM_VERSION, "drug_signature": _drug_signature(request)})


def _flags(output, training_source):
    _annotate(output, training_source)
    path = output / "endpoints.json"
    if path.exists():
        data = _read(path)
        data.update(observations_used=False, endpoint_estimator_uses_observations=False,
                    endpoint_estimator_fitted_pbpk_parameters=False,
                    development_curve_used_for_parameter_calibration=True,
                    comparison_observations_used=False, clearance_policy=CLEARANCE_POLICY)
        save(path, data)


def _infrastructure_problem(result):
    if result.get("status") in {"infrastructure_failed", "infrastructure_paused"}:
        return True
    def inspect(value):
        if isinstance(value, dict):
            if value.get("stop_reason") or value.get("timed_out"):
                return True
            if value.get("returncode") in {-9, 137}:
                return True
            return any(inspect(v) for v in value.values())
        if isinstance(value, list):
            return any(inspect(v) for v in value)
        if isinstance(value, str):
            return any(x in value.lower() for x in (
                "no space left", "out of memory", "cannot allocate memory", "filenotfounderror",
                "permissionerror", "timeouterror", "timeout expired", "timed out", "disk quota"))
        return False
    return inspect(result)


def _checked_completion(folder, parameters, root):
    completion = _read(folder / "completion.json")
    if completion["parameters"] != parameters:
        raise ValueError("Saved trial parameters differ from deterministic replay")
    _verify(completion["artifact_pins"], root)
    result = _read(root / completion["result"]["path"])
    if result.get("status") == "predicted":
        curve = completion.get("prediction")
        if curve is None or curve["sha256"] != result.get("prediction_sha256"):
            raise ValueError("Completed native prediction has no matching checksum")
        result = {**result, "curve": _read(root / curve["path"])}
    return completion, result


def fit_mapping(*, reference_request, training_source, output_dir, project_root,
                max_evaluations=24, before_trial=None):
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, Integral) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if before_trial is not None and not callable(before_trial):
        raise ValueError("before_trial must be callable")
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode="json", exclude_none=True)
    if req["dosing"]["route"] != "intravenous":
        raise ValueError("Single-dose IV development input required")
    training_source = deepcopy(training_source)
    observed = _training_document(training_source, req, root)
    if observed.get("human") != req["human"] or observed.get("dosing") != req["dosing"]:
        raise ValueError("Development source human/dosing must match the development request exactly")
    blueprint = assemble_evidence_blueprint(req, root)
    if blueprint["status"] != "assembled":
        raise ValueError("Required construction evidence is incomplete")
    allocation = blueprint["manifest"]["clearance_allocation"]
    target_cl = sum(allocation[k] for k in ("hepatic_ml_min_kg", "renal_ml_min_kg"))*req["human"]["weight_kg"]*.06
    logp = next(e["value"] for e in req["evidence"] if e["parameter"] == "logp")
    pins = _source_pins([req, training_source, observed], root)
    _verify(pins, root)
    code = _code_pins()
    fixed = {"version": VERSION, "algorithm_version": ALGORITHM_VERSION,
        "optimizer_version": OPTIMIZER_VERSION, "clearance_policy": CLEARANCE_POLICY,
        "mapping_signature": _signature(req), "request": req, "training_source": training_source,
        "maximum_evaluations": int(max_evaluations), "warm_start_source": None,
        "target_clearance_l_h": target_cl, "target_vss_l": None,
        "source_pins": pins, "code_pins": code, "infrastructure_retries_per_fit": INFRASTRUCTURE_RETRIES_PER_FIT}
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        input_path = output / "input.json"
        if input_path.exists():
            if _read(input_path) != fixed:
                raise ValueError("Resume input, code, policy, budget or source mismatch")
        else:
            if any(p.name != ".calibration.lock" for p in output.iterdir()):
                raise ValueError("Unrecognized existing fit directory")
            save(input_path, fixed)
        mapping_path = output / "mapping_locked.json"
        if mapping_path.exists():
            # A locked fit no longer needs unselected successful native trials.
            # Incomplete fits still replay every completed callback below.
            prior = _read(mapping_path)
            if (prior.get("version") != VERSION or prior.get("algorithm_version") != ALGORITHM_VERSION
                    or prior.get("clearance_policy") != CLEARANCE_POLICY
                    or prior.get("mapping_signature") != _signature(req)
                    or prior.get("input_sha256") != sha256(input_path)
                    or prior.get("source_pins") != pins or prior.get("code_pins") != code):
                raise ValueError("Completed mapping input, code or policy mismatch")
            _verify(prior["trial_artifact_pins"], root)
            if prior.get("selected_prediction"):
                _verify(_source_pins(prior["selected_prediction"], root), root)
            try:
                if before_trial is not None:
                    before_trial()
            except Exception as exc:
                raise InfrastructurePause(type(exc).__name__ + ": " + str(exc)) from exc
            return {**prior, "actual_new_calls": 0, "reused_calls": prior["evaluation_count"],
                "infra_retry_calls": 0, "completed_fit_reused": True,
                "mapping_path": str(mapping_path.relative_to(root)), "mapping": _ref(mapping_path, root)}
        started = time.monotonic()
        counters = {"actual_new_calls": 0, "reused_calls": 0, "infra_retry_calls": 0}
        ordinal = 0
        trial_artifacts, executions, diagnostic_evaluations = {}, {}, set()
        pause = None

        def stop(reason):
            nonlocal pause
            pause = str(reason)
            raise _AbortFit(pause)

        def checkpoint():
            if pause is not None:
                raise _AbortFit(pause)
            try:
                if before_trial is not None:
                    before_trial()
                _verify(pins, root); _check_code(code)
            except Exception as exc:
                stop(type(exc).__name__ + ": " + str(exc))

        def evaluate(lipophilicity, clearance_scale):
            nonlocal ordinal
            checkpoint()
            ordinal += 1
            parameters = {"evaluation": ordinal, "lipophilicity": lipophilicity,
                "clearance_scale": clearance_scale, "input_sha256": sha256(input_path),
                "clearance_policy": CLEARANCE_POLICY}
            trial = output / f"attempt_{ordinal:02d}"
            trial.mkdir(exist_ok=True)
            parameter_path = trial / "parameters.json"
            if parameter_path.exists() and _read(parameter_path) != parameters:
                stop("Saved parameter point disagrees with deterministic optimizer replay")
            if not parameter_path.exists():
                save(parameter_path, parameters)
            local_trial_pins = {str(parameter_path.relative_to(root)): sha256(parameter_path)}
            for slot in (0, 1):
                execution = trial / f"execution_{slot}"
                completion_path = execution / "completion.json"
                if completion_path.exists():
                    try:
                        completion, result = _checked_completion(execution, parameters, root)
                    except Exception as exc:
                        stop(type(exc).__name__ + ": " + str(exc))
                    local_trial_pins.update(completion["artifact_pins"])
                    local_trial_pins[str(completion_path.relative_to(root))] = sha256(completion_path)
                    if completion["infrastructure_failure"]:
                        diagnostic_evaluations.add(ordinal)
                        continue
                    counters["reused_calls"] += 1
                else:
                    if execution.exists():
                        diagnostic_evaluations.add(ordinal)
                        continue  # Unknown interrupted native process: never reuse its directory.
                    if slot:
                        prior_retries = sum(1 for _ in output.glob("attempt_*/execution_1"))
                        if prior_retries >= INFRASTRUCTURE_RETRIES_PER_FIT:
                            stop("Infrastructure retry limit exhausted; original diagnostics retained")
                        counters["infra_retry_calls"] += 1
                    checkpoint()
                    execution.mkdir()
                    save(execution / "parameters.json", parameters)
                    counters["actual_new_calls"] += 1
                    try:
                        native = snapshot_for_mapping(blueprint, lipophilicity, clearance_scale)
                        result = _run_development(native, req, execution, root, audit=False, require_vss=False)
                        _flags(execution, training_source)
                    except (OSError, TimeoutError, RuntimeError) as exc:
                        result = {"status": "infrastructure_failed", "reason": type(exc).__name__+": "+str(exc)}
                    except Exception as exc:
                        result = {"status": "native_failed", "reason": type(exc).__name__+": "+str(exc)}
                    save(execution / "result.json", result)
                    result_ref = _ref(execution / "result.json", root)
                    local_pins = {result_ref["path"]: result_ref["sha256"]}
                    prediction = None
                    if result.get("status") == "predicted":
                        prediction = _ref(execution / "prediction_locked.json", root)
                        if prediction["sha256"] != result.get("prediction_sha256"):
                            stop("Native prediction/result checksum mismatch")
                        local_pins[prediction["path"]] = prediction["sha256"]
                    for item in ("endpoints.json", "native/execution_manifest.json", "native/snapshot.json",
                                 "parameter_audit_result.json", "parameters.json"):
                        path = execution / item
                        if path.exists():
                            ref = _ref(path, root); local_pins[ref["path"]] = ref["sha256"]
                    receipt = execution / "native/execution_manifest.json"
                    if receipt.exists():
                        for curve in _read(receipt).get("curve_outputs", []):
                            path = inside(curve["path"], receipt.parent, exists=True)
                            local_pins[str(path.relative_to(root))] = curve["sha256"]
                    completion = {"parameters": parameters, "result": result_ref, "prediction": prediction,
                        "artifact_pins": local_pins, "infrastructure_failure": _infrastructure_problem(result)}
                    _verify(local_pins, root)
                    save(completion_path, completion)
                    local_trial_pins.update(local_pins)
                    local_trial_pins[str(completion_path.relative_to(root))] = sha256(completion_path)
                    if completion["infrastructure_failure"]:
                        diagnostic_evaluations.add(ordinal)
                        continue
                    if prediction:
                        result = {**result, "curve": _read(root / prediction["path"])}
                trial_artifacts[ordinal] = local_trial_pins
                executions[ordinal] = str(execution.relative_to(root))
                save(output / "progress.json", {"status": "running", "native_evaluations": ordinal,
                    "maximum_evaluations": max_evaluations, **counters, "latest": result.get("status")})
                return result
            stop("Native infrastructure trial remains incomplete or failed after its single permitted retry")

        def guarded_evaluate(lipophilicity, clearance_scale):
            try:
                return evaluate(lipophilicity, clearance_scale)
            except Exception as exc:
                # Includes disk/checksum/JSON errors outside the native call;
                # never let the optimizer consume the rest of its budget here.
                stop(type(exc).__name__ + ": " + str(exc))

        try:
            fit = optimize_pattern_curve(guarded_evaluate, observations=observed, logp_prior=logp,
                target_clearance_l_h=target_cl, max_evaluations=max_evaluations, warm_start=None)
            if pause is not None:
                raise _AbortFit(pause)
            checkpoint()
            _training_document(training_source, req, root)
            if assemble_evidence_blueprint(req, root)["status"] != "assembled":
                raise ValueError("Evidence changed during fitting")
        except _AbortFit as exc:
            save(output / "progress.json", {"status": "infrastructure_paused", "reason": str(exc),
                "native_evaluations": ordinal, "maximum_evaluations": max_evaluations, **counters})
            raise InfrastructurePause(str(exc)) from None
        elapsed = time.monotonic()-started
        if fit["decision"] is not None:
            execution = inside(executions[fit["decision"]["evaluation"]], root, exists=True)
            fit["decision"].update(native_output_dir=str(execution.relative_to(root)),
                prediction_path=str((execution / "prediction_locked.json").relative_to(root)))
            fit["selected_prediction"] = _ref(execution / "prediction_locked.json", root)
        retained = diagnostic_evaluations | {row["evaluation"] for row in fit["evaluations"] if row["status"] != "ok"}
        if fit["decision"] is not None:
            retained.add(fit["decision"]["evaluation"])
        artifacts = {path: digest for evaluation in sorted(retained)
                     for path, digest in trial_artifacts[evaluation].items()}
        fit.update(version=VERSION, algorithm_version=ALGORITHM_VERSION, optimizer_version=OPTIMIZER_VERSION,
            clearance_policy=CLEARANCE_POLICY, mapping_signature=_signature(req),
            compound_id=req["compound_id"], drug_signature=_drug_signature(req),
            construction_partition_method=req["partition_method"], reference_human=req["human"],
            reference_dosing=req["dosing"], evidence_ledger=req["evidence_ledger"],
            training_source=training_source, target_clearance_l_h=target_cl, target_vss_l=None,
            target_vss_anchor=None, warm_start_source=None, source_pins=pins, code_pins=code,
            input_sha256=sha256(input_path), trial_artifact_pins=artifacts,
            retained_evaluations=sorted(retained),
            disposable_successful_evaluations=sorted(set(trial_artifacts)-retained),
            development_curve_used=True, comparison_curve_used=False,
            endpoint_estimator_uses_observations=False, independent_validation=False,
            elapsed_seconds=elapsed, **counters,
            total_native_calls=sum(1 for _ in output.glob("attempt_*/execution_*")),
            limitations=["Fixed renal construction input is not a guarantee of clinical systemic renal clearance.",
                "Nonrenal elimination is a lumped liver-module input, not identified hepatic metabolism.",
                "Only the supplied development observation rows enter the loss; no warm start is permitted.",
                "Derived CL/Vss are diagnostics, not optimization targets; numerical convergence and identifiability are not guaranteed."])
        save(mapping_path, fit)
        save(output / "progress.json", {"status": fit["status"], "native_evaluations": ordinal,
            "maximum_evaluations": max_evaluations, "eligible": fit["eligible"], **counters})
        return {**fit, "mapping_path": str(mapping_path.relative_to(root)), "mapping": _ref(mapping_path, root)}


def predict_mapping(*, request, mapping_path, output_dir, project_root, renal_outputs=False):
    root = Path(project_root).resolve(strict=True)
    output, path = inside(output_dir, root), inside(mapping_path, root, exists=True)
    if not isinstance(renal_outputs, bool):
        raise ValueError("renal_outputs must be boolean")
    mapping = _read(path)
    mapping_hash = sha256(path)
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json", exclude_none=True)
    if (mapping.get("version") != VERSION or mapping.get("algorithm_version") != ALGORITHM_VERSION
            or mapping.get("clearance_policy") != CLEARANCE_POLICY or not mapping.get("eligible")
            or not mapping.get("decision") or mapping.get("mapping_signature") != _signature(req)
            or mapping.get("drug_signature") != _drug_signature(req)):
        raise ValueError("Mapping policy/version or drug evidence does not match this predictor")
    if req["dosing"]["route"] != "intravenous":
        raise ValueError("Single-dose IV prediction required")
    _check_code(mapping["code_pins"]); _verify(mapping["source_pins"], root)
    _verify(mapping["trial_artifact_pins"], root)
    _training_document(mapping["training_source"], req, root)
    if output.exists():
        raise FileExistsError("Existing final prediction directory is immutable")
    blueprint = assemble_evidence_blueprint(req, root)
    if blueprint["status"] != "assembled":
        raise ValueError("Required target evidence is incomplete")
    point = mapping["decision"]
    native = snapshot_for_mapping(blueprint, point["lipophilicity"], point["clearance_scale"])
    if renal_outputs:
        from .renal_extension import add_renal_outputs, read_renal_outputs
        native = add_renal_outputs(native)
    output.mkdir(parents=True)
    save(output / "request.json", req)
    save(output / "construction.json", {"version": VERSION, "clearance_policy": CLEARANCE_POLICY,
        "mapping": _ref(path, root), "mapping_signature": mapping["mapping_signature"],
        "request_sha256": _digest(req), "snapshot_sha256": _digest(native), "renal_outputs": renal_outputs})
    result = _run(native, req, output, root, audit=True)
    _flags(output, mapping["training_source"])
    if renal_outputs and result.get("status") == "predicted":
        renal = read_renal_outputs(output / "native/execution_manifest.json", native, root)
        save(output / "renal_prediction.json", renal)
        result["renal_prediction"] = _ref(output / "renal_prediction.json", root)
        if not renal.get("accepted"):
            result.update(status="renal_readback_failed", reason="Native renal output failed engineering checks")
    _check_code(mapping["code_pins"]); _verify(mapping["source_pins"], root)
    _training_document(mapping["training_source"], req, root)
    if sha256(path) != mapping_hash:
        raise ValueError("Selected mapping changed during prediction")
    result.update(method_version=VERSION, clearance_policy=CLEARANCE_POLICY,
        partition_method=req["partition_method"], development_curve_used=True, comparison_curve_used=False,
        observations_used_for_endpoint_estimation=False, endpoint_estimator_uses_observations=False,
        mapping_signature=mapping["mapping_signature"])
    if (output / "prediction_locked.json").exists():
        result.update(prediction_path=str((output / "prediction_locked.json").relative_to(root)),
            endpoints_path=str((output / "endpoints.json").relative_to(root)))
    save(output / "result.json", result)
    return result
