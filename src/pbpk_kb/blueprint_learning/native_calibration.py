"""Development-only PBPK calibration with optional learned starting proposals.

This separate package leaves historical ``blueprints`` modules and their file
set unchanged. The original evidence point is always evaluated first. At most
two learned proposals are projected into the declared bounds and evaluated at
the same native-call cost, followed by a bounded eight-direction pattern search.
No clinical comparison values, learned surrogate loss, or failed-run penalties
replace an actual native evaluation. All native attempts share one call budget.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
import time

from pbpk_kb.blueprints.evidence_preserving_calibration import (
    snapshot_for_mapping, _flags, _infrastructure_problem, InfrastructurePause,
)
from pbpk_kb.blueprints.development_curve_blueprint import _training_document, _run_development
from pbpk_kb.blueprints.development_curve_optimizer import _observations, _sample_curve
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, _digest
from pbpk_kb.blueprints.pk_anchored_blueprint import _drug_signature, _run
from pbpk_kb.blueprints.source_workflow import inside, save, sha256

VERSION = "learned-native-calibration-1"
CLEARANCE_POLICY = "only_nonrenal_when_net_renal_evidence"
MINIMUM_STEP = 1e-3
MAX_INFRASTRUCTURE_RETRIES = 1
CORE_NAMES = (
    "evidence_preserving_calibration.py", "evidence_blueprint.py", "development_curve_blueprint.py",
    "development_curve_optimizer.py", "pk_anchored_blueprint.py", "source_workflow.py",
    "design_runtime.py", "evidence_native_audit.py", "source_adaptation_audit.py",
    "pk_anchor_moments.py", "extended_prediction.py", "endpoint_prediction.py",
    "human_model.py", "human_model_snapshot.py", "evidence_evaluation.py",
)


class _BudgetExhausted(Exception):
    pass


def _read(path):
    return json.loads(path.read_text())


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return float(value)


def _ref(path, root):
    path = inside(path, root, exists=True)
    return {"path":str(path.relative_to(root)), "sha256":sha256(path)}


def _verify(pins, root):
    for name, expected in pins.items():
        if sha256(inside(name, root, exists=True)) != expected:
            raise ValueError("Frozen calibration resource changed: " + name)


def _embedded_pins(value, root):
    pins = {}
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = str(inside(value["path"], root, exists=True).relative_to(root))
            pins[path] = value["sha256"]
        for child in value.values():
            for path, digest in _embedded_pins(child, root).items():
                if path in pins and pins[path] != digest:
                    raise ValueError("Conflicting source checksums")
                pins[path] = digest
    elif isinstance(value, list):
        for child in value:
            for path, digest in _embedded_pins(child, root).items():
                if path in pins and pins[path] != digest:
                    raise ValueError("Conflicting source checksums")
                pins[path] = digest
    return pins


def _code_pins(root):
    # Explicit dependency names: adding unrelated learning code cannot invalidate
    # historical mappings, and adding a blueprints file is unnecessary.
    paths = [Path(__file__), *(root/"src/pbpk_kb/blueprints"/name for name in CORE_NAMES),
             root/"src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R"]
    return {str(inside(path,root,exists=True).relative_to(root)):sha256(path) for path in paths}


@contextmanager
def _exclusive(directory):
    with (directory/".calibration.lock").open("a+") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise InfrastructurePause("Another invocation owns this calibration directory") from exc
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)


def _initial_points(points, prior):
    if points is None: return []
    if not isinstance(points,(list,tuple)) or len(points)>2:
        raise ValueError("At most two learned proposals are allowed")
    normalized = []
    for point in points:
        if isinstance(point,dict):
            raw_lip,raw_scale = point.get("lipophilicity"),point.get("clearance_scale")
            provenance = {k:deepcopy(v) for k,v in point.items() if k not in {"lipophilicity","clearance_scale"}}
        elif isinstance(point,(list,tuple)) and len(point)==2:
            raw_lip,raw_scale = point; provenance = {}
        else: raise ValueError("Learned proposal needs lipophilicity and clearance_scale")
        raw_lip,raw_scale = _number(raw_lip,"learned lipophilicity"),_number(raw_scale,"learned scale",positive=True)
        lip,scale = min(prior+2.5,max(prior-2.5,raw_lip)),min(4.,max(.25,raw_scale))
        normalized.append({"lipophilicity":lip,"clearance_scale":scale,
            "unprojected":{"lipophilicity":raw_lip,"clearance_scale":raw_scale},
            "projected_to_bounds":lip!=raw_lip or scale!=raw_scale,"provenance":provenance})
    return normalized


def _point_key(input_sha, lip, scale):
    # Float hex preserves the exact evaluated parameter content, not rounded bins.
    return _digest({"input_sha256":input_sha,"lipophilicity_hex":float(lip).hex(),
                    "clearance_scale_hex":float(scale).hex(),"clearance_policy":CLEARANCE_POLICY})


def _score(result, curve, observed, lip, scale, prior, weight):
    import numpy as np
    times, values = _observations(observed)
    sampled = _sample_curve(curve,times)
    differences = np.log(sampled)-np.log(np.asarray(values))
    components = {"mean_squared_log_curve_error":float(np.mean(differences**2)),
        "clearance_input_prior":weight*(math.log(scale)/math.log(2))**2,
        "lipophilicity_prior":.01*((lip-prior)/2.5)**2}
    objective = math.fsum(components.values())
    if not math.isfinite(objective): raise ValueError("Nonfinite development objective")
    return {"status":"ok","eligible_for_selection":True,"objective":objective,
        "objective_components":components,"predicted_at_observed_times":sampled.tolist(),
        "sampled_log_ratios":differences.tolist(),
        "curve_metrics":{"gmfe":math.exp(float(np.mean(np.abs(differences)))),
                         "ln_rmse":float(np.sqrt(np.mean(differences**2))),"n":len(times)},
        "clearance_l_h":result.get("clearance_l_h"),"vss_l":result.get("vss_l")}


def _pattern(evaluate, prior, points):
    """Deterministic eight-direction search; evaluate owns cache and call budget."""
    records, cache, polls = [], {}, []
    cache_hits, converged, step = 0,False,.25
    reason = "Native call budget exhausted"

    def trial(lip,scale,purpose):
        nonlocal cache_hits
        key = (float(lip).hex(),float(scale).hex())
        if key in cache:
            cache_hits+=1;return cache[key]
        result = evaluate(float(lip),float(scale),purpose)
        records.append(result);cache[key]=result
        return result

    def best():
        return min((r for r in records if r.get("eligible_for_selection")),
                   key=lambda r:(r["objective"],r["evaluation"]),default=None)

    directions = ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))
    try:
        trial(prior,1.,"evidence_prior")
        for i,point in enumerate(points):
            trial(point["lipophilicity"],point["clearance_scale"],f"learned_proposal_{i+1}")
        while step >= MINIMUM_STEP:
            selected = best()
            anchor = (0.,0.) if selected is None else ((selected["lipophilicity"]-prior)/2.5,
                                                     math.log(selected["clearance_scale"])/math.log(4))
            before = None if selected is None else selected["objective"]
            count = len(records)
            for dx,dy in directions:
                x,y = max(-1.,min(1.,anchor[0]+dx*step)),max(-1.,min(1.,anchor[1]+dy*step))
                trial(prior+2.5*x,math.exp(math.log(4)*y),"pattern_poll")
            selected = best()
            improved = selected is not None and (before is None or selected["objective"] < before-1e-12)
            polls.append({"step":step,"new_parameter_points":len(records)-count,"improved":improved})
            if not improved: step*=.5
        converged = best() is not None
        reason = "Pattern step tolerance reached" if converged else "No feasible native parameter point"
    except _BudgetExhausted:
        pass
    selected = best()
    return {"status":"converged" if converged else "budget_exhausted" if selected else "no_feasible_point",
        "reason":reason,"converged":converged,"eligible":selected is not None,
        "decision":deepcopy(selected),"evaluations":records,"evaluation_count":len(records),
        "failed_evaluation_count":sum(r.get("status")!="ok" for r in records),"cache_hits":cache_hits,
        "optimizer":{"method":"bounded_eight_direction_pattern","minimum_step":MINIMUM_STEP,
                     "final_step":step,"polls":polls,"numerical_convergence_only":True}}


def _artifact_pins(execution, root):
    pins = {}
    for name in ("result.json","infrastructure_failure.json","prediction_locked.json","endpoints.json","parameter_audit_result.json",
                 "native/execution_manifest.json","native/snapshot.json"):
        path = execution/name
        if path.is_file(): pins[str(path.relative_to(root))]=sha256(path)
    receipt = execution/"native/execution_manifest.json"
    if receipt.is_file():
        for item in _read(receipt).get("curve_outputs",[]):
            path = inside(item["path"],receipt.parent,exists=True)
            if sha256(path)!=item["sha256"]: raise ValueError("Native curve CSV changed")
            pins[str(path.relative_to(root))]=item["sha256"]
    return pins


def _verify_locked_mapping(mapping, path, root):
    """The selected parameters must still be one of the scored native trials."""
    input_path=path.parent/"input.json"
    if mapping.get("version")!=VERSION or sha256(input_path)!=mapping.get("input_sha256"):
        raise ValueError("Locked mapping/input identity changed")
    fixed=_read(input_path)
    if (mapping.get("source_pins")!=fixed["source_pins"] or mapping.get("code_pins")!=fixed["code_pins"]
            or mapping.get("drug_signature")!=_drug_signature(fixed["request"])
            or mapping.get("construction_partition_method")!=fixed["request"]["partition_method"]):
        raise ValueError("Locked mapping resources or construction identity changed")
    for key in ("source_pins","code_pins","trial_artifact_pins"):
        _verify(mapping[key],root)
    for row in mapping["evaluations"]:
        compact=path.parent/"trials"/row["cache_key"]/"trial.json"
        if _read(compact)!=row or mapping["trial_artifact_pins"].get(str(compact.relative_to(root)))!=sha256(compact):
            raise ValueError("Locked evaluation differs from its immutable trial")
    best=min((r for r in mapping["evaluations"] if r.get("eligible_for_selection")),
        key=lambda r:(r["objective"],r["evaluation"]),default=None)
    if mapping.get("decision")!=best or mapping.get("eligible")!=(best is not None):
        raise ValueError("Locked decision differs from the best actual native trial")


def fit_mapping(*, reference_request, training_source, output_dir, project_root,
                max_evaluations=48, initial_points=None, before_trial=None, prior_weight=.1):
    root = Path(project_root).resolve(strict=True); output = inside(output_dir,root)
    if isinstance(max_evaluations,bool) or not isinstance(max_evaluations,Integral) or max_evaluations<1:
        raise ValueError("max_evaluations must be a positive integer")
    weight = _number(prior_weight,"prior_weight")
    if weight<0: raise ValueError("prior_weight must be nonnegative")
    if before_trial is not None and not callable(before_trial): raise ValueError("before_trial must be callable")
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode="json",exclude_none=True)
    observed = _training_document(training_source,req,root)
    if req["dosing"]["route"]!="intravenous" or any(observed.get(k)!=req[k] for k in ("human","dosing")):
        raise ValueError("Development IV administration/human must match the request")
    blueprint = assemble_evidence_blueprint(req,root)
    if blueprint["status"]!="assembled": raise ValueError("Incomplete construction evidence")
    prior = _number(next(e["value"] for e in req["evidence"] if e["parameter"]=="logp"),"evidence logP")
    if not prior-2.5 < prior < prior+2.5:
        raise ValueError("Evidence logP bounds must be representable")
    points = _initial_points(initial_points,prior)
    source_pins = _embedded_pins([req,training_source,observed,points],root)
    proposal_pins = _embedded_pins(points,root)
    forbidden = {p["sha256"] for p in req.get("forbidden_sources",[])}
    if forbidden.intersection(proposal_pins.values()): raise ValueError("Learning proposal directly references forbidden comparison data")
    code_pins = _code_pins(root);_verify(source_pins,root);_verify(code_pins,root)
    fixed = {"version":VERSION,"clearance_policy":CLEARANCE_POLICY,"request":req,
        "training_source":deepcopy(training_source),"maximum_evaluations":int(max_evaluations),
        "initial_points":points,"prior_weight":weight,"source_pins":source_pins,"code_pins":code_pins,
        "bounds":{"lipophilicity":[prior-2.5,prior+2.5],"clearance_scale":[.25,4.]},
        "infrastructure_retries_per_fit":MAX_INFRASTRUCTURE_RETRIES}
    output.mkdir(parents=True,exist_ok=True)
    with _exclusive(output):
        input_path = output/"input.json"
        if input_path.exists():
            if _read(input_path)!=fixed: raise ValueError("Resume input, source, code, initialization or budget changed")
        else:
            if any(p.name!=".calibration.lock" for p in output.iterdir()): raise ValueError("Unrecognized existing fit output")
            save(input_path,fixed)
        input_sha = sha256(input_path); mapping_path = output/"mapping_locked.json"
        if mapping_path.exists():
            mapping = _read(mapping_path)
            _verify_locked_mapping(mapping,mapping_path,root)
            return {**mapping,"mapping_path":str(mapping_path.relative_to(root)),"mapping":_ref(mapping_path,root),
                    "actual_new_calls":0,"reused_calls":len(mapping["evaluations"]),"completed_fit_reused":True}
        progress_path = output/"progress.json"
        progress = _read(progress_path) if progress_path.exists() else {"version":VERSION,"input_sha256":input_sha,"trial_pins":{}}
        if progress["input_sha256"]!=input_sha: raise ValueError("Progress belongs to another input")
        _verify(progress["trial_pins"],root)
        started = time.monotonic(); counters={"actual_new_calls":0,"reused_calls":0}
        trials = output/"trials";trials.mkdir(exist_ok=True)

        def calls(): return sum(1 for _ in trials.glob("*/execution_*"))

        def checkpoint(status="running",reason=None):
            progress.update(status=status,reason=reason,total_native_calls=calls(),**counters)
            save(progress_path,progress)

        def safety():
            try:
                if before_trial is not None: before_trial()
                _verify(source_pins,root);_verify(code_pins,root)
                if sha256(input_path)!=input_sha: raise ValueError("Frozen fit input changed")
            except Exception as exc:
                checkpoint("infrastructure_paused",str(exc))
                raise InfrastructurePause(str(exc)) from exc

        def evaluate(lip,scale,purpose):
            safety()
            key = _point_key(input_sha,lip,scale)
            folder = trials/key;folder.mkdir(exist_ok=True)
            compact = folder/"trial.json"
            if compact.exists():
                row = _read(compact)
                if row.get("cache_key")!=key or row.get("input_sha256")!=input_sha:
                    raise InfrastructurePause("Cached trial parameter/input identity changed")
                expected = progress["trial_pins"].get(str(compact.relative_to(root)))
                if expected is not None and sha256(compact)!=expected:
                    raise InfrastructurePause("Cached compact trial changed")
                # Also verifies orphan records written immediately before an
                # interrupted checkpoint. Unlocked fits retain all native files.
                _verify(row["artifact_pins"],root)
                progress["trial_pins"][str(compact.relative_to(root))]=sha256(compact)
                counters["reused_calls"]+=1;checkpoint();return row
            existing = list(folder.glob("execution_*"))
            retry_count = sum(max(0,len(list(p.glob("execution_*")))-1) for p in trials.iterdir() if p.is_dir())
            if existing and retry_count >= MAX_INFRASTRUCTURE_RETRIES:
                raise InfrastructurePause("Infrastructure retry allowance exhausted; diagnostic artifacts retained")
            if calls()>=max_evaluations:
                if existing:
                    raise InfrastructurePause("Interrupted native attempt exhausted its budget; cannot silently select an earlier fit")
                raise _BudgetExhausted
            execution = folder/f"execution_{len(existing)}";execution.mkdir()
            counters["actual_new_calls"]+=1
            number = len(progress["trial_pins"])+1
            row = {"evaluation":number,"cache_key":key,"input_sha256":input_sha,
                "lipophilicity":lip,"clearance_scale":scale,"purpose":purpose,
                "normalized_coordinates":[(lip-prior)/2.5,math.log(scale)/math.log(4)],
                "native_output_dir":str(execution.relative_to(root)),"eligible_for_selection":False}
            try:
                native = snapshot_for_mapping(blueprint,lip,scale)
                result = _run_development(native,req,execution,root,audit=False,require_vss=False)
                _flags(execution,training_source)
            except (OSError,TimeoutError,RuntimeError) as exc:
                result={"status":"infrastructure_failed","reason":type(exc).__name__+": "+str(exc)}
            except Exception as exc:
                save(execution/"infrastructure_failure.json",{"status":"infrastructure_paused",
                    "reason":type(exc).__name__+": "+str(exc)})
                checkpoint("infrastructure_paused",type(exc).__name__+": "+str(exc))
                raise InfrastructurePause("Unexpected native adapter failure: "+str(exc)) from exc
            save(execution/"result.json",result)
            if _infrastructure_problem(result):
                checkpoint("infrastructure_paused",str(result))
                raise InfrastructurePause("Native infrastructure failure; original diagnostic retained")
            row["callback_status"] = result.get("status")
            if result.get("status")=="predicted":
                prediction = execution/"prediction_locked.json"
                if not prediction.is_file() or sha256(prediction)!=result.get("prediction_sha256"):
                    raise InfrastructurePause("Native prediction/result SHA mismatch")
                try: row.update(_score(result,_read(prediction),observed,lip,scale,prior,weight))
                except (ValueError,OverflowError) as exc:
                    row.update(status="evaluation_failed",reason=str(exc))
                row["prediction"]=_ref(prediction,root)
                row["prediction_path"]=row["prediction"]["path"]
            else:
                row.update(status="evaluation_failed",reason=result.get("reason",result.get("status","Native execution failed")))
            row["artifact_pins"] = _artifact_pins(execution,root)
            _verify(row["artifact_pins"],root)
            save(compact,row);progress["trial_pins"][str(compact.relative_to(root))]=sha256(compact)
            checkpoint();return row

        try:
            result = _pattern(evaluate,prior,points)
            safety();_training_document(training_source,req,root)
            selected = result["decision"]
            retained = dict(progress["trial_pins"])
            for row in result["evaluations"]:
                if row["status"]!="ok" or selected and row["cache_key"]==selected["cache_key"]:
                    retained.update(row["artifact_pins"])
            completed={row["native_output_dir"] for row in result["evaluations"]}
            interrupted=[]
            for execution in sorted(trials.glob("*/execution_*")):
                if str(execution.relative_to(root)) not in completed:
                    pins=_artifact_pins(execution,root);retained.update(pins)
                    diagnostic=execution/"result.json"
                    if not diagnostic.is_file(): diagnostic=execution/"infrastructure_failure.json"
                    interrupted.append({"native_output_dir":str(execution.relative_to(root)),
                        "diagnostic":_read(diagnostic) if diagnostic.is_file() else {
                            "status":"interrupted","reason":"No complete native result was persisted"},
                        "artifact_pins":pins})
            result.update(version=VERSION,clearance_policy=CLEARANCE_POLICY,input_sha256=input_sha,
                drug_signature=_drug_signature(req),construction_partition_method=req["partition_method"],
                reference_human=req["human"],reference_dosing=req["dosing"],training_source=deepcopy(training_source),
                source_pins=source_pins,code_pins=code_pins,trial_artifact_pins=retained,
                initial_points=points,prior_weight=weight,max_evaluations=int(max_evaluations),
                total_native_calls=calls(),elapsed_seconds=time.monotonic()-started,**counters,
                infra_retry_calls=sum(max(0,len(list(p.glob("execution_*")))-1) for p in trials.iterdir() if p.is_dir()),
                infrastructure_attempts=interrupted,
                selected_prediction=selected.get("prediction") if selected else None,
                development_curve_used=True,comparison_curve_used=False,
                objective_spec="mean(log(P/O)^2)+prior_weight*(ln(scale)/ln(2))^2+0.01*((lip-logP)/2.5)^2",
                limitations=["Learned initialization is a proposed construction point, not measured evidence.",
                    "Numerical convergence is not clinical validity or parameter identifiability.",
                    "Fixed renal construction input does not establish observed systemic renal clearance."])
            save(mapping_path,result);checkpoint("locked")
            return {**result,"mapping_path":str(mapping_path.relative_to(root)),"mapping":_ref(mapping_path,root)}
        except InfrastructurePause as exc:
            checkpoint("infrastructure_paused",str(exc));raise
        except (OSError,ValueError,KeyError) as exc:
            checkpoint("infrastructure_paused",type(exc).__name__+": "+str(exc))
            raise InfrastructurePause("Calibration artifact/adapter validation failed: "+str(exc)) from exc


def predict_mapping(*,request,mapping_path,output_dir,project_root):
    """Execute a locked development mapping, or reuse an identical sealed result.

    A partial prior invocation is paused with its diagnostics intact. The caller
    may explicitly assign a fresh retry directory; this function never overwrites
    an interrupted prediction or reads comparison concentrations.
    """
    root=Path(project_root).resolve(strict=True);output=inside(output_dir,root);path=inside(mapping_path,root,exists=True)
    mapping=_read(path);mapping_sha=sha256(path)
    req=EvidenceBlueprintRequest.model_validate(request).model_dump(mode="json",exclude_none=True)
    if (mapping.get("version")!=VERSION or mapping.get("clearance_policy")!=CLEARANCE_POLICY
            or not mapping.get("eligible") or not mapping.get("decision")
            or mapping.get("drug_signature")!=_drug_signature(req)):
        raise ValueError("Selected learning mapping does not match the target drug construction")
    if req["dosing"]["route"]!="intravenous": raise ValueError("Only single-dose IV target administration is supported")
    _verify_locked_mapping(mapping,path,root)
    target_pins=_embedded_pins(req,root);_verify(target_pins,root)
    _training_document(mapping["training_source"],req,root)
    blueprint=assemble_evidence_blueprint(req,root)
    if blueprint["status"]!="assembled": raise ValueError("Incomplete target construction evidence")
    point=mapping["decision"];native=snapshot_for_mapping(blueprint,point["lipophilicity"],point["clearance_scale"])
    construction={"version":VERSION,"mapping":_ref(path,root),"clearance_policy":CLEARANCE_POLICY,
        "request_sha256":_digest(req),"snapshot_sha256":_digest(native),
        "decision":{k:point[k] for k in ("lipophilicity","clearance_scale","cache_key")},
        "target_source_pins":target_pins,"comparison_observations_used":False}
    output.mkdir(parents=True,exist_ok=True)
    with _exclusive(output):
        receipt=output/"completion.json"
        if receipt.exists():
            completion=_read(receipt)
            if (completion.get("version")!=VERSION or completion.get("construction")!=construction
                    or _read(output/"request.json")!=req or _read(output/"construction.json")!=construction):
                raise ValueError("Existing prediction request, mapping or construction differs")
            _verify(completion["artifact_pins"],root)
            result=_read(output/"result.json")
            if _infrastructure_problem(result): raise InfrastructurePause("Saved prediction is an infrastructure failure")
            if result.get("status")=="predicted" and sha256(output/"prediction_locked.json")!=result.get("prediction_sha256"):
                raise ValueError("Saved prediction/result checksum mismatch")
            return {**result,"actual_new_calls":0,"completed_prediction_reused":True}
        if any(p.name!=".calibration.lock" for p in output.iterdir()):
            raise InfrastructurePause("Incomplete prior target prediction retained; use an explicit fresh retry directory")
        save(output/"request.json",req);save(output/"construction.json",construction)
        started=time.monotonic()
        try:
            result=_run(native,req,output,root,audit=True);_flags(output,mapping["training_source"])
            if _infrastructure_problem(result):
                save(output/"result.json",result)
                raise InfrastructurePause("Target native infrastructure failure; output retained")
            _verify(mapping["source_pins"],root);_verify(mapping["code_pins"],root);_verify(target_pins,root)
            if sha256(path)!=mapping_sha: raise ValueError("Selected mapping changed during target execution")
            result.update(method_version=VERSION,clearance_policy=CLEARANCE_POLICY,
                development_curve_used=True,comparison_curve_used=False,mapping=_ref(path,root),
                actual_new_calls=1,elapsed_seconds=time.monotonic()-started)
            prediction=output/"prediction_locked.json"
            if result.get("status")=="predicted":
                if not prediction.is_file() or sha256(prediction)!=result.get("prediction_sha256"):
                    raise ValueError("Target prediction/result checksum mismatch")
                result.update(prediction=_ref(prediction,root),prediction_path=str(prediction.relative_to(root)))
            save(output/"result.json",result)
            artifacts=_artifact_pins(output,root)
            artifacts.update({str((output/name).relative_to(root)):sha256(output/name)
                              for name in ("request.json","construction.json")})
            save(receipt,{"version":VERSION,"construction":construction,"artifact_pins":artifacts})
            return result
        except InfrastructurePause: raise
        except Exception as exc:
            save(output/"infrastructure_failure.json",{"status":"infrastructure_paused",
                "reason":type(exc).__name__+": "+str(exc),"actual_native_call_attempted":True})
            raise InfrastructurePause("Target execution/artifact validation failed: "+str(exc)) from exc


def cleanup_unselected(*,output_dir,project_root):
    """Remove only unselected successful native folders owned by this new fit.

    All compact trial records, selected native artifacts and every failed trial
    remain. No historical fit, symlink, active/unlocked fit or parent is removed.
    """
    root=Path(project_root).resolve(strict=True);folder=inside(output_dir,root,exists=True)
    with _exclusive(folder):
        path=folder/"mapping_locked.json"
        if not path.is_file(): raise ValueError("Cleanup requires a locked learning fit")
        mapping=_read(path)
        if mapping.get("version")!=VERSION: raise ValueError("Cleanup does not own this historical fit")
        _verify_locked_mapping(mapping,path,root)
        selected=(mapping.get("decision") or {}).get("cache_key");removed=[]
        for row in mapping["evaluations"]:
            if row.get("status")!="ok" or row["cache_key"]==selected: continue
            path=root/row["native_output_dir"]
            expected=folder/"trials"/row["cache_key"]
            if (path.parent != expected or not path.name.startswith("execution_")
                    or not path.name.removeprefix("execution_").isdigit()
                    or any(p.is_symlink() for p in (path,expected,folder/"trials"))
                    or not path.resolve().is_relative_to(folder.resolve())):
                raise ValueError("Unsafe native cleanup path")
            if not path.exists(): continue
            if any(p.is_symlink() for p in path.rglob("*")): raise ValueError("Native cleanup rejects symlinks")
            _verify(row["artifact_pins"],root)
            if any((root/p).resolve().is_relative_to(path.resolve()) for p in mapping["trial_artifact_pins"]):
                raise ValueError("A retained artifact points into the cleanup candidate")
            shutil.rmtree(path);removed.append(str(path.relative_to(root)))
        return {"removed_native_directories":removed,"removed_count":len(removed),
                "compact_scores_preserved":len(mapping["evaluations"]),"failed_trials_preserved":True}
