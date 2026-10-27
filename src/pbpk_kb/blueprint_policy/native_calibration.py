"""Evidence-bound native PBPK calibration with optional local parameter screening.

The shared native executor and renal protection are unchanged. All diagnostics,
failed scientific trials and pattern-search trials share one native-call budget.
Screening is a local engineering diagnostic, not proof of identifiability.
"""
from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral
from pathlib import Path
import shutil
import time

from pbpk_kb.blueprint_learning.native_calibration import (
    _read, _number, _ref, _verify, _embedded_pins, _exclusive, _point_key,
    _score, _artifact_pins, _BudgetExhausted, CORE_NAMES, MINIMUM_STEP,
    MAX_INFRASTRUCTURE_RETRIES, CLEARANCE_POLICY,
)
from pbpk_kb.blueprints.evidence_preserving_calibration import (
    snapshot_for_mapping, _flags, _infrastructure_problem, InfrastructurePause,
)
from pbpk_kb.blueprints.development_curve_blueprint import _training_document, _run_development
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest, assemble_evidence_blueprint, _digest
from pbpk_kb.blueprints.pk_anchored_blueprint import _drug_signature, _run
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.design_diagnostics import analyze_identifiability

VERSION = "evidence-policy-native-calibration-1"
PARAMETERS = ("lipophilicity", "clearance_scale")
SCREEN_STEP = .05
SCREEN_RELATIVE_THRESHOLD = .05
SCREEN_MINIMUM_RMS_DERIVATIVE = .02


def _code_pins(root):
    paths = [Path(__file__), root/"src/pbpk_kb/blueprint_learning/native_calibration.py",
        root/"src/pbpk_kb/blueprints/design_diagnostics.py",
        *(root/"src/pbpk_kb/blueprints"/name for name in CORE_NAMES),
        root/"src/pbpk_kb/model_runtime/R/blueprint_snapshot_run.R"]
    return {str(inside(path,root,exists=True).relative_to(root)):sha256(path) for path in paths}


def _allowed(value):
    if isinstance(value,(str,bytes)):
        raise ValueError("allowed_parameters must be a sequence, not a string")
    try: names=list(value)
    except TypeError as exc: raise ValueError("allowed_parameters must be a sequence") from exc
    if any(name not in PARAMETERS for name in names) or len(names)!=len(set(names)):
        raise ValueError("Unknown or duplicate allowed parameter")
    return tuple(name for name in PARAMETERS if name in names)


def _screening_policy():
    return {"normalized_step":SCREEN_STEP,"relative_rank_threshold":SCREEN_RELATIVE_THRESHOLD,
        "minimum_rms_log_concentration_derivative":SCREEN_MINIMUM_RMS_DERIVATIVE,
        "observation_transform":"ln concentration at supplied development times",
        "parameter_scaling":{"lipophilicity":"(lipophilicity - evidence_logP) / 2.5",
                             "clearance_scale":"ln(clearance_scale) / ln(4)"},
        "minimum_response_interpretation":"Engineering threshold: RMS derivative 0.02 means approximately 0.001 lnC response to a normalized 0.05 perturbation; not clinical precision",
        "failed_probe_policy":"Retain original allowed axes and mark screen_failed",
        "selection_policy":"Diagnostic probes never become fitted candidates",
        "scope":"Local sensitivity at evidence origin; not global or practical identifiability"}


def _screen(trial, prior, allowed):
    """Use real development predictions, never their clinical target outcomes."""
    import numpy as np
    diagnostic={"status":"screening","requested_parameters":list(allowed),
        "selected_parameters":list(allowed),"probe_evaluations":[],"failed_probes":[],
        "policy":_screening_policy(),"fallback_used":False,
        "observation_source_verified":True,"source":"supplied development observations only"}
    if not allowed:
        diagnostic.update(status="no_allowed_parameters",selected_parameters=[])
        return (),diagnostic
    columns=[]
    try:
        for axis in allowed:
            pair=[]
            for sign in (-1,1):
                lip=prior+sign*2.5*SCREEN_STEP if axis=="lipophilicity" else prior
                scale=math.exp(sign*math.log(4)*SCREEN_STEP) if axis=="clearance_scale" else 1.
                row=trial(lip,scale,f"sensitivity_{axis}_{'minus' if sign<0 else 'plus'}")
                diagnostic["probe_evaluations"].append(row["evaluation"])
                if row.get("status")!="ok":
                    diagnostic["failed_probes"].append({"parameter":axis,"direction":sign,
                        "evaluation":row["evaluation"],"reason":row.get("reason",row.get("status"))})
                pair.append(row)
            if all(row.get("status")=="ok" for row in pair):
                minus,plus=(np.asarray(row["predicted_at_observed_times"],dtype=float) for row in pair)
                if minus.shape!=plus.shape or not np.isfinite(minus).all() or not np.isfinite(plus).all() or np.any(minus<=0) or np.any(plus<=0):
                    raise ValueError("Nonpositive, nonfinite or mismatched diagnostic development curve")
                columns.append((np.log(plus)-np.log(minus))/(2*SCREEN_STEP))
    except _BudgetExhausted:
        diagnostic.update(status="screen_failed",reason="Native budget exhausted during diagnostics",fallback_used=True)
        return allowed,diagnostic
    except (ValueError,ArithmeticError) as exc:
        diagnostic.update(status="screen_failed",reason=str(exc),fallback_used=True)
        return allowed,diagnostic
    if diagnostic["failed_probes"]:
        diagnostic.update(status="screen_failed",reason="At least one scientific probe failed; failure is not low sensitivity",fallback_used=True)
        return allowed,diagnostic
    matrix=np.column_stack(columns)
    rms=np.sqrt(np.mean(matrix**2,axis=0))
    responsive=[i for i,value in enumerate(rms) if value>=SCREEN_MINIMUM_RMS_DERIVATIVE]
    diagnostic.update(jacobian=matrix.tolist(),rms_derivative={name:float(rms[i]) for i,name in enumerate(allowed)},
        weak_response_parameters=[name for i,name in enumerate(allowed) if i not in responsive])
    if responsive:
        try:
            analysis=analyze_identifiability([allowed[i] for i in responsive],matrix[:,responsive],
                relative_threshold=SCREEN_RELATIVE_THRESHOLD,max_free_parameters=len(responsive))
        except ValueError as exc:
            diagnostic.update(status="screen_failed",reason=str(exc),fallback_used=True)
            return allowed,diagnostic
        analysis["observation_source_verified"]=True
        selected=tuple(name for name in allowed if name in analysis["selected_parameter_ids"])
        diagnostic["rank_analysis"]=analysis
    else:
        selected=()
        diagnostic["rank_analysis"]={"rank":0,"selected_parameter_ids":[],
            "reason":"No axis passes the absolute RMS response threshold"}
    diagnostic.update(status="screened",selected_parameters=list(selected),
        frozen_parameters=[name for name in allowed if name not in selected])
    return selected,diagnostic


def _pattern(evaluate, prior, allowed, screen_parameters):
    """Search only permitted axes; diagnostics and scientific failures cost calls."""
    records,cache,polls=[],{},[]
    cache_hits,converged,step=0,False,.25
    active=tuple(allowed)
    screening={"status":"not_requested","selected_parameters":list(active),
               "probe_evaluations":[],"fallback_used":False,"policy":_screening_policy()}
    reason="Native call budget exhausted"
    termination="budget_exhausted"

    def trial(lip,scale,purpose):
        nonlocal cache_hits
        key=(float(lip).hex(),float(scale).hex())
        if key in cache:
            cache_hits+=1
            return cache[key]
        row=evaluate(float(lip),float(scale),purpose)
        records.append(row);cache[key]=row
        return row

    def best():
        return min((row for row in records if row.get("eligible_for_selection")),
                   key=lambda row:(row["objective"],row["evaluation"]),default=None)

    try:
        trial(prior,1.,"evidence_prior")
        if screen_parameters:
            active,screening=_screen(trial,prior,active)
        directions=([(1,0),(-1,0)] if "lipophilicity" in active else [])
        directions+=([(0,1),(0,-1)] if "clearance_scale" in active else [])
        if len(active)==2:
            directions += [(1,1),(1,-1),(-1,1),(-1,-1)]
        while directions and step>=MINIMUM_STEP:
            selected=best()
            anchor=(0.,0.) if selected is None else ((selected["lipophilicity"]-prior)/2.5,
                math.log(selected["clearance_scale"])/math.log(4))
            before=None if selected is None else selected["objective"]
            count=len(records)
            failed_probes=0
            for dx,dy in directions:
                x=max(-1.,min(1.,anchor[0]+dx*step)) if "lipophilicity" in active else 0.
                y=max(-1.,min(1.,anchor[1]+dy*step)) if "clearance_scale" in active else 0.
                row=trial(prior+2.5*x,math.exp(math.log(4)*y),"pattern_poll")
                failed_probes+=row.get("status")!="ok"
            selected=best()
            improved=selected is not None and (before is None or selected["objective"]<before-1e-12)
            polls.append({"step":step,"new_parameter_points":len(records)-count,"improved":improved,
                          "failed_probes":failed_probes})
            if not improved:step*=.5
        converged=bool(active) and best() is not None and bool(polls) and polls[-1]["failed_probes"]==0
        termination="converged" if converged else "fixed_evidence_only" if not active else "no_feasible_improvement"
        reason="Pattern step tolerance reached" if converged else "No permitted parameters; evidence point retained" if not active else "Last poll includes native failures; convergence not established"
    except _BudgetExhausted:
        pass
    selected=best()
    status=termination if selected else "no_feasible_point"
    return {"status":status,"reason":reason,"converged":converged,"eligible":selected is not None,
        "decision":deepcopy(selected),"evaluations":records,"evaluation_count":len(records),
        "failed_evaluation_count":sum(row.get("status")!="ok" for row in records),"cache_hits":cache_hits,
        "active_parameters":list(active),"screening":screening,
        "optimizer":{"method":"bounded_evidence_policy_pattern","minimum_step":MINIMUM_STEP,
            "final_step":step,"polls":polls,"numerical_convergence_only":True,
            "active_parameters":list(active),"frozen_at_evidence_parameters":[name for name in PARAMETERS if name not in active]}}

def _verify_locked_mapping(mapping, path, root):
    """The selected parameters must still be one of the scored native trials."""
    input_path=path.parent/"input.json"
    if mapping.get("version")!=VERSION or sha256(input_path)!=mapping.get("input_sha256"):
        raise ValueError("Locked mapping/input identity changed")
    fixed=_read(input_path)
    if any(mapping.get(key)!=fixed.get(key) for key in ("allowed_parameters","screen_parameters","screening_policy")):
        raise ValueError("Locked parameter policy changed")
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
                max_evaluations=24, allowed_parameters=PARAMETERS, screen_parameters=False,
                before_trial=None, prior_weight=.1):
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
    allowed = _allowed(allowed_parameters)
    if not isinstance(screen_parameters,bool): raise ValueError("screen_parameters must be boolean")
    source_pins = _embedded_pins([req,training_source,observed],root)
    code_pins = _code_pins(root);_verify(source_pins,root);_verify(code_pins,root)
    fixed = {"version":VERSION,"clearance_policy":CLEARANCE_POLICY,"request":req,
        "training_source":deepcopy(training_source),"maximum_evaluations":int(max_evaluations),
        "allowed_parameters":list(allowed),"screen_parameters":screen_parameters,
        "screening_policy":_screening_policy(),"prior_weight":weight,"source_pins":source_pins,"code_pins":code_pins,
        "bounds":{"lipophilicity":[prior-2.5,prior+2.5],"clearance_scale":[.25,4.]},
        "infrastructure_retries_per_fit":MAX_INFRASTRUCTURE_RETRIES}
    output.mkdir(parents=True,exist_ok=True)
    with _exclusive(output):
        input_path = output/"input.json"
        if input_path.exists():
            if _read(input_path)!=fixed: raise ValueError("Resume input, source, code, parameter policy or budget changed")
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
            if purpose.startswith("sensitivity_"):
                row["eligible_for_selection"] = False
                row["selection_exclusion"] = "Local diagnostic probe; not a calibrated candidate"
            row["artifact_pins"] = _artifact_pins(execution,root)
            _verify(row["artifact_pins"],root)
            save(compact,row);progress["trial_pins"][str(compact.relative_to(root))]=sha256(compact)
            checkpoint();return row

        try:
            result = _pattern(evaluate,prior,allowed,screen_parameters)
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
                allowed_parameters=list(allowed),screen_parameters=screen_parameters,
                screening_policy=_screening_policy(),prior_weight=weight,max_evaluations=int(max_evaluations),
                total_native_calls=calls(),elapsed_seconds=time.monotonic()-started,**counters,
                infra_retry_calls=sum(max(0,len(list(p.glob("execution_*")))-1) for p in trials.iterdir() if p.is_dir()),
                infrastructure_attempts=interrupted,
                selected_prediction=selected.get("prediction") if selected else None,
                development_curve_used=True,comparison_curve_used=False,
                objective_spec="mean(log(P/O)^2)+prior_weight*(ln(scale)/ln(2))^2+0.01*((lip-logP)/2.5)^2",
                limitations=["No learned proposals are used; inactive calibration axes remain at their evidence point.",
                    "Screening is local and threshold-dependent; it does not establish practical identifiability.",
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
        raise ValueError("Selected policy mapping does not match the target drug construction")
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
        if not path.is_file(): raise ValueError("Cleanup requires a locked policy fit")
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
