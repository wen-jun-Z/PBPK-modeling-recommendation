"""Fixed joint CL/fu conditioning using training residuals, never clinical loss.

Three fu OOF fits are new. Shared Vss, frozen half-life and current complementary
CL OOF are reused. Parameter residual crossfit is developmental, not fully nested.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np

from . import clearance_complementary as complementary
from . import clearance_refinement as previous
from . import clinical_shared_prior as shared
from . import evidence_conditioned_prior as first
from . import rich_parameter_prior as rich

VERSION = "joint-evidence-conditioned-prior-1"
OUTPUT = "experiments/blueprint_design/evidence_conditioning_20261005/prior_joint"
SHRINKAGE = .5
RIDGE_PER_ROW = .01
PATTERNS = {"CL_only":("total_clearance",), "fu_only":("fraction_unbound",),
            "both":("total_clearance", "fraction_unbound")}
OUTPUTS = ("total_clearance", "fraction_unbound", "vss", "terminal_half_life")
CODE_FILES = tuple(dict.fromkeys((*first.CODE_FILES, *complementary.CODE_FILES,
    "src/pbpk_kb/blueprint_learning/joint_evidence_prior.py", "scripts/train_joint_evidence_prior.py")))


def residual_coefficients(errors, output_error):
    x, y = np.asarray(errors, dtype=float), np.asarray(output_error, dtype=float)
    if (x.ndim != 2 or not len(x) or x.shape[1] not in (1,2) or y.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("Aligned finite nonempty one/two-input residuals required")
    return np.linalg.solve(x.T @ x + len(x)*RIDGE_PER_ROW*np.eye(x.shape[1]), x.T @ y)


def crossfit_relation(errors, output_truth, output_center, folds, output):
    x = np.asarray(errors, dtype=float); y = np.asarray(output_truth, dtype=float)
    center = np.asarray(output_center, dtype=float); f = np.asarray(folds)
    residual_coefficients(x, y-center)
    if center.shape != y.shape or f.shape != y.shape or len(set(f.tolist())) != 3:
        raise ValueError("Three aligned training folds required")
    prediction = np.empty(len(y)); records = []
    for fold in sorted(set(f.tolist())):
        fit, held = f != fold, f == fold
        beta = residual_coefficients(x[fit], (y-center)[fit])
        prediction[held] = rich.physical_log_predictions(center[held] + SHRINKAGE*(x[held] @ beta), output)
        records.append({"fold":int(fold), "coefficients":beta.tolist(), "fit_count":int(fit.sum()),
            "held_count":int(held.sum()), "held_positions":np.flatnonzero(held).tolist()})
    return {"coefficients":residual_coefficients(x, y-center).tolist(),
        "folds":records, "original":first._metrics(y, center), "conditioned":first._metrics(y,prediction),
        "conditioned_oof_log10":prediction.tolist()}


def _read(root, relative):
    path = shared._inside(relative, root, exists=True); data = json.loads(path.read_text())
    if data.get("result_sha256") != shared._digest({k:v for k,v in data.items() if k != "result_sha256"}):
        raise ValueError("Frozen parameter artifact checksum changed")
    previous._verify(root, data["source_refs"]); previous._verify(root, data["code_pins"])
    for flag in ("target_labels_used", "target_observation_files_opened", "training_structure_overlap",
                 "training_generic_scaffold_overlap", "clinical_results_used_for_selection"):
        if data["audit"].get(flag) is not False: raise ValueError("Prior training isolation failed: "+flag)
    return data, path


def train(root, output_dir=OUTPUT, jobs=4, preference="auto", progress=None):
    root = Path(root).resolve(strict=True); output = shared._inside(output_dir, root)
    if isinstance(jobs,bool) or not isinstance(jobs,int) or not 1 <= jobs <= 4:
        raise ValueError("jobs must be an integer in [1,4]")
    output.mkdir(parents=True, exist_ok=True)
    with (output/"training.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(root, output, jobs, preference, progress)


def _train(root, output, jobs, preference, progress):
    started = time.monotonic(); old, _, wheel, refs = previous._read_source(root)
    a, ap = _read(root, first.OUTPUT+"/predictions.json")
    cl, cp = _read(root, complementary.OUTPUT+"/predictions.json")
    if a["original_target_priors"] != old["parameters"] or cl["selection"]["selected"] != complementary.MEAN:
        raise ValueError("Unexpected fixed input centers")
    for data,path in ((a,ap),(cl,cp)):
        for name,digest in data["source_refs"].items():
            if name in refs and refs[name] != digest: raise ValueError("Source hash conflict")
            refs[name] = digest
        refs[str(path.relative_to(root))] = shared._sha(path)
    code = {name:shared._sha(root/name) for name in CODE_FILES}; device = rich.choose_device(preference)
    config = {"version":VERSION, "source_refs":refs, "code_pins":code, "jobs":jobs, "device":device,
        "patterns":{k:list(v) for k,v in PATTERNS.items()}, "outputs":list(OUTPUTS),
        "fu_method":"rich_xgb_depth3", "fu_seed":shared.SEED,
        "ridge_per_row":RIDGE_PER_ROW, "shrinkage":SHRINKAGE,
        "CL_center":"current retained_animal_krr_equal_log_mean OOF and target CL",
        "other_centers":"Original shared Vss/fu/half; verified projection common rows for Vss",
        "selection":"Fixed joint residual ridge; no clinical weight tuning or hidden outcome rejection"}
    fingerprint = shared._digest(config); state_path = output/"training_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.pop("state_sha256",None) != shared._digest(state) or state["input_fingerprint"] != fingerprint:
            raise ValueError("Training checkpoint or frozen configuration changed")
    else: state = {"config":config,"input_fingerprint":fingerprint,"started_at":previous._now(),"fits":{}}
    result_path=output/"predictions.json"
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result.get("input_fingerprint") != fingerprint or result.get("result_sha256") != shared._digest({k:v for k,v in result.items()if k!="result_sha256"}):
            raise ValueError("Frozen joint-conditioning output changed")
        return result
    def emit(stage, **updates):
        state.update(stage=stage,pid=os.getpid(),updated_at=previous._now(),**updates)
        rich._atomic_json(state_path,{**state,"state_sha256":shared._digest(state)})
        if progress: progress({"stage":stage,**updates})
    emit("verifying_membership",planned_fits=3,completed_fits=len(state["fits"]))
    original=old["audit"]
    cohort=shared._cohort([{"compound_id":cid,"smiles":v["input_smiles"]}for cid,v in original["targets"].items()])
    rows,labels,membership=shared._training_data(wheel,cohort)
    if rows!=original["rows"] or membership["train_count"]!=1126: raise ValueError("Frozen 1126 structures changed")
    for parameter,values in labels.items():
        if sorted(values)!=original["parameter_training_indices"][parameter]:raise ValueError("Frozen label membership changed")
    trainrows=[r for r in rows if r["status"]=="train"];pos={r["index"]:i for i,r in enumerate(trainrows)}
    rowmap={r["index"]:r for r in trainrows};folds=original["folds"];fold_for={};seen=[]
    for f in folds:
        fit,held=f["train_indices"],f["validation_indices"]
        if (set(fit)&set(held) or (set(fit)|set(held))!=set(pos)
                or {rowmap[i]["scaffold_group"]for i in fit}&{rowmap[i]["scaffold_group"]for i in held}):raise ValueError("Invalid scaffold split")
        seen.extend(held);fold_for.update({i:f["fold"]for i in held})
    if len(seen)!=len(set(seen))or set(seen)!=set(pos):raise ValueError("OOF folds do not partition training structures")
    ca=cl["audit"];indices=ca["training_indices"]
    if indices!=sorted(labels["total_clearance"])or ca["observed_training_log10"]!=[labels["total_clearance"][i]for i in indices]or ca["oof_fold_ids"]!=[fold_for[i]for i in indices]:
        raise ValueError("Current complementary CL OOF truth/folds mismatch")
    centers={"total_clearance":dict(zip(indices,ca["candidate_oof_log10"][complementary.MEAN]))}
    vd=a["audit"]["residual_records"]["vss"];hd=a["audit"]["half_oof"]
    for record,parameter in ((vd,"vss"),(hd,"terminal_half_life")):
        truthkey="endpoint_observed_log10"if parameter=="vss"else"observed_log10"
        centerkey="endpoint_center_log10"if parameter=="vss"else"predicted_log10"
        if record[truthkey]!=[labels[parameter][i]for i in record["indices"]]or record["fold_ids"]!=[fold_for[i]for i in record["indices"]]:raise ValueError("Stored endpoint OOF truth/folds mismatch")
        centers[parameter]=dict(zip(record["indices"],record[centerkey]))
    emit("building_fu_features")
    x,features=rich.rich_features([r["identity_smiles"]for r in trainrows])
    if features["descriptor_names"]!=original["features"]["descriptor_names"]:raise ValueError("Frozen feature definition changed")
    fu={};parameter="fraction_unbound";method="rich_xgb_depth3"
    for f in folds:
        fit=[i for i in f["train_indices"]if i in labels[parameter]];held=[i for i in f["validation_indices"]if i in labels[parameter]]
        key=str(f["fold"]);metadata={"fold":f["fold"],"method":method,"train_indices":fit,"held_indices":held}
        if key in state["fits"]:
            record=state["fits"][key]
            if any(record[k]!=v for k,v in metadata.items()):raise ValueError("Fu checkpoint membership mismatch")
            pred=np.asarray(record["predicted_log10"],dtype=float)
        else:
            emit("fitting_fu_oof",current_fit=key,completed_fits=len(state["fits"]))
            begin=time.monotonic();model=shared._fit(method,x[[pos[i]for i in fit]],[labels[parameter][i]for i in fit],jobs,device)
            pred=rich.physical_log_predictions(model.predict(x[[pos[i]for i in held]]),parameter)
            state["fits"][key]={**metadata,"predicted_log10":pred.tolist(),"elapsed_seconds":time.monotonic()-begin}
            emit("fu_oof_fit_complete",completed_fits=len(state["fits"]))
        if pred.shape!=(len(held),)or not np.isfinite(pred).all()or np.any(pred>0):raise ValueError("Invalid fu OOF predictions")
        fu.update(zip(held,pred.tolist()))
    if set(fu)!=set(labels[parameter]):raise ValueError("Incomplete fu OOF")
    centers[parameter]=fu;coefficients={};stats={};records={}
    for pattern,inputs in PATTERNS.items():
        coefficients[pattern]={};stats[pattern]={};records[pattern]={}
        for output_parameter in OUTPUTS:
            if output_parameter in inputs:continue
            required=(*inputs,output_parameter)
            ids=sorted(set.intersection(*(set(centers[p])&set(labels[p])for p in required)))
            errors=np.array([[labels[p][i]-centers[p][i]for p in inputs]for i in ids])
            y=np.array([labels[output_parameter][i]for i in ids]);base=np.array([centers[output_parameter][i]for i in ids])
            report=crossfit_relation(errors,y,base,[fold_for[i]for i in ids],output_parameter)
            coefficients[pattern][output_parameter]={"inputs":list(inputs),"values":report["coefficients"],"n":len(ids)}
            stats[pattern][output_parameter]=report
            records[pattern][output_parameter]={"indices":ids,"fold_ids":[fold_for[i]for i in ids],
                "input_error_log10":errors.tolist(),"output_observed_log10":y.tolist(),"output_center_log10":base.tolist(),
                "omitted_training_indices":sorted(set(pos)-set(ids)),
                "omission_reason":"Required original/current verified OOF centers and endpoint labels must all exist; no imputation of training targets"}
    target=deepcopy(old["parameters"])
    if set(cl["predictions"])!=set(target):raise ValueError("Current CL target membership differs")
    for cid,value in cl["predictions"].items():target[cid]["total_clearance"]=value
    previous._verify(root,refs);previous._verify(root,code)
    result={"version":VERSION,"created_at":previous._now(),"input_fingerprint":fingerprint,
        "coefficients":coefficients,"shrinkage":SHRINKAGE,"ridge_per_row":RIDGE_PER_ROW,
        "current_target_priors":target,"original_target_priors":deepcopy(old["parameters"]),"prior_units":deepcopy(old["units"]),
        "half_log_uncertainty_q80":a["half_log_uncertainty_q80"],"half_uncertainty_log_base":"natural",
        "half_uncertainty_interpretation":"Inherited training uncertainty for audit only; joint conditioning does not use an uncertainty threshold",
        "source_refs":refs,"code_pins":code,"trainingstats":stats,
        "audit":{"target_labels_used":False,"target_observation_files_opened":False,"training_structure_overlap":False,
            "training_generic_scaffold_overlap":False,"clinical_results_used_for_selection":False,
            "training_structures":len(trainrows),"training_indices":sorted(pos),"target_count":len(target),
            "cl_training_labels":len(labels["total_clearance"]),"cl_oof_center":complementary.MEAN,
            "vss_oof_rows":len(centers["vss"]),"half_oof_rows":len(centers["terminal_half_life"]),"fu_oof_rows":len(fu),
            "new_fu_fits":len(state["fits"]),"new_half_fits":0,"new_CL_fits":0,"new_Vss_fits":0,
            "residual_records":records,"fu_oof":{"indices":sorted(fu),"fold_ids":[fold_for[i]for i in sorted(fu)],
                "observed_log10":[labels[parameter][i]for i in sorted(fu)],"predicted_log10":[fu[i]for i in sorted(fu)]},
            "limitations":["Residual crossfit inherits dependent base OOF from other folds; not fully nested independent validation",
                "Mixed-source training PK labels need not be matched studies; joint consistency is empirical rather than a physiological identity",
                "No clinical outcome threshold, weight search, or favorable-endpoint filtering"]},
        "elapsed_seconds":time.monotonic()-started}
    result["result_sha256"]=shared._digest(result);rich._atomic_json(result_path,result)
    emit("complete",result_path=str(result_path.relative_to(root)),result_sha256=shared._sha(result_path),elapsed_seconds=time.monotonic()-started)
    return result


def _factor(record, request):
    unit=str(record.get("unit","")).lower().replace(" ","");p=record["parameter"]
    if p=="total_clearance":return first.cl_ml_min_kg({"value":1.,"unit":unit},request["human"]["weight_kg"])
    if p=="fraction_unbound":
        if unit in {"fraction","dimensionless","1"}:return 1.
        if unit in {"%","percent"}:return .01
    if p=="vss":
        if unit=="l/kg":return 1.
        if unit=="l":return 1./first._positive(request["human"]["weight_kg"],"weight")
    if p=="terminal_half_life"and unit=="h":return 1.
    raise ValueError("Unsupported unit for "+p+": "+unit)


def _value(record,request):
    value=first._positive(record.get("value"),record["parameter"])*_factor(record,request)
    if record["parameter"]=="fraction_unbound"and value>1.:raise ValueError("fu must be in (0,1]")
    return value


def _qualify(record,request):
    c=record.get("context")or{};s=record.get("source")or{}
    if (record.get("compound_id")!=request["compound_id"]or c.get("species")!="human"or c.get("matrix")!="plasma"
            or c.get("independent_of_target")is False or c.get("study_role")in{"target","evaluation","test","validation","heldout"}):
        raise ValueError("Unqualified independent evidence context")
    if record["parameter"]=="total_clearance"and c.get("route")not in{"iv","intravenous"}:
        raise ValueError("Systemic CL needs an intravenous source")
    if not isinstance(s.get("path"),str)or not s["path"]or re.fullmatch(r"[0-9a-f]{64}",str(s.get("sha256","")))is None:
        raise ValueError("Bound evidence source required")
    if any(s.get("path")==f.get("path")or s.get("sha256")==f.get("sha256")for f in request.get("forbidden_sources",[])):
        raise ValueError("Target observation source cannot condition the prior")


def condition_inputs(request,sidecars,artifact,receipt):
    """Return cloned inputs and audit; measured/derived inputs are never edited."""
    req,sides=deepcopy(request),deepcopy(sidecars)
    audit={"version":VERSION,"applied":False,"pattern":"none","changed_parameters":[],"changes":[],
        "innovations_log10":{},"request_modified":False,"sidecars_modified":False,"clinical_observations_used":False}
    if artifact.get("version")!=VERSION or artifact.get("shrinkage")!=SHRINKAGE:
        raise ValueError("Unsupported frozen joint-conditioning artifact")
    selected={}
    for p in ("total_clearance","fraction_unbound"):
        rows=[r for r in req.get("evidence",[])if r.get("parameter")==p]
        if len(rows)!=1:raise ValueError("Exactly one construction "+p+" input required")
        selected[p]=rows[0]
    observed={p:r for p,r in selected.items()if r.get("basis")in{"measured","derived"}}
    if not observed:
        audit["reason"]="no_independent_CL_or_fu";return req,sides,audit
    if (selected["total_clearance"].get("basis")=="predicted"and
            any(r.get("parameter")=="renal_clearance"and r.get("basis")in{"measured","derived"}for r in req["evidence"])):
        raise ValueError("Unexpected predicted total CL with protected independent renal split")
    if (not isinstance(receipt,dict)or not isinstance(receipt.get("path"),str)or not receipt["path"]
            or re.fullmatch(r"[0-9a-f]{64}",str(receipt.get("sha256","")))is None):raise ValueError("Bound joint-prior receipt required")
    for record in observed.values():_qualify(record,req)
    pattern="both"if len(observed)==2 else "CL_only"if "total_clearance"in observed else "fu_only"
    cid=req["compound_id"];center=artifact["current_target_priors"][cid]
    innovation={p:math.log10(_value(r,req))-math.log10(first._positive(center[p],"frozen center"))for p,r in observed.items()}
    audit.update(pattern=pattern,innovations_log10=innovation,
        evidence_sources={p:deepcopy(r["source"])for p,r in observed.items()},
        independence_basis="Caller-verified frozen evidence policy plus explicit target/source exclusions")
    for p,coefficient in artifact["coefficients"][pattern].items():
        if p in observed:raise ValueError("Conditioning cannot edit an observed parameter")
        if tuple(coefficient["inputs"])!=PATTERNS[pattern]:raise ValueError("Coefficient input order changed")
        beta=np.asarray(coefficient["values"],dtype=float)
        if beta.shape!=(len(PATTERNS[pattern]),)or not np.isfinite(beta).all():raise ValueError("Invalid joint coefficients")
        shift=SHRINKAGE*float(beta@np.array([innovation[k]for k in coefficient["inputs"]]))
        if p in selected:records=[selected[p]];place="request"
        else:
            key={"vss":"vss","terminal_half_life":"half"}[p];records=sides.get(key,[]);place="sidecars"
            if not isinstance(records,list):raise ValueError("Expected sidecar record list")
        if any(r.get("basis")in{"measured","derived"}for r in records):continue
        for rec in records:
            if rec.get("basis")!="predicted":continue
            if rec.get("parameter")!=p or rec.get("compound_id")!=cid:raise ValueError("Prediction identity mismatch")
            current=_value(rec,req)
            if not math.isclose(current,first._positive(center[p],"current frozen prior"),rel_tol=1e-10,abs_tol=1e-12):
                raise ValueError("Predicted input differs from current frozen center")
            if shift==0.:continue
            log=math.log10(current)+shift
            if p=="fraction_unbound":log=min(log,0.)
            updated=first._positive(10.**log,p)/_factor(rec,req)
            if updated==rec["value"]:continue
            before=deepcopy(rec)
            rec.update(value=updated,source={**receipt,"location":f"{cid}/{pattern}/{p}: fixed training joint residual conditioning"},
                prediction_method={**rec.get("prediction_method",{}),"name":"joint_evidence_conditioned_prior","version":VERSION},
                notes="Predicted missing input conditioned on independent CL/fu; all measured inputs remain unchanged")
            audit["changed_parameters"].append(p);audit["changes"].append({"parameter":p,"location":place,
                "before":before["value"],"after":updated,"unit":rec["unit"],"shift_log10":shift,
                "original_source":before.get("source"),"conditioning_source":deepcopy(receipt)})
            audit[place+"_modified"]=True
    audit["applied"]=bool(audit["changes"]);audit["reason"]="predicted_missing_inputs_conditioned"if audit["applied"]else"no_predicted_prior_change"
    return req,sides,audit
