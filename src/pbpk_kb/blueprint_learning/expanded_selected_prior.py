"""Extend the selected fixed recipe, excluding every evaluation structure first.

The cohort contains identities only. No clinical files or target scalar labels
are accepted. Refit means new weights for the same algorithm, not optimization.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import time

import joblib
import numpy as np

from . import clearance_animal as animal
from . import clearance_complementary as complementary
from . import clearance_refinement as previous
from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich
from . import tanimoto_prior as kernel

VERSION = "expanded-selected-parameter-prior-1"
METHOD_NAME = "expanded_fixed_selected_parameter_prior"
BATCH = "experiments/blueprint_design/expanded_validation_20261007"
TARGETS = BATCH + "/targets.json"
OUTPUT = BATCH + "/prior"
SELECTED = "results/data/blueprint_selected.json"
CODE_FILES = tuple(dict.fromkeys((*complementary.CODE_FILES,
    "src/pbpk_kb/blueprint_learning/expanded_selected_prior.py",
    "scripts/train_expanded_selected_prior.py")))


def target_cohort(document):
    if (not isinstance(document, dict) or "targets" not in document
            or set(document) - {"targets","source_refs","version","rule"}
            or not isinstance(document["targets"], list)
            or any(not isinstance(row, dict) or not {"compound_id","smiles"} <= set(row)
                   or set(row) - {"compound_id","smiles","drug","role"}
                   for row in document["targets"])):
        raise ValueError("Targets accept identities and provenance metadata only, never numeric outcome labels")
    if (not isinstance(document.get("source_refs",{}),dict)
            or any(not isinstance(row.get(k,""),str) for row in document["targets"] for k in ("drug","role"))):
        raise ValueError("Invalid target provenance metadata")
    return shared._cohort([{k:r[k] for k in ("compound_id","smiles")} for r in document["targets"]])


def _exclusion(row, protected):
    if row["identity_smiles"] in {r["identity_smiles"] for r in protected.values()}:
        return "excluded_target_identity"
    if row["scaffold_group"] in {r["scaffold_group"] for r in protected.values()}:
        return "excluded_target_generic_scaffold"
    return None


def read_animal_rows(payload, cohort, *, interpret_labels=False):
    """Normalize and exclude before interpreting any numeric animal label.

    Human columns are neither selected nor converted. In overlap-only mode no
    numeric label is interpreted, including labels of otherwise eligible rows.
    """
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if not set(animal.ALLOWED_COLUMNS) <= set(reader.fieldnames or []):
        raise ValueError("Frozen animal source lacks required whitelisted columns")
    records, seen = [], set()
    for index, raw in enumerate(reader):
        row = {"index": index, **shared.normalize_structure(raw["smiles_r"])}
        if row["identity_smiles"] in seen:
            raise ValueError("Duplicate normalized animal identity")
        seen.add(row["identity_smiles"])
        row["status"] = _exclusion(row, cohort) or "train"
        if row["status"] == "train" and interpret_labels:
            labels, errors = {}, {}
            for species, column in animal.COLUMNS.items():
                value = raw[column].strip()
                if value.lower() in {"", "nan", "na", "null", "none"}:
                    errors[species] = "missing"; continue
                try:
                    number = float(value)
                except ValueError:
                    errors[species] = "not_numeric"; continue
                if not math.isfinite(number) or number <= 0:
                    errors[species] = "nonpositive_or_nonfinite"; continue
                labels[species] = math.log10(number)
            row.update(log10_cl=labels, label_exclusions=errors)
        records.append(row)
    return records


def overlap_audit(cohort, human_rows, animal_rows):
    audit = {}
    for cid, target in cohort.items():
        groups = {}
        for label, rows in (("human", human_rows), ("animal", animal_rows)):
            eligible = [r for r in rows if r["status"] == "train"]
            groups[label] = {
                "identity_indices": [r["index"] for r in eligible if r["identity_smiles"] == target["identity_smiles"]],
                "generic_scaffold_indices": [r["index"] for r in eligible if r["scaffold_group"] == target["scaffold_group"]]}
        audit[cid] = groups
    return audit


def requires_refit(overlaps):
    return any(indices for groups in overlaps.values() for values in groups.values() for indices in values.values())


def _receipt(root, path):
    path = shared._inside(path, root, exists=True)
    return {"path": str(path.relative_to(root)), "sha256": shared._sha(path)}


def _verify_receipt(root, receipt):
    previous._verify(root, {receipt["path"]: receipt["sha256"]})
    return shared._inside(receipt["path"], root, exists=True)


def _checked_json(root, receipt, *, result_digest=False):
    data = json.loads(_verify_receipt(root, receipt).read_text())
    if result_digest and data.get("result_sha256") != shared._digest({k:v for k,v in data.items() if k != "result_sha256"}):
        raise ValueError("Frozen source JSON checksum changed")
    return data


def _sources(root, target_path):
    selected_receipt = _receipt(root, SELECTED)
    selected = _checked_json(root, selected_receipt)
    if (selected["clearance_prior_version"] != complementary.VERSION
            or selected["core"]["design_version"] != "learned-exchange-design-2"):
        raise ValueError("Selected research algorithm changed")
    previous._verify(root, selected["core"]["code_pins"])
    old = _checked_json(root, selected["core"]["parameter_prior"], result_digest=True)
    cl = _checked_json(root, selected["clearance_prior"], result_digest=True)
    if cl["selection"]["selected"] != complementary.MEAN:
        raise ValueError("Expected the selected fixed three-component mean")
    for obj in (old, cl):
        for flag in ("target_labels_used", "target_observation_files_opened",
                     "training_structure_overlap", "training_generic_scaffold_overlap"):
            if obj["audit"].get(flag) is not False:
                raise ValueError("Selected source isolation audit failed: " + flag)
    previous._verify(root, cl["source_refs"]); previous._verify(root, cl["code_pins"])
    previous._verify(root, old["audit"]["frozen_input"]["code_sha256"])
    shared_receipt = {"path":str(shared._inside(old["model_path"],root,exists=True).relative_to(root)),
                      "sha256":old["model_sha256"]}
    _verify_receipt(root, shared_receipt); _verify_receipt(root, selected["clearance_model"])
    if selected["clearance_model"] != cl["model_receipt"]:
        raise ValueError("Selected complementary model receipt mismatch")
    animal_paths = [name for name,digest in cl["source_refs"].items()
                    if digest == animal.ANIMAL_SHA256 and name.endswith("animal_source.csv")]
    if len(animal_paths) != 1:
        raise ValueError("Expected one pinned local animal source")
    wheel = {"path":old["audit"]["frozen_input"]["source_path"],
             "sha256":old["audit"]["frozen_input"]["source_sha256"]}
    _verify_receipt(root, wheel)
    refs = dict(cl["source_refs"])
    for ref in (selected_receipt, selected["core"]["parameter_prior"], selected["clearance_prior"],
                shared_receipt, selected["clearance_model"], wheel, _receipt(root,target_path)):
        if ref["path"] in refs and refs[ref["path"]] != ref["sha256"]:
            raise ValueError("Conflicting source hashes")
        refs[ref["path"]] = ref["sha256"]
    return selected_receipt, selected, old, refs, wheel, animal_paths[0], shared_receipt


def _prepare(root, targets_path, output, jobs, preference):
    if isinstance(jobs,bool) or not isinstance(jobs,int) or not 1 <= jobs <= 8:
        raise ValueError("jobs must be an integer in [1,8]")
    target_path = shared._inside(targets_path,root,exists=True)
    document=json.loads(target_path.read_text());cohort = target_cohort(document)
    receipt, selected, old, refs, wheel, animal_path, original_shared = _sources(root,target_path)
    previous._verify(root,document.get("source_refs",{}))
    for name,digest in document.get("source_refs",{}).items():
        if name in refs and refs[name] != digest:raise ValueError("Conflicting cohort provenance hash")
        refs[name]=digest
    old_cohort = old["audit"]["targets"]
    if not set(old_cohort) <= set(cohort):
        raise ValueError("Expansion must retain the complete original twenty-drug cohort")
    if any(cohort[c]["identity_smiles"] != old_cohort[c]["identity_smiles"] for c in old_cohort):
        raise ValueError("Original evaluation compound identity changed")
    payload = shared._inside(animal_path,root,exists=True).read_bytes()
    original_animal = read_animal_rows(payload,old_cohort)
    overlaps = overlap_audit(cohort,old["audit"]["rows"],original_animal)
    mode = "refit_union_exclusion" if requires_refit(overlaps) else "reuse_frozen_weights"
    code = {name:shared._sha(root/name) for name in CODE_FILES}
    config = {"version":VERSION,"mode":mode,"targets":cohort,"source_refs":refs,"code_pins":code,
        "selected_receipt":receipt,"n_jobs":jobs,"device":rich.choose_device(preference),
        "shared_methods":{p:list(m) for p,m in shared.METHODS.items()},
        "shared_projection":"Unchanged training-only three-scaffold-fold projection rule",
        "complementary_formula":"equal mean of shared-projected, animal-augmented and Tanimoto-KRR log10 CL",
        "krr_alpha":.01,"animal_recipe":animal.RF_RECIPE,"seed":shared.SEED,
        "new_candidate_selection":False,"clinical_selection":False,
        "planned_fits":63 if mode == "refit_union_exclusion" else 0,
        "original_training_overlap":overlaps}
    fingerprint = shared._digest(config)
    path = output/"training_state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state.pop("state_sha256",None) != shared._digest(state) or state.get("input_fingerprint") != fingerprint:
            raise ValueError("Frozen expansion configuration or checkpoint changed")
    else:
        state = {"stage":"prepared","started_at":previous._now(),"updated_at":previous._now(),
                 "input_fingerprint":fingerprint,"config":config,"completed_fits":0}
        _state(path,state)
    return state,old,wheel,animal_path,original_shared,selected["clearance_model"]


def _state(path,state):
    rich._atomic_json(path,{**state,"state_sha256":shared._digest(state)})


def prepare(root,targets_path=TARGETS,output_dir=OUTPUT,jobs=8,preference="auto"):
    root=Path(root).resolve(strict=True);output=shared._inside(output_dir,root)
    output.mkdir(parents=True,exist_ok=True)
    with (output/"training.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _prepare(root,targets_path,output,jobs,preference)[0]


def predict_parameters(shared_bundle, complementary_bundle, cohort):
    """Pure model inference; no file/clinical access, no second CL projection."""
    ids=list(cohort);identities=[cohort[c]["identity_smiles"] for c in ids]
    x,features=rich.rich_features(identities)
    if (features["descriptor_names"] != shared_bundle["features"]["descriptor_names"]
            or features["descriptor_names"] != complementary_bundle["feature_audit"]["descriptor_names"]):
        raise ValueError("Frozen feature definition changed")
    logs={p:np.mean([rich.physical_log_predictions(shared_bundle["models"][p][m].predict(x),p)
                     for m in shared.METHODS[p]],axis=0) for p in shared.PARAMETERS}
    projection=shared_bundle["consistency_projection"]
    if projection["enabled"]:
        corrected=rich.project_consistency(np.column_stack([logs[p] for p in rich.CONSISTENCY_PARAMETERS]),
                                            projection["inference_residual_variances"])
        logs.update({p:corrected[:,j] for j,p in enumerate(rich.CONSISTENCY_PARAMETERS)})
    shared_parameters={cid:{p:float(10.**logs[p][i]) for p in shared.PARAMETERS} for i,cid in enumerate(ids)}
    if complementary_bundle["auxiliary_order"] != list(animal.SPECIES):
        raise ValueError("Frozen animal-feature order changed")
    aux=[complementary_bundle["animal_models"][s].predict(x) for s in animal.SPECIES]
    augmented=np.column_stack([x,*aux])
    animal_log=np.mean([complementary_bundle["human_models"][m].predict(augmented)
                        for m in shared.METHODS["total_clearance"]],axis=0)
    kernel_log=complementary_bundle["kernel_model"].predict(kernel.morgan_binary_features(identities))
    parts={"shared_projected":logs["total_clearance"],"animal_augmented":animal_log,"tanimoto_krr_001":kernel_log}
    selected=complementary.complementary_predictions(*parts.values())[complementary.MEAN]
    parameters=deepcopy(shared_parameters)
    for cid,value in zip(ids,10.**selected):parameters[cid]["total_clearance"]=float(value)
    if any(not math.isfinite(v) or v <= 0 for row in parameters.values() for v in row.values()):
        raise ValueError("Predicted parameters must be finite and positive")
    return parameters,shared_parameters,{name:dict(zip(ids,np.asarray(values).tolist())) for name,values in parts.items()}


def _refit_complement(root,output,cohort,shared_result,wheel,animal_path,jobs,device,emit,fingerprint):
    model_path=output/"model.joblib"
    if model_path.exists():
        raise ValueError("Uncommitted complementary model exists; inspect interrupted training before recovery")
    rows,labels,_=shared._training_data(shared._inside(wheel["path"],root,exists=True),cohort)
    if rows != shared_result["audit"]["rows"]:
        raise ValueError("Shared and complementary training membership differs")
    indices=sorted(labels["total_clearance"]);rowmap={r["index"]:r for r in rows}
    animal_rows=read_animal_rows(shared._inside(animal_path,root,exists=True).read_bytes(),cohort,interpret_labels=True)
    eligible=[r for r in animal_rows if r["status"] == "train"]
    if any(sum(s in r["log10_cl"] for r in eligible) < 2 for s in animal.SPECIES):
        raise ValueError("Insufficient animal labels after union exclusions")
    smiles=[rowmap[i]["identity_smiles"] for i in indices]
    ax_all,features=rich.rich_features(smiles+[r["identity_smiles"] for r in eligible])
    x,ax=ax_all[:len(indices)],ax_all[len(indices):]
    y=np.array([labels["total_clearance"][i] for i in indices]);animal_models={};human_models={};aux=[]
    animal_membership={}
    for j,species in enumerate(animal.SPECIES):
        emit("fitting_animal_final",current_fit=species,completed_fits=55+j)
        positions=[i for i,r in enumerate(eligible) if species in r["log10_cl"]]
        model=animal.make_animal_estimator(jobs)
        model.fit(ax[positions],[eligible[i]["log10_cl"][species] for i in positions])
        aux.append(model.predict(x));animal_models[species]=model
        animal_membership[species]=[eligible[i]["index"] for i in positions]
    augmented=np.column_stack([x,*aux])
    for j,member in enumerate(shared.METHODS["total_clearance"]):
        emit("fitting_augmented_human_final",current_fit=member,completed_fits=58+j)
        human_models[member]=shared._fit(member,augmented,y,jobs,device)
    emit("fitting_kernel_final",current_fit="tanimoto_krr_001",completed_fits=62)
    kernel_model=kernel.TanimotoKernelRidge(alpha=.01).fit(kernel.morgan_binary_features(smiles),y)
    bundle={"version":VERSION,"algorithm_version":complementary.VERSION,"selected":complementary.MEAN,
        "unit":"log10(mL/min/kg)","animal_models":animal_models,"human_models":human_models,
        "kernel_model":kernel_model,"auxiliary_order":list(animal.SPECIES),"feature_audit":features,
        "training_indices":indices,"training_identity_smiles":smiles,"animal_training_indices":animal_membership,
        "input_fingerprint":fingerprint,"retained_model_reference":_receipt(root,shared_result["model_path"])}
    rich._atomic_model(model_path,bundle)
    emit("complementary_fit_complete",completed_fits=63)
    return bundle,_receipt(root,model_path),animal_rows


def train(root,targets_path=TARGETS,output_dir=OUTPUT,jobs=8,preference="auto",progress=None):
    root=Path(root).resolve(strict=True);output=shared._inside(output_dir,root);output.mkdir(parents=True,exist_ok=True)
    with (output/"training.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train(root,targets_path,output,jobs,preference,progress)


def _train(root,targets_path,output,jobs,preference,progress):
    begin=time.monotonic()
    state,old,wheel,animal_path,old_shared,old_complement=_prepare(root,targets_path,output,jobs,preference)
    config=state["config"];cohort=config["targets"];result_path=output/"predictions.json"
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if (result.get("input_fingerprint") != state["input_fingerprint"]
                or result.get("result_sha256") != shared._digest({k:v for k,v in result.items() if k != "result_sha256"})):
            raise ValueError("Frozen expansion result changed")
        previous._verify(root,result["source_refs"]);previous._verify(root,result["code_pins"])
        for receipt in result["model_receipts"].values():_verify_receipt(root,receipt)
        return result
    def emit(stage,**updates):
        state.update(stage=stage,pid=os.getpid(),updated_at=previous._now(),**updates)
        _state(output/"training_state.json",state)
        if progress:progress({"stage":stage,**updates})
    try:
        emit("starting",planned_fits=config["planned_fits"],mode=config["mode"])
        if config["mode"] == "refit_union_exclusion":
            def shared_progress(event):
                emit("shared_training",shared_progress=event)
            shared_result=shared.fit_shared_prior([{"compound_id":cid,"smiles":row["input_smiles"]} for cid,row in cohort.items()],
                output/"shared",root,n_jobs=jobs,device=preference,progress=shared_progress)
            shared_receipt=_receipt(root,shared_result["model_path"])
            emit("shared_fit_complete",completed_fits=55)
            bundle,cl_receipt,animal_rows=_refit_complement(root,output,cohort,shared_result,wheel,animal_path,
                jobs,config["device"],emit,state["input_fingerprint"])
            shared_bundle=joblib.load(_verify_receipt(root,shared_receipt))
        else:
            shared_result=old;shared_receipt=old_shared;cl_receipt=old_complement
            shared_bundle=joblib.load(_verify_receipt(root,shared_receipt));bundle=joblib.load(_verify_receipt(root,cl_receipt))
            animal_rows=read_animal_rows(shared._inside(animal_path,root,exists=True).read_bytes(),old["audit"]["targets"])
        final_overlap=overlap_audit(cohort,shared_result["audit"]["rows"],animal_rows)
        if requires_refit(final_overlap):raise ValueError("Evaluation identities or scaffolds remain in fitted training members")
        emit("predicting_fixed_recipe")
        parameters,shared_values,parts=predict_parameters(shared_bundle,bundle,cohort)
        if config["mode"] == "refit_union_exclusion":
            for cid,row in shared_result["parameters"].items():
                if any(not math.isclose(value,shared_values[cid][p],rel_tol=1e-10,abs_tol=1e-12) for p,value in row.items()):
                    raise ValueError("Serialized shared model inference differs from committed training predictions")
        human=shared_result["audit"];human_excluded=[{k:r[k] for k in ("index","identity_smiles","scaffold_group","status") if k in r}
                    for r in human["rows"] if r["status"] != "train"]
        animal_excluded=[{k:r[k] for k in ("index","identity_smiles","scaffold_group","status")}
                         for r in animal_rows if r["status"] != "train"]
        refs=dict(config["source_refs"])
        if config["mode"] == "refit_union_exclusion":
            receipt=_receipt(root,output/"shared/predictions_audit.json");refs[receipt["path"]]=receipt["sha256"]
        previous._verify(root,refs);previous._verify(root,config["code_pins"])
        per_compound={cid:{"normalized_identity":row["identity_smiles"],"generic_scaffold":row["generic_scaffold_smiles"],
            "scaffold_group":row["scaffold_group"],"human_training_overlap":False,"animal_training_overlap":False,
            "original_training_overlap":config["original_training_overlap"][cid],"final_training_overlap":final_overlap[cid],
            "target_structure_and_scaffold_excluded":True,"is_original_twenty":cid in old["audit"]["targets"]}
            for cid,row in cohort.items()}
        result={"version":VERSION,"method_name":METHOD_NAME,"algorithm_version":complementary.VERSION,
            "mode":config["mode"],"created_at":previous._now(),"input_fingerprint":state["input_fingerprint"],
            "parameters":parameters,"units":deepcopy(shared.UNITS),"targets":cohort,"per_compound_audit":per_compound,
            "source_refs":refs,"code_pins":config["code_pins"],"selected_receipt":config["selected_receipt"],
            "model_receipts":{"shared":shared_receipt,"complementary":cl_receipt},"component_target_log10":parts,
            "training_exclusions":{"human":human_excluded,"animal":animal_excluded},
            "audit":{"target_count":len(cohort),"original_target_count":len(old["audit"]["targets"]),
                "added_target_count":len(cohort)-len(old["audit"]["targets"]),"target_labels_used":False,
                "target_observation_files_opened":False,"training_structure_overlap":False,"training_generic_scaffold_overlap":False,
                "animal_target_structure_overlap":False,"animal_target_generic_scaffold_overlap":False,
                "clinical_results_used_for_selection":False,"new_candidate_selection":False,
                "human_training_structures":human["train_count"],"human_parameter_training_indices":human["parameter_training_indices"],
                "animal_training_structures":sum(r["status"] == "train" for r in animal_rows),
                "animal_training_indices":{s:[r["index"] for r in animal_rows if r["status"] == "train" and s in r.get("log10_cl",{})]
                    for s in animal.SPECIES} if config["mode"] == "refit_union_exclusion" else "Frozen original animal model memberships; source and exclusions pinned",
                "new_fit_count":config["planned_fits"],"seed":shared.SEED,"n_jobs":jobs,"device":config["device"],
                "consistency_projection_enabled":human["consistency_projection"]["enabled"],
                "normalization":human["normalization"],"label_policy":"Both human and animal target identities/scaffolds excluded before numeric label interpretation",
                "clinical_files_accepted":False},
            "limitations":["Fixed selected algorithm; any refit changes weights and must not be pooled with historical old-weight predictions",
                "Training-only shared consistency projection inherits its documented nonnested crossfit limitation",
                "Structure/scaffold exclusion does not establish that clinical studies were previously unseen",
                "Complementary CL replaces shared CL only; the final five inputs are not reprojected as a joint physiological identity"],
            "elapsed_seconds":time.monotonic()-begin}
        result["result_sha256"]=shared._digest(result);rich._atomic_json(result_path,result)
        emit("complete",result_path=str(result_path.relative_to(root)),result_sha256=shared._sha(result_path),
             elapsed_seconds=time.monotonic()-begin,completed_fits=config["planned_fits"])
        return result
    except Exception as error:
        emit("failed",failure_type=type(error).__name__,failure_reason=str(error))
        raise
