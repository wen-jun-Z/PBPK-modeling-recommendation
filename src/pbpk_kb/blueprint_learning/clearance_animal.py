"""Animal-CL auxiliary features with target and outer-fold scaffold exclusion.

Only molecular structure and three nonhuman CL columns are interpreted from the
animal source. Human CL training labels come exclusively from the frozen shared
training source. No clinical observations or target human labels are inputs.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from . import clearance_refinement as previous
from . import clearance_similarity as gate
from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich

VERSION = "animal-auxiliary-clearance-prior-1"
OUTPUT = "experiments/blueprint_design/blueprint_cl_refinement_animal_20261004/prior"
ANIMAL_URL = "https://raw.githubusercontent.com/srijitseal/PKSmart/main/data/Animal_PK_data.csv"
ANIMAL_SHA256 = "07cabff1591c31de8bf325da3a2dde003c83fb40e78b29dc9af826b68da4a823"
SPECIES = ("dog", "monkey", "rat")
COLUMNS = {s: s + "_CL_mL_min_kg" for s in SPECIES}
ALLOWED_COLUMNS = ("smiles_r", *COLUMNS.values())
RF_RECIPE = {"n_estimators": 300, "min_samples_leaf": 2,
             "random_state": shared.SEED, "max_features": 1.0}
RAW = "human_only_raw_ensemble"
AUGMENTED = "animal_augmented_ensemble"
BLEND = "animal_augmented_half_blend"
CODE_FILES = (*gate.CODE_FILES, "src/pbpk_kb/blueprint_learning/clearance_animal.py",
              "scripts/train_clearance_animal.py")


def animal_rows(payload):
    """Parse only the explicit animal-label whitelist; never human columns."""
    frame = pd.read_csv(io.BytesIO(payload), usecols=list(ALLOWED_COLUMNS), dtype=str, na_filter=False)
    rows, seen = [], set()
    for index, raw in frame.iterrows():
        identity = shared.normalize_structure(raw["smiles_r"])
        if identity["identity_smiles"] in seen:
            raise ValueError("Animal source has duplicate normalized parents; freeze a deduplication rule first")
        seen.add(identity["identity_smiles"])
        labels, exclusions = {}, {}
        for species, column in COLUMNS.items():
            value = raw[column].strip()
            if value.lower() in {"", "nan", "na", "null", "none"}:
                exclusions[species] = "missing"
                continue
            try:
                number = float(value)
            except ValueError:
                exclusions[species] = "not_numeric"
                continue
            if not math.isfinite(number) or number <= 0:
                exclusions[species] = "nonpositive_or_nonfinite"
                continue
            labels[species] = math.log10(number)
        rows.append({"index": int(index), **identity, "log10_cl": labels, "label_exclusions": exclusions})
    return rows


def exclude_structures(rows, protected):
    """Identity AND generic-scaffold exclusion, independent of any labels."""
    identities = {r["identity_smiles"] for r in protected}
    scaffolds = {r["scaffold_group"] for r in protected}
    allowed, excluded = [], {}
    for row in rows:
        reason = ("protected_identity" if row["identity_smiles"] in identities else
                  "protected_generic_scaffold" if row["scaffold_group"] in scaffolds else None)
        if reason:
            excluded[str(row["index"])] = reason
        else:
            allowed.append(row)
    return allowed, excluded


def make_animal_estimator(jobs):
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("model", RandomForestRegressor(**RF_RECIPE, n_jobs=jobs))])


def _animal_source(root, output):
    path = output / "animal_source.csv"
    if not path.exists():
        with urllib.request.urlopen(ANIMAL_URL, timeout=30) as response:
            payload = response.read(2 * 1024**2 + 1)
        if len(payload) > 2 * 1024**2 or hashlib.sha256(payload).hexdigest() != ANIMAL_SHA256:
            raise ValueError("Animal source size/hash differs from the reviewed official release")
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    if shared._sha(path) != ANIMAL_SHA256:
        raise ValueError("Pinned animal source changed")
    return path


def _state_write(path, state):
    rich._atomic_json(path, {**state, "state_sha256": shared._digest(state)})


def _prepare(root, output, jobs, preference):
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 8:
        raise ValueError("jobs must be an integer in [1,8]")
    old, old_model, wheel, refs = previous._read_source(root)
    prior_path = shared._inside(gate.PRIOR, root, exists=True)
    prior = json.loads(prior_path.read_text())
    if prior["result_sha256"] != shared._digest({k:v for k,v in prior.items() if k != "result_sha256"}):
        raise ValueError("Stored full-1124 OOF checksum mismatch")
    previous._verify(root, prior["source_refs"]); previous._verify(root, prior["code_pins"])
    refs[str(prior_path.relative_to(root))] = shared._sha(prior_path)
    source = _animal_source(root, output)
    refs[str(source.relative_to(root))] = ANIMAL_SHA256
    original = old["audit"]
    cohort = shared._cohort([{"compound_id":cid, "smiles":r["input_smiles"]} for cid,r in original["targets"].items()])
    rows, labels, membership = shared._training_data(wheel, cohort)
    indices = sorted(labels["total_clearance"])
    if rows != original["rows"] or indices != prior["audit"]["training_indices"] or len(indices) != 1124:
        raise ValueError("Frozen human training rows/CL labels changed")
    if [labels["total_clearance"][i] for i in indices] != prior["audit"]["observed_training_log10"]:
        raise ValueError("Frozen human training CL values changed")
    animal = animal_rows(source.read_bytes())
    eligible, global_exclusions = exclude_structures(animal, cohort.values())
    if len(animal) != 371 or len(eligible) != 312:
        raise ValueError("Reviewed animal source/target exclusion counts changed")
    rowmap = {r["index"]:r for r in rows}
    fold_animal, folds_audit = {}, []
    expected_counts = {0: {"dog":152,"monkey":73,"rat":194},
                       1: {"dog":143,"monkey":70,"rat":175},
                       2: {"dog":125,"monkey":57,"rat":163}}
    for fold in original["folds"]:
        held = [rowmap[i] for i in fold["validation_indices"]]
        fit, exclusions = exclude_structures(eligible, held)
        counts = {s:sum(s in r["log10_cl"] for r in fit) for s in SPECIES}
        if counts != expected_counts[fold["fold"]]:
            raise ValueError("Reviewed animal fold sample counts changed")
        fold_animal[fold["fold"]] = fit
        folds_audit.append({"fold":fold["fold"], "animal_structure_count":len(fit),
            "animal_label_counts":counts, "animal_indices":[r["index"] for r in fit],
            "held_human_indices":fold["validation_indices"], "excluded_animal_indices":exclusions})
    code = {name:shared._sha(root/name) for name in CODE_FILES}
    config = {"version":VERSION,"source_refs":refs,"code_pins":code,"n_jobs":jobs,"device":rich.choose_device(preference),
        "animal_source":{"url":ANIMAL_URL,"sha256":ANIMAL_SHA256,"parsed_columns":list(ALLOWED_COLUMNS),
            "unit":"mL/min/kg","input_scale":"linear; positive labels transformed log10 once",
            "human_columns_interpreted":False,"raw_structures":len(animal),"eligible_structures":len(eligible),
            "global_label_counts":{s:sum(s in r['log10_cl'] for r in eligible) for s in SPECIES},
            "target_exclusions":global_exclusions,"folds":folds_audit},
        "animal_model":{"recipe":RF_RECIPE,"auxiliary_order":list(SPECIES),"auxiliary_unit":"log10(mL/min/kg)"},
        "human_model":{"members":list(shared.METHODS['total_clearance']),"recipes":rich.RECIPES,
                       "ensemble":"equal log mean; unchanged recipes with three appended animal predictions"},
        "candidates":[gate.BASE,RAW,AUGMENTED,BLEND],
        "selection":"Minimum CL OOF log-MAE, no worse overall/low CL/high CL/P90 than retained baseline",
        "clinical_adoption_gate":"Available and missing-CL curve GMFE/coverage cannot worsen; actual six-drug missing-CL drift and missing/available error ratio must also not worsen; no per-drug selection",
        "training_only":True,"human_training_rows":membership['train_count'],"human_cl_labels":len(indices),
        "auxiliary_fit_scope":"Animal models exclude all targets and entire human-held fold; training-human animal matches can occur, held/target animal matches cannot",
        "independent_post_selection_evaluation":False}
    fingerprint = shared._digest(config)
    path = output/'training_state.json'
    if path.exists():
        state = json.loads(path.read_text())
        if state.pop('state_sha256',None) != shared._digest(state):
            raise ValueError("Training checkpoint checksum mismatch")
        if state['input_fingerprint'] != fingerprint:
            raise ValueError("Training configuration changed; refusing mixed resume")
    else:
        state = {"stage":"prepared","started_at":previous._now(),"updated_at":previous._now(),
                 "input_fingerprint":fingerprint,"config":config,"fits":{}}
        _state_write(path,state)
    return state, old, old_model, prior, cohort, rows, labels, eligible, fold_animal


def prepare(root, output_dir=OUTPUT, jobs=8, preference='auto'):
    root = Path(root).resolve(strict=True); output = shared._inside(output_dir,root)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _prepare(root,output,jobs,preference)[0]


def fit_prior(root, output_dir=OUTPUT, jobs=8, preference='auto', progress=None):
    root = Path(root).resolve(strict=True); output = shared._inside(output_dir,root)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _fit(root,output,jobs,preference,progress)


def _fit(root,output,jobs,preference,progress):
    started=time.monotonic()
    state,old,old_model,prior,cohort,rows,labels,animal,fold_animal = _prepare(root,output,jobs,preference)
    config=state['config'];device=config['device'];refs=config['source_refs'];code=config['code_pins']
    result_path=output/'predictions.json'
    if result_path.exists():
        result=json.loads(result_path.read_text())
        if result['input_fingerprint']!=state['input_fingerprint'] or result['result_sha256']!=shared._digest({k:v for k,v in result.items() if k!='result_sha256'}):
            raise ValueError('Frozen animal-prior output mismatch')
        if result.get('model_receipt'):previous._verify(root,{result['model_receipt']['path']:result['model_receipt']['sha256']})
        return result
    def emit(stage,**updates):
        state.update(stage=stage,pid=os.getpid(),updated_at=previous._now(),**updates)
        _state_write(output/'training_state.json',state)
        if progress:progress({'stage':stage,**updates})
    indices=sorted(labels['total_clearance']);pos={ix:j for j,ix in enumerate(indices)}
    rowmap={r['index']:r for r in rows};apos={r['index']:j for j,r in enumerate(animal)}
    y=np.array([labels['total_clearance'][i] for i in indices])
    emit('building_features')
    all_x,feature_audit=rich.rich_features([rowmap[i]['identity_smiles'] for i in indices]+[r['identity_smiles'] for r in animal])
    x,ax=all_x[:len(indices)],all_x[len(indices):]
    folds=np.asarray(prior['audit']['oof_fold_ids']);augmented=np.full(len(indices),np.nan)
    raw_map={r['row_index']:r['predicted_log10'] for fold in old['audit']['consistency_oof']['total_clearance']['folds'].values() for r in fold['out_of_fold']}
    raw=np.array([raw_map[i] for i in indices])
    baseline=np.array(prior['audit']['candidate_oof_log10'][gate.BASE])
    for fold in old['audit']['folds']:
        fit=[i for i in fold['train_indices'] if i in pos];held=[i for i in fold['validation_indices'] if i in pos]
        it=np.array([pos[i] for i in fit]);iv=np.array([pos[i] for i in held])
        if set(fit)&set(held) or not np.all(folds[iv]==fold['fold']):raise ValueError('Human fold mismatch')
        fit_aux,held_aux=[],[]
        for species in SPECIES:
            source=[r for r in fold_animal[fold['fold']] if species in r['log10_cl']]
            key=f"{fold['fold']}/animal/{species}"
            metadata={'animal_training_indices':[r['index'] for r in source],'human_train_indices':fit,'human_held_indices':held,'species':species}
            if key in state['fits']:
                record=state['fits'][key]
                if any(record[k]!=v for k,v in metadata.items()):raise ValueError('Animal checkpoint membership mismatch')
                predicted=np.asarray(record['predicted_log10'])
            else:
                emit('fitting_oof',current_fit=key,completed_fits=len(state['fits']),planned_fits=21)
                then=time.monotonic();model=make_animal_estimator(jobs)
                model.fit(ax[[apos[r['index']] for r in source]],[r['log10_cl'][species] for r in source])
                predicted=model.predict(x[np.concatenate([it,iv])])
                state['fits'][key]={**metadata,'predicted_log10':predicted.tolist(),'elapsed_seconds':time.monotonic()-then}
                emit('oof_fit_complete',completed_fits=len(state['fits']))
            if predicted.shape!=(len(it)+len(iv),) or not np.isfinite(predicted).all():raise ValueError('Invalid animal auxiliary predictions')
            fit_aux.append(predicted[:len(it)]);held_aux.append(predicted[len(it):])
        train_x=np.column_stack([x[it],*fit_aux]);held_x=np.column_stack([x[iv],*held_aux]);member_predictions=[]
        for member in shared.METHODS['total_clearance']:
            key=f"{fold['fold']}/human/{member}"
            metadata={'train_indices':fit,'held_indices':held,'member':member,'auxiliary_order':list(SPECIES)}
            if key in state['fits']:
                record=state['fits'][key]
                if any(record[k]!=v for k,v in metadata.items()):raise ValueError('Human checkpoint membership mismatch')
                predicted=np.asarray(record['predicted_log10'])
            else:
                emit('fitting_oof',current_fit=key,completed_fits=len(state['fits']),planned_fits=21)
                then=time.monotonic();model=shared._fit(member,train_x,y[it],jobs,device)
                predicted=np.asarray(model.predict(held_x))
                state['fits'][key]={**metadata,'predicted_log10':predicted.tolist(),'elapsed_seconds':time.monotonic()-then}
                emit('oof_fit_complete',completed_fits=len(state['fits']))
            if predicted.shape!=(len(iv),) or not np.isfinite(predicted).all():raise ValueError('Invalid augmented human OOF predictions')
            member_predictions.append(predicted)
        augmented[iv]=np.mean(member_predictions,axis=0)
    candidates={gate.BASE:baseline,RAW:raw,AUGMENTED:augmented,BLEND:.5*(baseline+augmented)}
    selection=gate.evaluate_candidates(y,candidates,folds)
    if state.get('selection') and state['selection']!=selection:raise ValueError('Frozen training-only selection changed')
    emit('selection_locked',selection=selection,selection_locked_at=state.get('selection_locked_at',previous._now()))
    previous._verify(root,refs);previous._verify(root,code)
    selected=selection['selected'];receipt=None
    if selected==gate.BASE:
        predictions={cid:float(old['parameters'][cid]['total_clearance']) for cid in cohort}
    else:
        emit('predicting_selected_model')
        tx,_=rich.rich_features([r['identity_smiles'] for r in cohort.values()])
        if selected==RAW:
            bundle=joblib.load(old_model)
            logs=np.mean([bundle['models']['total_clearance'][m].predict(tx) for m in shared.METHODS['total_clearance']],axis=0)
        else:
            animal_models={};human_models={};train_aux=[];target_aux=[]
            for species in SPECIES:
                emit('fitting_final_animal',species=species)
                source=[r for r in animal if species in r['log10_cl']]
                model=make_animal_estimator(jobs)
                model.fit(ax[[apos[r['index']] for r in source]],[r['log10_cl'][species] for r in source])
                train_aux.append(model.predict(x));target_aux.append(model.predict(tx));animal_models[species]=model
            augmented_x=np.column_stack([x,*train_aux]);augmented_tx=np.column_stack([tx,*target_aux])
            final_members=[]
            for member in shared.METHODS['total_clearance']:
                emit('fitting_final_human',member=member)
                model=shared._fit(member,augmented_x,y,jobs,device)
                final_members.append(model.predict(augmented_tx));human_models[member]=model
            logs=np.mean(final_members,axis=0)
            if selected==BLEND:logs=.5*(logs+np.log10([old['parameters'][cid]['total_clearance'] for cid in cohort]))
            model_path=output/'model.joblib'
            rich._atomic_model(model_path,{'version':VERSION,'selected':selected,'animal_models':animal_models,
                'human_models':human_models,'auxiliary_order':list(SPECIES),'input_fingerprint':state['input_fingerprint'],
                'retained_model_reference':{'path':str(old_model.relative_to(root)),'sha256':shared._sha(old_model)}})
            receipt={'path':str(model_path.relative_to(root)),'sha256':shared._sha(model_path)}
        values=10.**logs
        if values.shape!=(len(cohort),) or not np.isfinite(values).all() or np.any(values<=0):raise ValueError('Invalid final target CL')
        predictions=dict(zip(cohort,values.tolist()))
    result={'version':VERSION,'unit':'mL/min/kg','predictions':predictions,'selected_using_training_only':True,
        'selection':selection,'oof_metrics':selection['scores'],'source_refs':refs,'code_pins':code,'model_receipt':receipt,
        'input_fingerprint':state['input_fingerprint'],'created_at':previous._now(),'elapsed_seconds':time.monotonic()-started,
        'audit':{'target_labels_used':False,'target_observation_files_opened':False,'training_structure_overlap':False,
            'training_generic_scaffold_overlap':False,'clinical_results_used_for_selection':False,
            'animal_target_structure_overlap':False,'animal_outer_validation_structure_overlap':False,
            'animal_human_label_columns_interpreted':False,'training_structures':1126,'cl_training_labels':len(indices),
            'training_indices':indices,'oof_fold_ids':folds.tolist(),'observed_training_log10':y.tolist(),
            'candidate_oof_log10':{k:v.tolist() for k,v in candidates.items()},'selection_locked_at':state['selection_locked_at'],
            'animal_source':config['animal_source'],'training_features':feature_audit,'target_count':len(cohort),
            'consistency_projection_inherited_limitation':prior['audit'].get('consistency_projection_inherited_limitation')},
        'interpretation':'Animal labels add training information only; no target animal match or clinical observations. Development OOF selection is not independent validation.'}
    previous._verify(root,refs);previous._verify(root,code)
    result['result_sha256']=shared._digest(result);rich._atomic_json(result_path,result)
    emit('complete',selected=selected,result_path=str(result_path.relative_to(root)),result_sha256=shared._sha(result_path),elapsed_seconds=time.monotonic()-started)
    return result
