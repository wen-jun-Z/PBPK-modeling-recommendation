"""Two fixed complementary CL fusions; no new OOF fitting or weight search.

This final development combination uses frozen training-only component OOF
predictions. Clinical target labels, curves and scores are not accepted.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from . import clearance_animal as animal
from . import clearance_refinement as previous
from . import clearance_similarity as gate
from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich
from . import tanimoto_prior as similarity

VERSION = 'fixed-complementary-clearance-prior-1'
OUTPUT = 'experiments/blueprint_design/blueprint_cl_refinement_complementary_20261004/prior'
MEAN = 'retained_animal_krr_equal_log_mean'
MEDIAN = 'retained_animal_krr_log_median'
KRR_CANDIDATE = 'tanimoto_krr_001'
CODE_FILES = tuple(dict.fromkeys((*animal.CODE_FILES, *gate.CODE_FILES,
    'src/pbpk_kb/blueprint_learning/clearance_complementary.py',
    'scripts/train_clearance_complementary.py')))


def complementary_predictions(baseline, augmented, kernel):
    arrays = [np.asarray(v,dtype=float) for v in (baseline,augmented,kernel)]
    if (arrays[0].ndim != 1 or not arrays[0].size
            or any(v.shape != arrays[0].shape or not np.isfinite(v).all() for v in arrays)):
        raise ValueError('Three finite aligned log-CL prediction arrays are required')
    return {gate.BASE:arrays[0],MEAN:np.mean(arrays,axis=0),MEDIAN:np.median(arrays,axis=0)}


def _read_result(root,name):
    path=shared._inside(name,root,exists=True);result=json.loads(path.read_text())
    if result['result_sha256'] != shared._digest({k:v for k,v in result.items() if k!='result_sha256'}):
        raise ValueError('Frozen component result checksum mismatch')
    if result.get('unit')!='mL/min/kg' or result.get('selected_using_training_only') is not True:
        raise ValueError('Component must be a training-only CL prior with explicit units')
    previous._verify(root,result['source_refs']);previous._verify(root,result['code_pins'])
    for flag in ('target_labels_used','target_observation_files_opened','training_structure_overlap','training_generic_scaffold_overlap','clinical_results_used_for_selection'):
        if result['audit'][flag] is not False:raise ValueError('Component training isolation failed: '+flag)
    return result,path


def align_component_oof(animal_result,kernel_result):
    aa,ka=animal_result['audit'],kernel_result['audit']
    for key in ('training_indices','oof_fold_ids','observed_training_log10'):
        if aa[key]!=ka[key]:raise ValueError('Component OOF rows, folds or labels differ: '+key)
    if aa['cl_training_labels']!=1124 or ka['cl_training_labels']!=1124:
        raise ValueError('All original 1124 CL labels must be retained')
    baseline=np.asarray(aa['candidate_oof_log10'][gate.BASE],dtype=float)
    if not np.array_equal(baseline,ka['candidate_oof_log10'][gate.BASE]):
        raise ValueError('Components use different retained baseline OOF predictions')
    candidates=complementary_predictions(baseline,aa['candidate_oof_log10'][animal.AUGMENTED],
                                         ka['candidate_oof_log10'][KRR_CANDIDATE])
    return np.asarray(aa['observed_training_log10']),np.asarray(aa['oof_fold_ids']),candidates


def _prepare(root,output,jobs,preference):
    if isinstance(jobs,bool) or not isinstance(jobs,int) or not 1<=jobs<=8:raise ValueError('jobs must be 1..8')
    old,old_model,_,refs=previous._read_source(root)
    av,ap=_read_result(root,animal.OUTPUT+'/predictions.json')
    kv,kp=_read_result(root,gate.OUTPUT+'/predictions.json')
    for component,path in ((av,ap),(kv,kp)):
        for name,digest in component['source_refs'].items():
            if name in refs and refs[name]!=digest:raise ValueError('Component source mismatch')
            refs[name]=digest
        refs[str(path.relative_to(root))]=shared._sha(path)
    y,folds,candidates=align_component_oof(av,kv)
    code={name:shared._sha(root/name) for name in CODE_FILES}
    config={'version':VERSION,'source_refs':refs,'code_pins':code,'n_jobs':jobs,'device':rich.choose_device(preference),
        'candidates':[gate.BASE,MEAN,MEDIAN],'component_order':[gate.BASE,animal.AUGMENTED,KRR_CANDIDATE],
        'formula':{'mean':'(retained_logCL + animal_logCL + krr_logCL)/3',
                   'median':'median(retained_logCL, animal_logCL, krr_logCL)'},
        'new_oof_fits':0,'cl_training_labels':1124,'folds':3,'krr_alpha':.01,
        'animal_recipe':animal.RF_RECIPE,'human_members':list(shared.METHODS['total_clearance']),
        'selection':'Minimum CL OOF log-MAE, subject to no worse overall/low CL/high CL/P90 than retained baseline',
        'clinical_adoption_gate':'Available and missing-CL curve GMFE/coverage cannot worsen; actual six-drug missing-CL drift and missing/available error ratio must also not worsen; no per-drug selection',
        'stop_rule':'If neither fixed combination qualifies, retain baseline and do not try other weights in this round',
        'selection_scope':'Sequential development search using already viewed training OOF; not independent post-selection evaluation'}
    fingerprint=shared._digest(config);path=output/'training_state.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state.pop('state_sha256',None)!=shared._digest(state):raise ValueError('Checkpoint checksum mismatch')
        if state['input_fingerprint']!=fingerprint:raise ValueError('Protocol changed; refusing mixed resume')
    else:
        state={'stage':'prepared','input_fingerprint':fingerprint,'config':config,'started_at':previous._now(),'updated_at':previous._now()}
        animal._state_write(path,state)
    return state,old,old_model,av,kv,y,folds,candidates


def prepare(root,output_dir=OUTPUT,jobs=8,preference='auto'):
    root=Path(root).resolve(strict=True);output=shared._inside(output_dir,root);output.mkdir(parents=True,exist_ok=True)
    with (output/'training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _prepare(root,output,jobs,preference)[0]


def fit_prior(root,output_dir=OUTPUT,jobs=8,preference='auto',progress=None):
    root=Path(root).resolve(strict=True);output=shared._inside(output_dir,root);output.mkdir(parents=True,exist_ok=True)
    with (output/'training.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _fit(root,output,jobs,preference,progress)


def _fit(root,output,jobs,preference,progress):
    started=time.monotonic();state,old,old_model,av,kv,y,folds,candidates=_prepare(root,output,jobs,preference)
    config=state['config'];refs=config['source_refs'];code=config['code_pins'];device=config['device']
    path=output/'predictions.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['input_fingerprint']!=state['input_fingerprint'] or result['result_sha256']!=shared._digest({k:v for k,v in result.items() if k!='result_sha256'}):
            raise ValueError('Frozen complementary result mismatch')
        if result.get('model_receipt'):previous._verify(root,{result['model_receipt']['path']:result['model_receipt']['sha256']})
        return result
    def emit(stage,**updates):
        state.update(stage=stage,pid=os.getpid(),updated_at=previous._now(),**updates)
        animal._state_write(output/'training_state.json',state)
        if progress:progress({'stage':stage,**updates})
    selection=gate.evaluate_candidates(y,candidates,folds)
    if state.get('selection') and state['selection']!=selection:raise ValueError('Frozen selection changed')
    emit('selection_locked',selection=selection,selection_locked_at=state.get('selection_locked_at',previous._now()))
    previous._verify(root,refs);previous._verify(root,code)
    selected=selection['selected'];receipt=None;component_predictions=None
    if selected==gate.BASE:
        predictions={cid:float(r['total_clearance']) for cid,r in old['parameters'].items()}
        emit('baseline_retained_no_new_training',reason='No fixed combination passed all four predeclared training gates')
    else:
        # Read the completed animal preparation only; never mutate its state.
        _,_,_,_,cohort,rows,labels,animal_rows,_=animal._prepare(root,shared._inside(animal.OUTPUT,root),jobs,preference)
        indices=sorted(labels['total_clearance'])
        if indices!=av['audit']['training_indices']:raise ValueError('Final human training membership changed')
        rowmap={r['index']:r for r in rows};apos={r['index']:i for i,r in enumerate(animal_rows)}
        smiles=[rowmap[i]['identity_smiles'] for i in indices];targets=[r['identity_smiles'] for r in cohort.values()]
        emit('building_final_features')
        all_x,feature_audit=rich.rich_features(smiles+[r['identity_smiles'] for r in animal_rows]+targets)
        x=all_x[:len(indices)];ax=all_x[len(indices):len(indices)+len(animal_rows)];tx=all_x[len(indices)+len(animal_rows):]
        animal_models={};human_models={};aux_train=[];aux_target=[]
        for species in animal.SPECIES:
            emit('fitting_final_animal',species=species)
            source=[r for r in animal_rows if species in r['log10_cl']]
            model=animal.make_animal_estimator(jobs)
            model.fit(ax[[apos[r['index']] for r in source]],[r['log10_cl'][species] for r in source])
            aux_train.append(model.predict(x));aux_target.append(model.predict(tx));animal_models[species]=model
        augmented_x=np.column_stack([x,*aux_train]);augmented_tx=np.column_stack([tx,*aux_target]);human_predictions=[]
        for member in shared.METHODS['total_clearance']:
            emit('fitting_final_human',member=member)
            model=shared._fit(member,augmented_x,y,jobs,device)
            human_predictions.append(model.predict(augmented_tx));human_models[member]=model
        animal_log=np.mean(human_predictions,axis=0)
        emit('fitting_final_kernel',alpha=.01)
        bits=similarity.morgan_binary_features(smiles);target_bits=similarity.morgan_binary_features(targets)
        kernel_model=similarity.TanimotoKernelRidge(alpha=.01).fit(bits,y)
        kernel_log=kernel_model.predict(target_bits)
        baseline_log=np.log10([old['parameters'][cid]['total_clearance'] for cid in cohort])
        target_candidates=complementary_predictions(baseline_log,animal_log,kernel_log)
        final=10.**target_candidates[selected]
        if final.shape!=(len(cohort),) or not np.isfinite(final).all() or np.any(final<=0):raise ValueError('Invalid final CL predictions')
        predictions=dict(zip(cohort,final.tolist()))
        component_predictions={name:dict(zip(cohort,values.tolist())) for name,values in (
            (gate.BASE,baseline_log),(animal.AUGMENTED,animal_log),(KRR_CANDIDATE,kernel_log))}
        model_path=output/'model.joblib'
        rich._atomic_model(model_path,{'version':VERSION,'selected':selected,'unit':'log10(mL/min/kg)',
            'animal_models':animal_models,'human_models':human_models,'kernel_model':kernel_model,
            'auxiliary_order':list(animal.SPECIES),'component_order':config['component_order'],'formula':config['formula'],
            'training_indices':indices,'training_identity_smiles':smiles,'target_identity_smiles':dict(zip(cohort,targets)),
            'input_fingerprint':state['input_fingerprint'],'feature_audit':feature_audit,
            'retained_model_reference':{'path':str(old_model.relative_to(root)),'sha256':shared._sha(old_model)},
            'retained_target_cl':{cid:old['parameters'][cid]['total_clearance'] for cid in cohort}})
        receipt={'path':str(model_path.relative_to(root)),'sha256':shared._sha(model_path)}
    result={'version':VERSION,'unit':'mL/min/kg','predictions':predictions,'selected_using_training_only':True,
        'selection':selection,'oof_metrics':selection['scores'],'source_refs':refs,'code_pins':code,
        'model_receipt':receipt,'component_target_log10':component_predictions,'input_fingerprint':state['input_fingerprint'],
        'created_at':previous._now(),'elapsed_seconds':time.monotonic()-started,
        'audit':{'target_labels_used':False,'target_observation_files_opened':False,'training_structure_overlap':False,
            'training_generic_scaffold_overlap':False,'clinical_results_used_for_selection':False,
            'animal_target_structure_overlap':False,'animal_outer_validation_structure_overlap':False,
            'animal_human_label_columns_interpreted':False,'new_oof_fits':0,'training_structures':1126,'cl_training_labels':1124,
            'training_indices':av['audit']['training_indices'],'oof_fold_ids':folds.tolist(),'observed_training_log10':y.tolist(),
            'candidate_oof_log10':{k:v.tolist() for k,v in candidates.items()},'selection_locked_at':state['selection_locked_at'],
            'target_count':len(predictions),'animal_source':av['audit']['animal_source'],
            'consistency_projection_inherited_limitation':av['audit'].get('consistency_projection_inherited_limitation')},
        'interpretation':'Fixed final complementary development selection. Reusing viewed training OOF is not independent validation; no other weights searched in this round.'}
    previous._verify(root,refs);previous._verify(root,code)
    result['result_sha256']=shared._digest(result);rich._atomic_json(path,result)
    emit('complete',selected=selected,result_path=str(path.relative_to(root)),result_sha256=shared._sha(path),elapsed_seconds=time.monotonic()-started)
    return result
