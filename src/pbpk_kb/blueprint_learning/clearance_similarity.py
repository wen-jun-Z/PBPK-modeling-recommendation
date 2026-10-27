"""Structure-only CL candidates with explicit extreme-label non-regression gates.

The held-out clinical cohort never supplies model-selection labels. This is a
development search on existing training scaffold folds, not external validation.
The retained comparator inherits its documented non-nested consistency weights.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from sklearn.svm import SVR

from . import clearance_refinement as previous
from . import clinical_shared_prior as shared
from . import rich_parameter_prior as rich
from . import tanimoto_prior as similarity

VERSION = 'clearance-similarity-tail-gated-1'
OUTPUT = 'experiments/blueprint_design/blueprint_cl_refinement_similarity_20261004/prior'
PRIOR = previous.OUTPUT + '/predictions.json'
BASE = 'retained_projected_cl'
RECIPES = {
    'tanimoto_krr_001': {'kind': 'krr', 'alpha': .01},
    'tanimoto_krr_01': {'kind': 'krr', 'alpha': .1},
    'tanimoto_krr_1': {'kind': 'krr', 'alpha': 1.},
    'tanimoto_svr_1': {'kind': 'svr', 'C': 1., 'epsilon': .1},
    'tanimoto_svr_10': {'kind': 'svr', 'C': 10., 'epsilon': .1},
    'xgb_depth3_unweighted': {'kind': 'xgb', 'method': 'rich_xgb_depth3', 'balanced': False},
    'xgb_depth3_balanced': {'kind': 'xgb', 'method': 'rich_xgb_depth3', 'balanced': True},
    'xgb_depth5_balanced': {'kind': 'xgb', 'method': 'rich_xgb_depth5', 'balanced': True},
}
CODE_FILES = (*previous.CODE_FILES, 'src/pbpk_kb/blueprint_learning/tanimoto_prior.py',
    'src/pbpk_kb/blueprint_learning/clearance_similarity.py', 'scripts/train_clearance_similarity.py')


def tail_weights(y):
    """Bounded inverse-sqrt group frequency, computed from training labels only."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError('Finite nonempty training log-CL required')
    groups = np.digitize(y, [0., math.log10(15.)])
    counts = np.bincount(groups, minlength=3)
    weights = np.sqrt(len(y) / counts[groups])
    weights /= weights.mean()
    return np.minimum(weights, 3.)


def evaluate_candidates(y, candidates, folds):
    y, folds = np.asarray(y, dtype=float), np.asarray(folds)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all() or folds.shape != y.shape or BASE not in candidates:
        raise ValueError('Aligned finite labels, folds and baseline are required')
    groups = {'low_cl': y < 0., 'middle_cl': (y >= 0.) & (y < math.log10(15.)),
              'high_cl': y >= math.log10(15.)}
    if any(not np.any(mask) for mask in groups.values()):
        raise ValueError('All three clearance groups must have training OOF labels')
    scores = {}
    for name, values in candidates.items():
        predicted = np.asarray(values, dtype=float)
        if predicted.shape != y.shape or not np.isfinite(predicted).all():
            raise ValueError('Every candidate must predict all CL-labeled OOF rows')
        errors = np.abs(predicted - y)
        scores[name] = {'n': len(y), 'log10_mae': float(errors.mean()), 'gmfe': float(10**errors.mean()),
            'log10_error_p90': float(np.quantile(errors, .9)),
            'fold_error_p90': float(np.quantile(10**errors, .9)),
            'fold_error_p95': float(np.quantile(10**errors, .95)),
            'fold_error_max': float(np.max(10**errors)),
            'within_counts': {str(t): int(np.sum(errors < math.log10(t))) for t in (2, 3, 5)},
            'threshold_comparison': '<',
            'groups': {group: {'n': int(mask.sum()), 'log10_mae': float(errors[mask].mean()),
                               'mean_signed_log10_error': float((predicted-y)[mask].mean()),
                               'gmfe': float(10**errors[mask].mean())} for group, mask in groups.items()},
            'fold_log10_mae': {str(f): float(errors[folds == f].mean()) for f in sorted(set(folds.tolist()))}}
    baseline = scores[BASE]; eligible = []; tol = 1e-12
    for name, s in scores.items():
        checks = {'overall_not_worse': s['log10_mae'] <= baseline['log10_mae'] + tol,
            'low_cl_not_worse': s['groups']['low_cl']['log10_mae'] <= baseline['groups']['low_cl']['log10_mae'] + tol,
            'high_cl_not_worse': s['groups']['high_cl']['log10_mae'] <= baseline['groups']['high_cl']['log10_mae'] + tol,
            'p90_not_worse': s['fold_error_p90'] <= baseline['fold_error_p90'] + tol}
        s['eligibility_checks'] = checks
        if all(checks.values()): eligible.append(name)
    selected = min(eligible, key=lambda name: (scores[name]['log10_mae'], list(candidates).index(name)))
    if scores[selected]['log10_mae'] >= baseline['log10_mae'] - tol: selected = BASE
    return {'selected': selected, 'eligible': eligible, 'scores': scores, 'training_only': True,
        'criterion': 'Minimum overall OOF log-MAE subject to no worse overall, low/high CL and P90 than retained baseline',
        'tail_groups_unit': 'mL/min/kg', 'tail_groups': {'low': '<1', 'middle': '[1,15)', 'high': '>=15'},
        'clinical_values_used': False, 'independent_post_selection_evaluation': False}


def fit_prior(root, output_dir=OUTPUT, jobs=8, preference='auto', progress=None):
    root = Path(root).resolve(strict=True); output = shared._inside(output_dir, root)
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 8:
        raise ValueError('jobs must be 1..8')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'training.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _fit(root, output, jobs, preference, progress)


def _fit(root, output, jobs, preference, progress):
    started = time.monotonic(); old, old_model, wheel, refs = previous._read_source(root)
    prior_path = shared._inside(PRIOR, root, exists=True)
    prior = json.loads(prior_path.read_text())
    if prior['result_sha256'] != shared._digest({k:v for k,v in prior.items() if k != 'result_sha256'}):
        raise ValueError('Stored full-1124 OOF checksum mismatch')
    previous._verify(root, prior['source_refs']); previous._verify(root, prior['code_pins'])
    refs[str(prior_path.relative_to(root))] = shared._sha(prior_path)
    code = {name: shared._sha(root/name) for name in CODE_FILES}
    device = rich.choose_device(preference)
    config = {'version': VERSION, 'source_refs': refs, 'code_pins': code, 'recipes': RECIPES,
        'fixed_blends': 'Each new predictor also evaluated at 0.5 log weight with retained CL',
        'selection': 'Overall, low CL, high CL and P90 non-regression; then minimum overall log-MAE',
        'clinical_adoption_gate': 'Available and missing-CL curve GMFE/coverage cannot worsen; actual six-drug missing-CL drift and missing/available error ratio must also not worsen. No per-drug selection.',
        'weighting': 'Training-only <1/[1,15)/>=15 inverse square-root group counts; mean-normalize then cap at 3',
        'n_jobs': jobs, 'device': device}
    fingerprint = shared._digest(config); state_path = output/'training_state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.pop('state_sha256', None) != shared._digest(state): raise ValueError('Training checkpoint checksum mismatch')
        if state['input_fingerprint'] != fingerprint: raise ValueError('Training protocol changed; refusing mixed resume')
    else: state = {'input_fingerprint': fingerprint, 'config': config, 'started_at': previous._now(), 'fits': {}}
    def emit(stage, **updates):
        state.update(stage=stage, pid=os.getpid(), updated_at=previous._now(), **updates)
        payload = {**state, 'state_sha256': shared._digest(state)}
        rich._atomic_json(state_path, payload)
        if progress: progress({'stage': stage, **updates})
    if (output/'predictions.json').exists():
        result = json.loads((output/'predictions.json').read_text())
        if result['input_fingerprint'] != fingerprint or result['result_sha256'] != shared._digest({k:v for k,v in result.items() if k!='result_sha256'}):
            raise ValueError('Frozen output mismatch')
        if result.get('model_receipt'): previous._verify(root, {result['model_receipt']['path']:result['model_receipt']['sha256']})
        return result
    emit('verifying_membership')
    original = old['audit']
    cohort = shared._cohort([{'compound_id': cid, 'smiles': rec['input_smiles']} for cid, rec in original['targets'].items()])
    rows, labels, membership = shared._training_data(wheel, cohort)
    if rows != original['rows']: raise ValueError('Training membership changed')
    indices = sorted(labels['total_clearance']); pa = prior['audit']
    if indices != pa['training_indices'] or len(indices) != 1124: raise ValueError('OOF labels differ')
    y = np.array([labels['total_clearance'][i] for i in indices])
    if not np.array_equal(y, pa['observed_training_log10']): raise ValueError('Stored OOF labels changed')
    rowmap = {r['index']:r for r in rows}; positions = {ix:j for j,ix in enumerate(indices)}
    smiles = [rowmap[i]['identity_smiles'] for i in indices]
    emit('building_features')
    x, feature_audit = rich.rich_features(smiles)
    bits = similarity.morgan_binary_features(smiles)
    kernel = similarity.tanimoto_kernel(bits)
    candidates = {BASE: np.asarray(pa['candidate_oof_log10'][BASE], dtype=float)}
    raw = {name: np.full(len(indices), np.nan) for name in RECIPES}
    folds = np.asarray(pa['oof_fold_ids'])
    for f in original['folds']:
        fit = [i for i in f['train_indices'] if i in positions]; held = [i for i in f['validation_indices'] if i in positions]
        it = np.array([positions[i] for i in fit]); iv = np.array([positions[i] for i in held])
        if set(fit)&set(held) or set(rowmap[i]['scaffold_group'] for i in fit)&set(rowmap[i]['scaffold_group'] for i in held):
            raise ValueError('Training and validation identity/scaffold overlap')
        if not np.all(folds[iv] == f['fold']): raise ValueError('Stored fold assignment mismatch')
        for name, recipe in RECIPES.items():
            key = f"{f['fold']}/{name}"; metadata = {'train_indices':fit,'held_indices':held,'recipe':recipe}
            if key in state['fits']:
                record = state['fits'][key]
                if any(record[k]!=v for k,v in metadata.items()): raise ValueError('Checkpoint membership mismatch')
                predicted = np.asarray(record['predicted_log10'], dtype=float)
            else:
                emit('fitting_oof', current_fit=key, completed_fits=len(state['fits']), planned_fits=24)
                then = time.monotonic()
                estimator = make_estimator(recipe, jobs, device)
                if recipe['kind']=='krr':
                    estimator.fit(bits[it], y[it], train_kernel=kernel[np.ix_(it,it)])
                    predicted = estimator.predict(bits[iv], cross_kernel=kernel[np.ix_(iv,it)])
                elif recipe['kind']=='svr':
                    estimator.fit(kernel[np.ix_(it,it)], y[it]); predicted = estimator.predict(kernel[np.ix_(iv,it)])
                else:
                    kwargs = {'model__sample_weight':tail_weights(y[it])} if recipe['balanced'] else {}
                    estimator.fit(x[it],y[it],**kwargs); rich._verify_fitted_device(estimator,recipe['method'],device)
                    predicted = estimator.predict(x[iv])
                state['fits'][key] = {**metadata,'predicted_log10':np.asarray(predicted).tolist(),'elapsed_seconds':time.monotonic()-then}
                emit('oof_fit_complete',completed_fits=len(state['fits']))
            if predicted.shape != (len(iv),) or not np.isfinite(predicted).all(): raise ValueError('Invalid OOF predictions')
            raw[name][iv] = predicted
    for name, values in raw.items():
        candidates[name] = values; candidates[name+'_half_blend'] = .5*(values+candidates[BASE])
    selection = evaluate_candidates(y,candidates,folds)
    if state.get('selection') and state['selection'] != selection: raise ValueError('Frozen selection changed')
    emit('selection_locked',selection=selection,selection_locked_at=state.get('selection_locked_at',previous._now()))
    previous._verify(root,refs); previous._verify(root,code)
    selected = selection['selected']; model_receipt = None
    if selected == BASE:
        predictions = {cid:float(old['parameters'][cid]['total_clearance']) for cid in cohort}
    else:
        emit('fitting_selected_final_model')
        name = selected.removesuffix('_half_blend'); recipe = RECIPES[name]
        estimator = make_estimator(recipe,jobs,device)
        target_smiles = [r['identity_smiles'] for r in cohort.values()]
        target_bits = similarity.morgan_binary_features(target_smiles)
        target_kernel = similarity.tanimoto_kernel(target_bits,bits)
        if recipe['kind']=='krr':
            estimator.fit(bits,y,train_kernel=kernel); target_log = estimator.predict(target_bits,cross_kernel=target_kernel)
        elif recipe['kind']=='svr':
            estimator.fit(kernel,y); target_log = estimator.predict(target_kernel)
        else:
            kwargs = {'model__sample_weight':tail_weights(y)} if recipe['balanced'] else {}
            estimator.fit(x,y,**kwargs); rich._verify_fitted_device(estimator,recipe['method'],device)
            target_x,_ = rich.rich_features(target_smiles); target_log = estimator.predict(target_x)
        if selected.endswith('_half_blend'):
            target_log = .5*(target_log+np.log10([old['parameters'][cid]['total_clearance'] for cid in cohort]))
        values = 10.**target_log
        if not np.isfinite(values).all() or np.any(values<=0): raise ValueError('Invalid target CL')
        predictions = dict(zip(cohort,values.tolist()))
        model_path = output/'model.joblib'
        rich._atomic_model(model_path,{'version':VERSION,'selected':selected,'model':estimator,
            'training_fingerprints':bits if recipe['kind']=='svr' else None,
            'retained_model_reference':{'path':str(old_model.relative_to(root)),'sha256':shared._sha(old_model)},
            'unit':'log10(mL/min/kg)','input_fingerprint':fingerprint})
        model_receipt = {'path':str(model_path.relative_to(root)),'sha256':shared._sha(model_path)}
    result = {'version':VERSION,'unit':'mL/min/kg','predictions':predictions,'selected_using_training_only':True,
        'selection':selection,'oof_metrics':selection['scores'],'source_refs':refs,'code_pins':code,
        'model_receipt':model_receipt,'input_fingerprint':fingerprint,'created_at':previous._now(),
        'elapsed_seconds':time.monotonic()-started,
        'audit':{'target_labels_used':False,'target_observation_files_opened':False,'training_structure_overlap':False,
            'training_generic_scaffold_overlap':False,'clinical_results_used_for_selection':False,
            'training_structures':membership['train_count'],'cl_training_labels':len(indices),'training_indices':indices,
            'oof_fold_ids':folds.tolist(),'observed_training_log10':y.tolist(),
            'candidate_oof_log10':{k:v.tolist() for k,v in candidates.items()},'selection_locked_at':state['selection_locked_at'],
            'training_features':feature_audit,'target_count':len(cohort),
            'consistency_projection_inherited_limitation':pa.get('consistency_projection_inherited_limitation')},
        'interpretation':'Training development selection under extreme-CL gates, not external clinical validation; no clinical or per-drug model switch'}
    previous._verify(root,refs); previous._verify(root,code)
    result['result_sha256'] = shared._digest(result); rich._atomic_json(output/'predictions.json',result)
    emit('complete',selected=selected,result_path=str((output/'predictions.json').relative_to(root)),
         result_sha256=shared._sha(output/'predictions.json'),elapsed_seconds=time.monotonic()-started)
    return result


def make_estimator(recipe, jobs, device):
    if recipe['kind']=='krr': return similarity.TanimotoKernelRidge(alpha=recipe['alpha'])
    if recipe['kind']=='svr': return SVR(kernel='precomputed',C=recipe['C'],epsilon=recipe['epsilon'])
    return rich.make_estimator(recipe['method'],shared.SEED,jobs,device)
