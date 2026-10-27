"""Development-only selection and tail-balanced calibration of the frozen flow-exchange PBPK model."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from pbpk_kb.blueprints.development_curve_blueprint import _training_document
from pbpk_kb.blueprints.pk_anchored_blueprint import _drug_signature
from pbpk_kb.blueprints.evidence_preserving_calibration import _exclusive
from pbpk_kb.blueprint_learning.native_calibration import _embedded_pins, _verify
from .axes_calibration import _base, _read, _ref, _artifacts

from . import robust_exchange_calibration as base
from .perfusion_calibration import clearance_inputs, prediction_grid, _physiology

VERSION = 'tail-development-exchange-calibration-1'
PARAMETERS = {name: base.PARAMETERS['evidence_flexible']
              for name in ('tail_balanced', 'development_guarded')}
CONFIG = {
    **deepcopy(base.CONFIG), 'starts': 16, 'cv_starts': 8,
    'max_evaluations_per_start': 400,
    'physical_family': 'evidence_flexible',
    'exchange': 'PS_i = Q_i * shared_scale; unchanged frozen flow-exchange equations',
    'exchange_alpha': 0.,
    'tail': {'postinfusion': 'time_h > infusion_end_h',
        'tail_count': 'max(3,ceil(number_of_postinfusion_points/3))',
        'weights': 'body logMSE/2 + tail logMSE/2',
        'fallback': 'Uniform logMSE if fewer than three postinfusion points or no body'},
    'cv': {'rf_penalty_weights': [.01, .1], 'observed_penalty_weights': [.1],
        'holdout_count': 'min(3,max(1,n//4),n-6)',
        'minimum_training_points': 6,
        'folds': ['Final k time points',
            'k non-tail points nearest equally spaced interior log-time strata; earliest and nearest infusion-end protected'],
        'initialization': 'Eight fixed starts generated independently for every fold; no fitted warm starts',
        'score': 'Equal mean of the two held-out logMSEs, no penalty in validation score',
        'tie': 'Exact score tie prefers stronger clearance prior; no target outcomes',
        'unsupported': 'Fewer than seven distinct development times raises an explicit error; no full-data fallback'},
    'loss': 'tail_balanced: phase-weighted logMSE; development_guarded: uniform logMSE; unchanged K/PS penalties',
    'prior_selection': 'tail: evidence-only; guarded: development-only CV within each formula',
}


def _code_pins(root):
    pins = base._code_pins(root)
    path = Path(root)/'src/pbpk_kb/blueprint_policy/tail_exchange_calibration.py'
    pins[str(path.relative_to(root))] = sha256(path)
    return pins


def clearance_policy(request):
    return base.clearance_policy(request)


def parameter_starts(variant, request, *, count=None):
    keys = PARAMETERS[variant]
    bounds = np.asarray([clearance_policy(request)['scale_bounds'],
        CONFIG['bounds']['partition_scales'], CONFIG['bounds']['exchange_scale']], float)
    lower, upper = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    count = CONFIG['starts'] if count is None else count
    rng = np.random.default_rng(CONFIG['seed'])
    return keys, lower, upper, np.vstack([np.zeros(3), rng.uniform(lower, upper, size=(count-1, 3))])


def _simulate(physiology, request, times_h, variant, keys, x):
    if variant not in PARAMETERS:
        raise ValueError('Unknown tail/development calibration variant')
    return base._simulate(physiology, request, times_h, 'evidence_flexible', keys, x)


def _times(times):
    values = np.asarray(times, float)
    if (values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(np.diff(values) <= 0)):
        raise ValueError('Development times must be distinct, finite and increasing')
    return values


def tail_weights(times_h, infusion_h):
    times = _times(times_h); n = len(times)
    post = np.flatnonzero(times > infusion_h)
    count = max(3, math.ceil(len(post)/3))
    tail = post[-count:].tolist() if len(post) >= 3 else []
    body = [i for i in range(n) if i not in tail]
    supported = bool(tail and body)
    weights = np.full(n, 1/n)
    if supported:
        weights[body] = .5/len(body); weights[tail] = .5/len(tail)
    return {'weights': weights.tolist(), 'body_indices': body, 'tail_indices': tail,
        'supported': supported, 'fallback': None if supported else 'uniform_logMSE_insufficient_phase_support',
        'selection_uses_concentrations': False}


def development_folds(times_h, infusion_h):
    times = _times(times_h); n = len(times)
    if n < 7:
        raise ValueError('Development CV requires at least seven distinct times; no full-data fallback')
    k = min(3, max(1, n//4), n-6)
    tail = list(range(n-k, n))
    peak = min(range(n), key=lambda i: (abs(times[i]-infusion_h), times[i]))
    protected = {0, peak}
    candidates = [i for i in range(n-k) if i not in protected]
    if len(candidates) < k:
        raise ValueError('Insufficient non-tail support for the fixed CV strata')
    # A positive scale avoids undefined logarithms at a possible predose point.
    positive = times[times > 0]
    if not len(positive):
        raise ValueError('Positive development sampling times are required')
    lt = np.log(np.maximum(times, positive[0]/2))
    strata = np.linspace(lt[candidates[0]], lt[candidates[-1]], k+2)[1:-1]
    selected = []
    for center in strata:
        selected.append(min((i for i in candidates if i not in selected),
                            key=lambda i: (abs(lt[i]-center), times[i])))
    folds = []
    for name, held in (('terminal_block', tail), ('interior_time_strata', sorted(selected))):
        train = [i for i in range(n) if i not in held]
        if len(train) < 6:
            raise ValueError('A development fold has fewer than six training points')
        folds.append({'name': name, 'training_indices': train, 'heldout_indices': held,
            'protected_interior_indices': sorted(protected), 'selection_uses_concentrations': False})
    return folds


def _plan(variant, request, observed):
    times = [r['time_min']/60 for r in observed['observations']]
    infusion = request['dosing']['infusion_minutes']/60
    policy = clearance_policy(request)
    return {'tail': tail_weights(times, infusion),
        'folds': development_folds(times, infusion) if variant == 'development_guarded' else [],
        'prior_weights': (CONFIG['cv']['rf_penalty_weights'] if policy['rule'] == 'qualified_rf'
            else CONFIG['cv']['observed_penalty_weights']) if variant == 'development_guarded'
            else [policy['penalty_weight']],
        'development_points': len(times), 'target_observations_used': False}


def _best(rows):
    eligible = [r for r in rows if r.get('eligible')]
    return min(eligible, key=lambda r: (r['objective'], r['index'])) if eligible else None


def _cv_selection(stages, plan):
    candidates = []
    for weight in plan['prior_weights']:
        scores = [stages[f'cv_{weight:g}_{i}']['heldout_score'] for i in range(len(plan['folds']))]
        valid = all(x is not None and math.isfinite(x) for x in scores)
        candidates.append({'penalty_weight': weight, 'fold_scores': scores,
            'eligible': valid, 'score': float(np.mean(scores)) if valid else None})
    eligible = [x for x in candidates if x['eligible']]
    selected = min(eligible, key=lambda x: (x['score'], -x['penalty_weight'])) if eligible else None
    return {'candidates': candidates, 'selected': selected, 'target_observations_used': False}


def _validated(mapping, path, root):
    fixed = _read(path.parent/'input.json'); progress = _read(path.parent/'progress.json')
    if (mapping['version'] != VERSION or fixed['version'] != VERSION or
            mapping['input_sha256'] != sha256(path.parent/'input.json') or
            mapping['optimizer_config'] != CONFIG or fixed['optimizer_config'] != CONFIG):
        raise ValueError('Locked tail/development fit changed')
    for key in ('source_pins', 'code_pins'):
        if mapping[key] != fixed[key]: raise ValueError('Locked fit dependency records changed')
        _verify(mapping[key], root)
    _verify(mapping['retained_artifact_pins'], root)
    observed = _training_document(fixed['training_source'], fixed['request'], root)
    plan = _plan(fixed['variant'], fixed['request'], observed)
    keys, lower, upper, full_initial = parameter_starts(fixed['variant'], fixed['request'])
    if (fixed['development_plan'] != plan or mapping['development_plan'] != plan or
            fixed['clearance_input'] != clearance_inputs(fixed['request']) or
            fixed['clearance_prior_policy'] != clearance_policy(fixed['request']) or
            mapping['variant'] != fixed['variant'] or mapping['drug_signature'] != _drug_signature(fixed['request']) or
            mapping['training_source'] != fixed['training_source'] or
            mapping['reference_human'] != fixed['request']['human'] or mapping['reference_dosing'] != fixed['request']['dosing'] or
            mapping['construction_partition_method'] != fixed['request']['partition_method'] or
            mapping['clearance_input'] != fixed['clearance_input'] or mapping['active_parameters'] != list(keys)):
        raise ValueError('Locked mapping differs from development records or inputs')
    expected_bounds = {'lower': lower.tolist(), 'upper': upper.tolist()}
    if fixed['effective_log_bounds'] != expected_bounds or mapping['effective_log_bounds'] != expected_bounds:
        raise ValueError('Locked clearance bounds changed')
    if (progress['status'] != 'locked' or mapping['stages'] != progress['stages'] or
            any(mapping[k] != progress[k] for k in ('linear_solver_calls', 'failed_evaluation_count', 'cost_count_exact'))):
        raise ValueError('Locked mapping differs from development records')
    expected_stages = ({f'cv_{w:g}_{i}' for w in plan['prior_weights'] for i in range(len(plan['folds']))})
    selection = _cv_selection(progress['stages'], plan) if plan['folds'] else None
    if selection != mapping['cv_selection']:
        raise ValueError('Development selection changed')
    selected_weight = (selection['selected']['penalty_weight'] if selection and selection['selected']
        else None if selection else fixed['clearance_prior_policy']['penalty_weight'])
    if selected_weight is not None: expected_stages.add('full')
    if set(progress['stages']) != expected_stages:
        raise ValueError('Incomplete or unrecognized optimization stages')
    for name, stage in progress['stages'].items():
        count = CONFIG['starts'] if name == 'full' else CONFIG['cv_starts']
        initial = full_initial if name == 'full' else parameter_starts(fixed['variant'], fixed['request'], count=count)[3]
        if stage['status'] != 'complete' or sorted(x['index'] for x in stage['starts']) != list(range(count)):
            raise ValueError('Incomplete optimization starts')
        for row in stage['starts']:
            x = np.asarray(row['log_parameters'])
            if row['initial_log_parameters'] != initial[row['index']].tolist() or np.any(x < lower-1e-12) or np.any(x > upper+1e-12):
                raise ValueError('Stage used an invalid or fitted warm start')
        if stage['decision'] != _best(stage['starts']): raise ValueError('Stage selection changed')
        if name == 'full':
            train, held = list(range(plan['development_points'])), []
            weight = selected_weight
            weights = plan['tail']['weights'] if fixed['variant'] == 'tail_balanced' else [1/len(train)]*len(train)
        else:
            _, w, i = name.split('_'); fold = plan['folds'][int(i)]
            train, held = fold['training_indices'], fold['heldout_indices']; weight = float(w)
            weights = [1/len(train)]*len(train)
        if (stage['training_indices'] != train or stage['heldout_indices'] != held or stage['penalty_weight'] != weight
                or stage['data_weights'] != weights):
            raise ValueError('Fold data, loss or regularization changed')
        if held and stage['heldout_score'] is not None:
            errors = np.asarray(stage['heldout_log_errors'], float)
            if len(errors) != len(held) or not math.isclose(stage['heldout_score'], float(np.mean(errors**2)), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('Held-out development score changed')
    rows = progress['stages'].get('full', {}).get('starts', [])
    best = _best(rows); score = selection['selected']['score'] if selection and selection['selected'] else best['objective'] if best else None
    prior = {**fixed['clearance_prior_policy'], 'penalty_weight': selected_weight}
    if (mapping['starts'] != rows or mapping['decision'] != best or mapping['eligible'] != (best is not None)
            or mapping['formula_selection_score'] != score or mapping['clearance_prior_policy'] != prior):
        raise ValueError('Locked final selection differs from development records')


def fit_mapping(*, reference_request, training_source, output_dir, project_root, variant, before_trial=None):
    root = Path(project_root).resolve(strict=True); output = inside(output_dir, root)
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode='json', exclude_none=True)
    prior = clearance_policy(req); keys, lower, upper, _ = parameter_starts(variant, req)
    observed = _training_document(training_source, req, root)
    if (observed['human'] != req['human'] or observed['dosing'] != req['dosing'] or
            req['dosing']['route'] != 'intravenous' or float(req['dosing'].get('start_h', 0)) != 0):
        raise ValueError('Development conditions differ or dose does not start at zero')
    plan = _plan(variant, req, observed); cl = clearance_inputs(req)
    pins, code = _embedded_pins([req, training_source, observed], root), _code_pins(root)
    _verify(pins, root); _verify(code, root)
    fixed = {'version': VERSION, 'request': req, 'training_source': deepcopy(training_source),
        'variant': variant, 'optimizer_config': CONFIG, 'source_pins': pins, 'code_pins': code,
        'clearance_input': cl, 'clearance_prior_policy': prior, 'external_warm_start_used': False,
        'development_plan': plan, 'effective_log_bounds': {'lower': lower.tolist(), 'upper': upper.tolist()}}
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        input_path, mapping_path, progress_path = output/'input.json', output/'mapping_locked.json', output/'progress.json'
        if input_path.exists():
            if _read(input_path) != fixed: raise ValueError('Resume request, method, budget or dependencies changed')
        elif any(x.name != '.calibration.lock' for x in output.iterdir()):
            raise ValueError('Unrecognized existing tail/development fit folder')
        else: save(input_path, fixed)
        if mapping_path.exists():
            result = _read(mapping_path); _validated(result, mapping_path, root)
            return {**result, 'mapping': _ref(mapping_path, root), 'actual_new_calls': 0}
        started = time.monotonic(); physiology = _physiology(req, output, root, before_trial)
        mw = next(float(e['value']) for e in req['evidence'] if e['parameter'] == 'molecular_weight')
        times = np.array([r['time_min']/60 for r in observed['observations']], float)
        values = np.array([r['concentration_umol_per_l']*mw/1000 for r in observed['observations']], float)
        if np.any(values <= 0) or not np.isfinite(values).all(): raise ValueError('Positive development values required')
        progress = _read(progress_path) if progress_path.exists() else {'stages': {}, 'linear_solver_calls': 0,
            'failed_evaluation_count': 0, 'wall_seconds': 0., 'cost_count_exact': True}
        if progress.get('incomplete_start') is not None:
            progress['cost_count_exact'] = False
            progress.setdefault('interrupted_starts', []).append(progress['incomplete_start'])
        last_check = [0.]

        def run_stage(name, train, held, weight, data_weights, count):
            stage = progress['stages'].setdefault(name, {'starts': [], 'training_indices': train,
                'heldout_indices': held, 'penalty_weight': weight, 'data_weights': data_weights})
            if stage.get('status') == 'complete': return
            _, _, _, initials = parameter_starts(variant, req, count=count)
            target = values[train]; weights = np.asarray(data_weights); counters = {'calls': 0, 'failed': 0}
            def simulate_at(x, indices):
                stamp = time.monotonic()
                if before_trial and stamp-last_check[0] >= 2:
                    before_trial(); last_check[0] = stamp
                counters['calls'] += 1
                predicted = np.asarray(_simulate(physiology, req, times[indices], variant, keys, x)['concentration_mg_l'])
                if predicted.shape != (len(indices),) or np.any(predicted <= 0) or not np.isfinite(predicted).all():
                    raise ValueError('Invalid positive fitted concentration')
                return predicted
            def residual(x):
                try:
                    errors = np.log(simulate_at(x, train))-np.log(target)
                    penalty = [math.sqrt(weight)*x[0]/math.log(2), .1*x[1]/math.log(10), .1*x[2]/math.log(100)]
                    return np.r_[errors*np.sqrt(weights), penalty]
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    counters['failed'] += 1
                    if len(progress.setdefault('diagnostic_failures', [])) < 20: progress['diagnostic_failures'].append(str(exc))
                    return np.full(len(train)+3, 1e6)
            for index, initial in enumerate(initials):
                if any(row['index'] == index for row in stage['starts']): continue
                progress.update(status='running', incomplete_start={'stage': name, 'index': index})
                save(progress_path, progress); begin = time.monotonic(); old_calls, old_failed = counters['calls'], counters['failed']
                try:
                    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=CONFIG['max_evaluations_per_start'],
                        ftol=1e-8, xtol=1e-8, gtol=1e-8)
                    res = residual(fit.x); objective = float(res@res); eligible = math.isfinite(objective) and objective < 1e10
                    raw = res[:len(train)]/np.sqrt(weights)
                    row = {'index': index, 'initial_log_parameters': initial.tolist(), 'log_parameters': fit.x.tolist(),
                        'parameters': dict(zip(keys, np.exp(fit.x).tolist())), 'objective': objective, 'eligible': eligible,
                        'status': 'converged' if fit.success else 'budget_exhausted', 'optimizer_message': fit.message,
                        'nfev': int(fit.nfev), 'data_jacobian_singular_values': np.linalg.svd(fit.jac[:len(train)], compute_uv=False).tolist(),
                        'jacobian_weighting': 'training data weights, excluding regularization rows',
                        'data_only_ln_rmse': float(np.sqrt(np.mean(raw**2))), 'weighted_data_loss': float(res[:len(train)]@res[:len(train)]),
                        'curve_metrics': {'gmfe': math.exp(float(np.mean(abs(raw)))) if eligible else None,
                            'ln_rmse': float(np.sqrt(np.mean(raw**2))), 'n': len(train)},
                        'linear_solver_calls': counters['calls']-old_calls, 'failed_evaluation_count': counters['failed']-old_failed,
                        'elapsed_seconds': time.monotonic()-begin}
                except Exception:
                    progress['linear_solver_calls'] += counters['calls']-old_calls
                    progress['failed_evaluation_count'] += counters['failed']-old_failed
                    progress['wall_seconds'] += time.monotonic()-begin; save(progress_path, progress); raise
                stage['starts'].append(row)
                progress['linear_solver_calls'] += row['linear_solver_calls']; progress['failed_evaluation_count'] += row['failed_evaluation_count']
                progress['wall_seconds'] += row['elapsed_seconds']; progress['incomplete_start'] = None; save(progress_path, progress)
            stage['decision'] = _best(stage['starts']); stage['heldout_score'] = None
            if held and stage['decision']:
                begin = time.monotonic(); old_calls, old_failed = counters['calls'], counters['failed']
                progress['incomplete_start'] = {'stage': name, 'index': 'heldout'}; save(progress_path, progress)
                try:
                    predicted = simulate_at(stage['decision']['log_parameters'], held)
                    errors = np.log(predicted/values[held]); stage['heldout_log_errors'] = errors.tolist()
                    stage['heldout_score'] = float(np.mean(errors**2))
                    stage['heldout_predicted_mg_l'] = predicted.tolist()
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    counters['failed'] += 1; stage['heldout_failure'] = str(exc)
                finally:
                    progress['linear_solver_calls'] += counters['calls']-old_calls
                    progress['failed_evaluation_count'] += counters['failed']-old_failed
                    progress['wall_seconds'] += time.monotonic()-begin
            stage['status'] = 'complete'; progress['incomplete_start'] = None; save(progress_path, progress)

        if plan['folds']:
            for weight in plan['prior_weights']:
                for i, fold in enumerate(plan['folds']):
                    train, held = fold['training_indices'], fold['heldout_indices']
                    run_stage(f'cv_{weight:g}_{i}', train, held, weight, [1/len(train)]*len(train), CONFIG['cv_starts'])
            selection = _cv_selection(progress['stages'], plan)
            chosen_weight = selection['selected']['penalty_weight'] if selection['selected'] else None
        else:
            selection = None; chosen_weight = prior['penalty_weight']
        if chosen_weight is not None:
            weights = plan['tail']['weights'] if variant == 'tail_balanced' else [1/len(times)]*len(times)
            run_stage('full', list(range(len(times))), [], chosen_weight, weights, CONFIG['starts'])
        rows = progress['stages'].get('full', {}).get('starts', []); best = _best(rows)
        score = selection['selected']['score'] if selection and selection['selected'] else best['objective'] if best else None
        progress['status'] = 'locked'; save(progress_path, progress)
        _verify(pins, root); _verify(code, root); retained = _artifacts(output, root)
        result = {'version': VERSION, 'variant': variant, 'physical_family': 'evidence_flexible', 'status': 'fitted' if best else 'failed',
            'eligible': best is not None, 'decision': best, 'starts': rows, 'stages': progress['stages'], 'cv_selection': selection,
            'formula_selection_score': score, 'formula_selection_basis': 'development_CV_heldout_logMSE' if plan['folds'] else 'tail_balanced_penalized_objective',
            'source_pins': pins, 'code_pins': code, 'retained_artifact_pins': retained, 'input_sha256': sha256(input_path),
            'optimizer_config': CONFIG, 'drug_signature': _drug_signature(req), 'construction_partition_method': req['partition_method'],
            'training_source': deepcopy(training_source), 'clearance_input': cl, 'reference_human': req['human'], 'reference_dosing': req['dosing'],
            'active_parameters': list(keys), 'clearance_prior_policy': {**prior, 'penalty_weight': chosen_weight},
            'effective_log_bounds': fixed['effective_log_bounds'], 'development_plan': plan, 'total_native_calls': 1, 'actual_solver_calls': 1,
            'linear_solver_calls': progress['linear_solver_calls'], 'cost_count_exact': progress['cost_count_exact'],
            'elapsed_seconds': max(progress['wall_seconds'], time.monotonic()-started),
            'failed_evaluation_count': progress['failed_evaluation_count'], 'development_curve_used': True, 'comparison_curve_used': False,
            'limitations': ['Previously viewed development drugs, not blind validation.',
                'Within-study development CV cannot guarantee cross-study generalization.',
                'Tail weighting changes fitting priorities; it does not establish clinical terminal identifiability.',
                'Unchanged effective flow-exchange physics and fixed net renal input; no tissue concentration validation.']}
        save(mapping_path, result); _validated(result, mapping_path, root)
        return {**result, 'mapping': _ref(mapping_path, root), 'actual_new_calls': 1}


def predict_mapping(*, request, mapping_path, output_dir, project_root, before_trial=None):
    from pbpk_kb.blueprints.extended_prediction import summarize_extended_prediction
    root = Path(project_root).resolve(strict=True); output = inside(output_dir, root)
    path = inside(mapping_path, root, exists=True); mapping = _read(path)
    _validated(mapping, path, root)
    req = EvidenceBlueprintRequest.model_validate(request).model_dump(mode='json', exclude_none=True)
    if not mapping['eligible'] or _drug_signature(req) != mapping['drug_signature']:
        raise ValueError('Prediction drug inputs differ from locked finite-exchange fit')
    if float(req['dosing'].get('start_h', 0)) != 0:
        raise ValueError('This pilot requires dose time zero')
    pins = _embedded_pins(req, root); _verify(pins, root)
    construction = {'version': VERSION, 'mapping': _ref(path, root), 'request': req,
        'parameters': mapping['decision']['parameters'], 'target_outcomes_used': False}
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        receipt = output/'completion.json'
        if receipt.exists():
            previous = _read(receipt)
            if previous['construction'] != construction:
                raise ValueError('Prediction inputs changed')
            _verify(previous['artifact_pins'], root)
            return {**_read(output/'result.json'), 'actual_new_calls': 0}
        if any(p.name != '.calibration.lock' for p in output.iterdir()):
            raise ValueError('Partial prediction retained; use explicit retry directory')
        save(output/'request.json', req)
        started = time.monotonic()
        physiology = _physiology(req, output, root, before_trial)
        if before_trial:
            before_trial()
        times = prediction_grid(req)
        variant = mapping['variant']; keys = PARAMETERS[variant]
        result = {'status': 'failed', 'actual_new_calls': 1, 'actual_native_solver_calls': 1,
            'linear_solver_calls': 1, 'parameters_fitted_to_development_only': True, 'comparison_curve_used': False}
        try:
            predicted = _simulate(physiology, req, times, variant, keys, mapping['decision']['log_parameters'])
            mw = next(float(r['value']) for r in req['evidence'] if r['parameter'] == 'molecular_weight')
            curve = {'time_min': (times*60).tolist(), 'concentration': (np.asarray(predicted['concentration_mg_l'])*1000/mw).tolist(),
                'time_unit': 'min', 'unit': 'umol/l', 'observation': predicted['observation']}
            # Validate endpoint/curve arithmetic before publishing the prediction.
            endpoints = summarize_extended_prediction(curve, req)
            endpoints.update(development_curve_used=True, comparison_observations_used=False,
                endpoint_estimator_uses_observations=False, parameters_fitted=True)
            save(output/'prediction_locked.json', curve)
            save(output/'endpoints.json', endpoints)
            result.update(status='predicted', prediction=_ref(output/'prediction_locked.json', root),
                mass_balance=predicted['mass_balance'], observation=predicted['observation'],
                effective_total_clearance_l_h=predicted['effective_total_clearance_l_h'])
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            result.update(failure_category='scientific_failed', reason=str(exc))
        _verify(pins, root); _validated(mapping, path, root)
        result['elapsed_seconds'] = time.monotonic()-started
        save(output/'result.json', result)
        save(receipt, {'construction': construction, 'artifact_pins': _artifacts(output, root)})
        return result


def cleanup_unselected(*, output_dir, project_root):
    """No per-evaluation files are written; retain compact fit and source model."""
    root = Path(project_root).resolve(strict=True); folder = inside(output_dir, root, exists=True)
    _validated(_read(folder/'mapping_locked.json'), folder/'mapping_locked.json', root)
    return {'removed_count': 0, 'per_trial_artifacts_generated': False,
            'compact_start_scores_preserved': CONFIG['starts'], 'selected_model_and_sources_preserved': True}
