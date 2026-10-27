"""Evidence-capability pilot: choose calibratable axes before fitting.

Isolated adapter of the frozen transport calibrator; the original source is
retained because it is hash-bound to completed experiments. Structural
equations, bounds, starts and penalties are unchanged. Selection is a local
engineering heuristic, not proof of global identifiability.

Frozen-input calibration of finite vascular–extravascular PBPK exchange.

The native model supplies human physiology and uncalibrated drug partitioning.
Transport equations and the aggregate systemic clearance location are replaced,
not silently interpreted as changes to an organ's intrinsic clearance.
"""
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

VERSION = 'capability-adaptive-flow-exchange-1'
PARAMETERS = {'flow_exchange': ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale')}
CONFIG = {
    'starts': 16, 'max_evaluations_per_start': 400, 'seed': 20260930,
    'initialization': 'One identity and fifteen deterministic log-uniform starts, independent of observations',
    'bounds': {'clearance_scale': [.25, 4.], 'partition_scales': [.1, 10.], 'exchange_scale': [.01, 100.]},
    'loss': 'mean(log(P/O)^2)+0.1*(log(CLscale)/log(2))^2+0.01*(log(Kscale)/log(10))^2+0.01*(log(PSscale)/log(100))^2',
    'solver': 'Linear vascular–extravascular exchange equations, double precision',
    'clearance': 'Same aggregate venous plasma sink as perfusion comparison; fixed renal input and scaled nonrenal component',
    'observation': 'Same native peripheral venous weights, applied to vascular outflow',
    'exchange': 'PS_i = organ effective blood flow times one shared scale; liver uses total portal plus arterial flow',
    'exchange_interpretation': 'Predetermined effective exchange hypothesis, not measured cellular permeability',
    'extravascular_scaling_only': True, 'target_observations_used': False,
    'axis_selection': {'minimum_points_per_axis': 2, 'relative_singular_threshold': 1e-3,
        'absolute_column_norm_threshold': 1e-6, 'central_difference_log_step': 1e-3,
        'point': 'identity', 'priority': ['clearance_scale', 'partition_scale', 'bidirectional_exchange_scale'],
        'scaling': 'd(log concentration)/d(log scale)/sqrt(n); data rows only',
        'frozen_scale': 1., 'target_values_used': False},
}



def _code_pins(root):
    from .perfusion_calibration import _code_pins as original
    pins = original(root)
    for name in ('transport_calibration.py', 'permeability_model.py', 'capability_calibration.py'):
        path = root / 'src/pbpk_kb/blueprint_policy' / name
        pins[str(path.relative_to(root))] = sha256(path)
    return pins


from .perfusion_calibration import clearance_inputs, prediction_grid, _physiology


def parameter_starts(variant):
    keys = PARAMETERS[variant]
    bounds = np.array([CONFIG['bounds']['clearance_scale']] +
                      [CONFIG['bounds']['partition_scales'], CONFIG['bounds']['exchange_scale']], float)
    lower, upper = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    rng = np.random.default_rng(CONFIG['seed'])
    return keys, lower, upper, np.vstack([np.zeros(len(keys)),
        rng.uniform(lower, upper, size=(CONFIG['starts']-1, len(keys)))])


def _simulate(physiology, request, times_h, variant, keys, x):
    from .permeability_model import simulate
    cl = clearance_inputs(request)
    return simulate(physiology, times_h=np.asarray(times_h, float),
        dose_mg=float(request['dosing']['dose_mg']),
        infusion_duration_h=float(request['dosing']['infusion_minutes'])/60,
        total_clearance_l_h=cl['total_clearance_l_h'], renal_clearance_l_h=cl['renal_clearance_l_h'],
        family='exchange',
        log_parameters=dict(zip(keys, np.asarray(x, float).tolist())), observation='auto')


def select_active(data_jacobian, n):
    """Greedy, ordered local information screen; no outcome or scoring input."""
    jac = np.asarray(data_jacobian, float)
    keys = PARAMETERS['flow_exchange']
    if type(n) is not int or n < 1 or jac.shape != (n, 3) or not np.isfinite(jac).all():
        raise ValueError('Expected a finite n-by-three data-only Jacobian')
    cfg = CONFIG['axis_selection']
    active, decisions = [], []
    for index, key in enumerate(keys):
        candidate = active + [index]
        norm = float(np.linalg.norm(jac[:, index]))
        singular = np.linalg.svd(jac[:, candidate], compute_uv=False)
        ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.
        if n < cfg['minimum_points_per_axis'] * len(candidate):
            reason = 'insufficient_observations_per_axis'
        elif norm < cfg['absolute_column_norm_threshold']:
            reason = 'negligible_local_sensitivity'
        elif ratio < cfg['relative_singular_threshold']:
            reason = 'locally_dependent_axes'
        else:
            reason = 'selected'
            active.append(index)
        decisions.append({'parameter': key, 'reason': reason, 'column_norm': norm,
            'candidate_singular_values': singular.tolist(), 'candidate_singular_ratio': ratio})
    return {'active_indices': active, 'active_parameters': [keys[i] for i in active],
        'frozen_parameters': [k for i, k in enumerate(keys) if i not in active],
        'decisions': decisions, 'config': deepcopy(cfg), 'observation_count': n,
        'interpretation': 'Local identity-point engineering screen, not global identifiability proof'}


def diagnose_axes(physiology, request, times, before_trial=None):
    keys = PARAMETERS['flow_exchange']
    step = CONFIG['axis_selection']['central_difference_log_step']
    columns = []
    for i in range(3):
        if before_trial:
            before_trial()
        x = np.zeros(3); x[i] = step
        plus = np.asarray(_simulate(physiology, request, times, 'flow_exchange', keys, x)['concentration_mg_l'])
        minus = np.asarray(_simulate(physiology, request, times, 'flow_exchange', keys, -x)['concentration_mg_l'])
        if np.any(plus <= 0) or np.any(minus <= 0):
            raise ValueError('Local screen requires positive model concentrations')
        columns.append((np.log(plus)-np.log(minus))/(2*step*math.sqrt(len(times))))
    jac = np.column_stack(columns)
    result = select_active(jac, len(times))
    result['data_jacobian'] = jac.tolist()
    result['linear_solver_calls'] = 6
    return result


def _validated(mapping, path, root):
    fixed = _read(path.parent/'input.json')
    if (mapping['version'] != VERSION or mapping['input_sha256'] != sha256(path.parent/'input.json')
            or mapping['variant'] != fixed['variant'] or mapping['optimizer_config'] != CONFIG
            or mapping['drug_signature'] != _drug_signature(fixed['request'])
            or mapping['source_pins'] != fixed['source_pins'] or mapping['code_pins'] != fixed['code_pins']):
        raise ValueError('Locked finite-exchange fit changed')
    _verify(mapping['source_pins'], root)
    _verify(mapping['code_pins'], root)
    _verify(mapping['retained_artifact_pins'], root)
    progress = _read(path.parent/'progress.json')
    if (progress['status'] != 'locked' or mapping['starts'] != progress['starts']
            or any(mapping[k] != progress[k] for k in ('linear_solver_calls', 'failed_evaluation_count', 'cost_count_exact'))
            or mapping['construction_partition_method'] != fixed['request']['partition_method']
            or mapping['training_source'] != fixed['training_source']
            or mapping['reference_human'] != fixed['request']['human']
            or mapping['reference_dosing'] != fixed['request']['dosing']
            or mapping['clearance_input'] != fixed['clearance_input']
            or mapping['axis_selection'] != progress['axis_selection']
            or mapping['active_parameters'] != mapping['axis_selection']['active_parameters']):
        raise ValueError('Locked mapping differs from its development records or input metadata')
    selected = select_active(np.asarray(mapping['axis_selection']['data_jacobian']), mapping['axis_selection']['observation_count'])
    if any(mapping['axis_selection'][k] != selected[k] for k in selected):
        raise ValueError('Locked axis selection changed')
    inactive = set(range(3)) - set(selected['active_indices'])
    if any(any(row['log_parameters'][i] != 0. for i in inactive) for row in mapping['starts']):
        raise ValueError('A frozen multiplier was changed')
    eligible = [row for row in mapping['starts'] if row.get('eligible')]
    best = min(eligible, key=lambda row: (row['objective'], row['index'])) if eligible else None
    if mapping['decision'] != best or mapping['eligible'] != (best is not None):
        raise ValueError('Locked selection differs from recorded development starts')


def fit_mapping(*, reference_request, training_source, output_dir, project_root,
                variant, before_trial=None):
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    keys, lower, upper, starts = parameter_starts(variant)
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode='json', exclude_none=True)
    observed = _training_document(training_source, req, root)
    if (observed['human'] != req['human'] or observed['dosing'] != req['dosing']
            or req['dosing']['route'] != 'intravenous' or float(req['dosing'].get('start_h', 0)) != 0):
        raise ValueError('Development conditions differ or administration does not start at zero')
    cl = clearance_inputs(req)
    pins, code = _embedded_pins([req, training_source, observed], root), _code_pins(root)
    _verify(pins, root); _verify(code, root)
    fixed = {'version': VERSION, 'request': req, 'training_source': deepcopy(training_source),
        'variant': variant, 'optimizer_config': CONFIG, 'source_pins': pins, 'code_pins': code,
        'clearance_input': cl, 'external_warm_start_used': False}
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        input_path, mapping_path = output/'input.json', output/'mapping_locked.json'
        if input_path.exists():
            if _read(input_path) != fixed:
                raise ValueError('Resume request, method, budget or dependencies changed')
        elif any(p.name != '.calibration.lock' for p in output.iterdir()):
            raise ValueError('Unrecognized existing finite-exchange fit folder')
        else:
            save(input_path, fixed)
        if mapping_path.exists():
            result = _read(mapping_path); _validated(result, mapping_path, root)
            return {**result, 'mapping': _ref(mapping_path, root), 'actual_new_calls': 0}
        started = time.monotonic()
        physiology = _physiology(req, output, root, before_trial)
        mw = next(float(r['value']) for r in req['evidence'] if r['parameter'] == 'molecular_weight')
        times = np.array([r['time_min']/60 for r in observed['observations']], float)
        values = np.array([r['concentration_umol_per_l'] * mw/1000 for r in observed['observations']], float)
        if np.any(values <= 0) or not np.isfinite(values).all():
            raise ValueError('Log concentration objective requires positive development values')
        progress_path = output/'progress.json'
        progress = _read(progress_path) if progress_path.exists() else {'starts': [], 'linear_solver_calls': 0,
            'failed_evaluation_count': 0, 'wall_seconds': 0., 'cost_count_exact': True}
        if progress.get('incomplete_start') is not None:
            # A hard shutdown can lose an unfinished start's in-memory counter.
            progress['cost_count_exact'] = False
            progress.setdefault('interrupted_starts', []).append(progress['incomplete_start'])
        if 'axis_selection' not in progress:
            progress['axis_selection'] = diagnose_axes(physiology, req, times, before_trial)
            progress['linear_solver_calls'] += progress['axis_selection']['linear_solver_calls']
            save(progress_path, progress)
        axes = progress['axis_selection']['active_indices']
        def expand(z):
            x = np.zeros(3)
            x[axes] = np.asarray(z, float)
            return x
        counts = {'calls': 0, 'failed': 0}
        failures = []
        last_check = 0.
        def residual(z):
            nonlocal last_check
            x = expand(z)
            # The shared controller persists state. Avoid one disk rewrite per
            # cheap linear solve while still checking pause every two seconds.
            stamp = time.monotonic()
            if before_trial and stamp-last_check >= 2:
                before_trial()
                last_check = stamp
            counts['calls'] += 1
            try:
                prediction = _simulate(physiology, req, times, variant, keys, x)
                p = np.asarray(prediction['concentration_mg_l'], float)
                if p.shape != values.shape or not np.isfinite(p).all() or np.any(p <= 0):
                    raise ValueError('Invalid positive fitted concentration')
                data = (np.log(p)-np.log(values))/math.sqrt(len(values))
                penalty = np.r_[math.sqrt(.1)*x[0]/math.log(2), .1*x[1]/math.log(10), .1*x[2]/math.log(100)]
                return np.r_[data, penalty]
            except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                counts['failed'] += 1
                if len(failures) < 20:
                    failures.append(str(exc))
                return np.full(len(values)+len(keys), 1e6)
        for index, x0 in enumerate(starts):
            if any(row['index'] == index for row in progress['starts']):
                continue
            progress.update(status='running', incomplete_start=index)
            save(progress_path, progress)
            before_calls, before_failed = counts['calls'], counts['failed']
            begin = time.monotonic()
            try:
                if axes:
                    fitted = least_squares(residual, x0[axes], bounds=(lower[axes], upper[axes]),
                        max_nfev=CONFIG['max_evaluations_per_start'], ftol=1e-8, xtol=1e-8, gtol=1e-8)
                    x_full = expand(fitted.x)
                    r = residual(fitted.x)
                    singular = np.linalg.svd(fitted.jac[:len(values)], compute_uv=False).tolist()
                    success, message, nfev = bool(fitted.success), str(fitted.message), int(fitted.nfev)
                else:
                    x_full = expand([]); r = residual([]); singular = []
                    success, message, nfev = True, 'No locally supported axis; retain all prior scales', 1
                objective = float(r@r)
                eligible = math.isfinite(objective) and objective < 1e10
                row = {'index': index, 'initial_log_parameters': expand(x0[axes]).tolist(), 'log_parameters': x_full.tolist(),
                    'parameters': dict(zip(keys, np.exp(x_full).tolist())), 'objective': objective,
                    'eligible': eligible, 'status': 'converged' if success else 'budget_exhausted',
                    'optimizer_message': message, 'nfev': nfev,
                    'data_jacobian_singular_values': singular,
                    'data_only_ln_rmse': float(np.linalg.norm(r[:len(values)])),
                    'curve_metrics': {'gmfe': math.exp(float(np.mean(abs(r[:len(values)]))*math.sqrt(len(values)))) if eligible else None,
                        'ln_rmse': float(np.linalg.norm(r[:len(values)])), 'n': len(values)},
                    'linear_solver_calls': counts['calls']-before_calls,
                    'failed_evaluation_count': counts['failed']-before_failed,
                    'elapsed_seconds': time.monotonic()-begin}
            except Exception:
                progress['linear_solver_calls'] += counts['calls']-before_calls
                progress['failed_evaluation_count'] += counts['failed']-before_failed
                progress['wall_seconds'] += time.monotonic()-begin
                save(progress_path, progress)
                raise
            progress['starts'].append(row)
            progress['linear_solver_calls'] += row['linear_solver_calls']
            progress['failed_evaluation_count'] += row['failed_evaluation_count']
            progress['wall_seconds'] += row['elapsed_seconds']
            progress.update(incomplete_start=None, diagnostic_failures=failures)
            save(progress_path, progress)
        eligible = [r for r in progress['starts'] if r['eligible']]
        best = min(eligible, key=lambda r: (r['objective'], r['index'])) if eligible else None
        progress['status'] = 'locked'
        save(progress_path, progress)
        _verify(pins, root); _verify(code, root)
        retained = _artifacts(output, root)
        result = {'version': VERSION, 'variant': variant, 'status': 'fitted' if best else 'failed',
            'eligible': best is not None, 'decision': best, 'starts': progress['starts'],
            'source_pins': pins, 'code_pins': code, 'retained_artifact_pins': retained,
            'input_sha256': sha256(input_path), 'optimizer_config': CONFIG,
            'drug_signature': _drug_signature(req), 'construction_partition_method': req['partition_method'],
            'training_source': deepcopy(training_source), 'clearance_input': cl,
            'reference_human': req['human'], 'reference_dosing': req['dosing'],
            'active_parameters': progress['axis_selection']['active_parameters'],
            'axis_selection': progress['axis_selection'], 'total_native_calls': 1, 'actual_solver_calls': 1,
            'linear_solver_calls': progress['linear_solver_calls'], 'cost_count_exact': progress['cost_count_exact'],
            'elapsed_seconds': progress['wall_seconds'] + max(0., time.monotonic()-started-sum(r['elapsed_seconds'] for r in progress['starts'])),
            'failed_evaluation_count': progress['failed_evaluation_count'],
            'development_curve_used': True, 'comparison_curve_used': False,
            'limitations': ['Previously viewed development cohort, not independent blinded validation.',
                'Aggregate systemic clearance does not identify hepatic, renal or tissue metabolism.',
                'Flow-proportional reversible exchange is a hypothesis, not measured tissue permeability or exposure.',
                'Local identity-point Jacobian screen and n>=2k are unvalidated engineering regularizers, not global parameter identifiability.']}
        save(mapping_path, result)
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
                endpoint_estimator_uses_observations=False,
                parameters_fitted=bool(mapping['active_parameters']),
                calibration_axis_selection=deepcopy(mapping['axis_selection']))
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
