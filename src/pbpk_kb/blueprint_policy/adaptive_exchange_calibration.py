"""Tail-balanced calibration with relaxed partitioning and optional exchange shape.

Both candidates retain the native human template and fixed net-renal input.
The optional fourth coordinate is signed alpha, equivalently log(exp(alpha));
it changes exchange heterogeneity rather than adding a clearance compartment.
Only source-separated development curves enter fitting. No target-based selection.
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
from .tail_exchange_calibration import tail_weights
from .robust_exchange_calibration import clearance_policy

VERSION = 'adaptive-evidence-exchange-calibration-1'
PARAMETERS = {'relaxed_tail': ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale')}
PARAMETERS['adaptive_exchange'] = PARAMETERS['relaxed_tail'] + ('exchange_heterogeneity_scale',)
CONFIG = {
    'starts': 16, 'max_evaluations_per_start': 400, 'seed': 20260930,
    'alpha_seed': 20260931,
    'initialization': 'One identity and fifteen deterministic starts; identical first three axes across variants; independent signed-alpha stream',
    'bounds': {'partition_scales': [.02, 10.], 'exchange_scale': [.01, 100.],
               'exchange_heterogeneity_alpha': [-.5, .5]},
    'clearance_policies': {
        'qualified_rf': {'scale_bounds': [.05, 20.], 'penalty_weight': .01},
        'qualified_observed': {'scale_bounds': [.25, 4.], 'penalty_weight': .1}},
    'loss': '0.5*body_mean_logMSE+0.5*tail_mean_logMSE+w_CL*(log(CLscale)/log(2))^2+0.01*(log(Kscale)/log(10))^2+0.01*(log(PSscale)/log(100))^2+0.1*(alpha/0.5)^2 when present; unsupported phases use uniform logMSE',
    'tail': 'Frozen tail_exchange_calibration.tail_weights: strict postinfusion; final max(3,ceil(npost/3)); body/tail equally weighted',
    'alpha_penalty_weight': .1,
    'parameterization': 'First three coordinates are log multipliers; fourth is signed alpha and equals log(exchange_heterogeneity_scale)',
    'diagnostics': {'jacobian': 'Unweighted log-concentration residuals divided by sqrt(n); no prior rows',
                    'relative_rank_threshold': 1e-3, 'bound_absolute_tolerance': 1e-5},
    'solver': 'Linear vascular–extravascular exchange equations, double precision',
    'clearance': 'Same aggregate venous plasma sink as perfusion comparison; fixed renal input and scaled nonrenal component',
    'observation': 'Same native peripheral venous weights, applied to vascular outflow',
    'exchange': {'relaxed_tail': 'Frozen evidence_flexible PS_i = Q_i * shared_scale',
        'adaptive_exchange': 'Learned signed heterogeneity on the same vascular-tissue circulation; see pinned adaptive_exchange_model'},
    'exchange_alpha': {'relaxed_tail': 0., 'adaptive_exchange': 'fitted signed coordinate in [-0.5,0.5]'},
    'prior_selection': 'Qualified total-clearance evidence provenance only; no target values or metrics',
    'exchange_interpretation': 'Effective exchange hypothesis, not measured cellular permeability',
    'extravascular_scaling_only': True, 'target_observations_used': False,
}



def _code_pins(root):
    from .perfusion_calibration import _code_pins as original
    pins = original(root)
    for name in ('transport_calibration.py', 'permeability_model.py', 'evidence_decision.py',
                 'robust_exchange_calibration.py', 'robust_exchange_model.py',
                 'tail_exchange_calibration.py', 'adaptive_exchange_calibration.py', 'adaptive_exchange_model.py'):
        path = root / 'src/pbpk_kb/blueprint_policy' / name
        pins[str(path.relative_to(root))] = sha256(path)
    return pins


from .perfusion_calibration import clearance_inputs, prediction_grid, _physiology


def parameter_starts(variant, request):
    keys = PARAMETERS[variant]
    bounds = np.array([clearance_policy(request)['scale_bounds']] +
                      [CONFIG['bounds']['partition_scales'], CONFIG['bounds']['exchange_scale']], float)
    lower, upper = np.log(bounds[:, 0]), np.log(bounds[:, 1])
    rng = np.random.default_rng(CONFIG['seed'])
    starts = np.vstack([np.zeros(3), rng.uniform(lower, upper, size=(CONFIG['starts']-1, 3))])
    if variant == 'adaptive_exchange':
        low, high = CONFIG['bounds']['exchange_heterogeneity_alpha']
        alpha = np.r_[0., np.random.default_rng(CONFIG['alpha_seed']).uniform(low, high, CONFIG['starts']-1)]
        starts = np.column_stack((starts, alpha)); lower = np.r_[lower, low]; upper = np.r_[upper, high]
    return keys, lower, upper, starts


def parameter_penalty(x, prior):
    x = np.asarray(x, float)
    if x.shape not in ((3,), (4,)) or not np.isfinite(x).all():
        raise ValueError('Three scales and an optional signed alpha are required')
    penalty = np.r_[math.sqrt(prior['penalty_weight'])*x[0]/math.log(2),
                    .1*x[1]/math.log(10), .1*x[2]/math.log(100)]
    return np.r_[penalty, math.sqrt(CONFIG['alpha_penalty_weight'])*x[3]/.5] if len(x) == 4 else penalty


def _diagnostics(data_jacobian, x, lower, upper, keys):
    singular = np.linalg.svd(data_jacobian, compute_uv=False)
    tolerance = CONFIG['diagnostics']['relative_rank_threshold']
    rank = int(np.count_nonzero(singular > singular[0]*tolerance)) if len(singular) and singular[0] > 0 else 0
    boundary = CONFIG['diagnostics']['bound_absolute_tolerance']
    return {'data_jacobian_singular_values': singular.tolist(), 'data_jacobian_rank': rank,
        'data_jacobian_columns': len(keys), 'data_jacobian_rows': len(data_jacobian),
        'data_jacobian_includes_prior_rows': False,
        'bound_hits': {name: ('lower' if abs(x[i]-lower[i]) <= boundary else 'upper'
            if abs(x[i]-upper[i]) <= boundary else None) for i, name in enumerate(keys)}}


def _simulate(physiology, request, times_h, variant, keys, x):
    if variant == 'relaxed_tail':
        from .robust_exchange_model import simulate
        family = 'evidence_flexible'
    elif variant == 'adaptive_exchange':
        from .adaptive_exchange_model import simulate
        family = 'adaptive_exchange'
    else:
        raise ValueError('Unknown adaptive calibration variant')
    cl = clearance_inputs(request)
    return simulate(physiology, times_h=np.asarray(times_h, float),
        dose_mg=float(request['dosing']['dose_mg']),
        infusion_duration_h=float(request['dosing']['infusion_minutes'])/60,
        total_clearance_l_h=cl['total_clearance_l_h'], renal_clearance_l_h=cl['renal_clearance_l_h'],
        family=family,
        log_parameters=dict(zip(keys, np.asarray(x, float).tolist())), observation='auto')


def _validated(mapping, path, root):
    fixed = _read(path.parent/'input.json')
    if (mapping['version'] != VERSION or mapping['input_sha256'] != sha256(path.parent/'input.json')
            or mapping['variant'] != fixed['variant'] or mapping['optimizer_config'] != CONFIG
            or mapping['drug_signature'] != _drug_signature(fixed['request'])
            or fixed['version'] != VERSION or fixed['optimizer_config'] != CONFIG
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
            or mapping['active_parameters'] != list(PARAMETERS[fixed['variant']])
            or mapping['clearance_prior_policy'] != fixed['clearance_prior_policy']
            or mapping['clearance_prior_policy'] != clearance_policy(fixed['request'])
            or mapping['effective_log_bounds'] != fixed['effective_log_bounds']
            or mapping['loss_plan'] != fixed['loss_plan']
            or mapping['parameterization'] != CONFIG['parameterization']):
        raise ValueError('Locked mapping differs from its development records or input metadata')
    keys, lower, upper, initial = parameter_starts(fixed['variant'], fixed['request'])
    expected_bounds = {'lower': lower.tolist(), 'upper': upper.tolist()}
    if mapping['effective_log_bounds'] != expected_bounds:
        raise ValueError('Locked evidence-dependent calibration bounds changed')
    observed = _training_document(fixed['training_source'], fixed['request'], root)
    times = [row['time_min']/60 for row in observed['observations']]
    expected_plan = tail_weights(times, fixed['request']['dosing']['infusion_minutes']/60)
    if mapping['loss_plan'] != expected_plan or [row['index'] for row in mapping['starts']] != list(range(CONFIG['starts'])):
        raise ValueError('Locked tail weighting or complete fixed-start sequence changed')
    weights = np.asarray(expected_plan['weights'], float)
    for row in mapping['starts']:
        x = np.asarray(row['log_parameters'], float)
        if (x.shape != (len(keys),) or not np.isfinite(x).all()
                or np.any(x < lower-1e-12) or np.any(x > upper+1e-12)
                or row['initial_log_parameters'] != initial[row['index']].tolist()):
            raise ValueError('Locked start or solution violates the evidence-dependent bounds')
        expected_parameters = dict(zip(keys, np.exp(x).tolist()))
        alpha = float(x[3]) if len(x) == 4 else 0.
        if row['parameters'] != expected_parameters or row['exchange_heterogeneity_alpha'] != alpha:
            raise ValueError('Locked alpha or exponentiated parameter representation changed')
        if row['eligible']:
            errors = np.asarray(row['development_log_errors'], float)
            if errors.shape != weights.shape or not np.isfinite(errors).all():
                raise ValueError('Locked development log residuals changed')
            loss = float(weights @ (errors**2))
            penalty = parameter_penalty(x, mapping['clearance_prior_policy'])
            objective = loss + float(penalty @ penalty)
            if (not math.isclose(row['weighted_data_loss'], loss, rel_tol=1e-10, abs_tol=1e-12)
                    or not math.isclose(row['objective'], objective, rel_tol=1e-10, abs_tol=1e-12)
                    or not math.isclose(row['data_only_ln_rmse'], float(np.sqrt(np.mean(errors**2))), rel_tol=1e-10, abs_tol=1e-12)
                    or not math.isclose(row['curve_metrics']['gmfe'], math.exp(float(np.mean(abs(errors)))), rel_tol=1e-10, abs_tol=1e-12)
                    or row['data_jacobian_includes_prior_rows'] is not False
                    or row['data_jacobian_rows'] != len(weights) or row['data_jacobian_columns'] != len(keys)):
                raise ValueError('Locked weighted loss, ordinary metric or data-only Jacobian contract changed')
    eligible = [row for row in mapping['starts'] if row.get('eligible')]
    best = min(eligible, key=lambda row: (row['objective'], row['index'])) if eligible else None
    if (mapping['decision'] != best or mapping['eligible'] != (best is not None)
            or mapping['formula_selection_score'] != (best['objective'] if best else None)
            or mapping['exchange_heterogeneity_alpha'] != (best['exchange_heterogeneity_alpha'] if best else None)):
        raise ValueError('Locked selection differs from recorded development starts')


def fit_mapping(*, reference_request, training_source, output_dir, project_root,
                variant, before_trial=None):
    root = Path(project_root).resolve(strict=True)
    output = inside(output_dir, root)
    req = EvidenceBlueprintRequest.model_validate(reference_request).model_dump(mode='json', exclude_none=True)
    prior = clearance_policy(req)
    keys, lower, upper, starts = parameter_starts(variant, req)
    observed = _training_document(training_source, req, root)
    if (observed['human'] != req['human'] or observed['dosing'] != req['dosing']
            or req['dosing']['route'] != 'intravenous' or float(req['dosing'].get('start_h', 0)) != 0):
        raise ValueError('Development conditions differ or administration does not start at zero')
    cl = clearance_inputs(req)
    loss_plan = tail_weights([r['time_min']/60 for r in observed['observations']],
                            req['dosing']['infusion_minutes']/60)
    pins, code = _embedded_pins([req, training_source, observed], root), _code_pins(root)
    _verify(pins, root); _verify(code, root)
    fixed = {'version': VERSION, 'request': req, 'training_source': deepcopy(training_source),
        'variant': variant, 'optimizer_config': CONFIG, 'source_pins': pins, 'code_pins': code,
        'clearance_input': cl, 'external_warm_start_used': False, 'clearance_prior_policy': prior,
        'effective_log_bounds': {'lower': lower.tolist(), 'upper': upper.tolist()},
        'loss_plan': loss_plan, 'parameterization': CONFIG['parameterization']}
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
        weights = np.asarray(loss_plan['weights'], float)
        progress_path = output/'progress.json'
        progress = _read(progress_path) if progress_path.exists() else {'starts': [], 'linear_solver_calls': 0,
            'failed_evaluation_count': 0, 'wall_seconds': 0., 'cost_count_exact': True}
        if progress.get('incomplete_start') is not None:
            # A hard shutdown can lose an unfinished start's in-memory counter.
            progress['cost_count_exact'] = False
            progress.setdefault('interrupted_starts', []).append(progress['incomplete_start'])
        counts = {'calls': 0, 'failed': 0}
        failures = []
        last_check = 0.
        def residual(x):
            nonlocal last_check
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
                data = (np.log(p)-np.log(values))*np.sqrt(weights)
                penalty = parameter_penalty(x, prior)
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
                fitted = least_squares(residual, x0, bounds=(lower, upper), max_nfev=CONFIG['max_evaluations_per_start'],
                    ftol=1e-8, xtol=1e-8, gtol=1e-8)
                r = residual(fitted.x)
                objective = float(r@r)
                eligible = math.isfinite(objective) and objective < 1e10
                # Undo stage weights before reporting ordinary curve metrics
                # and local data Jacobian rank. Prior residual rows are excluded.
                errors = r[:len(values)]/np.sqrt(weights)
                ordinary_rmse = float(np.sqrt(np.mean(errors**2))) if eligible else None
                jacobian = np.asarray(fitted.jac[:len(values)], float)/np.sqrt(weights[:, None]*len(values))
                diagnostics = _diagnostics(jacobian, fitted.x, lower, upper, keys)
                row = {'index': index, 'initial_log_parameters': x0.tolist(), 'log_parameters': fitted.x.tolist(),
                    'parameters': dict(zip(keys, np.exp(fitted.x).tolist())), 'objective': objective,
                    'exchange_heterogeneity_alpha': float(fitted.x[3]) if len(keys) == 4 else 0.,
                    'eligible': eligible, 'status': ('converged' if fitted.success else 'budget_exhausted') if eligible else 'scientific_failed',
                    'optimizer_message': fitted.message, 'nfev': int(fitted.nfev),
                    **diagnostics, 'data_jacobian_valid': eligible,
                    'development_log_errors': errors.tolist() if eligible else None,
                    'weighted_data_loss': float(r[:len(values)]@r[:len(values)]) if eligible else None,
                    'penalty_loss': float(r[len(values):]@r[len(values):]) if eligible else None,
                    'data_only_ln_rmse': ordinary_rmse,
                    'curve_metrics': {'gmfe': math.exp(float(np.mean(abs(errors)))) if eligible else None,
                        'ln_rmse': ordinary_rmse, 'n': len(values)},
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
            'formula_selection_score': best['objective'] if best else None,
            'formula_selection_basis': 'penalized_tail_development_objective_only',
            'exchange_heterogeneity_alpha': best['exchange_heterogeneity_alpha'] if best else None,
            'source_pins': pins, 'code_pins': code, 'retained_artifact_pins': retained,
            'input_sha256': sha256(input_path), 'optimizer_config': CONFIG,
            'drug_signature': _drug_signature(req), 'construction_partition_method': req['partition_method'],
            'training_source': deepcopy(training_source), 'clearance_input': cl,
            'reference_human': req['human'], 'reference_dosing': req['dosing'],
            'active_parameters': list(keys), 'clearance_prior_policy': prior,
            'loss_plan': loss_plan, 'parameterization': CONFIG['parameterization'],
            'effective_log_bounds': {'lower': lower.tolist(), 'upper': upper.tolist()},
            'total_native_calls': 1, 'actual_solver_calls': 1,
            'linear_solver_calls': progress['linear_solver_calls'], 'cost_count_exact': progress['cost_count_exact'],
            'elapsed_seconds': progress['wall_seconds'] + max(0., time.monotonic()-started-sum(r['elapsed_seconds'] for r in progress['starts'])),
            'failed_evaluation_count': progress['failed_evaluation_count'],
            'development_curve_used': True, 'comparison_curve_used': False,
            'limitations': ['Previously viewed development cohort, not independent blinded validation.',
                'Aggregate systemic clearance does not identify hepatic, renal or tissue metabolism.',
                'Effective reversible exchange and fitted heterogeneity are hypotheses, not measured tissue permeability or exposure.',
                'RF clearance relaxation reflects weaker provenance, not a calibrated confidence interval.',
                'Local Jacobian diagnostics do not prove global parameter identifiability.']}
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
    if req['dosing']['route'] != 'intravenous' or float(req['dosing'].get('start_h', 0)) != 0:
        raise ValueError('This pilot requires intravenous administration at dose time zero')
    pins = _embedded_pins(req, root); _verify(pins, root)
    construction = {'version': VERSION, 'mapping': _ref(path, root), 'request': req,
        'parameters': mapping['decision']['parameters'], 'target_outcomes_used': False,
        'exchange_heterogeneity_alpha': mapping['exchange_heterogeneity_alpha'],
        'loss_plan': mapping['loss_plan']}
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
                endpoint_estimator_uses_observations=False, parameters_fitted=True,
                calibration_version=VERSION, calibration_variant=variant,
                exchange_heterogeneity_alpha=mapping['exchange_heterogeneity_alpha'])
            save(output/'prediction_locked.json', curve)
            save(output/'endpoints.json', endpoints)
            result.update(status='predicted', prediction=_ref(output/'prediction_locked.json', root),
                mass_balance=predicted['mass_balance'], observation=predicted['observation'],
                effective_total_clearance_l_h=predicted['effective_total_clearance_l_h'],
                exchange_heterogeneity_alpha=mapping['exchange_heterogeneity_alpha'])
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
