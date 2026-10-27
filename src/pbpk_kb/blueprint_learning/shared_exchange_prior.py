"""Grouped, static three-axis priors and five-point linear PBPK adaptation.

Teacher targets are development-fitted log multipliers, not clinical parameter
truth. The caller must exclude compound/study aliases before decoding training
records, verify their hashes, and supply input-only native physiology exports.
No native execution, source-file access or target-outcome scoring occurs here.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import math
import time

import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pbpk_kb.blueprint_policy import robust_exchange_model
from pbpk_kb.blueprint_policy.perfusion_calibration import clearance_inputs
from pbpk_kb.blueprint_policy.tail_exchange_calibration import tail_weights
from .curve_prior import _reject_forbidden

VERSION = 'shared-static-exchange-prior-1'
PARTITION_METHOD = 'PKSimStandard'
PARAMETERS = ('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale')
FEATURE_NAMES = ('log_molecular_weight_g_mol', 'logp', 'log_fraction_unbound',
    'log_clearance_ml_min_kg', 'acid_pka_count', 'base_pka_count', 'minimum_pka', 'maximum_pka')
SCALE_BOUNDS = ((.05, 20.), (.1, 10.), (.01, 100.))
CONFIG = {'partition_method': PARTITION_METHOD, 'ridge_alphas': [1., 10., 100.],
    'inner_validation': 'Leave one connected compound/study group out; train-only weighted scaler; teacher MSE',
    'inner_score_aggregation': 'Validation-fold teacher MSE weighted by distinct held-out compound count',
    'ridge_tie': 'Exact tie favors larger alpha', 'compound_equal_weight': True,
    'starts': 16, 'max_evaluations_per_start': 400, 'seed': 20260930,
    'bounds': {name: list(bound) for name, bound in zip(PARAMETERS, SCALE_BOUNDS)},
    'prior_weight': .1, 'prior_log_scales': [math.log(2.), math.log(10.), math.log(100.)],
    'five_loss': 'tail-balanced logMSE + 0.1*mean(((log_scales-prior_center)/log([2,10,100]))^2)',
    'initialization': 'Identity and fifteen fixed log-uniform starts; no teacher/full-curve/learned warm start',
    'teacher_is_clinical_ground_truth': False, 'native_execution_required': False}


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(name + ' must be numeric')
    value = float(value)
    if not math.isfinite(value) or positive and value <= 0:
        raise ValueError(name + ' must be finite' + (' and positive' if positive else ''))
    return value


def _evidence(request):
    if request.get('partition_method') != PARTITION_METHOD:
        raise ValueError('The shared prior requires fixed PKSimStandard partitioning')
    entries = request['evidence']
    records = {entry['parameter']: entry for entry in entries}
    if len(records) != len(entries):
        raise ValueError('Duplicate evidence parameter')
    if any(entry.get('compound_id') != request['compound_id'] for entry in entries):
        raise ValueError('Evidence compound identity mismatch')
    if {'renal_clearance', 'hepatic_clearance'} & set(records):
        raise ValueError('Clinical renal/hepatic clearance is excluded from this shared-prior protocol')
    for name in ('fraction_unbound', 'total_clearance'):
        if records.get(name, {}).get('basis') != 'predicted':
            raise ValueError('Shared-prior fu and total clearance must be RF predicted inputs')
    _reject_forbidden(entries, request)
    return records


def _value(records, name, unit, *, positive=False):
    row = records[name]
    if row['unit'] != unit:
        raise ValueError('Unexpected unit for ' + name)
    return _number(row['value'], name, positive=positive)


def static_features(request):
    """Eight static features; no curve, person, dose, identity or endpoint feature."""
    evidence = _evidence(request)
    mw = _value(evidence, 'molecular_weight', 'g/mol', positive=True)
    logp = _value(evidence, 'logp', 'Log Units')
    fu = _value(evidence, 'fraction_unbound', 'fraction', positive=True)
    cl = _value(evidence, 'total_clearance', 'ml/min/kg', positive=True)
    if fu > 1.:
        raise ValueError('Fraction unbound exceeds one')
    pkas = evidence.get('pka_types', {}).get('value', [])
    if not isinstance(pkas, (list, tuple)) or any(p.get('Type') not in ('Acid', 'Base') for p in pkas):
        raise ValueError('Invalid pKa types')
    values = [_number(p['Pka'], 'pKa') for p in pkas]
    return np.asarray([math.log(mw), logp, math.log(fu), math.log(cl),
        sum(p['Type'] == 'Acid' for p in pkas), sum(p['Type'] == 'Base' for p in pkas),
        min(values, default=0.), max(values, default=0.)], dtype=np.float64)


def _ids(values, name, *, allow_empty=False):
    if (not isinstance(values, (list, tuple, set, frozenset))
            or any(not isinstance(item, str) or not item for item in values)
            or not allow_empty and not values):
        raise ValueError(name + ' must contain explicit nonempty identifiers')
    return set(values)


def _bounds():
    bounds = np.asarray(SCALE_BOUNDS, float)
    return np.log(bounds[:, 0]), np.log(bounds[:, 1])


def _logs(value, name):
    if isinstance(value, Mapping):
        if set(value) != set(PARAMETERS):
            raise ValueError(name + ' must contain exactly the three exchange parameters')
        value = [value[key] for key in PARAMETERS]
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 3:
        raise ValueError(name + ' must contain three log multipliers')
    array = np.asarray([_number(x, name) for x in value], float)
    lower, upper = _bounds()
    if np.any(array < lower-1e-12) or np.any(array > upper+1e-12):
        raise ValueError(name + ' exceeds shared exchange bounds')
    return np.clip(array, lower, upper)


def _weights(compounds):
    counts = Counter(compounds)
    return np.asarray([len(compounds) / (len(counts) * counts[c]) for c in compounds], float)


def _study_groups(rows):
    """Join rows sharing either a compound or a study, including transitive links."""
    groups = []
    for index, row in enumerate(rows):
        connected = {index}
        compounds, studies = {row['compound_id']}, set(row['study_ids'])
        keep = []
        for group in groups:
            if compounds & group['compounds'] or studies & group['studies']:
                connected |= group['indices']; compounds |= group['compounds']; studies |= group['studies']
            else:
                keep.append(group)
        # A later merge can connect a previously skipped group transitively.
        changed = True
        while changed:
            changed = False
            for group in list(keep):
                if compounds & group['compounds'] or studies & group['studies']:
                    connected |= group['indices']; compounds |= group['compounds']; studies |= group['studies']
                    keep.remove(group); changed = True
        groups = keep + [{'indices': connected, 'compounds': compounds, 'studies': studies}]
    return sorted(groups, key=lambda group: (sorted(group['compounds']), sorted(group['studies'])))


def fit_prior(rows, *, excluded_compound_ids, excluded_study_ids):
    """Fit serializable pooled/ridge priors to already excluded training rows.

    All held-out rows are rejected before accessing their requests or teacher
    targets. Inner splits additionally group related compounds and studies.
    """
    excluded_compounds = _ids(excluded_compound_ids, 'excluded_compound_ids')
    excluded_studies = _ids(excluded_study_ids, 'excluded_study_ids')
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('Explicit training rows are required')
    for row in rows:
        if not isinstance(row.get('compound_id'), str) or not row['compound_id']:
            raise ValueError('Training compound identity required')
        studies = _ids(row.get('study_ids'), 'study_ids')
        if row['compound_id'] in excluded_compounds or studies & excluded_studies:
            raise ValueError('Held-out compound or study row cannot enter prior training')
    compounds = [row['compound_id'] for row in rows]
    if len(set(compounds)) < 3:
        raise ValueError('At least three independent training compounds are required')
    for row in rows:
        if row['request']['compound_id'] != row['compound_id']:
            raise ValueError('Training row/request compound identity mismatch')
    x = np.asarray([static_features(row['request']) for row in rows])
    y = np.asarray([_logs(row['log_parameters'], 'teacher log_parameters') for row in rows])
    groups = _study_groups(rows)
    if len(groups) < 3:
        raise ValueError('At least three connected compound/study groups are required')
    alphas = CONFIG['ridge_alphas']
    scores = {alpha: [] for alpha in alphas}
    folds = []
    for group in groups:
        validation = sorted(group['indices'])
        training = [i for i in range(len(rows)) if i not in group['indices']]
        train_compounds = [compounds[i] for i in training]
        validation_compounds = [compounds[i] for i in validation]
        weights = _weights(train_compounds)
        scaler = StandardScaler().fit(x[training], sample_weight=weights)
        train_x, validation_x = scaler.transform(x[training]), scaler.transform(x[validation])
        fold_scores = {}
        for alpha in alphas:
            estimator = Ridge(alpha=alpha, solver='svd').fit(train_x, y[training], sample_weight=weights)
            errors = np.mean((estimator.predict(validation_x)-y[validation])**2, axis=1)
            score = float(np.average(errors, weights=_weights(validation_compounds)))
            scores[alpha].append(score); fold_scores[str(alpha)] = score
        folds.append({'training_row_indices': training, 'validation_row_indices': validation,
            'training_compound_ids': sorted(set(train_compounds)),
            'validation_compound_ids': sorted(set(validation_compounds)),
            'training_study_ids': sorted({s for i in training for s in rows[i]['study_ids']}),
            'validation_study_ids': sorted(group['studies']),
            'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
            'teacher_mse_by_alpha': fold_scores})
    validation_counts = [len(fold['validation_compound_ids']) for fold in folds]
    mean_scores = {alpha: float(np.average(values, weights=validation_counts)) for alpha, values in scores.items()}
    selected = min(alphas, key=lambda alpha: (mean_scores[alpha], -alpha))
    weights = _weights(compounds)
    scaler = StandardScaler().fit(x, sample_weight=weights)
    estimator = Ridge(alpha=selected, solver='svd').fit(scaler.transform(x), y, sample_weight=weights)
    return {'version': VERSION, 'config': deepcopy(CONFIG), 'feature_names': list(FEATURE_NAMES),
        'parameter_names': list(PARAMETERS), 'partition_method': PARTITION_METHOD,
        'excluded_compound_ids': sorted(excluded_compounds), 'excluded_study_ids': sorted(excluded_studies),
        'training_compound_ids': sorted(set(compounds)),
        'training_study_ids': sorted({s for row in rows for s in row['study_ids']}),
        'training_rows': [{'compound_id': r['compound_id'], 'study_ids': sorted(set(r['study_ids'])),
            'weight': float(w)} for r, w in zip(rows, weights)],
        'pooled_mean': np.average(y, axis=0, weights=weights).tolist(),
        'ridge': {'alpha': selected, 'coef': estimator.coef_.tolist(), 'intercept': estimator.intercept_.tolist(),
            'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()},
        'inner_folds': folds,
        'inner_teacher_mse_by_alpha': {str(alpha): value for alpha, value in mean_scores.items()},
        'teacher_is_clinical_ground_truth': False, 'clinical_target_values_used': False,
        'source_hashes_verified_by_this_module': False, 'native_execution_required': False,
        'limitations': ['Teacher decisions are regularized development fits, not clinical parameter truth.',
            'Caller must exclude identity/study aliases before decoding training records and verify all source hashes.',
            'Previously viewed drugs permit an exploratory grouped evaluation, not a new blind clinical validation.']}


def predict_prior(bundle, request, method='ridge'):
    """Return three bounded log multipliers using static inputs only."""
    if (bundle.get('version') != VERSION or bundle.get('feature_names') != list(FEATURE_NAMES)
            or bundle.get('parameter_names') != list(PARAMETERS) or bundle.get('config') != CONFIG):
        raise ValueError('Incompatible shared-prior schema or frozen configuration')
    compound = request['compound_id']
    if compound in bundle['training_compound_ids'] or compound not in bundle['excluded_compound_ids']:
        raise ValueError('Prediction must concern an explicitly excluded compound')
    features = static_features(request)
    if method == 'rf_only':
        prediction = np.zeros(3)
    elif method == 'pooled_mean':
        prediction = np.asarray(bundle['pooled_mean'], float)
    elif method == 'ridge':
        ridge = bundle['ridge']
        standardized = (features-np.asarray(ridge['scaler_mean']))/np.asarray(ridge['scaler_scale'])
        prediction = np.asarray(ridge['coef']) @ standardized + np.asarray(ridge['intercept'])
    else:
        raise ValueError('Unknown shared-prior method')
    if prediction.shape != (3,) or not np.isfinite(prediction).all():
        raise ValueError('Invalid learned log multipliers')
    return np.clip(prediction, *_bounds())


def _five_document(document, request):
    if (document.get('role') != 'calibration_data_reconstruction' or document.get('matrix') != 'plasma'
            or document.get('time_unit') != 'min' or document.get('unit') != 'umol/l'):
        raise ValueError('A declared development plasma curve in min and umol/l is required')
    if document.get('compound_id') != request['compound_id']:
        raise ValueError('Five-point/request compound identity mismatch')
    if document.get('human') != request.get('human') or document.get('dosing') != request.get('dosing'):
        raise ValueError('Five-point/request human or dosing conditions differ')
    if request['dosing']['route'] != 'intravenous' or request['dosing'].get('start_h', 0.) != 0.:
        raise ValueError('Five-point exchange fitting requires intravenous dosing at zero')
    forbidden_keys = {'raw_observations', 'full_observations', 'full_curve', 'teacher_parameters',
        'teacher', 'clinical_reference', 'endpoints', 'target_label', 'target_labels', 'heldout_labels',
        'initial_points', 'warm_start', 'full_fit_parameters'}
    def reject_hidden(value):
        if isinstance(value, Mapping):
            if forbidden_keys & set(value):
                raise ValueError('Full-curve, teacher or held-out labels cannot enter five-point adaptation')
            for child in value.values(): reject_hidden(child)
        elif isinstance(value, (list, tuple)):
            for child in value: reject_hidden(child)
    reject_hidden(document)
    _reject_forbidden(document, request)
    rows = document.get('observations', [])
    if len(rows) != 5:
        raise ValueError('Five-point adaptation requires exactly five supplied observations')
    if any(row.get('unit', 'umol/l') != 'umol/l' or row.get('time_unit', 'min') != 'min' for row in rows):
        raise ValueError('Individual observation units differ')
    times = np.asarray([_number(row['time_min'], 'observation time')/60. for row in rows])
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('Five observation times must be nonnegative and strictly increasing')
    mw = _value(_evidence(request), 'molecular_weight', 'g/mol', positive=True)
    if 'molecular_weight_g_mol' in document and not math.isclose(
            _number(document['molecular_weight_g_mol'], 'document MW', positive=True), mw, rel_tol=1e-8):
        raise ValueError('Development/request molecular weight mismatch')
    values = np.asarray([_number(row['concentration_umol_per_l'], 'concentration', positive=True)
                         for row in rows]) * mw / 1000.
    return times, values


def fit_five(physiology, request, document, prior_center, before_trial=None):
    """Sixteen identical-budget fits around a frozen static prior, without native calls."""
    static_features(request)
    times, values = _five_document(document, request)
    center = _logs(prior_center, 'prior_center')
    lower, upper = _bounds()
    duration = _number(request['dosing']['infusion_minutes'], 'infusion minutes')/60.
    if duration < 0:
        raise ValueError('Infusion duration must be nonnegative')
    dose = _number(request['dosing']['dose_mg'], 'dose', positive=True)
    clearance = clearance_inputs(request)
    if clearance['renal_clearance_l_h'] is not None:
        raise ValueError('No clinical renal clearance is allowed in shared adaptation')
    plan = tail_weights(times, duration)
    weights = np.asarray(plan['weights'])
    penalty_scale = math.sqrt(CONFIG['prior_weight']/3.)/np.asarray(CONFIG['prior_log_scales'])
    starts = np.vstack([np.zeros(3), np.random.default_rng(CONFIG['seed']).uniform(
        lower, upper, size=(CONFIG['starts']-1, 3))])
    counters = {'calls': 0, 'failed': 0}; failures = []; last_check = 0.
    mass_error = 0.
    def residual(x):
        nonlocal last_check, mass_error
        stamp = time.monotonic()
        if before_trial and stamp-last_check >= 2.:
            before_trial(); last_check = stamp
        counters['calls'] += 1
        try:
            prediction = robust_exchange_model.simulate(physiology, times_h=times, dose_mg=dose,
                infusion_duration_h=duration, total_clearance_l_h=clearance['total_clearance_l_h'],
                renal_clearance_l_h=None, family='evidence_flexible',
                log_parameters=dict(zip(PARAMETERS, x)), observation='auto')
            predicted = np.asarray(prediction['concentration_mg_l'])
            if predicted.shape != values.shape or not np.isfinite(predicted).all() or np.any(predicted <= 0):
                raise ValueError('Invalid positive five-point concentration')
            mass_error = max(mass_error, prediction['mass_balance_relative_error'])
            return np.r_[(np.log(predicted)-np.log(values))*np.sqrt(weights), (x-center)*penalty_scale]
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
            counters['failed'] += 1
            if len(failures) < 20: failures.append(str(exc))
            return np.full(8, 1e6)
    rows = []; begin = time.monotonic()
    for index, initial in enumerate(starts):
        if before_trial: before_trial()
        old_calls, old_failed = counters['calls'], counters['failed']; started = time.monotonic()
        fitted = least_squares(residual, initial, bounds=(lower, upper),
            max_nfev=CONFIG['max_evaluations_per_start'], ftol=1e-8, xtol=1e-8, gtol=1e-8)
        result = residual(fitted.x); objective = float(result @ result)
        eligible = math.isfinite(objective) and objective < 1e10
        data_jacobian = fitted.jac[:5] / np.sqrt(weights[:, None]) / math.sqrt(5.)
        rows.append({'index': index, 'initial_log_parameters': initial.tolist(),
            'log_parameters': fitted.x.tolist(), 'parameters': dict(zip(PARAMETERS, np.exp(fitted.x).tolist())),
            'objective': objective, 'weighted_data_loss': float(result[:5] @ result[:5]),
            'prior_loss': float(result[5:] @ result[5:]), 'eligible': eligible,
            'status': 'converged' if fitted.success else 'budget_exhausted',
            'nfev': int(fitted.nfev), 'optimizer_message': str(fitted.message),
            'development_log_errors': (result[:5]/np.sqrt(weights)).tolist() if eligible else None,
            'data_jacobian_singular_values': np.linalg.svd(data_jacobian, compute_uv=False).tolist(),
            'data_jacobian_includes_prior_rows': False,
            'bound_hits': {key: 'lower' if abs(fitted.x[i]-lower[i]) <= 1e-5 else 'upper'
                if abs(fitted.x[i]-upper[i]) <= 1e-5 else None for i, key in enumerate(PARAMETERS)},
            'linear_solver_calls': counters['calls']-old_calls,
            'failed_evaluation_count': counters['failed']-old_failed,
            'elapsed_seconds': time.monotonic()-started})
    eligible = [row for row in rows if row['eligible']]
    best = min(eligible, key=lambda row: (row['objective'], row['index'])) if eligible else None
    return {**(deepcopy(best) if best else {'log_parameters': None, 'parameters': None}),
        'version': VERSION, 'status': 'fitted' if best else 'failed', 'eligible': best is not None,
        'decision': deepcopy(best), 'starts': rows, 'prior_center': center.tolist(), 'loss_plan': plan,
        'optimizer_config': deepcopy(CONFIG), 'linear_solver_calls': counters['calls'],
        'failed_evaluation_count': counters['failed'], 'diagnostic_failures': failures,
        'maximum_mass_balance_relative_error': mass_error, 'elapsed_seconds': time.monotonic()-begin,
        'native_solver_calls': 0, 'development_points': 5, 'full_curve_used': False,
        'external_warm_start_used': False, 'clinical_target_values_used': False}
