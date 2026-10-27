"""Matched-reference evaluation of 11 endpoints from single-dose IV studies.

This module performs no fitting, simulation, source acquisition or file writes.
Reference qualification can run before prediction. The caller must verify source
files and freeze the case, terminal window and prediction file. All derived
references are NCA summaries of observations, not 11 independent measurements.

Observations declare compound_id, matrix='plasma', time_unit='min', unit='umol/l',
and observations[{time_min, concentration_umol_per_l}]. Genuine measured zeroes
are allowed. A missing dose-time observation is never silently supplied. The
optional initial_condition declares method='single_infusion_zero' or
'single_bolus_back_extrapolation', source_url and rationale. This is an NCA
boundary assumption, not an observation. Bolus C0 uses the first two positive
observations; matched predictions independently use those same two sample times.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from numbers import Real
from urllib.parse import urlparse

import numpy as np

from .endpoint_prediction import predict_requested_endpoints

SCALAR_FIELDS = {
    'auc': ('value_mg_h_per_l', 'mg*h/L'),
    'cmax': ('value_mg_per_l', 'mg/L'),
    'tmax': ('value_h', 'h'),
    'c_last': ('value_mg_per_l', 'mg/L'),
    'terminal_half_life': ('value_h', 'h'),
    'auc_inf': ('value_mg_h_per_l', 'mg*h/L'),
    'clearance': ('value_l_per_h', 'L/h'),
    'vz': ('value_l', 'L'),
    'mrt': ('value_h', 'h'),
    'vss': ('value_l', 'L'),
}
ENDPOINTS = ('plasma_concentration_time', *SCALAR_FIELDS)
TOTAL_ENDPOINTS = ('auc_inf', 'clearance', 'vz', 'mrt', 'vss')
TERMINAL_ENDPOINTS = ('terminal_half_life', *TOTAL_ENDPOINTS)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(label+' must be a finite number')
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(label+' must be finite and '+('positive' if positive else 'nonnegative'))
    return value


def _series(times, values, name):
    if not isinstance(times, (list, tuple)) or not isinstance(values, (list, tuple)) or len(times) < 2 or len(times) != len(values):
        raise ValueError(name+' needs at least two aligned samples')
    t = np.asarray([_number(x, name+' time') for x in times])
    c = np.asarray([_number(x, name+' concentration') for x in values])
    if np.any(np.diff(t) <= 0):
        raise ValueError(name+' times must be strictly increasing')
    return t, c


def _bolus_boundary(times, values, indices=None):
    indices = list(np.flatnonzero(values > 0)[:2]) if indices is None else indices
    if len(indices) != 2:
        raise ValueError('Bolus back-extrapolation requires two positive actual samples')
    first, second = indices
    t1, t2, c1, c2 = map(float, (times[first], times[second], values[first], values[second]))
    if not (0 < t1 < t2 and 0 < c2 < c1):
        raise ValueError('Bolus back-extrapolation requires strictly declining positive concentrations at the declared two sample times')
    slope = (math.log(c2)-math.log(c1))/(t2-t1)
    try:
        c0 = _number(math.exp(math.log(c1)-slope*t1), 'back-extrapolated C0', positive=True)
    except OverflowError as exc:
        raise ValueError('Back-extrapolated C0 exceeds finite numeric range') from exc
    return {'concentration_umol_per_l': c0,
            'sample_indices': [int(first), int(second)], 'sample_times_min': [t1, t2],
            'initial_decline_rate_per_h': -slope*60.,
            'calculation': 'log-linear back-extrapolation from the declared two actual sampling times'}


def _prepare(observations, request, terminal_fit, initial_condition):
    if not isinstance(observations, Mapping) or not isinstance(request, Mapping):
        raise ValueError('Observation and request documents are required')
    if (observations.get('compound_id') != request.get('compound_id') or not request.get('compound_id')
            or observations.get('matrix') != 'plasma' or observations.get('time_unit') != 'min'
            or observations.get('unit') != 'umol/l'):
        raise ValueError('Observation identity, plasma matrix and canonical units must match')
    rows = observations.get('observations')
    if not isinstance(rows, list) or any(not isinstance(r, Mapping) for r in rows):
        raise ValueError('observations must be a list of records')
    for row in rows:
        if row.get('unit', 'umol/l') != 'umol/l' or row.get('time_unit', 'min') != 'min':
            raise ValueError('Individual observation units disagree with the document')
        if row.get('below_quantification') or row.get('censored'):
            raise ValueError('Censored observations require a separate declared analysis; do not substitute zero')
    times, values = _series([r.get('time_min') for r in rows],
                           [r.get('concentration_umol_per_l') for r in rows], 'observed')
    dose = request.get('dosing', {})
    if dose.get('route') != 'intravenous' or dose.get('single_dose', True) is not True:
        raise ValueError('This evaluator supports single-dose intravenous cases only')
    if _number(dose.get('start_h', 0.), 'dose start') != 0:
        raise ValueError('This evaluator currently requires the actual dose start to be time zero')
    context = {'route': 'intravenous', 'single_dose': True, 'start_h': 0.,
               'dose_mg': _number(dose.get('dose_mg'), 'dose', positive=True),
               'infusion_minutes': _number(dose.get('infusion_minutes', 0.), 'infusion duration')}
    weights = [e for e in request.get('evidence', []) if e.get('parameter') == 'molecular_weight']
    if len(weights) != 1:
        raise ValueError('Exactly one molecular-weight evidence record is required')
    if weights[0].get('unit') != 'g/mol':
        raise ValueError('Molecular-weight evidence must declare g/mol')
    mw = _number(weights[0].get('value'), 'molecular weight', positive=True)
    registered = request.get('target', {}).get('terminal_fit')
    if terminal_fit is None:
        terminal_fit = registered
    elif registered is not None and dict(terminal_fit) != registered:
        raise ValueError('Explicit terminal window differs from the request freeze')
    if not isinstance(terminal_fit, Mapping):
        raise ValueError('A fixed terminal_fit must be supplied before prediction; no window search is performed')
    fit = deepcopy(dict(terminal_fit))
    if not (times[0]/60 <= _number(fit.get('start_h'), 'terminal start')
            < _number(fit.get('end_h'), 'terminal end') == times[-1]/60):
        raise ValueError('Fixed terminal window must use the observed interval and end at its last sample')
    policy = {'method': 'strict', 'boundary_added': False, 'boundary_is_observed': False}
    if initial_condition is not None:
        supported = ('single_infusion_zero', 'single_bolus_back_extrapolation')
        if not isinstance(initial_condition, Mapping) or initial_condition.get('method') not in supported:
            raise ValueError('Only explicit single_infusion_zero or single_bolus_back_extrapolation NCA boundaries are supported')
        url = initial_condition.get('source_url', '')
        rationale = initial_condition.get('rationale', '')
        if (not isinstance(url, str) or urlparse(url).scheme not in ('http', 'https') or not urlparse(url).netloc
                or not isinstance(rationale, str) or not rationale.strip()):
            raise ValueError('The initial-condition assumption needs a methodological source URL and study-specific rationale')
        if times[0] == 0:
            raise ValueError('A dose-time observation already exists; do not override it with an assumption')
        method = initial_condition['method']
        if method == 'single_infusion_zero':
            if context['infusion_minutes'] <= 0:
                raise ValueError('The zero-boundary option cannot be used for bolus dosing')
            boundary = {'concentration_umol_per_l': 0.}
        else:
            if context['infusion_minutes'] != 0:
                raise ValueError('Bolus back-extrapolation cannot be used for infusion dosing')
            boundary = _bolus_boundary(times, values)
        policy = {**deepcopy(dict(initial_condition)), 'boundary_added': True,
                  'boundary_is_observed': False, 'time_min': 0., **boundary,
                  'interpretation': 'Explicit NCA boundary assumption; excluded from observed-point counts, log errors and terminal samples'}
    return times, values, context, mw, fit, policy


def _curve(times, values):
    return {'time_min': times.tolist(), 'concentration': values.tolist(), 'time_unit': 'min', 'unit': 'umol/l'}


def _endpoints(times, values, context, mw, fit, *, total_times=None, total_values=None):
    target = {'endpoints': list(SCALAR_FIELDS), 'primary_endpoint': 'auc',
              'window_start_h': float(times[0]/60), 'window_end_h': float(times[-1]/60), 'terminal_fit': fit}
    result = predict_requested_endpoints(_curve(times, values), target, mw, context=context)['endpoints']
    if total_times is not None:
        target.update(endpoints=list(TOTAL_ENDPOINTS), primary_endpoint='auc_inf', window_start_h=0.)
        extended = predict_requested_endpoints(_curve(total_times, total_values), target, mw, context=context)['endpoints']
        result.update(extended)
    return result


def _compact(endpoint, field):
    available = endpoint.get('status', 'completed') == 'completed'
    value = endpoint.get(field) if available else None
    if available:
        value = _number(value, 'derived endpoint')
    return {'status': 'qualified' if available else 'not_estimable', 'value': value,
            'reason': endpoint.get('reason'),
            'diagnostics': {k: v for k, v in endpoint.get('diagnostics', {}).items() if k != 'fit_times_h'}}


def qualify_endpoint_references(observations, request, *, terminal_fit=None, initial_condition=None):
    """Qualify all references without accepting or reading any predictions.

    Source-file checks are the caller's responsibility. ``all_11_qualified``
    means the declared numerical/reference rules passed, not clinical validation.
    Optional observations.terminal_phase_qualified=False rejects all six
    terminal-derived references even when their numerical regression is good.
    """
    t, o, context, mw, fit, policy = _prepare(observations, request, terminal_fit, initial_condition)
    nt, no = (np.r_[0., t], np.r_[policy['concentration_umol_per_l'], o]) if policy['boundary_added'] else (None, None)
    raw = _endpoints(t, o, context, mw, fit, total_times=nt, total_values=no)
    positive_count = int(np.sum(o > 0))
    references = {'plasma_concentration_time': {
        'status': 'qualified' if positive_count >= 2 else 'not_estimable', 'unit': 'umol/l',
        'value': None, 'reference_type': 'observed_concentration_time',
        'reason': None if positive_count >= 2 else 'At least two positive actual observations are required for log errors'}}
    for name, (field, unit) in SCALAR_FIELDS.items():
        entry = _compact(raw[name], field)
        kind = ('direct_sample' if name in ('cmax', 'tmax', 'c_last') else
                'observed_curve_trapezoid' if name == 'auc' else 'observed_curve_NCA')
        if name in TOTAL_ENDPOINTS and policy['boundary_added']:
            kind += '_with_declared_initial_condition'
        entry.update(unit=unit, reference_type=kind)
        if observations.get('terminal_phase_qualified') is False and name in TERMINAL_ENDPOINTS:
            entry.update(status='not_estimable', value=None,
                reason=observations.get('terminal_phase_reason', 'Source review did not establish a terminal phase'))
        references[name] = entry
    for entry in references.values():
        entry['source'] = deepcopy(observations.get('source'))
    if policy['boundary_added']:
        first_auc = float(t[0]/60 * (no[0]+o[0])/2 * mw/1000)
        nca_auc_last = float(np.sum(np.diff(nt)/60 * (no[:-1]+no[1:])/2)*mw/1000)
        policy.update(first_interval_h=[0., float(t[0]/60)], first_interval_auc_mg_h_l=first_auc,
            first_interval_fraction_of_auc_last=first_auc/nca_auc_last if nca_auc_last > 0 else None)
    elif t[0] == 0:
        policy.update(boundary_is_observed=True)
    inputs = {'observations': observations, 'request': request, 'terminal_fit': fit,
              'initial_condition': initial_condition}
    unavailable = [name for name, value in references.items() if value['status'] != 'qualified']
    return {'version': 'full-observed-endpoints-1.0', 'status': 'qualified' if not unavailable else 'incomplete_references',
        'all_11_qualified': not unavailable, 'qualified_endpoint_count': len(ENDPOINTS)-len(unavailable),
        'unavailable_endpoints': unavailable, 'endpoints': references, 'terminal_fit': fit,
        'initial_condition': policy, 'observed_window_h': [float(t[0]/60), float(t[-1]/60)],
        'observed_point_count': len(t), 'positive_observed_point_count': positive_count,
        'zero_observed_point_count': len(t)-positive_count, 'input_content_sha256': _digest(inputs),
        'source_integrity_verification': 'caller must verify and freeze original source files',
        'independent_validation': False,
        'interpretation': 'Numerically qualified observed or observation-derived references; scalar NCA endpoints from one curve are statistically dependent'}


def score_full_endpoints(observations, request, locked_prediction, *, terminal_fit=None,
                         initial_condition=None, prediction_failure_reason=None):
    """Score the unchanged locked curve at original observation times.

    A missing/invalid prediction stays in the returned 11-endpoint denominator.
    The caller verifies the file lock; prediction_content_sha256 binds the parsed
    document here and is explicitly not claimed to be the original file hash.
    """
    reference = qualify_endpoint_references(observations, request, terminal_fit=terminal_fit,
                                            initial_condition=initial_condition)
    t, o, context, mw, fit, policy = _prepare(observations, request, terminal_fit, initial_condition)
    failure = prediction_failure_reason
    predicted = raw = None
    content_hash = None
    prediction_initial_condition = None
    try:
        if locked_prediction is None:
            raise ValueError(failure or 'Prediction unavailable')
        if not isinstance(locked_prediction, Mapping) or locked_prediction.get('time_unit') != 'min' or locked_prediction.get('unit') != 'umol/l':
            raise ValueError('Locked prediction must declare min and umol/l')
        pt, pc = _series(locked_prediction.get('time_min'), locked_prediction.get('concentration'), 'predicted')
        required_start = 0. if policy['method'] == 'single_infusion_zero' else t[0]
        if pt[0] > required_start or pt[-1] < t[-1]:
            raise ValueError('Locked prediction does not cover the scoring interval; extrapolation is forbidden')
        predicted = np.interp(t, pt, pc)
        if np.any(predicted[o > 0] <= 0):
            raise ValueError('Nonpositive prediction at a positive observed concentration; no log floor is substituted')
        raw = _endpoints(t, predicted, context, mw, fit)
        if policy['boundary_added']:
            try:
                if policy['method'] == 'single_bolus_back_extrapolation':
                    prediction_initial_condition = _bolus_boundary(t, predicted, policy['sample_indices'])
                else:
                    prediction_initial_condition = {'concentration_umol_per_l': float(np.interp(0., pt, pc)),
                        'calculation': 'Actual locked model concentration at dose time; not replaced by the reference assumption'}
                nt = np.r_[0., t]
                npred = np.r_[prediction_initial_condition['concentration_umol_per_l'], predicted]
                first_auc = float(t[0]/60*(npred[0]+predicted[0])/2*mw/1000)
                total_auc = float(np.sum(np.diff(nt)/60*(npred[:-1]+npred[1:])/2)*mw/1000)
                prediction_initial_condition.update(status='available',
                    first_interval_auc_mg_h_l=first_auc,
                    first_interval_fraction_of_auc_last=first_auc/total_auc if total_auc > 0 else None)
                raw = _endpoints(t, predicted, context, mw, fit, total_times=nt, total_values=npred)
            except (ValueError, OverflowError) as exc:
                prediction_initial_condition = {'status': 'not_estimable', 'reason': str(exc)}
                for name in TOTAL_ENDPOINTS:
                    raw[name] = {'status': 'not_estimable', 'reason': str(exc)}
        content_hash = _digest(locked_prediction)
        failure = None
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        failure = str(exc)
        predicted = raw = None
    outcomes = {}
    for name in ENDPOINTS:
        ref = reference['endpoints'][name]
        pred = None if raw is None or name == 'plasma_concentration_time' else _compact(raw[name], SCALAR_FIELDS[name][0])
        ref_ok = ref['status'] == 'qualified'
        pred_ok = raw is not None and (name == 'plasma_concentration_time' or pred['status'] == 'qualified')
        status = ('completed' if ref_ok and pred_ok else 'reference_unavailable' if pred_ok else
                  'prediction_unavailable' if ref_ok else 'reference_and_prediction_unavailable')
        row = {'status': status, 'unit': ref['unit'], 'reference': ref['value'],
               'prediction': pred['value'] if pred is not None else None,
               'reference_type': ref['reference_type'], 'source': deepcopy(ref['source']),
               'reference_status': ref['status'], 'reference_reason': ref['reason'],
               'prediction_status': 'available' if pred_ok else 'not_estimable',
               'prediction_reason': failure if raw is None else pred.get('reason') if pred is not None else None,
               'fold_error': None, 'absolute_error_h': None}
        if name == 'plasma_concentration_time':
            row.update(observed_point_count=len(t), positive_points_scored=int(np.sum(o > 0)),
                       zero_observations_excluded_from_log_error=int(np.sum(o == 0)),
                       gmfe=None, log_rmse=None, nca_boundary_excluded_from_point_count=policy['boundary_added'])
            if status == 'completed':
                logs = np.log(predicted[o > 0])-np.log(o[o > 0])
                row.update(gmfe=float(np.exp(np.mean(np.abs(logs)))), log_rmse=float(np.sqrt(np.mean(logs**2))))
        elif status == 'completed':
            if name == 'tmax':
                row['absolute_error_h'] = abs(row['prediction']-row['reference'])
            elif row['reference'] > 0 and row['prediction'] > 0:
                row['fold_error'] = max(row['prediction']/row['reference'], row['reference']/row['prediction'])
            else:
                row.update(status='nonpositive_fold_endpoint', prediction_reason='A positive pair is required for fold error')
        outcomes[name] = row
    complete = sum(row['status'] == 'completed' for row in outcomes.values())
    return {'version': 'full-observed-endpoints-1.0', 'status': 'completed' if complete == len(ENDPOINTS) else 'completed_with_unavailable_endpoints',
        'endpoints': outcomes, 'reference_qualification': reference,
        'attempted_endpoints': len(ENDPOINTS), 'scored_endpoints': complete,
        'prediction_available': raw is not None, 'prediction_failure_reason': failure,
        'prediction_initial_condition': prediction_initial_condition,
        'prediction_content_sha256': content_hash,
        'input_content_sha256': reference['input_content_sha256'],
        'independent_validation': False,
        'interpretation': 'Same observed times and definitions; no native dense-grid endpoint substitution, fitting or model selection'}
