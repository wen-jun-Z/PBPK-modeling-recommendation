"""Source-honest, bolus-safe clinical scoring for the sampling expansion.

Actual observed times define finite-window comparisons. No zero-time clinical
sample is manufactured. NCA of a reported aggregate curve is explicitly NOT a
paper-reported scalar or the mean of individual-patient NCA. Full AUC infinity
and its derived clearance/volume/moment references require observed time zero;
otherwise they remain unavailable. Model predictions may use their own genuine
time-zero value, with that difference recorded, never filling clinical labels.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pbpk_kb.blueprints.clinical_prediction_nca import UNITS, _integrals, _tail_fit, _sample
from pbpk_kb.blueprint_policy import learned_sampling_design as learner
from blueprint_gap_design_data import _conditions, _molecular_weight
from summarize_additional_clinical_baselines import prediction_in_hours, weak_endpoints
from run_clinical_method_comparison import clinical_unit_factor

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'sampling-expansion-observed-curve-nca-1'
GOALS = ('cmax', 'auc_inf', 'terminal_half_life')
DEPENDENT = ('auc_inf', 'clearance', 'vz', 'mrt', 'vss')


def _num(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(name + ' must be finite numeric')
    value = float(value)
    if value < 0 or positive and value == 0:
        raise ValueError(name + ' must be nonnegative/positive')
    return value


def _times(values):
    result = [_num(t, 'time') for t in values]
    if len(result) < 2 or any(b <= a for a, b in zip(result, result[1:])):
        raise ValueError('At least two strictly increasing actual sampling times required')
    return result


def read_reference(value, project_root=ROOT):
    """Decode only a project-local SHA-bound reference, or an explicit dict."""
    if not isinstance(value, dict):
        raise ValueError('Require a document or SHA-bound project reference')
    if 'path' not in value:
        return deepcopy(value)
    root = Path(project_root).resolve()
    path = (root / value['path']).resolve()
    if not path.is_relative_to(root) or not isinstance(value.get('sha256'), str):
        raise ValueError('Source path/hash is invalid or outside project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('Source hash changed: ' + value['path'])
    data = json.loads(raw)
    pointer = value.get('json_pointer', '')
    if pointer and not pointer.startswith('/'):
        raise ValueError('Invalid JSON pointer')
    for token in pointer.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        data = data[int(token)] if isinstance(data, list) else data[token]
    return data


def _clinical(document, case_id):
    if 'cases' in document:
        matched = [c for c in document['cases'] if c.get('case_id') == case_id]
        if len(matched) != 1:
            raise ValueError('Ambiguous clinical case')
        document = matched[0]
    if document.get('case_id') not in (None, case_id):
        raise ValueError('Clinical case identity differs')
    return document.get('clinical_reference', document)


def _curve_view(raw, *, metadata_only=False):
    """Normalize curated observation rows without reading values during design."""
    if isinstance(raw.get('curve'), dict):
        curve = raw['curve']
        result = {k: deepcopy(curve[k]) for k in ('time_h', 'unit', 'matrix', 'statistic') if k in curve}
        if not metadata_only:
            result['concentration'] = deepcopy(curve['concentration'])
        return result
    if (raw.get('time_unit') != 'min' or raw.get('unit') != 'umol/l'
            or not isinstance(raw.get('observations'), list)):
        raise ValueError('Require a clinical curve or curated min/umol/l observation document')
    result = dict(time_h=[_num(row['time_min'], 'time') / 60 for row in raw['observations']],
                  unit='umol/l', matrix=raw.get('matrix', 'plasma'),
                  statistic=raw.get('statistic', 'repository_aggregate_or_individual'))
    if not metadata_only:
        result['concentration'] = [row['concentration_umol_per_l'] for row in raw['observations']]
    return result


def build_target_contract(case_config, clinical_document, request, molecular_weight_g_mol=None):
    """Export ONLY whitelisted study settings/times, never concentration values."""
    human, dosing = _conditions(request)
    cid = case_config['case_id']; raw = _clinical(clinical_document, cid)
    if raw.get('study_id') not in (None, case_config.get('study_id')):
        raise ValueError('Clinical study identity differs')
    if raw.get('compound_id') not in (None, request['compound_id']):
        raise ValueError('Clinical compound identity differs')
    for key in ('human', 'dosing'):
        if key in raw and raw[key] != request[key]:
            raise ValueError('Clinical target conditions differ: ' + key)
    curve = _curve_view(raw, metadata_only=True)
    times = _times(curve['time_h'])
    mw = _molecular_weight(request)
    if molecular_weight_g_mol is not None and not math.isclose(mw, molecular_weight_g_mol, rel_tol=1e-12):
        raise ValueError('Molecular weight differs')
    policy = case_config.get('nca_policy', {})
    cutoff, cutoff_basis = 0., 'not_reported_positive_points_only'
    for key, basis in (('lloq_ng_ml', 'reported_lloq'), ('lod_ng_ml', 'reported_lod_proxy'),
                       ('calibration_lower_bound_ng_ml', 'reported_assay_lower_bound_proxy')):
        if policy.get(key) is not None:
            cutoff, cutoff_basis = _num(policy[key], key, positive=True), basis
            break
    return dict(version=VERSION, case_id=cid, study_id=case_config.get('study_id'),
        compound_id=request['compound_id'], human=human, dosing=dosing,
        molecular_weight_g_mol=mw, sampling_time_h=times,
        cutoff_ng_ml=cutoff, cutoff_basis=cutoff_basis,
        integration='linear_up_log_down', endpoint_units={g: UNITS[g] for g in GOALS},
        source_kind='target_protocol_metadata_only', target_observation_values_used=False,
        actual_observed_time_zero=times[0] == 0,
        clinical_matrix=curve.get('matrix', 'plasma'),
        clinical_statistic=curve.get('statistic', 'unspecified'))


def _validate_contract(contract, request):
    expected = {'version', 'case_id', 'study_id', 'compound_id', 'human', 'dosing',
        'molecular_weight_g_mol', 'sampling_time_h', 'cutoff_ng_ml', 'cutoff_basis',
        'integration', 'endpoint_units', 'source_kind', 'target_observation_values_used',
        'actual_observed_time_zero', 'clinical_matrix', 'clinical_statistic'}
    human, dosing = _conditions(request)
    if (set(contract) != expected or contract.get('version') != VERSION
            or contract.get('source_kind') != 'target_protocol_metadata_only'
            or contract.get('target_observation_values_used') is not False
            or contract['human'] != human or contract['dosing'] != dosing
            or contract['compound_id'] != request['compound_id']
            or contract['endpoint_units'] != {g: UNITS[g] for g in GOALS}
            or contract['integration'] != 'linear_up_log_down'):
        raise ValueError('Untrusted or changed metadata-only target contract')
    ts = _times(contract['sampling_time_h'])
    if contract['actual_observed_time_zero'] != (ts[0] == 0):
        raise ValueError('Time-zero metadata differs')
    if not math.isclose(contract['molecular_weight_g_mol'], _molecular_weight(request), rel_tol=1e-12):
        raise ValueError('Molecular weight differs')
    _num(contract['cutoff_ng_ml'], 'cutoff')


def observed_curve_nca(*, time_h, concentration_ng_ml, dose_ng, infusion_h,
                       weight_kg, cutoff_ng_ml=0., model_initial_condition_known=False):
    """Same estimator for matched model samples and actual clinical samples.

    Finite AUC covers the first to last supplied sample, including zeros. No
    concentration-dependent window changes. Tail fitting uses positive samples
    above a predeclared cutoff, strictly after infusion and sampled Cmax, and
    exactly the frozen Best Fit suffix rule. Without supplied t=0 only terminal
    half-life is inferable: partial AUC is not misreported as total AUC infinity.
    """
    ts = _times(time_h); cs = [_num(c, 'concentration') for c in concentration_ng_ml]
    if len(ts) != len(cs):
        raise ValueError('Concentration/time lengths differ')
    dose = _num(dose_ng, 'dose', positive=True)
    infusion = _num(infusion_h, 'infusion duration')
    weight = _num(weight_kg, 'weight', positive=True)
    cutoff = _num(cutoff_ng_ml, 'cutoff')
    if not isinstance(model_initial_condition_known, bool):
        raise ValueError('Initial-condition declaration must be boolean')
    initial_known = ts[0] == 0 and (infusion > 0 or cs[0] > 0 or model_initial_condition_known)
    endpoints = {g: dict(value=None, unit=u, status='unavailable', reason='No valid terminal fit') for g, u in UNITS.items()}
    def put(g, value):
        if math.isfinite(value) and (value > 0 or g == 'tmax' and value == 0):
            endpoints[g] = dict(value=float(value), unit=UNITS[g], status='completed', reason='')
    auc, aumc = _integrals(ts, cs)
    peak = max(range(len(ts)), key=lambda i: cs[i])
    for key, value in dict(auc=auc, cmax=cs[peak], tmax=ts[peak], c_last=cs[-1]).items():
        put(key, value)
    diag = dict(method='aggregate_curve_standard_nca', integration='linear_up_log_down',
        sampled_time_h=ts, sampled_concentration_ng_ml=cs,
        window_start_h=ts[0], window_end_h=ts[-1], actual_supplied_time_zero=ts[0] == 0,
        complete_initial_condition=initial_known,
        manufactured_time_zero=False, cutoff_ng_ml=cutoff, dose_ng=dose,
        infusion_h=infusion, weight_kg=weight, auc_window_ng_h_ml=auc,
        aumc_window_ng_h2_ml=aumc, target_values_used_for_model_fitting=False,
        terminal_selection='Post-infusion/post-peak positive suffixes >=3; max adjusted R2; longest within 0.0001',
        clinical_interpretation='NCA of one aggregate curve, not original paper scalar or mean individual NCA')
    tail = [i for i, c in enumerate(cs) if c > 0 and c >= cutoff and ts[i] > infusion and ts[i] > ts[peak]]
    fits = []
    for n in range(3, len(tail) + 1):
        ids = tail[-n:]
        fit = _tail_fit([ts[i] for i in ids], [cs[i] for i in ids])
        if fit is not None:
            fits.append(fit)
    diag['terminal_candidates'] = fits
    if fits:
        best = max(f['adjusted_r_squared'] for f in fits)
        fit = max((f for f in fits if f['adjusted_r_squared'] >= best - .0001), key=lambda f: f['n'])
        lam = fit['lambda_z_per_h']; half = math.log(2) / lam
        put('terminal_half_life', half)
        diag.update(selected_tail=fit, lambda_z_per_h=lam,
            terminal_span_in_half_lives=(fit['end_h'] - fit['start_h']) / half)
        # A zero/BLQ final point cannot be used to invent the terminal tail area.
        if cs[-1] > 0 and cs[-1] >= cutoff:
            tail_auc = cs[-1] / lam
            tail_aumc = cs[-1] * (ts[-1] / lam + 1 / lam**2)
            partial = auc + tail_auc
            diag['first_sample_to_infinity_auc_ng_h_ml'] = partial
            if initial_known and partial > 0:
                total_moment = aumc + tail_aumc
                cl = dose / partial / 1000
                mrt = total_moment / partial - infusion / 2
                for key, value in dict(auc_inf=partial, clearance=cl / weight,
                                      vz=cl / lam / weight, mrt=mrt, vss=cl * mrt / weight).items():
                    put(key, value)
                diag.update(auc_extrapolated_fraction=tail_auc / partial,
                    aumc_extrapolated_fraction=tail_aumc / total_moment,
                    absolute_clearance_l_h=cl)
            elif ts[0] > 0:
                for key in DEPENDENT:
                    endpoints[key]['reason'] = 'No observed time zero: initial exposure unknown; no clinical area imputation'
            else:
                for key in DEPENDENT:
                    endpoints[key]['reason'] = 'A zero at bolus time is not an observed post-bolus initial concentration'
        else:
            for key in DEPENDENT:
                endpoints[key]['reason'] = 'Final supplied concentration is zero/below declared cutoff; no tail area imputation'
    if diag.get('terminal_span_in_half_lives', float('inf')) < 1:
        weak = ['terminal_half_life']
    else:
        weak = []
    weak = sorted(set(weak + weak_endpoints(diag)))
    return dict(status='completed' if all(e['value'] is not None for e in endpoints.values()) else 'partial',
                endpoints=endpoints, diagnostics=diag, weak_endpoints=weak)


def _native_samples(contract, predicted_curve):
    if predicted_curve.get('unit') != 'umol/l' or predicted_curve.get('time_unit') != 'min':
        raise ValueError('Native predictions must use min and umol/l')
    times, values, merged = prediction_in_hours(predicted_curve, contract['molecular_weight_g_mol'])
    _times(times)
    if len(times) != len(values) or any(_num(v, 'native concentration') < 0 for v in values):
        raise ValueError('Invalid native prediction')
    actual = contract['sampling_time_h']
    sampled = [_sample(times, values, t) for t in actual]
    return times, values, sampled, merged


def prediction_nca(contract, predicted_curve, request):
    _validate_contract(contract, request)
    times, values, sampled, merged = _native_samples(contract, predicted_curve)
    kwargs = dict(dose_ng=request['dosing']['dose_mg'] * 1e6,
                  infusion_h=request['dosing']['infusion_minutes'] / 60,
                  weight_kg=request['human']['weight_kg'], cutoff_ng_ml=contract['cutoff_ng_ml'],
                  model_initial_condition_known=True)
    result = observed_curve_nca(time_h=contract['sampling_time_h'], concentration_ng_ml=sampled, **kwargs)
    # The model has an actual initial condition. Use it only for model-only
    # full-exposure endpoints, leaving matched finite-window endpoints intact.
    if contract['sampling_time_h'][0] > 0 and times[0] == 0:
        complete = observed_curve_nca(time_h=[0., *contract['sampling_time_h']],
            concentration_ng_ml=[values[0], *sampled], **kwargs)
        for key in DEPENDENT:
            result['endpoints'][key] = complete['endpoints'][key]
        result['diagnostics']['model_full_exposure_nca'] = complete['diagnostics']
        result['weak_endpoints'] = sorted(set(result['weak_endpoints'] + complete['weak_endpoints']))
    result['diagnostics'].update(coalesced_float_time_duplicates=merged,
        finite_endpoints_matched_to_actual_times=True, model_initial_condition_used_for_full_endpoints=True)
    result['status'] = 'completed' if all(e['value'] is not None for e in result['endpoints'].values()) else 'partial'
    return result


def model_goals(contract, predicted_curve, request):
    result = prediction_nca(contract, predicted_curve, request)
    return dict(version=VERSION, values={g: result['endpoints'][g]['value'] for g in GOALS},
        units={g: UNITS[g] for g in GOALS}, reasons={g: result['endpoints'][g]['reason'] for g in GOALS},
        nca_status=result['status'], diagnostics=result['diagnostics'], weak_endpoints=result['weak_endpoints'],
        target_observation_values_used=False)


def acquisition_features(**kwargs):
    """Zero-duration extension: reference-time offset cancels in linear ranking."""
    duration = _num(kwargs['infusion_end'], 'infusion end')
    args = dict(kwargs)
    if duration == 0:
        args['infusion_end'] = 1.
        result = learner.features(**args)
        result[:, learner.FEATURE_NAMES.index('after_infusion')] = 1.
        return result
    return learner.features(**args)


def _curve_ng_ml(curve, mw):
    unit = str(curve.get('unit', '')).replace('µ', 'u').replace('μ', 'u')
    factors = {'ng/mL': 1., 'ng/ml': 1., 'mg/L': 1000., 'mg/l': 1000.,
               'ug/mL': 1000., 'ug/ml': 1000., 'ug/L': 1., 'ug/l': 1.,
               'umol/l': mw, 'umol/L': mw, 'nmol/L': mw / 1000., 'nmol/l': mw / 1000.}
    if unit not in factors:
        raise ValueError('Unsupported observed concentration unit: ' + unit)
    ts = _times(curve['time_h'])
    cs = [_num(c, 'observed concentration') * factors[unit] for c in curve['concentration']]
    if len(ts) != len(cs):
        raise ValueError('Observed arrays differ')
    return ts, cs


def score_case(case, prediction_unit, *, project_root=ROOT):
    """Called only after all prediction locks; never performs fitting/selection."""
    request = read_reference(prediction_unit['request'], project_root)
    native = read_reference(prediction_unit['prediction'], project_root)
    source = read_reference(case['reference_source'], project_root)
    raw = _clinical(source, case['case_id'])
    contract = (read_reference(case['target_contract'], project_root) if case.get('target_contract')
                else build_target_contract(case, source, request))
    _validate_contract(contract, request)
    clinical_curve = _curve_view(raw)
    times, observed = _curve_ng_ml(clinical_curve, contract['molecular_weight_g_mol'])
    if times != contract['sampling_time_h']:
        raise ValueError('Observed times changed after metadata lock')
    matrix = clinical_curve.get('matrix', 'plasma')
    approximations = set(case.get('approximations', [])) | set(source.get('approximations', []))
    if matrix != 'plasma' and not (matrix == 'serum' and 'serum_as_plasma' in approximations):
        raise ValueError('Clinical matrix differs from plasma without explicit approximation')
    pred = prediction_nca(contract, native, request)
    obs = observed_curve_nca(time_h=times, concentration_ng_ml=observed,
        dose_ng=request['dosing']['dose_mg'] * 1e6, infusion_h=request['dosing']['infusion_minutes'] / 60,
        weight_kg=request['human']['weight_kg'], cutoff_ng_ml=contract['cutoff_ng_ml'])
    _, _, matched, _ = _native_samples(contract, native)
    valid = [(p, o) for p, o in zip(matched, observed) if o > 0]
    if not valid or any(p <= 0 for p, _ in valid):
        metrics = dict(gmfe=None, ln_rmse=None, n=len(valid), reason='No positive prediction/observation pairs')
    else:
        logs = [math.log(p / o) for p, o in valid]
        metrics = dict(gmfe=math.exp(math.fsum(abs(x) for x in logs) / len(logs)),
            ln_rmse=math.sqrt(math.fsum(x*x for x in logs) / len(logs)), n=len(logs),
            observed_points=len(times), zero_observations_not_log_scored=len(times)-len(logs))
    output, reported_sensitivity = {}, {}
    for key, unit in UNITS.items():
        p, o = pred['endpoints'][key], obs['endpoints'][key]
        pv, ov = p['value'], o['value']
        metric = 'absolute_error_h' if key == 'tmax' else 'fold_error'
        error = None if pv is None or ov is None else abs(pv-ov) if key == 'tmax' else max(pv/ov, ov/pv)
        output[key] = dict(value=pv, unit=unit, error=error, error_metric=metric,
            weak_tail=key in pred['weak_endpoints'], reference_weak_tail=key in obs['weak_endpoints'],
            reason=p['reason'], reference=ov, reference_reason=o['reason'],
            reference_kind='observed_curve_nca' if ov is not None else 'unavailable',
            reference_curve_statistic=clinical_curve.get('statistic', 'unspecified'),
            reference_is_original_reported_scalar=False)
        original = raw.get('endpoints', {}).get(key, {})
        if original.get('value') is not None and original.get('kind') == 'clinical_study_reported':
            value = _num(original['value'], 'reported endpoint', positive=key != 'tmax')
            factor = clinical_unit_factor(key, original['unit'], request['human']['weight_kg'])
            converted = pv * factor if pv is not None else None
            reported_sensitivity[key] = dict(prediction=converted, reference=value, unit=original['unit'],
                reference_kind='source_reported', reference_source=case['reference_source'],
                error=(None if converted is None else abs(converted-value) / (60 if original['unit']=='min' else 1)
                       if key == 'tmax' else max(converted/value, value/converted)),
                error_metric=metric)
    return dict(endpoints=output, curve_metrics=metrics, nca_status=pred['status'],
        nca_diagnostics=pred['diagnostics'], reference_nca_diagnostics=obs['diagnostics'],
        weak_endpoints=pred['weak_endpoints'], reference_weak_endpoints=obs['weak_endpoints'],
        source_reported_sensitivity=reported_sensitivity,
        reference_semantics='NCA of the actual observed curve and stated aggregation; no synthetic observed t0; not mean individual PK',
        checks=dict(actual_observed_schedule_matched=True, reference_values_used_for_prediction=False))
