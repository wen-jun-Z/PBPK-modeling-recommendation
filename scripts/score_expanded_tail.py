#!/usr/bin/env python3
"""Score locked predictions without conflating reported and curve-derived PK.

The legacy cohort keeps its original clinical-NCA protocol. For curve-only
references, four finite-window summaries use the same unaltered observation
times for P and O. No predose point, assay threshold or terminal clinical label
is invented. Model-only terminal predictions remain visible but unscored.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS, _integrals
from pbpk_kb.blueprints.evidence_blueprint import EvidenceBlueprintRequest
from render_current_clinical_table import native_value
from render_pbpk_method_comparison import curve_metrics, interpolate
import report_blueprint_validation as legacy

VERSION = 'expanded-tail-clinical-scoring-1'
WINDOW_ENDPOINTS = frozenset({'auc', 'cmax', 'tmax', 'c_last'})
REFERENCE_KIND = 'observation_derived'
CODE_FILES = ('scripts/score_expanded_tail.py', 'scripts/report_blueprint_validation.py',
    'scripts/render_current_clinical_table.py', 'scripts/render_pbpk_method_comparison.py',
    'src/pbpk_kb/blueprints/clinical_prediction_nca.py')


def _finite(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _equal_conditions(actual, expected, label):
    if set(actual) != set(expected):
        raise ValueError(label + ' fields differ from the reference conditions')
    for key, value in expected.items():
        current = actual[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            equal = isinstance(current, (int, float)) and not isinstance(current, bool) and math.isclose(
                current, value, rel_tol=1e-12, abs_tol=1e-12)
        else:
            equal = current == value
        if not equal:
            raise ValueError(label + '/' + key + ' differs from the reference conditions')


def _arrays(times, values, label):
    if (not isinstance(times, list) or not isinstance(values, list) or len(times) != len(values)
            or len(times) < 2 or not all(_finite(x) for x in times + values)
            or any(b <= a for a, b in zip(times, times[1:]))
            or sum(x > 0 for x in values) < 2):
        raise ValueError(label + ' must contain ordered finite paired concentrations and two positive values')


def _window_values(times, values):
    # This integrates only the observed interval, with no extrapolation to zero.
    peak = max(range(len(values)), key=lambda i: values[i])
    auc, _ = _integrals(times, values)
    return {'auc': auc, 'cmax': values[peak], 'tmax': times[peak], 'c_last': values[-1]}


def _verify_completion(computation, request, store):
    if 'completion' not in computation:
        raise ValueError('A completed prediction receipt is required before scoring')
    receipt = store.ref(computation['completion'])
    construction = receipt.get('construction', {})
    if construction.get('request') != request or construction.get('mapping') != computation.get('mapping'):
        raise ValueError('Prediction receipt belongs to another request or mapping')
    pins = receipt.get('artifact_pins', {})
    for name in ('prediction', 'dense_endpoints'):
        reference = computation[name]
        if pins.get(reference['path']) != reference['sha256']:
            raise ValueError('Prediction/endpoint output differs from its locked completion receipt')
    for path, digest in pins.items():
        store.checksum(path, digest)


def _observed_curve(case, computation, store):
    reference = store.ref(case['reference_source'])
    request_raw = store.ref(computation['request'])
    request = EvidenceBlueprintRequest.model_validate(request_raw).model_dump(mode='json', exclude_none=True)
    if (reference.get('compound_id') != case['compound_id']
            or reference.get('study_id') != case['study_id']
            or request['compound_id'] != case['compound_id']):
        raise ValueError('Reference curve, case and prediction identities differ')
    if reference.get('role') != 'historical_cross_study_reference':
        raise ValueError('A separately declared reference curve is required')
    if (reference.get('matrix'), reference.get('unit'), reference.get('time_unit')) != ('plasma', 'umol/l', 'min'):
        raise ValueError('Reference must be plasma in min and umol/l')
    if request['dosing']['route'] != 'intravenous' or request['dosing'].get('start_h', 0) != 0:
        raise ValueError('This reference protocol supports single intravenous dosing from time zero only')
    _equal_conditions(request['human'], reference['human'], 'human')
    _equal_conditions(request['dosing'], reference['dosing'], 'dosing')
    store.verify_refs(reference)
    _verify_completion(computation, request_raw, store)
    mw = reference.get('molecular_weight_g_mol')
    drug_mw = [r['value'] for r in request['evidence'] if r['parameter'] == 'molecular_weight']
    physchem = store.ref(case['physchem']) if isinstance(case['physchem'], dict) else store.read(case['physchem'])
    if (not _finite(mw, positive=True) or len(drug_mw) != 1
            or not math.isclose(mw, drug_mw[0], rel_tol=1e-12)
            or not math.isclose(mw, physchem['identity']['molecular_weight_g_mol'], rel_tol=1e-12)):
        raise ValueError('Reference, request and physicochemical molecular weights differ')
    rows = reference['observations']
    if any(not _finite(r['time_min']) or not _finite(r['concentration_umol_per_l']) for r in rows):
        raise ValueError('Reference observation fields must be finite nonnegative numbers')
    observed_times = [r['time_min'] / 60 for r in rows]
    observed = [r['concentration_umol_per_l'] * mw for r in rows]  # umol/L -> ng/mL
    _arrays(observed_times, observed, 'Reference curve')
    if any(r.get('unit', 'umol/l') != 'umol/l' or r.get('time_unit', 'min') != 'min' for r in rows):
        raise ValueError('Observation row units disagree with the reference')
    policy = case['nca_policy']
    schedule = policy['sampling_time_h']
    if len(schedule) != len(observed_times) or any(not _finite(t) or not math.isclose(t, o, rel_tol=1e-12, abs_tol=1e-12)
                                                for t, o in zip(schedule, observed_times)):
        raise ValueError('Frozen digitized sampling times differ from the reference curve')
    for key, value in [('weight_kg', request['human']['weight_kg']),
                       ('dose_ng', request['dosing']['dose_mg'] * 1e6),
                       ('infusion_h', request['dosing']['infusion_minutes'] / 60)]:
        if key in policy and (not _finite(policy[key]) or not math.isclose(policy[key], value, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Frozen sampling policy has inconsistent ' + key)
    native = store.ref(computation['prediction'])
    if native.get('unit') != 'umol/l' or native.get('time_unit') != 'min':
        raise ValueError('Native curve must be in min and umol/l')
    if (len(native.get('time_min', [])) != len(native.get('concentration', []))
            or not all(_finite(v) for v in native.get('time_min', []) + native.get('concentration', []))):
        raise ValueError('Native curve requires finite paired arrays of equal length')
    times, concentrations, merged = legacy.prediction_in_hours(native, mw)
    _arrays(times, concentrations, 'Prediction curve')
    prediction = [interpolate(times, concentrations, t) for t in observed_times]
    _arrays(observed_times, prediction, 'Matched prediction')
    if any(p <= 0 and o > 0 for p, o in zip(prediction, observed)):
        raise ValueError('Prediction is nonpositive at a positive reference concentration')
    # These labels describe paired series, not an asserted equivalence between
    # a representative adult and a mean/individual with unknown covariates.
    common = {'unit': 'ng/mL', 'matrix': 'plasma', 'statistic': 'paired_reference_series'}
    curve = {**common, 'time_h': times, 'concentration': concentrations}
    measured = {**common, 'time_h': observed_times, 'concentration': observed}
    metrics = curve_metrics(curve, measured, set())
    metrics.update(reference_kind='clinical_observed_curve', reference_statistic=case.get('reference_statistic', 'unspecified'))
    dense = store.ref(computation['dense_endpoints'])
    if dense.get('comparison_observations_used') is not False or dense.get('endpoint_estimator_uses_observations') is not False:
        raise ValueError('Native endpoint receipt does not establish observation-free endpoint estimation')
    observed_values, matched_values = _window_values(observed_times, observed), _window_values(observed_times, prediction)
    result = {}
    definitions = {'auc': 'linear_up_log_down_AUC_first_to_last_observed_time',
        'cmax': 'maximum_on_observed_time_grid_not_continuous_Cmax',
        'tmax': 'earliest_maximum_on_observed_time_grid_not_continuous_Tmax',
        'c_last': 'concentration_at_last_observed_time_not_assay_based_last_quantifiable'}
    for endpoint, unit in UNITS.items():
        native_endpoint = dense['endpoints'][endpoint]
        native_prediction, _, reason = native_value(native_endpoint, endpoint, unit, request['human']['weight_kg'])
        has_reference = endpoint in WINDOW_ENDPOINTS
        value = matched_values[endpoint] if has_reference else native_prediction
        observed_value = observed_values[endpoint] if has_reference else None
        error, metric = legacy.endpoint_error(value, observed_value, endpoint, unit)
        result[endpoint] = {'value': value, 'unit': unit, 'reference': observed_value, 'error': error,
            'error_metric': metric, 'weak_tail': False, 'reason': reason if value is None else '',
            'reference_kind': REFERENCE_KIND if has_reference else 'unavailable',
            'reference_reason': '' if has_reference else 'Original clinical scalar not curated; no terminal label derived from the sparse observed curve.',
            'prediction_basis': 'matched_observed_grid' if has_reference else 'native_model_only_unscored',
            'definition': definitions.get(endpoint, native_endpoint.get('definition', 'native_model_terminal_summary')),
            'native_prediction': native_prediction, 'native_prediction_reason': reason,
            'reference_window_h': [observed_times[0], observed_times[-1]] if has_reference else None}
    return {'curve_metrics': metrics, 'endpoints': result, 'scoring_status': 'scored',
        'reference_kind': 'observed_curve', 'nca_status': 'finite_window_only', 'weak_endpoints': [],
        'nca_diagnostics': {'method': VERSION, 'sampling_basis': 'unchanged_digitized_reference_times',
            'sampled_time_h': observed_times, 'sampled_concentration_ng_ml': prediction,
            'reference_concentration_ng_ml': observed, 'window_h': [observed_times[0], observed_times[-1]],
            'integration': 'linear_up_log_down', 'assay_lloq_ng_ml': None, 'synthetic_predose_added': False,
            'terminal_reference_reconstruction': False, 'source': case['reference_source'],
            'coalesced_float_time_duplicates': merged, 'reference_statistic': case.get('reference_statistic', 'unspecified'),
            'aggregation_equivalence_claimed': False,
            'limitation': 'Summaries of one digitized reference series; not original reported population PK statistics or a validated true peak.'},
        'checks': {'target_dosing_matches_frozen_policy': True, 'target_human_matches_frozen_reference': True,
                   'units_valid': True, 'target_curve_times_preserved': True, 'reference_scalar_provenance_separated': True}}


def score_case(case, computation, store):
    """Called only after the runner locks the complete prediction/attempt set."""
    if computation.get('status') != 'predicted':
        raise ValueError('Only a completed prediction can be scored')
    kind = case.get('reference_kind', 'original_reported')
    if kind == 'observed_curve':
        return _observed_curve(case, computation, store)
    if kind not in ('original_reported', 'clinical_reported'):
        raise ValueError('Unrecognized clinical reference kind')
    clinical, curve, endpoints = legacy._case_reference(case, store)
    result, _, _ = legacy._score_prediction(case, clinical, curve, endpoints, computation, store)
    result = deepcopy(result); result.pop('curve', None)
    result.update(scoring_status='scored' if legacy.positive(result['curve_metrics'].get('gmfe')) else 'partial',
                  reference_kind='original_reported')
    for endpoint in result['endpoints'].values():
        endpoint['reference_kind'] = 'original_reported' if endpoint['reference'] is not None else 'unavailable'
    return result
