#!/usr/bin/env python3
"""Score a completely locked cross-drug pilot, without fitting or imputation.

Prediction bundles are SHA-pinned once, and their fields are addressed using
JSON pointers. Original reported PK and summaries of digitized curves remain
distinct. Missing clinical terminal values never acquire synthetic labels.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprints.source_workflow import inside, save, sha256
from pbpk_kb.blueprints.clinical_prediction_nca import UNITS
from render_current_clinical_table import native_value
from render_pbpk_method_comparison import curve_metrics, interpolate
import report_blueprint_validation as legacy
from score_expanded_tail import _arrays, _equal_conditions, _window_values, WINDOW_ENDPOINTS

VERSION = 'cross-drug-clinical-report-1'
CODE_FILES = ('scripts/score_cross_drug_blueprint.py', 'scripts/score_expanded_tail.py',
    'scripts/report_blueprint_validation.py', 'scripts/render_current_clinical_table.py',
    'scripts/render_pbpk_method_comparison.py', 'scripts/run_clinical_method_comparison.py',
    'scripts/summarize_additional_clinical_baselines.py',
    'src/pbpk_kb/blueprints/clinical_prediction_nca.py',
    'src/pbpk_kb/blueprints/extended_prediction.py')
METHODS = ('rf_only', 'pooled_mean', 'ridge')
CONDITIONS = ('zero', 'five')
EXCLUDED = frozenset({'Dexmedetomidine', 'Midazolam', 'Nemonoxacin',
                     'Sufentanil', 'Sildenafil', 'Isoniazid', 'Digoxin'})


def _pointer(reference, *tokens):
    return {**reference, 'json_pointer': '/' + '/'.join(
        str(t).replace('~', '~0').replace('/', '~1') for t in tokens)}


def validate_locked(protocol, state):
    """This gate runs before any clinical reference is decoded."""
    if state.get('all_predictions_locked_before_scoring') is not True:
        raise ValueError('All predictions must be locked before scoring')
    cases = protocol['cases']
    if len(cases) != 20 or len({c['case_id'] for c in cases}) != 20:
        raise ValueError('The predeclared twenty distinct cases are required')
    if set(state.get('prediction_bundles', {})) != {c['case_id'] for c in cases}:
        raise ValueError('Prediction bundle coverage differs from protocol')
    if set(protocol['methods']) != set(METHODS) or set(protocol['conditions']) != set(CONDITIONS):
        raise ValueError('Changed method or observation conditions')


def _native_fields(entry):
    request, native, dense = entry['request'], entry['curve'], entry['endpoints']
    if native.get('unit') != 'umol/l' or native.get('time_unit') != 'min':
        raise ValueError('Native curve units differ')
    _arrays(native['time_min'], native['concentration'], 'Native curve')
    if (dense.get('comparison_observations_used') is not False or
            dense.get('endpoint_estimator_uses_observations') is not False):
        raise ValueError('Endpoint estimation must not use target observations')
    evidence = [e for e in request['evidence'] if e['parameter'] == 'molecular_weight']
    if len(evidence) != 1 or evidence[0]['unit'] != 'g/mol' or not legacy.positive(evidence[0]['value']):
        raise ValueError('Unique canonical molecular weight required')
    if request['dosing']['route'] != 'intravenous' or request['dosing'].get('start_h', 0) != 0:
        raise ValueError('Only single IV dosing from zero is in scope')
    return request, native, dense, evidence[0]['value']


def score_curve_reference(case, entry, store):
    """Finite observed window only: do not extrapolate missing clinical labels."""
    request, native, dense, mw = _native_fields(entry)
    ref = store.ref(case['reference_source'])
    if (ref.get('compound_id') != case['compound_id'] or ref.get('study_id') != case['study_id']
            or request['compound_id'] != case['compound_id'] or
            ref.get('role') != 'historical_cross_study_reference'):
        raise ValueError('Clinical case/study identity differs')
    if (ref.get('matrix'), ref.get('unit'), ref.get('time_unit')) != ('plasma', 'umol/l', 'min'):
        raise ValueError('Clinical curve units or matrix differ')
    _equal_conditions(request['human'], ref['human'], 'human')
    _equal_conditions(request['dosing'], ref['dosing'], 'dosing')
    if not math.isclose(mw, ref['molecular_weight_g_mol'], rel_tol=1e-12):
        raise ValueError('Clinical/request molecular weights differ')
    store.verify_refs(ref)
    t = [r['time_min']/60 for r in ref['observations']]
    o = [r['concentration_umol_per_l']*mw for r in ref['observations']]
    _arrays(t, o, 'Clinical curve')
    schedule = case['nca_policy']['sampling_time_h']
    if len(schedule) != len(t) or not np.allclose(schedule, t, rtol=1e-12, atol=1e-12):
        raise ValueError('Clinical sampling schedule differs from protocol')
    times, values, merged = legacy.prediction_in_hours(native, mw)
    _arrays(times, values, 'Prediction curve')
    p = [interpolate(times, values, x) for x in t]
    if any(a <= 0 and b > 0 for a, b in zip(p, o)):
        raise ValueError('Nonpositive prediction at a positive observation')
    common = {'unit': 'ng/mL', 'matrix': 'plasma', 'statistic': 'paired_reference_series'}
    metrics = curve_metrics({**common, 'time_h': times, 'concentration': values},
                            {**common, 'time_h': t, 'concentration': o}, set())
    matched, observed = _window_values(t, p), _window_values(t, o)
    endpoints = {}
    for endpoint, unit in UNITS.items():
        value_native, _, reason = native_value(dense['endpoints'][endpoint], endpoint, unit,
                                              request['human']['weight_kg'])
        available = endpoint in WINDOW_ENDPOINTS
        value, actual = (matched[endpoint], observed[endpoint]) if available else (value_native, None)
        error, metric = legacy.endpoint_error(value, actual, endpoint, unit)
        endpoints[endpoint] = {'value': value, 'reference': actual, 'unit': unit,
            'error': error, 'error_metric': metric, 'native_prediction': value_native,
            'reference_kind': 'observation_derived' if available else 'unavailable',
            'prediction_basis': 'matched_observed_grid' if available else 'native_model_only_unscored',
            'reason': reason if value is None else '', 'weak_tail': False,
            'reference_reason': '' if available else 'No curated original reported scalar; no terminal label manufactured.'}
    return {'curve_metrics': metrics, 'endpoints': endpoints, 'scoring_status': 'scored',
            'reference_kind': 'observed_curve', 'nca_status': 'finite_window_only', 'weak_endpoints': [],
            'nca_diagnostics': {'time_h': t, 'prediction_ng_ml': p, 'reference_ng_ml': o,
                'coalesced_float_time_duplicates': merged, 'synthetic_predose_added': False,
                'terminal_reference_reconstruction': False}}


def summarize(records, cases):
    summaries, comparisons = [], {}
    cohorts = {'all_20': cases, 'original_10': [c for c in cases if c['cohort'] == 'original'],
        'extension_10': [c for c in cases if c['cohort'] == 'extension'],
        'source_sensitivity': [c for c in cases if c['drug'] not in EXCLUDED]}
    for name, group in cohorts.items():
        ids = {c['case_id'] for c in group}
        clusters = {c['case_id']: c.get('cross_drug_cluster_id', c['cluster_id']) for c in group}
        scores = {}
        for condition in CONDITIONS:
            for method in METHODS:
                rows = [r for r in records if r['case_id'] in ids and r['condition'] == condition and r['method'] == method]
                values = {r['case_id']: r['curve_metrics']['gmfe'] for r in rows
                    if legacy.positive(r.get('curve_metrics', {}).get('gmfe'))}
                scores[condition, method] = values
                ep = {}
                for row in rows:
                    for key, value in row.get('endpoints', {}).items():
                        kind = value['reference_kind']
                        item = ep.setdefault(key+'__'+kind, {'endpoint': key, 'reference_kind': kind,
                            'requested': len(group), 'predicted': 0, 'reference_available': 0,
                            'weak_scored': 0, 'errors': []})
                        item['predicted'] += value.get('value') is not None
                        item['reference_available'] += value.get('reference') is not None
                        if value.get('error') is not None:
                            item['errors'].append(value['error'])
                            item['weak_scored'] += bool(value.get('weak_tail'))
                for item in ep.values():
                    errors = item.pop('errors'); item['scored'] = len(errors)
                    if errors:
                        if item['endpoint'] == 'tmax': item['mean_absolute_error_h'] = float(np.mean(errors))
                        else:
                            item['gmfe'] = math.exp(float(np.mean(np.log(errors))))
                            item['within_twofold'] = sum(e <= 2 for e in errors)
                summaries.append({'cohort': name, 'condition': condition, 'method': method,
                    'requested': len(group), 'scored': len(values), 'failed_or_unscorable': len(group)-len(values),
                    'curve_gmfe': legacy.clustered_geomean(values, {k: clusters[k] for k in values}, seed=20260930),
                    'within_twofold': sum(v <= 2 for v in values.values()), 'endpoint_errors': ep})
        comparisons[name] = {
            'learned_over_rf': {cond: {m: legacy.paired_ratio(scores[cond,m], scores[cond,'rf_only'],
                clusters, expected_ids=ids, seed=20260930) for m in METHODS[1:]} for cond in CONDITIONS},
            'five_over_zero': {m: legacy.paired_ratio(scores['five',m], scores['zero',m], clusters,
                expected_ids=ids, seed=20260930) for m in METHODS}}
    return summaries, comparisons


def score(run, project_root=ROOT):
    root = Path(project_root).resolve(); run = inside(run, root, exists=True)
    store = legacy.Store(root)
    protocol = store.read(str((run/'protocol.json').relative_to(root)))
    state = store.read(str((run/'state.json').relative_to(root)))
    if store.ref(state['protocol']) != protocol:
        raise ValueError('Protocol receipt differs')
    validate_locked(protocol, state)
    for name, digest in {**protocol['source_pins'], **protocol['paper_guard']}.items():
        store.checksum(name, digest)
    # Validate EVERY unit before decoding ANY clinical reference.
    loaded = []
    for case in protocol['cases']:
        reference = state['prediction_bundles'][case['case_id']]
        bundle = store.ref(reference)
        if bundle.get('case_id') != case['case_id']:
            raise ValueError('Prediction bundle case differs')
        entries = bundle['entries']
        iterator = entries.items() if isinstance(entries, dict) else enumerate(entries)
        seen = set()
        for token, entry in iterator:
            pair = (entry['method'], entry['condition'])
            if pair in seen or pair[0] not in METHODS or pair[1] not in CONDITIONS:
                raise ValueError('Duplicate or unknown prediction unit')
            seen.add(pair)
            if entry['status'] not in ('predicted', 'failed'):
                raise ValueError('Unfinished prediction')
            if entry['status'] == 'predicted':
                _native_fields(entry)
                if entry['request']['compound_id'] != case['compound_id']:
                    raise ValueError('Prediction compound differs')
            loaded.append((case, reference, token, entry))
        if seen != {(m, c) for m in METHODS for c in CONDITIONS}:
            raise ValueError('Incomplete prediction condition matrix')
    rows = []
    for case, reference, token, entry in loaded:
        row = {k: case[k] for k in ('case_id', 'compound_id', 'drug', 'cohort', 'cluster_id')}
        row.update(method=entry['method'], condition=entry['condition'], attempt_status=entry['status'],
                   prediction_source=_pointer(reference, 'entries', token),
                   scoring_status='failed', failure=entry.get('failure'), endpoints={})
        if entry['status'] == 'predicted':
            try:
                if case['reference_kind'] == 'observed_curve':
                    row.update(score_curve_reference(case, entry, store))
                else:
                    clinical, curve, endpoints = legacy._case_reference(case, store)
                    unit = {'prediction': _pointer(reference, 'entries', token, 'curve'),
                            'request': _pointer(reference, 'entries', token, 'request')}
                    output, _, _ = legacy._score_prediction(case, clinical, curve, endpoints, unit, store)
                    output.pop('curve', None)
                    output['scoring_status'] = 'scored' if legacy.positive(output['curve_metrics'].get('gmfe')) else 'partial'
                    output['reference_kind'] = 'original_reported'
                    for endpoint, value in output['endpoints'].items():
                        value['reference_kind'] = 'original_reported' if value['reference'] is not None else 'unavailable'
                        value['native_prediction'], _, value['native_prediction_reason'] = native_value(
                            entry['endpoints']['endpoints'][endpoint], endpoint, value['unit'], entry['request']['human']['weight_kg'])
                    row.update(output)
            except (ValueError, ArithmeticError, KeyError) as exc:
                if 'Source lock mismatch:' in str(exc) or 'Source changed during reporting:' in str(exc):
                    raise
                row.update(scoring_status='scoring_failed', failure=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    summary, comparisons = summarize(rows, protocol['cases'])
    result = {'version': VERSION, 'protocol': state['protocol'], 'units': rows,
        'summary': summary, 'paired_comparisons': comparisons,
        'predictions_locked_before_scoring': True, 'paper_updated': False,
        'automatic_promotion': False, 'clinical_risk_calibrated': False,
        'limitations': ['Previously inspected twenty drugs; grouped development cross-validation, not an untouched test.',
            'No target clinical curve or reported scalar used for parameter fitting.',
            'Zero-point and five-point tasks must be reported separately.',
            'Teachers are PBPK fits to training-drug development curves, not observed parameter labels.',
            'RF excludes all twenty drugs; pooled/ridge stage has at most nineteen training drugs.',
            'No accuracy is assigned to endpoints without clinical references; no organ exposure validation.',
            'Protocol changes inputs and formula: do not compare its scores directly to prior full-curve GMFE.'],
        # The controller updates state with the result receipt after return;
        # preserve immutable bundle/protocol pins, not that mutable state hash.
        'source_pins': {p: h for p, h in store.pins.items()
                        if p != str((run/'state.json').relative_to(root))}}
    store.unchanged()
    save(run/'results.json', result)
    fields = ['drug','case_id','method','condition','status','endpoint','reference_kind','unit',
              'prediction','native_prediction','observation','error','error_metric','curve_gmfe','curve_ln_rmse','n','weak_tail','reason']
    with (run/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in rows:
            base = {k: row[k] for k in ('drug','case_id','method','condition')}
            metrics = row.get('curve_metrics', {})
            writer.writerow({**base, 'status': row['scoring_status'], 'endpoint': 'concentration_time_curve',
                'reference_kind': 'clinical_observed_curve', 'curve_gmfe': metrics.get('gmfe'),
                'curve_ln_rmse': metrics.get('ln_rmse'), 'n': metrics.get('n'), 'reason': row.get('failure')})
            for endpoint in UNITS:
                item = row.get('endpoints', {}).get(endpoint, {})
                writer.writerow({**base, 'status': row['scoring_status'], 'endpoint': endpoint,
                    'reference_kind': item.get('reference_kind','unavailable'), 'unit': item.get('unit',UNITS[endpoint]),
                    'prediction': item.get('value'), 'native_prediction': item.get('native_prediction'),
                    'observation': item.get('reference'), 'error': item.get('error'), 'error_metric': item.get('error_metric'),
                    'weak_tail': item.get('weak_tail'), 'reason': item.get('reason') or item.get('reference_reason') or row.get('failure')})
    return result
