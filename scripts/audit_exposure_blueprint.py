#!/usr/bin/env python3
"""Read-only independent audit. Refuse every result/target read until complete."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/blueprint_design/exposure_profile_20261008'
NEW = ('fixed_prior', 'direct_relaxed', 'exposure_constrained')
METHODS = NEW + ('paper_pbpk_reference', 'two_compartment_reference')
SAMPLINGS = ('full', 'five')
FORMULAS = ('PKSimStandard', 'RodgersRowland')
ENDPOINTS = ('auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life', 'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
ALL_ENDPOINTS = set(ENDPOINTS) | {'concentration_time_curve'}
PINS = {}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def path(name):
    result = (ROOT / name).resolve()
    check(result.is_relative_to(ROOT), 'Outside project: ' + str(name))
    return result


def read(name):
    return json.loads(path(name).read_text())


def checksum(name, expected):
    item = path(name)
    actual = hashlib.sha256(item.read_bytes()).hexdigest()
    check(actual == expected, 'Hash mismatch: ' + str(name))
    if str(item) in PINS:
        check(PINS[str(item)] == actual, 'File changed during audit: ' + str(name))
    PINS[str(item)] = actual
    return item


def ref(record):
    data = json.loads(checksum(record['path'], record['sha256']).read_text())
    for token in record.get('json_pointer', '').split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        data = data[int(token)] if isinstance(data, list) else data[token]
    return data


def verify_refs(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
            checksum(value['path'], value['sha256'])
        for child in value.values():
            verify_refs(child)
    elif isinstance(value, list):
        for child in value:
            verify_refs(child)


def same(a, b, label, rtol=2e-10, atol=1e-12):
    if a is None or b is None:
        check(a is None and b is None, label + ': null differs')
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        check(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rtol, abs_tol=atol), label + ': numeric mismatch')
    else:
        check(a == b, label + ': mismatch')


def clearance(request):
    weight = request['human']['weight_kg']
    entries = {row['parameter']: row for row in request['evidence']}
    def get(name):
        if name not in entries:
            return None
        row = entries[name]
        check(row['unit'] == 'ml/min/kg', 'Unexpected clearance source unit')
        check(all(row['context'].get(k) == v for k, v in {'species': 'human', 'matrix': 'plasma', 'route': 'intravenous'}.items()), 'Unqualified clearance')
        return row['value'] * weight * 60 / 1000
    renal = get('renal_clearance')
    total = get('total_clearance')
    if total is None:
        total = (get('hepatic_clearance') or 0.) + (renal or 0.)
    check(total > (renal or 0.), 'Nonpositive nonrenal source clearance')
    return total, renal


def bootstrap(values, clusters):
    """Independent drug-equal estimate and whole-study resampling arithmetic."""
    keys = sorted(values)
    names = sorted({clusters[k] for k in keys})
    if not keys:
        return {'estimate': None, 'ci95': None, 'n_cases': 0, 'n_clusters': 0}
    logs = {key: math.log(values[key]) for key in keys}
    result = {'estimate': math.exp(math.fsum(logs.values()) / len(keys)), 'n_cases': len(keys), 'n_clusters': len(names)}
    if len(names) < 2:
        result['ci95'] = None
        return result
    blocks = [[logs[k] for k in keys if clusters[k] == name] for name in names]
    indices = np.random.default_rng(20260929).integers(len(names), size=(2000, len(names)))
    draws = []
    for selection in indices:
        chosen = [value for index in selection for value in blocks[int(index)]]
        draws.append(math.exp(math.fsum(chosen) / len(chosen)))
    result.update(ci95=np.quantile(draws, [.025, .975]).tolist(), resamples=2000, seed=20260929)
    return result


def compare_bootstrap(actual, expected, label):
    for key, value in expected.items():
        if key == 'ci95' and value is not None:
            check(isinstance(actual.get(key), list) and len(actual[key]) == 2, label + ': missing CI')
            for i in range(2):
                same(actual[key][i], value[i], label + ': CI')
        else:
            same(actual.get(key), value, label + ': ' + key)


def paired(actual, numerator, denominator, clusters, label):
    paired_values = {key: numerator[key] / denominator[key] for key in clusters
                     if numerator.get(key) is not None and denominator.get(key) is not None}
    compare_bootstrap(actual, bootstrap(paired_values, clusters), label)
    same(actual['attempted_pairs'], len(clusters), label + ': attempted')
    same(actual['complete_pairs'], len(paired_values), label + ': complete')
    same(actual['incomplete_case_ids'], sorted(set(clusters) - set(paired_values)), label + ': incomplete IDs')
    check(set(actual['per_case_ratio']) == set(paired_values), label + ': ratio cases')
    for key, value in paired_values.items():
        same(actual['per_case_ratio'][key], value, label + ': case ratio')
    same(actual['ratio_below_one'], sum(v < 1 for v in paired_values.values()), label + ': improved')
    same(actual['ratio_above_one'], sum(v > 1 for v in paired_values.values()), label + ': worsened')


def independent_curve_metrics(native, request, observed):
    check(native['unit'] == 'umol/l' and native['time_unit'] == 'min', 'Native curve units')
    mw = next(row['value'] for row in request['evidence'] if row['parameter'] == 'molecular_weight')
    multiplier = {'ng/mL': 1., 'mg/L': .001, 'ug/mL': .001, 'ng/L': 1000.}[observed['unit']]
    raw_t = np.asarray(native['time_min'], dtype=float) / 60.
    raw_c = np.asarray(native['concentration'], dtype=float) * mw * multiplier
    check(raw_t.shape == raw_c.shape and np.isfinite(raw_c).all(), 'Native curve shape/finite')
    times, values = [], []
    for t, c in zip(raw_t, raw_c):
        if times and t == times[-1]:
            same(float(c), values[-1], 'Duplicate float time concentrations', rtol=1e-12)
            continue
        check(not times or t > times[-1], 'Unsorted prediction')
        times.append(float(t)); values.append(float(c))
    target_t = np.asarray(observed['time_h'], dtype=float)
    target_c = np.asarray(observed['concentration'], dtype=float)
    check(target_t.min() >= times[0] and target_t.max() <= times[-1], 'Prediction does not span observations')
    matched = np.interp(target_t, times, values)
    valid = target_c > 0
    check(np.all(target_c >= 0) and np.all(matched[valid] > 0), 'Nonpositive scored concentration')
    logs = np.log(matched[valid]) - np.log(target_c[valid])
    return {'gmfe': math.exp(float(np.mean(np.abs(logs)))), 'ln_rmse': math.sqrt(float(np.mean(logs**2))),
            'n': int(valid.sum()), 'observed_points': len(target_t), 'zero_observations_not_log_scored': int((~valid).sum())}


def csv_scalar(value):
    return None if value == '' else float(value)


def main():
    # This is the sole read before the gate. Do not import report/scoring code.
    state = read(RUN / 'state.json')
    if state.get('status') != 'complete':
        raise SystemExit('Audit refused: state.status must be complete before any result/target read.')
    check(state.get('all_predictions_locked_before_scoring') is True, 'Missing prediction lock declaration')
    protocol = ref(state['protocol'])
    for name, digest in protocol['source_pins'].items(): checksum(name, digest)
    for name, digest in state['output_pins'].items(): checksum(name, digest)
    verify_refs(protocol); verify_refs(state)
    results = read(RUN / 'results.json')
    verify_refs(results)
    same(results['protocol'], state['protocol'], 'Result protocol')
    same(results['locked_at'], state['locked_at'], 'Global lock time')
    check(results['scored_at'] >= state['locked_at'], 'Scored before global lock')
    check(state['completed_at'] >= results['scored_at'], 'Completion before scoring')
    cases = {case['case_id']: case for case in protocol['cases']}
    clusters = {key: case['cluster_id'] for key, case in cases.items()}
    check(len(cases) == 10 and len({c['drug'] for c in cases.values()}) == 10, 'Expected ten unique drugs')
    pletz = [case['cluster_id'] for case in cases.values() if case['drug'].lower() in {'ertapenem', 'ceftriaxone'}]
    check(len(pletz) == 2 and len(set(pletz)) == 1, 'Same-study Ertapenem/Ceftriaxone must share cluster')
    expected_jobs = {f'{cid}__{sampling}__{method}' for cid in cases for sampling in SAMPLINGS for method in NEW}
    check(set(protocol['jobs']) == set(state['units']) == expected_jobs, 'Incomplete 60-unit attempt matrix')
    same(protocol['objective_calls_per_formula'], 16384, 'Formula budget')
    same(protocol['starts_per_formula'], 16, 'Start count')
    same(protocol['development_only_preflight']['actual_simulations'], 21999, 'Separate preflight calls')
    same(protocol['development_only_preflight']['new_native'], 0, 'Preflight native calls')
    check(protocol['development_only_preflight']['target_read'] is False, 'Preflight target leakage')
    sys.path.insert(0, str(ROOT / 'src'))
    from pbpk_kb.blueprint_policy.permeability_model import build_system
    fit_count = calls = renal_checks = transfers = 0
    exposure_errors = []
    for key in sorted(expected_jobs):
        unit = state['units'][key]; job = protocol['jobs'][key]; case = cases[job['case_id']]
        check(unit['status'] in {'predicted', 'failed'}, key + ': nonterminal status')
        same(unit, read(RUN / 'work' / key / 'result.json'), key + ': state/result equality')
        same(unit['protocol_sha256'], state['protocol']['sha256'], key + ': protocol hash')
        check({f['formula'] for f in unit['formula_scores']} == set(FORMULAS), key + ': formula matrix')
        development = ref(case['requests']['evidence']['development'])
        training = ref(job['training_source'])
        check(job['training_source'] == case['sampling_sources'][job['sampling']], key + ': wrong sampling source')
        check(training['role'] == 'calibration_data_reconstruction', key + ': not development observations')
        if job['sampling'] == 'five':
            check(len(training['observations']) == 5, key + ': five-point count')
            full = ref(case['sampling_sources']['full'])
            available = {(r['time_min'], r['concentration_umol_per_l']) for r in full['observations']}
            check(all((r['time_min'], r['concentration_umol_per_l']) in available for r in training['observations']), key + ': five not subset')
        local_calls = local_failures = 0
        for formula in unit['formula_scores']:
            fitted = ref(formula['fit']); fit_count += 1
            same(fitted['protocol_sha256'], state['protocol']['sha256'], key + ': fit protocol')
            same(fitted['training_source'], job['training_source'], key + ': fit uses different data')
            check(fitted.get('no_warm_start') is True, key + ': warm start not excluded')
            check(len(fitted['starts']) == 16 and fitted['call_limit'] == 16384, key + ': fit budget')
            check(fitted['calls'] <= 16384 and fitted['calls'] == sum(s['calls'] for s in fitted['starts']), key + ': uncharged root/Jacobian calls')
            check(all(0 <= s['calls'] <= s['call_limit'] == 1024 for s in fitted['starts']), key + ': start cap')
            check(sum(s['call_limit'] for s in fitted['starts']) == 16384, key + ': allocated budget')
            same(formula['calls'], fitted['calls'], key + ': formula calls')
            same(formula['best'], fitted['best'], key + ': formula best')
            local_calls += fitted['calls']; local_failures += fitted['failures']
            valid = [s['best'] for s in fitted['starts'] if s['best'] is not None]
            if valid:
                same(fitted['best']['objective'], min(v['objective'] for v in valid), key + ': best start score')
            best = fitted['best']
            if best is None: continue
            source_total, source_renal = clearance(fitted['request']); source_nr = source_total - (source_renal or 0.)
            multiplier = math.exp(best['log_parameters'][0])
            if job['method'] == 'exposure_constrained':
                check(fitted.get('target_observations_used') is False and fitted['added_zero_time_or_extrapolated_tail'] is False, key + ': exposure scope')
                check(fitted['visible_training_points'] == len(training['observations']), key + ': visible-point count')
                t = np.array([r['time_min'] / 60 for r in training['observations']])
                mw = next(r['value'] for r in fitted['request']['evidence'] if r['parameter'] == 'molecular_weight')
                c = np.array([r['concentration_umol_per_l'] * mw / 1000 for r in training['observations']])
                observed_auc = float(np.trapezoid(c, t))
                same(fitted['observed_window_auc_mg_h_l'], observed_auc, key + ': visible AUC')
                same(best['observed_window_auc_mg_h_l'], observed_auc, key + ': best AUC')
                relative = abs(best['predicted_window_auc_mg_h_l'] - observed_auc) / observed_auc
                same(best['relative_auc_error'], relative, key + ': AUC error')
                check(relative <= 1e-8, key + ': AUC constraint broken')
                exposure_errors.append(relative)
                same(best['profiled_nonrenal_clearance_l_h'], source_nr * multiplier, key + ': absolute CL')
                same(best['fixed_renal_clearance_l_h'], source_renal, key + ': fixed renal')
                same(fitted['source_clearance']['renal_clearance_l_h'], source_renal, key + ': original renal ledger')
                same(fitted['source_clearance']['total_clearance_l_h'], source_total, key + ': original CL ledger')
        calls += local_calls
        same(unit['resources']['objective_calls'], local_calls, key + ': resource calls')
        same(unit['resources']['failed_calls'], local_failures, key + ': resource failures')
        if unit['status'] != 'predicted': continue
        check(unit['target_observations_used'] is False, key + ': target-used flag')
        selection = ref(unit['selection'])
        check(selection['basis'] == 'development_only' and selection['locked_at'] <= state['locked_at'], key + ': late/invalid selection')
        check(path(unit['prediction']['path']).stat().st_mtime <= state['locked_at'] + .001, key + ': prediction modified after locking')
        check(selection['locked_at'] <= path(unit['prediction']['path']).stat().st_mtime + .001, key + ': prediction predates selection')
        same(selection['parameters'], unit['parameters'], key + ': parameter selection')
        target = ref(unit['request']); total, renal = clearance(target)
        params = unit['parameters']['log_parameters']; effective = (renal or 0.) + (total - (renal or 0.)) * math.exp(params[0])
        same(unit['effective_total_clearance_l_h'], effective, key + ': target CL transfer')
        cache_key = f"{job['case_id']}__target__{unit['formula']}"
        if cache_key in protocol['physiology_cache']:
            physiology = ref(protocol['physiology_cache'][cache_key])
        else:
            physiology = read(RUN / 'native_cache' / cache_key / 'physiology/perfusion_physiology.json')
        system = build_system(physiology, total_clearance_l_h=total, renal_clearance_l_h=renal, family='exchange',
                              log_parameters=dict(zip(('clearance_scale', 'partition_scale', 'bidirectional_exchange_scale'), params)), observation='auto')
        same(system.metadata['clearance_components_l_h']['qualified_renal_fixed'], renal or 0., key + ': engine renal readback')
        same(system.metadata['clearance_components_l_h']['scaled_nonrenal'], effective - (renal or 0.), key + ': engine nonrenal readback')
        renal_checks += int(renal is not None)
        dt, dr = clearance(development)
        sw, tw = development['human']['weight_kg'], target['human']['weight_kg']
        if sw != tw:
            if math.isclose(dt / sw, total / tw, rel_tol=1e-12) and math.isclose((dr or 0.) / sw, (renal or 0.) / tw, rel_tol=1e-12):
                same(effective / ((dr or 0.) + (dt - (dr or 0.)) * math.exp(params[0])), tw / sw, key + ': bodyweight CL transfer')
                transfers += 1
    same(fit_count, 120, 'Total formula fits')
    all_units = {**state['units'], **protocol['controls']}
    records = {(r['case_id'], r['sampling'], r['method']): r for r in results['records']}
    expected_records = {(cid, sampling, method) for cid in cases for sampling in SAMPLINGS for method in METHODS}
    check(len(results['records']) == len(records) == 100 and set(records) == expected_records, 'Incomplete scored attempt matrix')
    with (RUN / 'results.csv').open(newline='') as stream: csv_rows = list(csv.DictReader(stream))
    csv_index = {(r['drug'], r['sampling'], r['method'], r['endpoint']): r for r in csv_rows}
    check(len(csv_rows) == len(csv_index) == 1100, 'CSV must preserve all 100 x 11 rows')
    curve_points = scalar_checks = 0
    for cid, case in cases.items():
        clinical = ref(case['reference_source'])
        if 'cases' in clinical:
            matches = [r for r in clinical['cases'] if r['case_id'] == cid]
            check(len(matches) == 1, cid + ': clinical identity ambiguity'); clinical = matches[0]
        observed = clinical['clinical_reference']
        check(observed['case_id'] == cid and observed['study_id'] == case['study_id'], cid + ': target reference identity')
        reference_curve = observed['curve']; reference_endpoints = observed['endpoints']
        for sampling in SAMPLINGS:
            for method in METHODS:
                key = f'{cid}__{sampling}__{method}'; record = records[cid, sampling, method]; unit = all_units[key]
                same(record['status'], unit['status'], key + ': record status')
                same(record['resources'], unit.get('resources', {}), key + ': reported resources')
                check(set(record['endpoints']) == set(ENDPOINTS), key + ': ten scalar endpoint classes')
                present = {r['endpoint'] for r in csv_rows if r['drug'] == case['drug'] and r['sampling'] == sampling and r['method'] == method}
                check(present == ALL_ENDPOINTS, key + ': eleven CSV endpoint classes')
                if unit['status'] == 'predicted':
                    metrics = independent_curve_metrics(ref(unit['prediction']), ref(unit['request']), reference_curve)
                    for name, value in metrics.items(): same(record['curve_metrics'][name], value, key + ': ' + name)
                    curve_points += metrics['n']
                else:
                    check(record['curve_metrics'].get('gmfe') is None, key + ': failed curve scored')
                    check(all(v['value'] is None and v['error'] is None for v in record['endpoints'].values()), key + ': failed endpoints scored')
                curve_row = csv_index[case['drug'], sampling, method, 'concentration_time_curve']
                same(csv_scalar(curve_row['error']), record['curve_metrics'].get('gmfe'), key + ': CSV curve GMFE')
                same(curve_row['unit'], reference_curve['unit'], key + ': CSV curve unit')
                check(curve_row['prediction'] == curve_row['reference'] == '' and curve_row['metric'] == 'GMFE', key + ': curve scalar fiction')
                for endpoint, pred in record['endpoints'].items():
                    original = reference_endpoints.get(endpoint, {'value': None})
                    same(pred['reference'], original['value'], key + ': original reference ' + endpoint)
                    if original['value'] is not None:
                        same(pred['unit'], original['unit'], key + ': endpoint reference unit ' + endpoint)
                    p, o = pred['value'], pred['reference']
                    if p is None or o is None: error = None
                    elif endpoint == 'tmax': error = abs(p - o) / (60 if pred['unit'] == 'min' else 1)
                    else:
                        check(p > 0 and o > 0, key + ': nonpositive FE values'); error = max(p / o, o / p)
                    same(pred['error'], error, key + ': independent endpoint error ' + endpoint)
                    scalar_checks += int(error is not None)
                    row = csv_index[case['drug'], sampling, method, endpoint]
                    same(row['unit'], pred.get('unit') or '', key + ': CSV endpoint unit ' + endpoint)
                    for label, value in (('prediction', p), ('reference', o), ('error', error)):
                        same(csv_scalar(row[label]), value, key + ': CSV ' + endpoint + '/' + label)
                    same(row['status'], record['status'], key + ': CSV failure status')
    values = {(s, m): {cid: records[cid, s, m]['curve_metrics'].get('gmfe') for cid in cases} for s in SAMPLINGS for m in METHODS}
    for sampling in SAMPLINGS:
        for method in METHODS:
            available = {cid: value for cid, value in values[sampling, method].items() if value is not None}
            summary = results['summary'][sampling][method]
            compare_bootstrap(summary, bootstrap(available, clusters), sampling + '/' + method)
            same(summary['attempted'], 10, 'Failure denominator')
            same(summary['failed_or_unscorable'], 10 - len(available), 'Unscorable denominator')
            same(summary['within_twofold'], sum(v <= 2 for v in available.values()), 'Curve twofold count')
            if method != 'exposure_constrained':
                paired(results['comparisons'][sampling][method], values[sampling, 'exposure_constrained'], values[sampling, method], clusters, 'method comparison')
            subset = {cid: cluster for cid, cluster in clusters.items() if cases[cid]['drug'].lower() not in {'nemonoxacin', 'dexmedetomidine', 'midazolam'}}
            paired(results['source_sensitivity_excluding_nemonoxacin_dexmedetomidine_midazolam'][sampling][method], values[sampling, 'exposure_constrained'], values[sampling, method], subset, 'source sensitivity')
            for endpoint in ENDPOINTS:
                entries = [records[cid, sampling, method]['endpoints'][endpoint] for cid in cases]
                errors = [entry['error'] for entry in entries if entry['error'] is not None]
                summary_ep = results['endpoint_summary'][sampling][method][endpoint]
                expected = {'attempted': 10, 'reference_available': sum(e['reference'] is not None for e in entries), 'predicted': sum(e['value'] is not None for e in entries), 'scored': len(errors), 'weak_scored': sum(bool(e.get('weak_tail')) and e['error'] is not None for e in entries)}
                if errors:
                    if endpoint == 'tmax': expected['mean_absolute_error_h'] = math.fsum(errors) / len(errors)
                    else: expected.update(gmfe=math.exp(math.fsum(math.log(v) for v in errors) / len(errors)), within_twofold=sum(v <= 2 for v in errors))
                for field, value in expected.items(): same(summary_ep[field], value, 'Endpoint summary ' + endpoint + '/' + field)
    for method in METHODS:
        paired(results['five_over_full_gmfe'][method], values['five', method], values['full', method], clusters, 'five/full')
        selected = [record for record in records.values() if record['method'] == method]
        for field in ('objective_calls', 'failed_calls', 'new_native', 'wall_seconds'):
            same(results['resources'][method][field], sum(record['resources'].get(field, 0) for record in selected), method + ': aggregate resource ' + field)
    # Recheck every source/output already accessed at the end, not only initially.
    for name, digest in list(PINS.items()): checksum(name, digest)
    print(json.dumps({'audit': 'passed', 'new_units': 60, 'formula_fits': fit_count, 'records': len(records), 'csv_rows': len(csv_rows),
        'hashes_verified': len(PINS), 'curve_observation_comparisons': curve_points, 'scalar_errors_recomputed': scalar_checks,
        'main_actual_simulations': calls, 'preflight_actual_simulations': 21999, 'exposure_auc_max_relative_error': max(exposure_errors),
        'fixed_renal_target_readbacks': renal_checks, 'bodyweight_transfer_checks': transfers, 'bootstrap_resamples': 2000,
        'study_clusters': len(set(clusters.values())), 'target_scoring_after_all_prediction_locks': True,
        'failure_counts': dict(Counter(r['status'] for r in records.values())),
        'limitations': ['Independent arithmetic audit; historical access cannot be proven by timestamps alone.',
                       'Scalar errors are recomputed from recorded predictions and original clinical references; NCA estimator is not independently reimplemented.']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
