#!/usr/bin/env python3
"""Validate frozen CL comparisons and publish one appendix table, without fitting.

The audit independently aggregates stored per-case errors and paired cluster
bootstraps. It also recalculates curve/scalar scores from the locked trajectories
and clinical references using the frozen, scoring-only endpoint implementation.
``--check-only`` is strictly read-only, including when no table exists yet.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
RUN = ROOT / 'experiments/blueprint_design/rf_gap_comparison_20261012'
TABLE = ROOT / 'results/tables/blueprint_cl_gap_comparison.tex'
METHODS = ('rf', 'tabpfn', 'complementary')
NAMES = {'rf': 'RF (common refit)', 'tabpfn': 'TabPFNv2', 'complementary': 'Complementary CL'}
ENDPOINTS = ('curve', 'auc', 'cmax', 'tmax', 'c_last', 'terminal_half_life',
             'auc_inf', 'clearance', 'vz', 'mrt', 'vss')
COMMON = ENDPOINTS[:6]
UNITS = dict(auc='ng.h/mL', cmax='ng/mL', tmax='h', c_last='ng/mL',
             terminal_half_life='h', auc_inf='ng.h/mL', clearance='L/h/kg',
             vz='L/kg', mrt='h', vss='L/kg')
SEED, DRAWS = 20261012, 2000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    result = (ROOT / value).resolve()
    require(result.is_relative_to(ROOT), 'Outside project: ' + str(value))
    return result


def read(value):
    return json.loads(path(value).read_text())


def sha(value):
    with path(value).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def near(actual, expected, context):
    if actual is None or expected is None:
        require(actual is expected, context + ': missing value mismatch')
    else:
        require(math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=2e-10, abs_tol=1e-12),
                context + ': numeric mismatch')


class Sources:
    """Verify every declared pin, caching only identical path/hash pairs."""
    def __init__(self):
        self.verified = {}

    def check(self, source):
        name, expected = source['path'], source['sha256']
        previous = self.verified.get(name)
        require(previous is None or previous == expected, 'Conflicting source pins: ' + name)
        if previous is None:
            require(sha(name) == expected, 'Source hash changed: ' + name)
            self.verified[name] = expected

    def document(self, source):
        self.check(source)
        return read(source['path'])

    def pins(self, pins):
        for name, digest in pins.items():
            self.check(dict(path=name, sha256=digest))

    def references(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                self.check(value)
            for child in value.values():
                self.references(child)
        elif isinstance(value, list):
            for child in value:
                self.references(child)


def clustered(values, clusters):
    """Independent drug-equal estimate, whole-cluster percentile bootstrap."""
    require(set(values) == set(clusters), 'Misaligned cluster values')
    require(all(math.isfinite(v) and v > 0 for v in values.values()), 'Invalid fold error')
    if not values:
        return dict(n_cases=0, n_clusters=0, estimate=None, ci95=None)
    keys, groups = sorted(values), sorted(set(clusters.values()))
    logs = {k: math.log(values[k]) for k in keys}
    sums = np.asarray([math.fsum(logs[k] for k in keys if clusters[k] == g) for g in groups])
    counts = np.asarray([sum(clusters[k] == g for k in keys) for g in groups])
    result = dict(n_cases=len(keys), n_clusters=len(groups),
                  estimate=math.exp(math.fsum(logs.values()) / len(keys)))
    if len(groups) < 2:
        return dict(**result, ci95=None)
    draws = np.random.default_rng(SEED).integers(len(groups), size=(DRAWS, len(groups)))
    estimates = np.exp(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
    return dict(**result, ci95=np.quantile(estimates, [.025, .975]).tolist(),
                resamples=DRAWS, seed=SEED)


def verify_statistics(actual, stored, context):
    for name in ('n_cases', 'n_clusters', 'resamples', 'seed'):
        if name in stored:
            require(actual.get(name) == stored[name], context + ': ' + name)
    near(actual['estimate'], stored['estimate'], context)
    if actual['ci95'] is None or stored['ci95'] is None:
        require(actual['ci95'] == stored['ci95'], context + ': interval availability')
    else:
        require(len(actual['ci95']) == len(stored['ci95']) == 2, context + ': malformed interval')
        for a, b in zip(actual['ci95'], stored['ci95']):
            near(a, b, context + ': interval')


def parameter_audit(sources):
    p = read(RUN / 'parameter_protocol.json')
    scores = read(RUN / 'parameter_metrics.json')
    validation = read(RUN / 'parameter_validation.json')
    sources.pins(p['source_pins'])
    sources.pins(validation['dependency_pins'])
    lock = sources.document(scores['prediction_lock'])
    sources.references(lock)
    require(p['methods'] == list(METHODS), 'Parameter method set changed')
    require(not lock['parameter_test_labels_scored'], 'Parameter labels opened before lock')
    require(datetime.fromisoformat(lock['all_methods_locked_at']) < datetime.fromisoformat(scores['scored_at']),
            'Parameter scoring precedes lock')
    require(not p['clinical_observation_files_opened'] and not p['label_imputation'], 'Parameter leakage flags')
    require(p['train_cl_n'] == 1006 and p['train_structures'] == 1008 and p['test_structures'] == 100,
            'Unexpected molecular cohort')
    require(p['tabpfn']['input_dimension'] == 500 and not p['tabpfn']['hyperparameter_search'],
            'TabPFN adapter/search changed')
    require('training-only variance' in p['tabpfn']['feature_adapter'], 'Missing training-only feature policy')
    split_ref = read(RUN / 'clinical_inputs.json')['source_split']
    split = sources.document(split_ref)
    rows = {r['index']: r for r in split['rows']}
    require(p['test_indices'] == split['test_indices'], 'Parameter test set changed')
    require(p['train_cl_indices'] == split['label_audit']['human_CL_mL_min_kg']['train_indices'],
            'CL training rows changed')
    train, test = set(p['train_cl_indices']), set(p['test_indices'])
    require(len(train) == 1006 and len(test) == 100 and not train & test, 'Parameter row overlap')
    clinical_id = {r['identity_smiles'] for r in p['clinical_targets']}
    clinical_sc = {r['scaffold_group'] for r in p['clinical_targets']}
    require(len(p['clinical_targets']) == 30, 'Clinical exclusion cohort changed')
    for field, targets in [('identity_smiles', clinical_id), ('scaffold_group', clinical_sc)]:
        require(not ({rows[i][field] for i in train | test} & targets), 'Clinical molecule enters parameter split')
        require(not ({rows[i][field] for i in train} & {rows[i][field] for i in test}), 'Train/test molecule overlap')
    source = split['source']
    sources.check(dict(path=str(path(source['source_path']).relative_to(ROOT)), sha256=source['source_sha256']))
    with zipfile.ZipFile(path(source['source_path'])) as archive:
        raw_csv = archive.read(source['csv_member'])
    require(hashlib.sha256(raw_csv).hexdigest() == source['csv_sha256'], 'Human CL source CSV changed')
    label_rows = list(csv.DictReader(io.StringIO(raw_csv.decode())))
    require(source['endpoint_definitions']['human_CL_mL_min_kg'] == dict(unit='mL/min/kg', input_scale='log10'),
            'Human CL units changed')
    metrics, errors, priors = {}, {}, {}
    stored = {(r['method'], r['row_index']): r for r in scores['predictions']}
    require(len(stored) == len(scores['predictions']) == 300, 'Missing/duplicate parameter prediction')
    for method in METHODS:
        prior = sources.document(lock['priors'][method]); priors[method] = prior
        sources.references(prior)
        require(prior['protocol_sha256'] == sha(RUN / 'parameter_protocol.json'), 'Parameter protocol not locked')
        require(prior['unit'] == 'mL/min/kg' and prior['parameter'] == 'total_clearance', 'Prior CL units')
        require(prior['train_count'] == 1006 and not prior['test_labels_used_for_fit_or_selection']
                and not prior['clinical_observation_files_opened'], 'Prior training policy changed')
        require(set(prior['predictions']) == {r['compound_id'] for r in p['clinical_targets']}, 'Prior clinical coverage')
        require(all(math.isfinite(v) and v > 0 for v in prior['predictions'].values()), 'Invalid CL prior')
        require(datetime.fromisoformat(prior['prediction_locked_at']) <= datetime.fromisoformat(lock['all_methods_locked_at']),
                'Prior locked after all-method lock')
        predictions = {r['row_index']: r['predicted_log10'] for r in prior['parameter_test_predictions']}
        require(set(predictions) == test and len(predictions) == len(prior['parameter_test_predictions']), 'Prior test coverage')
        diffs = []
        errors[method] = {}
        for index in p['test_indices']:
            observed = float(label_rows[index]['human_CL_mL_min_kg'])
            frozen = stored[(method, index)]
            near(observed, frozen['observed_log10'], 'Raw parameter label')
            near(predictions[index], frozen['predicted_log10'], 'Locked parameter prediction')
            delta = predictions[index] - observed; diffs.append(delta)
            errors[method][index] = 10 ** abs(delta)
        values = dict(n=len(diffs), gmfe=10 ** (math.fsum(abs(d) for d in diffs) / len(diffs)),
                      within_2fold=sum(abs(d) <= math.log10(2) for d in diffs) / len(diffs),
                      log10_rmse=math.sqrt(math.fsum(d*d for d in diffs) / len(diffs)))
        for field, value in values.items():
            near(value, scores['metrics'][method][field], 'Parameter ' + method + '/' + field)
        metrics[method] = values
        check = validation['checks'][method]
        require(check['prior_sha256'] == lock['priors'][method]['sha256']
                and check['saved_model_reload_max_log10_difference'] <= check['reload_log10_tolerance'],
                'Saved-model reload receipt failed')
    columns = priors['tabpfn']['audit']['feature_columns']
    require(len(columns) == 500 and len(set(columns)) == 500 and columns == sorted(columns)
            and columns[-6:] == list(range(1024, 1030)), 'Invalid TabPFN feature selection receipt')
    clusters = {i: rows[i]['scaffold_group'] for i in test}
    pairs = {}
    for method in METHODS[1:]:
        ratios = {i: errors[method][i] / errors['rf'][i] for i in sorted(test)}
        actual = clustered(ratios, clusters)
        prior_stats = validation['paired_descriptive_bootstrap']['results'][method]
        near(actual['estimate'], prior_stats['candidate_over_rf_gmfe_ratio'], 'Parameter paired ratio')
        for a, b in zip(actual['ci95'], prior_stats['ci95']):
            near(a, b, 'Parameter paired interval')
        pairs[method] = actual
    return dict(metrics=metrics, paired_to_rf=pairs, n_train_structures=1008, n_train_labels=1006,
                n_test=100, n_test_scaffolds=len(set(clusters.values())),
                train_cl_indices=p['train_cl_indices'], test_indices=p['test_indices'],
                parameter_test_previously_viewed=True, all_30_clinical_molecules_excluded=True,
                feature_policy=p['tabpfn']['feature_adapter'],
                animal_information_is_additional=True, software=validation['software'],
                historical_rf_refit=validation['historical_rf_refit']), lock


def endpoint_error(score, endpoint):
    if score is None:
        return None
    if endpoint == 'curve':
        return score['curve_metrics']['gmfe']
    record = score['endpoints'][endpoint]
    require(record['unit'] == UNITS[endpoint], 'Endpoint unit changed: ' + endpoint)
    predicted, observed = record['value'], record['reference']
    if predicted is None or observed is None:
        require(bool(record['reason'] or record['reference_reason']), 'Missing endpoint reason')
        value = None
    elif endpoint == 'tmax':
        value = abs(predicted - observed)
    else:
        require(predicted > 0 and observed > 0, 'Nonpositive endpoint')
        value = max(predicted / observed, observed / predicted)
    near(value, record['error'], 'Per-case ' + endpoint)
    require(record['error_metric'] == ('absolute_error_h' if endpoint == 'tmax' else 'fold_error'),
            'Endpoint metric changed')
    return value


def clinical_audit(sources, parameter_lock):
    p, state = read(RUN / 'protocol.json'), read(RUN / 'state.json')
    expected = {(m, c) for m in METHODS for c in p['jobs']}
    require(state['status'] == 'complete' and len(expected) == 90
            and set(state['jobs_done']) == {m+'__'+c for m, c in expected}, 'All 90 attempts must finish before publication')
    sources.pins(p['source_pins'])
    sources.references(p)
    result = sources.document(state['results'])
    sources.references(result)
    require(result['protocol']['sha256'] == sha(RUN / 'protocol.json'), 'Clinical protocol changed')
    require(result['priors'] == state['priors'] == parameter_lock['priors'], 'Clinical priors differ from locked parameter predictions')
    require(result['predictions_locked_at'] == state['predictions_locked_at'] < result['scored_at'], 'Clinical scoring before prediction lock')
    require(result['independent_evidence_protected'], 'Independent evidence protection flag')
    require(p['analysis']['cluster_resamples'] == DRAWS and p['analysis']['seed'] == SEED
            and p['analysis']['primary'] == 'curve' and p['analysis']['ties_relative_tolerance'] == 1e-6,
            'Analysis protocol changed')
    records = {(r['method'], r['case_id']): r for r in result['records']}
    require(set(records) == expected and len(result['records']) == 90, 'Clinical rows missing/duplicated')
    require(sum(j['replaceable'] for j in p['jobs'].values()) == 24, 'Changed-input cohort changed')
    # Import only after lock/source validation; this scorer performs NCA, never fitting.
    import blueprint_sampling_expansion_data as scorer
    cases = {c['case_id']: c for c in p['cases']}
    values, failures, fit_failures, bounds, direct_rows, shared_requests = {}, [], [], {}, [], {}
    for method, cid in sorted(expected):
        row, job = records[(method, cid)], p['jobs'][cid]
        rec = state['jobs_done'][method+'__'+cid]
        require(row['result'] == rec['result'] and row['reused'] == rec['reused'] == (not job['replaceable']), 'Incorrect fit reuse')
        require(row['replaceable'] == job['replaceable'] and row['cluster_id'] == cases[cid]['cluster_id'], 'Cohort/cluster mismatch')
        unit = sources.document(row['result'])
        sources.references(unit)
        sources.pins(unit.get('output_pins', {}))
        training = sources.document(job['training'])
        require(len(training['observations']) == 5, 'Development point count changed')
        require(unit['status'] == row['status'], 'Attempt status mismatch')
        if job['replaceable']:
            require(path(row['result']['path']).is_relative_to(RUN / 'fits'), 'Old RF fit reused in changed-input cohort')
            require(unit['prior'] == parameter_lock['priors'][method] and unit['training'] == job['training'], 'Fit inputs changed')
            require(not unit['target_observations_used'] and unit['prediction_locked_at'] <= state['predictions_locked_at'], 'Fit did not lock before scoring')
            require(len(unit['changes']) == 1 and unit['changes'][0]['parameter'] == 'total_clearance'
                    and unit['changes'][0]['unit'].lower() == 'ml/min/kg', 'CL-only replacement failed')
            prior = sources.document(unit['prior'])
            near(unit['changes'][0]['replacement'], prior['predictions'][job['compound_id']], 'CL replacement value')
            for fit in unit['fits']:
                if fit['best'] is None:
                    fit_failures.append(dict(method=method, case_id=cid, formula=fit['formula'],
                                             status=fit.get('status', 'no_feasible_fit'), reason=fit.get('reason', 'No feasible optimum')))
                if 'fit' in fit:
                    detail = fit['fit']
                    require(detail['visible_points'] == 5 and detail['no_warm_start']
                            and not detail['target_observations_used'], 'Calibration protocol violation')
                    key = (cid, fit['formula'])
                    absolute = detail['nonrenal_clearance_bounds_l_h']
                    if key in bounds:
                        for a, b in zip(absolute, bounds[key]): near(a, b, 'Absolute CL bounds changed')
                    bounds[key] = absolute
        else:
            require(row['result'] == job['source'], 'Protected CL case changed')
        if unit['status'] == 'predicted':
            if job['replaceable']:
                require(not unit['selection']['target_values_used']
                        and unit['selection']['locked_at'] <= unit['prediction_locked_at'], 'Formula selection lock failed')
                request = sources.document(unit['request'])
                cl = [e for e in request['evidence'] if e['parameter'] == 'total_clearance']
                require(len(cl) == 1 and cl[0]['unit'].lower() == 'ml/min/kg', 'Target request CL unit changed')
                near(cl[0]['value'], unit['changes'][0]['replacement'], 'Target request CL prior')
                unchanged = deepcopy(request)
                unchanged.pop('evidence_ledger')
                unchanged.pop('partition_method')  # Each formula is selected on the same visible points.
                changed_cl = next(e for e in unchanged['evidence'] if e['parameter'] == 'total_clearance')
                for key in ('value', 'source', 'prediction_method', 'notes'):
                    changed_cl.pop(key, None)
                require(cid not in shared_requests or shared_requests[cid] == unchanged,
                        'Non-CL target request differs between methods: ' + cid)
                shared_requests[cid] = unchanged
            recalculated = scorer.score_case(cases[cid], unit)
            for endpoint in ENDPOINTS:
                near(endpoint_error(row['score'], endpoint), endpoint_error(recalculated, endpoint), 'Rescored ' + cid + '/' + endpoint)
                if endpoint != 'curve':
                    for field in ('value', 'reference'):
                        near(row['score']['endpoints'][endpoint][field], recalculated['endpoints'][endpoint][field],
                             'Rescored ' + cid + '/' + endpoint + '/' + field)
            for key in ('n', 'observed_points', 'zero_observations_not_log_scored'):
                require(row['score']['curve_metrics'].get(key) == recalculated['curve_metrics'].get(key), 'Curve scoring coverage changed')
            near(row['score']['curve_metrics']['ln_rmse'], recalculated['curve_metrics']['ln_rmse'], 'Curve RMSE')
        else:
            require(row['score'] is None, 'Failed attempt has a score')
            failures.append(dict(method=method, case_id=cid, status=unit['status'], reason=unit.get('reason', 'Unavailable')))
        errors = {e: endpoint_error(row['score'], e) for e in ENDPOINTS}
        values[(method, cid)] = errors
        direct_rows.append(dict(method=method, case_id=cid, cluster_id=row['cluster_id'],
                                replaceable=row['replaceable'], status=row['status'], errors=errors,
                                weak_endpoints=(row['score'] or {}).get('weak_endpoints', []),
                                reference_weak_endpoints=(row['score'] or {}).get('reference_weak_endpoints', []),
                                resources=unit.get('resources')))
    # CSV must expose every attempted output, including unavailable references/failures.
    with (RUN / 'endpoints.csv').open(newline='') as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_map = {(r['method'], r['case_id'], r['endpoint']): r for r in csv_rows}
    require(len(csv_rows) == len(csv_map) == 990 and set(csv_map) == {(m,c,e) for m,c in expected for e in ENDPOINTS},
            'Endpoint CSV does not retain all 990 attempted outputs')
    for (method, cid, endpoint), csv_row in csv_map.items():
        row = records[(method, cid)]
        require(csv_row['status'] == row['status'] and csv_row['replaceable'] == str(row['replaceable']), 'CSV status mismatch')
        near(float(csv_row['error']) if csv_row['error'] else None, values[(method,cid)][endpoint], 'CSV error')
        if endpoint != 'curve' and row['score'] is not None:
            ep = row['score']['endpoints'][endpoint]
            require(csv_row['unit'] == ep['unit'] and csv_row['error_metric'] == ep['error_metric'], 'CSV units')
            for field, key in [('prediction', 'value'), ('reference', 'reference')]:
                near(float(csv_row[field]) if csv_row[field] else None, ep[key], 'CSV scalar')
    summary = {}
    for scope, target_n in [('all_thirty', 30), ('changed_twenty_four', 24)]:
        ids = sorted(c for c,j in p['jobs'].items() if scope == 'all_thirty' or j['replaceable'])
        require(len(ids) == target_n, 'Scope count changed')
        summary[scope] = {}
        for method in METHODS:
            groups = {}
            for endpoint in ENDPOINTS:
                selected = {c: values[(method,c)][endpoint] for c in ids if values[(method,c)][endpoint] is not None}
                if endpoint == 'tmax':
                    actual = dict(n=len(selected), mae_h=math.fsum(selected.values())/len(selected) if selected else None)
                    stored = result['summary'][scope][method]['metrics'][endpoint]
                    require(actual['n'] == stored['n'], 'Tmax coverage changed'); near(actual['mae_h'], stored['mae_h'], 'Tmax MAE')
                else:
                    actual = clustered(selected, {c: cases[c]['cluster_id'] for c in selected})
                    actual['within_2fold'] = sum(v <= 2 for v in selected.values())
                    stored = result['summary'][scope][method]['metrics'][endpoint]
                    verify_statistics(actual, stored, scope+'/'+method+'/'+endpoint)
                    require(actual['within_2fold'] == stored['within_2fold'], 'Twofold count changed')
                groups[endpoint] = actual
            predicted = sum(records[(method,c)]['status'] == 'predicted' for c in ids)
            require(result['summary'][scope][method]['enrolled'] == target_n
                    and result['summary'][scope][method]['predicted'] == predicted, 'Published attempt denominator changed')
            summary[scope][method] = dict(enrolled=target_n, predicted=predicted, metrics=groups)
            if method != 'rf':
                pairs = {}
                for endpoint in ENDPOINTS:
                    if endpoint == 'tmax': continue
                    ratios = {c: values[(method,c)][endpoint]/values[('rf',c)][endpoint] for c in ids
                              if values[(method,c)][endpoint] is not None and values[('rf',c)][endpoint] is not None}
                    actual = clustered(ratios, {c: cases[c]['cluster_id'] for c in ratios})
                    actual.update(improved=sum(v < 1-1e-6 for v in ratios.values()),
                                  worsened=sum(v > 1+1e-6 for v in ratios.values()),
                                  tied=sum(abs(v-1) <= 1e-6 for v in ratios.values()))
                    stored = result['summary'][scope][method]['paired_to_rf'][endpoint]
                    verify_statistics(actual, stored, 'Paired '+scope+'/'+method+'/'+endpoint)
                    for key in ('improved', 'worsened', 'tied'): require(actual[key] == stored[key], 'Paired counts changed')
                    actual.update(attempted_pairs=target_n, incomplete_case_ids=[c for c in ids if c not in ratios], per_case_ratio=ratios)
                    pairs[endpoint] = actual
                summary[scope][method]['paired_to_rf'] = pairs
    reasons = Counter()
    for c in cases:
        score = records[('rf',c)]['score']
        if score:
            for endpoint in ENDPOINTS[6:]: reasons[score['endpoints'][endpoint]['reference_reason']] += 1
    coverage = {m: {e: dict(attempted=30,
                           model_available=sum(records[(m,c)]['score'] is not None and
                               (records[(m,c)]['score']['curve_metrics']['gmfe'] if e == 'curve' else
                                records[(m,c)]['score']['endpoints'][e]['value']) is not None for c in cases),
                           error_available=sum(values[(m,c)][e] is not None for c in cases))
                    for e in ENDPOINTS} for m in METHODS}
    return dict(summary=summary, records=direct_rows, failed_attempts=failures,
                failed_formula_attempts=fit_failures, common_reference_classes=list(COMMON),
                missing_reference_reasons=dict(reasons), endpoint_csv_rows=len(csv_rows),
                endpoint_coverage=coverage,
                actual_new_fits=sum(not r['reused'] for r in records.values()),
                retained_independent_cases=sum(not j['replaceable'] for j in p['jobs'].values()),
                full_endpoint_csv=str((RUN/'endpoints.csv').relative_to(ROOT)),
                all_case_scores_recomputed_from_locked_trajectories=True,
                all_scalar_errors_independently_recomputed=True,
                fixed_absolute_clearance_bounds_verified=True)


def table_text(audit):
    parameter, clinical = audit['parameter'], audit['clinical']
    lines = [r'\begingroup', r'\captionsetup[table]{name=Table}', r'\begin{table}[htbp]',
             r'\centering', r'\caption{CL gap-filler accuracy and propagation through the same five-point PBPK calibration.}',
             r'\label{tab:blueprint_cl_gap_comparison}', r'\fontsize{8}{10}\selectfont',
             r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.12}',
             r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
             r'\multicolumn{6}{@{}l}{\textbf{A. Common 100-drug CL holdout (' + str(parameter['n_test_scaffolds']) + r' scaffolds)}}\\',
             r'Predictor & $n$ & GMFE & $\leq2$-fold & $\log_{10}$ RMSE & Ratio to RF [95\% CI]\\', r'\midrule']
    for method in METHODS:
        m = parameter['metrics'][method]
        pair = parameter['paired_to_rf'].get(method)
        ratio = '--' if pair is None else f"{pair['estimate']:.3f} [{pair['ci95'][0]:.3f}, {pair['ci95'][1]:.3f}]"
        lines.append(f"{NAMES[method]} & {m['n']} & {m['gmfe']:.3f} & {m['within_2fold']*100:.0f}\\% & {m['log10_rmse']:.3f} & {ratio}" + r'\\')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\vspace{6pt}',
              r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr@{}}', r'\toprule',
              r'\multicolumn{6}{@{}l}{\textbf{B. Selected clinical endpoints after identical five-point calibration}}\\',
              r'Predictor / scope & $n$ & Curve & $C_{\max}$ & AUC$_{\rm window}$ & $t_{1/2}$\\', r'\midrule']
    for scope, description in [('all_thirty', 'All 30 drugs'), ('changed_twenty_four', '24 drugs with replaced CL')]:
        lines.append(r'\multicolumn{6}{@{}l}{\emph{' + description + r'}}\\')
        for method in METHODS:
            record = clinical['summary'][scope][method]; mm = record['metrics']
            cells = [NAMES[method], str(record['enrolled'])]
            for endpoint in ('curve', 'cmax', 'auc', 'terminal_half_life'):
                m = mm[endpoint]
                cells.append('--' if m['estimate'] is None else f"{m['estimate']:.3f}" + (f" ({m['n_cases']}/{record['enrolled']})" if m['n_cases'] != record['enrolled'] else ''))
            lines.append(' & '.join(cells) + r'\\')
        if scope == 'all_thirty': lines.append(r'\addlinespace[2pt]')
    lines += [r'\bottomrule', r'\end{tabular*}', r'\vspace{4pt}', r'\begin{minipage}{\linewidth}',
              r'\fontsize{7.5}{9.2}\selectfont\raggedright']
    failed = len(clinical['failed_attempts'])
    lines.append(r'\textit{Design.} Each predictor uses the same 1,006 human CL labels from 1,008 training structures; the 100 holdout drugs and all 30 clinical identities/scaffolds are excluded from training. RF uses the current recipe refitted under scikit-learn 1.6.1. These are newly calibrated comparison results, not Table~13 values or reused historical RF fits for the 24 changed cases. Six independently supported CL inputs are retained and their fits reused. All three priors lock before clinical scoring; only CL changes, with the same five development points, other inputs, PBPK structure, fit budget and visible-data repair.\par')
    lines.append(r'\textit{Predictors.} TabPFNv2 uses 494 Morgan bits selected by training-only variance plus six descriptors (500 features). Complementary CL is a fixed CL-only adaptation: equal log-scale combination of an ET/SVR/two-XGB base, its animal-augmented counterpart, and Tanimoto KRR. Additional animal CL labels make this a practical pipeline comparison, not an algorithm-only test.\par')
    lines.append(r'\textit{Scoring and coverage.} GMFE is the geometric mean of per-drug fold errors (curve: exponentiated mean absolute log ratio per drug). AUC uses actual observed windows; scalars derive from aggregate-curve NCA. Six common-reference classes are available: curve, AUC, $C_{\max}$, $T_{\max}$, $C_{\rm last}$ and $t_{1/2}$. All 11 outputs for all 90 method--drug attempts remain in \textnormal{endpoints.csv}; AUC$_\infty$, CL, $V_z$, MRT and $V_{ss}$ lack common clinical references and are not error-scored. $T_{\max}$ uses absolute hours. ' + f'Failed predictions: {failed}/90; unavailable values and failed formula attempts remain in the audit.' + r'\par')
    pairs = [clinical['summary']['changed_twenty_four'][m]['paired_to_rf']['curve'] for m in METHODS[1:]]
    max_shift = max(abs(s['estimate']-1) for s in pairs) * 100
    ci_low = min(s['ci95'][0] for s in pairs); ci_high = max(s['ci95'][1] for s in pairs)
    lines.append(r'\textit{Interpretation.} The 100-drug holdout was previously viewed and the 30-drug clinical cohort is retrospective. Paired 95\% CIs use 2,000 scaffold/study-cluster bootstrap draws (parameter/clinical), without multiplicity adjustment or significance claims. On the 24 changed cases, curve-error ratios to RF are 1.000 at reported precision; ' + f'the largest mean change is {max_shift:.5f}\\%, with both CIs within [{ci_low:.6f}, {ci_high:.6f}]. ' + r'Freely recalibrated CL with fixed absolute bounds absorbs the prior differences: lower parameter error does not produce a material downstream gain in this workflow.')
    lines += [r'\end{minipage}', r'\end{table}', r'\endgroup', '']
    return '\n'.join(lines)


def main():
    # One current table: once the no-calibration arm completes, report both
    # information regimes. Keep the original numerical audit functions reusable.
    direct_state = RUN / 'direct_prediction' / 'state.json'
    if direct_state.exists() and read(direct_state).get('status') == 'complete':
        from report_direct_cl_comparison import main as direct_main
        return direct_main()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate all frozen inputs and calculations without writing any file')
    args = parser.parse_args()
    sources = Sources()
    parameter, parameter_lock = parameter_audit(sources)
    clinical = clinical_audit(sources, parameter_lock)
    for name in ('parameter_protocol.json', 'parameter_metrics.json', 'parameter_validation.json',
                 'parameter_predictions_locked.json', 'clinical_inputs.json', 'protocol.json', 'state.json',
                 'results.json', 'endpoints.csv'):
        file = RUN / name
        sources.check(dict(path=str(file.relative_to(ROOT)), sha256=sha(file)))
    script = str(Path(__file__).resolve().relative_to(ROOT))
    sources.check(dict(path=script, sha256=sha(script)))
    audit = dict(version='cl-gap-publication-1', status='validated', parameter=parameter, clinical=clinical,
                 source_pins=sources.verified, methods=list(METHODS), bootstrap_draws=DRAWS, bootstrap_seed=SEED,
                 intervals_are_descriptive=True, multiplicity_adjusted=False,
                 no_training_refitting_or_simulation_performed=True,
                 validation='Source hashes, prediction locks, units, direct saved-trajectory rescoring, independent per-case aggregation and paired 2,000-cluster bootstrap agree.')
    latex = table_text(audit)
    audit['table'] = dict(path=str(TABLE.relative_to(ROOT)), sha256=hashlib.sha256(latex.encode()).hexdigest())
    if args.check_only:
        if TABLE.exists(): require(TABLE.read_text() == latex, 'Existing table differs from independently validated values')
        if (RUN / 'publication.json').exists():
            require(read(RUN / 'publication.json') == audit, 'Existing publication audit differs from current validation')
    else:
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        TABLE.write_text(latex)
        (RUN / 'publication.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='validated', check_only=args.check_only, verified_sources=len(sources.verified),
                          clinical_rows=90, endpoint_csv_rows=990, failed_attempts=len(clinical['failed_attempts']),
                          table=str(TABLE.relative_to(ROOT))), sort_keys=True))


if __name__ == '__main__':
    main()
